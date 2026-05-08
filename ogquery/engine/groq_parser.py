"""
Groq Query Parser  ─  v5
======================================
Responsibilities
----------------
1. Build a grounded context block (DataContextBuilder).
2. Call Groq LLM with context + query → receive raw JSON string.
3. Parse + validate the JSON → canonical dict with safe defaults.
4. Hand the validated dict to QueryPlanner → typed ExecutionPlan.

The LLM is ONLY a parser. No LLM call happens after step 2.
No raw dict ever escapes this module — only a typed ExecutionPlan.

Changes vs v3:
    - Default limit changed from 10 → 3 (system decides, not users)
    - Limit in system prompt updated to match
    - _validate: limit fallback = 3
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from groq import Groq

if TYPE_CHECKING:
    from ogquery.engine.data_context import DataContextBuilder
    from ogquery.engine.planner import ExecutionPlan, QueryPlanner
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Rule-based fallback  (used when Groq API is unavailable)
# ══════════════════════════════════════════════════════════════════════════════

_OBJECTIVE_WORDS: Dict[str, str] = {
    "cheapest": "minimize", "cheap": "minimize", "affordable": "minimize",
    "lowest": "minimize", "minimum": "minimize", "least expensive": "minimize",
    "most expensive": "maximize", "expensive": "maximize",
    "highest": "maximize", "maximum": "maximize", "priciest": "maximize",
    "top": "top_k", "best": "top_k",
    "all": "list", "list": "list",
    "count": "count", "how many": "count",
}

_NUMERIC_HINTS: Dict[str, tuple] = {
    "cheapest": ("price", "min"), "cheap": ("price", "min"),
    "affordable": ("price", "min"), "expensive": ("price", "max"),
    "priciest": ("price", "max"), "largest": ("size", "max"),
    "smallest": ("size", "min"), "biggest": ("area", "max"),
}


def _rule_based_dict(query: str, context_block: str) -> Dict[str, Any]:
    q = query.lower()

    real_columns: List[str] = []
    for line in context_block.splitlines():
        m = re.match(r"\s*-\s+(\S+)\s+\(", line)
        if m:
            real_columns.append(m.group(1).lower())

    objective = "list"
    target_col: Optional[str] = None
    for kw, obj in _OBJECTIVE_WORDS.items():
        if kw in q:
            objective = obj
            break

    for kw, (col_guess, _) in _NUMERIC_HINTS.items():
        if kw in q:
            target_col = next(
                (c for c in real_columns if col_guess in c), col_guess
            )
            break

    numeric_filters: Dict[str, Any] = {}
    if target_col:
        for pattern, bound_key in [
            (r"(?:under|below|less than)\s+([\d,]+)", "max"),
            (r"(?:above|over|more than|at least)\s+([\d,]+)", "min"),
        ]:
            m = re.search(pattern, q)
            if m:
                val = float(m.group(1).replace(",", ""))
                entry = numeric_filters.get(target_col, {})
                entry[bound_key] = val
                numeric_filters[target_col] = entry

    return {
        "filters": {},
        "numeric_filters": numeric_filters,
        "objective": objective,
        "target_column": target_col,
        "sort_by": target_col,
        "sort_ascending": objective == "minimize",
        "limit": 3,   # system default
        "raw_query": query,
        "_parser": "rule_based",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Groq system prompt
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are a schema-grounded query parser for a structured dataset query engine.

You will receive:
  1. A DATASET CONTEXT block with exact column names, types, sample rows,
     known categorical values, and numeric ranges.
  2. A natural language QUERY.

Rules:
  - Use ONLY column names from the DATASET CONTEXT.
  - Use ONLY filter values matching or close to known values.
  - Return ONLY valid JSON — no markdown, no explanation, no preamble.
  - "filters"         → { col: value }  for categorical/text columns.
  - "numeric_filters" → { col: {"min": float, "max": float} } for numeric cols.
  - "objective"       → "minimize" | "maximize" | "list" | "count" | "top_k"
  - "target_column"   → numeric column for objective, or null.
  - "sort_by"         → column to sort by, or null.
  - "sort_ascending"  → true = ascending (cheapest first).
  - "limit"           → default 3. Only increase if query explicitly asks for more.

Output schema (nothing else):
{
  "filters":         {"column_name": "value"},
  "numeric_filters": {"column_name": {"min": 0.0, "max": 9999.0}},
  "objective":       "minimize",
  "target_column":   "column_name",
  "sort_by":         "column_name",
  "sort_ascending":  true,
  "limit":           3
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# GroqParser
# ══════════════════════════════════════════════════════════════════════════════

class GroqParser:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("GROQ_API_KEY")
        self._client: Optional[Groq] = None
        if self._api_key:
            try:
                self._client = Groq(api_key=self._api_key)
            except Exception:
                self._client = None

    def parse(
        self,
        query: str,
        context_builder: "DataContextBuilder",
        planner: "QueryPlanner",
    ) -> "ExecutionPlan":
        context_block = context_builder.build()

        raw_dict: Dict[str, Any]
        if self._client is not None:
            try:
                raw_dict = self._call_groq(query, context_block)
            except Exception as exc:
                print(f"[GroqParser] API error: {exc} — using rule-based fallback.")
                raw_dict = _rule_based_dict(query, context_block)
        else:
            raw_dict = _rule_based_dict(query, context_block)

        validated = self._validate(raw_dict, query)
        return planner.plan(validated)

    def _call_groq(self, query: str, context_block: str) -> Dict[str, Any]:
        user_msg = f"{context_block}\nQUERY:\n\"{query}\""

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=512,
        )

        raw = response.choices[0].message.content.strip()
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        clean = re.sub(r"^```[a-z]*\n?", "", raw)
        clean = re.sub(r"\n?```$", "", clean)
        result = json.loads(clean)
        if not isinstance(result, dict):
            raise ValueError(f"LLM returned non-dict JSON: {type(result)}")
        return result

    import re

    @staticmethod
    def _validate(plan: Dict[str, Any], query: str) -> Dict[str, Any]:

        validated: Dict[str, Any] = {
            "filters": {},
            "numeric_filters": {},
            "objective": "list",
            "target_column": None,
            "sort_by": None,
            "sort_ascending": True,
            "limit": 3,   # ← system default (not 10)
            "raw_query": query,
            "_parser": plan.get("_parser", "groq"),
        }

        if isinstance(plan.get("filters"), dict):
            validated["filters"] = plan["filters"]

        if isinstance(plan.get("numeric_filters"), dict):
            nf: Dict[str, Any] = {}
            for col, bounds in plan["numeric_filters"].items():
                if not isinstance(bounds, dict):
                    continue
                cleaned: Dict[str, float] = {}
                for bound_key in ("min", "max"):
                    if bound_key in bounds:
                        try:
                            cleaned[bound_key] = float(bounds[bound_key])
                        except (TypeError, ValueError):
                            pass
                if cleaned:
                    nf[col] = cleaned
            validated["numeric_filters"] = nf

        if plan.get("objective") in {
            "minimize", "maximize", "list", "count", "top_k",
        }:
            validated["objective"] = plan["objective"]

        if plan.get("target_column") is not None:
            validated["target_column"] = str(plan["target_column"])

        if plan.get("sort_by") is not None:
            validated["sort_by"] = str(plan["sort_by"])

        if isinstance(plan.get("sort_ascending"), bool):
            validated["sort_ascending"] = plan["sort_ascending"]

        # Limit: try parser output, then query text, then default=3
        limit = None
        raw_limit = plan.get("limit")
        if raw_limit is not None:
            try:
                limit = int(str(raw_limit).strip())
            except (TypeError, ValueError):
                limit = None

        if limit is None:
            patterns = [
                r"\blimit\s+(\d+)",
                r"\btop\s+(\d+)",
                r"\bfirst\s+(\d+)",
                r"\bonly\s+(\d+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    try:
                        limit = int(match.group(1))
                        break
                    except ValueError:
                        pass

        if limit is None:
            limit = 3  # system default

        validated["limit"] = max(1, min(limit, 1000))
        return validated
