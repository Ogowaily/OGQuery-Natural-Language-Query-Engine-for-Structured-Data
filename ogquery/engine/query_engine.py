"""
Query Engine  ─  v5 (dataset-centric, structured output)
==========================================================
Execution flow
--------------
  1. DataContextBuilder.build()     → grounded context block
  2. GroqParser.parse()             → typed ExecutionPlan  (LLM ends here)
  3. QueryEngine._execute(plan)     → hybrid filter execution (zero LLM)
     a. SemanticFilters  → FAISS candidate row_id sets
     b. ExactFilters     → inverted-index row_id sets
     c. Intersect all sets
     d. NumericRangeFilters + sort via NumericStore
  4. RowStore.fetch_rows()          → final result dicts

Design changes vs v4:
    - Default limit = 3 (not 10). Engine auto-scales based on query intent.
    - Adaptive limit detection: "compare"/"list all"/"show more" → higher limits.
    - Answer generator now produces a SHORT structured summary (not chatbot prose).
    - No user-facing limit parameter — limit is a system-level decision only.
    - Output includes ranked results + execution insight for structured UI rendering.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ogquery.engine.answer_generator import AnswerGenerator
from ogquery.engine.data_context import DataContextBuilder
from ogquery.engine.groq_parser import GroqParser
from ogquery.engine.planner import (
    ExecutionPlan,
    ExactFilter,
    NumericRangeFilter,
    QueryPlanner,
    SemanticFilter,
)
from ogquery.embeddings.embedder import Embedder
from ogquery.schema.registry import SchemaRegistry
from ogquery.storage.mapping_store import MappingStore
from ogquery.storage.numeric_store import NumericStore
from ogquery.storage.row_store import RowStore
from ogquery.storage.semantic_store import SemanticStore


SEMANTIC_THRESHOLD = 0.6

# ── Adaptive limit policy ─────────────────────────────────────────────────────
DEFAULT_LIMIT = 3

_EXPANSION_PATTERNS = [
    (re.compile(r"\b(compare|comparison|vs\.?|versus)\b", re.I), 5),
    (re.compile(r"\b(list all|show all|all results|every)\b", re.I), 10),
    (re.compile(r"\b(show more|more results|more examples|what else)\b", re.I), 10),
    (re.compile(r"\b(top\s+(\d+))\b", re.I), None),      # extract number
    (re.compile(r"\b(first\s+(\d+))\b", re.I), None),
]

_AGGREGATION_PATTERNS = re.compile(
    r"\b(count|how many|total|sum|average|avg|minimum|maximum|overall)\b", re.I
)


def _resolve_adaptive_limit(query: str, plan_limit: int) -> int:
    """
    Determine result limit from query intent.
    Users never set this — it's a system decision.

    Priority:
      1. Explicit number in query ("top 10", "first 5")
      2. Expansion intent pattern
      3. Aggregation intent → full dataset (planner handles via objective)
      4. Default = 3
    """
    # Check for explicit number first
    for pattern, fixed_limit in _EXPANSION_PATTERNS:
        m = pattern.search(query)
        if m:
            if fixed_limit is None:
                # Extract number from group 2
                try:
                    return max(1, min(int(m.group(2)), 100))
                except (IndexError, ValueError):
                    pass
            else:
                return fixed_limit

    # Aggregation queries return full set (engine will summarize)
    if _AGGREGATION_PATTERNS.search(query):
        return 50  # capped — AnswerGenerator will summarize, not list all

    # Fall back to planner's parsed limit only if > DEFAULT_LIMIT
    # (LLM sometimes over-estimates; we trust the query intent more)
    if plan_limit > DEFAULT_LIMIT:
        return plan_limit

    return DEFAULT_LIMIT


class QueryEngine:
    """
    Top-level query orchestrator.

    After __init__ all stores are read-only during query execution.
    Thread-safe for concurrent reads (SQLite check_same_thread=False).
    """

    def __init__(
        self,
        row_store: RowStore,
        semantic_store: SemanticStore,
        numeric_store: NumericStore,
        mapping_store: MappingStore,
        registry: SchemaRegistry,
        embedder: Embedder,
        parser: GroqParser,
        context_builder: DataContextBuilder,
    ) -> None:
        self._row_store = row_store
        self._semantic_store = semantic_store
        self._numeric_store = numeric_store
        self._mapping_store = mapping_store
        self._registry = registry
        self._embedder = embedder
        self._parser = parser
        self._context_builder = context_builder
        self._planner = QueryPlanner(registry)
        self._answer_generator = AnswerGenerator()

    # ------------------------------------------------------------------
    # Public properties (required by FastAPI /schema and /context)
    # ------------------------------------------------------------------

    @property
    def registry(self) -> SchemaRegistry:
        return self._registry

    @property
    def context_builder(self) -> DataContextBuilder:
        return self._context_builder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, natural_query: str) -> Dict[str, Any]:
        """
        Execute a natural language query end-to-end.

        Returns a structured result dict:
          - results: ranked list of matching rows
          - execution_plan: what filters/sorts were applied
          - execution_trace: step-by-step trace for debugging
          - answer: short structured summary (NOT chatbot prose)
          - total_matches, returned, elapsed_ms

        Limit is determined adaptively from query intent — never user-controlled.
        """
        t0 = time.perf_counter()

        # ── Step 1-2: Parse → typed ExecutionPlan (only LLM step) ─────
        plan: ExecutionPlan = self._parser.parse(
            natural_query,
            self._context_builder,
            self._planner,
        )

        # ── Adaptive limit resolution ───────────────────────────────────
        effective_limit = _resolve_adaptive_limit(natural_query, plan.limit)

        # ── Step 3: Execute hybrid filters (zero LLM) ──────────────────
        result_ids, trace = self._execute(plan)

        # ── Step 4: Score + rank candidates, then fetch top rows ──────
        all_ids = list(result_ids)
        scored = self._score_results(all_ids, plan)
        # scored is [(relevance_score, row_id), ...] descending
        top_scored = scored[:effective_limit]
        limited_ids = [rid for _, rid in top_scored]
        limited_rows = self._row_store.fetch_rows(limited_ids)

        # Attach relevance score to each row for the caller
        score_map = {rid: round(score, 4) for score, rid in top_scored}
        for row, rid in zip(limited_rows, limited_ids):
            row["_relevance_score"] = score_map.get(rid, 0.0)

        # ── Step 5: Generate structured summary ────────────────────────
        summary = self._answer_generator.generate(
            natural_query,
            limited_rows,
            metadata={
                "total_matches": len(result_ids),
                "returned": len(limited_rows),
                "limit": effective_limit,
            },
        ) if limited_rows else "No results found."

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "query": natural_query,
            "execution_plan": {
                "objective":        plan.objective,
                "limit":            effective_limit,
                "sort_by":          plan.sort_by,
                "sort_ascending":   plan.sort_ascending,
                "target_column":    plan.target_column,
                "exact_filters":    [
                    {"column": f.column, "value": f.value}
                    for f in plan.exact_filters
                ],
                "semantic_filters": [
                    {"column": f.column, "value": f.value}
                    for f in plan.semantic_filters
                ],
                "numeric_filters":  [
                    {"column": f.column, "min": f.min, "max": f.max}
                    for f in plan.numeric_filters
                ],
            },
            "execution_trace":  trace,
            "total_matches":    len(result_ids),
            "returned":         len(limited_rows),
            "results":          limited_rows,
            "elapsed_ms":       elapsed_ms,
            "parser_used":      plan.parser_used,
            "answer":           summary,
        }

    # ------------------------------------------------------------------
    # Relevance scoring
    # ------------------------------------------------------------------

    def _score_results(
        self,
        row_ids: List[int],
        plan: ExecutionPlan,
    ) -> List[Tuple[float, int]]:
        """
        Score every candidate row_id and return sorted list
        [(score_desc, row_id), ...].

        Scoring breakdown (all components are additive, max ≈ 1.0):
          - Semantic match bonus   (+0.40 per filter, capped at 0.40)
          - Exact match bonus      (+0.30 per filter, capped at 0.30)
          - Numeric sort position  (+0.30, decays linearly with rank)

        Rows that pass more filters score higher.
        When a sort column exists, position in the sorted order is
        used as a tiebreaker within the numeric component.
        """
        if not row_ids:
            return []

        id_set = set(row_ids)
        scores: Dict[int, float] = {rid: 0.0 for rid in row_ids}

        # ── Semantic component ────────────────────────────────────────
        # Distribute score proportional to cosine similarity of the
        # matched value, shared across all rows that carry that value.
        sem_weight = 0.40 / max(len(plan.semantic_filters), 1)
        for f in plan.semantic_filters:
            if not self._semantic_store.column_indexed(f.column):
                continue
            query_vec = self._embedder.embed(f.value)
            matches = self._semantic_store.search(
                f.column, query_vec, top_k=50, threshold=0.0,
            )
            max_score = matches[0][1] if matches else 1.0
            for _val, sim, matched_ids in matches:
                normalised = sim / (max_score + 1e-9)
                for rid in matched_ids:
                    if rid in id_set:
                        scores[rid] += sem_weight * normalised

        # ── Exact match component ─────────────────────────────────────
        exact_weight = 0.30 / max(len(plan.exact_filters), 1)
        for f in plan.exact_filters:
            matched = self._mapping_store.exact_lookup(f.column, f.value)
            if not matched:
                matched = self._mapping_store.contains_lookup(f.column, f.value)
            for rid in matched:
                if rid in id_set:
                    scores[rid] += exact_weight

        # ── Numeric sort position component ──────────────────────────
        # The row at the "best" end of the sort gets +0.30, decaying
        # linearly to 0.0 at the worst end.
        if plan.sort_by and self._numeric_store.column_indexed(plan.sort_by):
            ordered = self._numeric_store.top_k(
                plan.sort_by,
                k=len(row_ids),
                ascending=plan.sort_ascending,
                row_ids=id_set,
            )
            n = len(ordered)
            for rank, (_, rid) in enumerate(ordered):
                decay = 1.0 - (rank / n) if n > 1 else 1.0
                scores[rid] += 0.30 * decay

        # ── Sort descending by score, stable tiebreak by row_id ───────
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(score, rid) for rid, score in ranked]

    # ------------------------------------------------------------------
    # Internal execution  (deterministic — no LLM beyond this point)
    # ------------------------------------------------------------------

    def _execute(
        self, plan: ExecutionPlan
    ) -> Tuple[List[int], List[str]]:
        """
        Run all filters in the plan and return an ordered list of row_ids
        together with a human-readable execution trace.

        Return type is List[int] (not Set) so that sort order is preserved
        when NumericStore.top_k() has already determined the ranking.
        """
        trace: List[str] = []
        active: Optional[Set[int]] = None

        active = self._apply_semantic_filters(plan.semantic_filters, plan.exact_filters, active, trace)
        active = self._apply_exact_filters(plan.exact_filters, active, trace)
        active = self._apply_numeric_filters(plan.numeric_filters, active, trace)

        if active is None:
            all_ids = set(self._row_store.fetch_all_ids())
            trace.append(f"[NO FILTER] returning all {len(all_ids)} rows")
            active = all_ids

        if plan.sort_by and self._numeric_store.column_indexed(plan.sort_by):
            top = self._numeric_store.top_k(
                plan.sort_by,
                k=plan.limit,
                ascending=plan.sort_ascending,
                row_ids=active,
            )
            ordered_ids: List[int] = [rid for _, rid in top]
            direction = "ASC" if plan.sort_ascending else "DESC"
            trace.append(
                f"[SORT] '{plan.sort_by}' {direction} "
                f"→ top {len(ordered_ids)} of {len(active)} rows"
            )
            return ordered_ids, trace

        return list(active), trace

    # ------------------------------------------------------------------
    # Filter execution helpers
    # ------------------------------------------------------------------

    def _apply_semantic_filters(
        self,
        filters: List[SemanticFilter],
        exact_filters: List[ExactFilter],
        active: Optional[Set[int]],
        trace: List[str],
    ) -> Optional[Set[int]]:

        exact_pairs = {
            (f.column.lower(), str(f.value).lower())
            for f in exact_filters
        }

        for f in filters:
            pair = (f.column.lower(), str(f.value).lower())
            if pair in exact_pairs:
                trace.append(
                    f"[SEMANTIC] skipped duplicate exact filter "
                    f"'{f.column}'='{f.value}'"
                )
                continue

            if not self._semantic_store.column_indexed(f.column):
                trace.append(f"[SEMANTIC] '{f.column}' not indexed")
                continue

            query_vec = self._embedder.embed(f.value)
            matches = self._semantic_store.search(
                f.column, query_vec, top_k=20, threshold=SEMANTIC_THRESHOLD,
            )

            ids: Set[int] = set()
            if matches:
                for _val, _score, row_ids in matches:
                    ids.update(row_ids)
                best_val, best_score, _ = matches[0]
                trace.append(
                    f"[SEMANTIC] '{f.column}'='{f.value}' "
                    f"→ top='{best_val}' (score={best_score:.2f}) rows={len(ids)}"
                )
            else:
                ids = self._mapping_store.contains_lookup(f.column, f.value)
                trace.append(
                    f"[SEMANTIC→CONTAINS] '{f.column}'='{f.value}' → {len(ids)} rows"
                )

            active = ids if active is None else (active & ids)

        return active

    def _apply_exact_filters(
        self,
        filters: List[ExactFilter],
        active: Optional[Set[int]],
        trace: List[str],
    ) -> Optional[Set[int]]:
        for f in filters:
            ids = self._mapping_store.exact_lookup(f.column, f.value)
            if not ids:
                ids = self._mapping_store.contains_lookup(f.column, f.value)
            trace.append(f"[EXACT] '{f.column}'='{f.value}' → {len(ids)} rows")
            active = ids if active is None else (active & ids)
        return active

    def _apply_numeric_filters(
        self,
        filters: List[NumericRangeFilter],
        active: Optional[Set[int]],
        trace: List[str],
    ) -> Optional[Set[int]]:
        for f in filters:
            ids = self._numeric_store.range_query(f.column, f.lo, f.hi, active)
            trace.append(
                f"[RANGE] '{f.column}' ∈ [{f.lo}, {f.hi}] → {len(ids)} rows"
            )
            active = ids
        return active
