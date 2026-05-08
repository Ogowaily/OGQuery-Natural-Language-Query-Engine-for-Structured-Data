"""
Answer Generator  ─  v5 (structured output)
=============================================
Design principle: output must feel like a DATA INTELLIGENCE RESULT, not a chatbot response.

Output format:
    1. Short key summary (1-3 lines)
    2. Key observations from the data

Rules:
    - NEVER use chatbot-style prose
    - NEVER hallucinate counts or rows
    - Always trust metadata (total_matches, returned)
    - For ≤3 results: use deterministic fallback (no LLM cost)
    - For >3 results: use LLM with strict structured prompt
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class AnswerGenerator:
    """
    Structured dataset summary generator.

    Output style: analytics dashboard / search engine result page.
    NOT a conversational assistant.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = Groq(
            api_key=api_key or os.getenv("GROQ_API_KEY")
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Smart summary prompt (used for ALL result counts)
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        results: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> str:
        total = int(metadata.get("total_matches", len(results)))
        returned = int(metadata.get("returned", len(results)))
        sample = json.dumps(results[:5], indent=2, default=str)

        return f"""You are a DATA INTELLIGENCE ENGINE producing executive-level insight summaries.

OUTPUT RULES (STRICT):
- 2 sentences ONLY — no more, no less
- Sentence 1: State the total count and what was queried, using natural phrasing.
              Example: "Found 917 ROSCOSMOS missions matching the query."
              Example: "3 NASA missions were found involving SpaceX partnerships."
- Sentence 2: Describe the most notable pattern, distribution, or characteristic visible in the returned rows.
              Example: "The returned results include ongoing, successful, and failed missions launched mainly from Baikonur and Plesetsk."
              Example: "Most missions used Saturn V launch vehicles and occurred between 1965–1972."

HARD CONSTRAINTS:
- TOTAL MATCHES = {total} → Sentence 1 MUST use this number
- RETURNED ROWS = {returned} → Sentence 2 may reference this if relevant
- Use real column values from the data sample — do NOT invent anything
- Do NOT start with "I", "Here", or "Based on"
- Do NOT add bullet points, headers, or a third sentence

QUERY: {query}

DATA ({returned} rows):
{sample}

OUTPUT (exactly 2 sentences):"""

    # ------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        results: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:

        metadata = metadata or {}

        if not results:
            return f"No results found for: {query}"

        prompt = self._build_prompt(query, results, metadata)

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise data analysis engine. "
                            "Output is 2 structured sentences. "
                            "No conversational tone. No chatbot language."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=120,
            )

            content = response.choices[0].message.content
            if not content or not content.strip():
                return self._emergency_fallback(query, results, metadata)

            return content.strip()

        except Exception:
            return self._emergency_fallback(query, results, metadata)

    # ------------------------------------------------------------------
    # Emergency fallback (only if LLM completely fails)
    # ------------------------------------------------------------------

    def _emergency_fallback(
        self,
        query: str,
        results: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> str:
        total = int(metadata.get("total_matches", len(results)))
        returned = int(metadata.get("returned", len(results)))
        # Build a minimal but meaningful sentence from first row's values
        first = results[0] if results else {}
        values = [str(v) for k, v in list(first.items())[:3] if not k.startswith("_")]
        preview = ", ".join(values)
        word = "match" if total == 1 else "matches"
        return f"Found {total} {word} for the query. The top result includes: {preview}."
