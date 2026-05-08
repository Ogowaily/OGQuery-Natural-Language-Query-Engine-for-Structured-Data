"""
Mapping Store
Fast exact-match index: value → set of row_ids.
Handles categorical columns with potentially high cardinality.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Set, Optional
import pandas as pd


class MappingStore:
    """
    Inverted index: { column → { normalized_value → {row_id, ...} } }

    Built during ingestion, queried during execution.
    Zero LLM calls — purely deterministic.
    """

    def __init__(self):
        # col → value → frozenset of row_ids
        self._index: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame, columns: List[str]) -> None:
        """Build the inverted index for the specified columns."""
        self._index.clear()
        for col in columns:
            if col not in df.columns:
                continue
            for row_id, val in enumerate(df[col]):
                if pd.isna(val):
                    continue
                normalized = self._normalize(str(val))
                self._index[col][normalized].add(row_id)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def exact_lookup(self, col: str, value: str) -> Set[int]:
        """Exact match after normalization."""
        normalized = self._normalize(value)
        return self._index.get(col, {}).get(normalized, set())

    def prefix_lookup(self, col: str, prefix: str) -> Set[int]:
        """All row_ids where the value starts with prefix."""
        prefix = self._normalize(prefix)
        result: Set[int] = set()
        for val, ids in self._index.get(col, {}).items():
            if val.startswith(prefix):
                result.update(ids)
        return result

    def contains_lookup(self, col: str, substring: str) -> Set[int]:
        """All row_ids where the value contains substring."""
        substring = self._normalize(substring)
        result: Set[int] = set()
        for val, ids in self._index.get(col, {}).items():
            if substring in val:
                result.update(ids)
        return result

    def all_values(self, col: str) -> List[str]:
        """Return all indexed values for a column."""
        return list(self._index.get(col, {}).keys())

    def column_indexed(self, col: str) -> bool:
        return col in self._index

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(value: str) -> str:
        return value.strip().lower()
