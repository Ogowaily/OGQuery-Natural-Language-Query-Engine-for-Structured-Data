"""
Numeric Store
Sorted index for numeric columns enabling O(log n) range queries.
All operations are deterministic — no LLM involved.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
import bisect


class NumericStore:
    """
    Maintains a sorted array of (value, row_id) per numeric column.
    Supports:  min, max, range, percentile, top-k
    """

    def __init__(self):
        # col → sorted list of (value, row_id)
        self._sorted: Dict[str, List[Tuple[float, int]]] = {}
        # col → numpy array of values (parallel to _sorted)
        self._values: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame, columns: List[str]) -> None:
        """Build sorted index for each numeric column."""
        self._sorted.clear()
        self._values.clear()

        for col in columns:
            if col not in df.columns:
                continue
            pairs = []
            for row_id, val in enumerate(df[col]):
                if pd.isna(val):
                    continue
                try:
                    pairs.append((float(val), row_id))
                except (ValueError, TypeError):
                    continue
            pairs.sort(key=lambda x: x[0])
            self._sorted[col] = pairs
            self._values[col] = np.array([p[0] for p in pairs])

    # ------------------------------------------------------------------
    # Aggregation (over full column or filtered row_ids)
    # ------------------------------------------------------------------

    def min(self, col: str, row_ids: Optional[Set[int]] = None) -> Optional[float]:
        return self._agg(col, row_ids, "min")

    def max(self, col: str, row_ids: Optional[Set[int]] = None) -> Optional[float]:
        return self._agg(col, row_ids, "max")

    def avg(self, col: str, row_ids: Optional[Set[int]] = None) -> Optional[float]:
        return self._agg(col, row_ids, "avg")

    def sum(self, col: str, row_ids: Optional[Set[int]] = None) -> Optional[float]:
        return self._agg(col, row_ids, "sum")

    def _agg(self, col: str, row_ids: Optional[Set[int]], op: str) -> Optional[float]:
        pairs = self._sorted.get(col, [])
        if not pairs:
            return None
        if row_ids is not None:
            values = [v for v, rid in pairs if rid in row_ids]
        else:
            values = [v for v, _ in pairs]
        if not values:
            return None
        arr = np.array(values)
        return float({"min": arr.min, "max": arr.max, "avg": arr.mean, "sum": arr.sum}[op]())

    # ------------------------------------------------------------------
    # Range query → row_ids
    # ------------------------------------------------------------------

    def range_query(
        self, col: str, lo: float, hi: float, row_ids: Optional[Set[int]] = None
    ) -> Set[int]:
        """Return row_ids where lo ≤ value ≤ hi."""
        pairs = self._sorted.get(col, [])
        vals = self._values.get(col, np.array([]))
        if len(vals) == 0:
            return set()

        left = bisect.bisect_left(vals, lo)
        right = bisect.bisect_right(vals, hi)

        result = set()
        for v, rid in pairs[left:right]:
            if row_ids is None or rid in row_ids:
                result.add(rid)
        return result

    # ------------------------------------------------------------------
    # Top-K (ascending = cheapest / smallest)
    # ------------------------------------------------------------------

    def top_k(
        self,
        col: str,
        k: int = 10,
        ascending: bool = True,
        row_ids: Optional[Set[int]] = None,
    ) -> List[Tuple[float, int]]:
        """Return up to k (value, row_id) pairs."""
        pairs = self._sorted.get(col, [])
        if row_ids is not None:
            pairs = [(v, rid) for v, rid in pairs if rid in row_ids]
        if not ascending:
            pairs = list(reversed(pairs))
        return pairs[:k]

    def column_indexed(self, col: str) -> bool:
        return col in self._sorted
