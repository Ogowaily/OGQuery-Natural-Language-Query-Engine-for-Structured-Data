"""
Schema Registry
Defines column types, behaviors, and routing strategy.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


NUMERIC_TYPES = {"int64", "float64", "int32", "float32"}
HIGH_CARDINALITY_THRESHOLD = 500   # unique values above this → exact match only
SEMANTIC_MAX_UNIQUE = 500           # above this → no embeddings


@dataclass
class ColumnMeta:
    name: str
    dtype: str                          # pandas dtype string
    col_type: str                       # "numeric" | "categorical" | "text" | "id"
    semantic_enabled: bool = False
    indexed: bool = False
    operations: List[str] = field(default_factory=list)
    unique_count: int = 0
    sample_values: List[Any] = field(default_factory=list)
    null_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class SchemaRegistry:
    """
    Analyzes a DataFrame and builds a column-level schema registry.
    """

    def __init__(self):
        self.columns: Dict[str, ColumnMeta] = {}
        self._dataset_name: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame, dataset_name: str = "dataset") -> None:
        """Ingest a DataFrame and classify every column."""
        self._dataset_name = dataset_name
        self.columns.clear()

        for col in df.columns:
            meta = self._classify_column(df, col)
            self.columns[col] = meta

    def get(self, col: str) -> Optional[ColumnMeta]:
        return self.columns.get(col)

    def numeric_columns(self) -> List[str]:
        return [c for c, m in self.columns.items() if m.col_type == "numeric"]

    def semantic_columns(self) -> List[str]:
        return [c for c, m in self.columns.items() if m.semantic_enabled]

    def categorical_columns(self) -> List[str]:
        return [c for c, m in self.columns.items() if m.col_type == "categorical"]

    def summary(self) -> dict:
        return {
            "dataset": self._dataset_name,
            "total_columns": len(self.columns),
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
        }

    def to_llm_prompt_schema(self) -> str:
        """Compact schema description for Groq query parser."""
        lines = [f"Dataset: {self._dataset_name}", "Columns:"]
        for name, m in self.columns.items():
            ops = ", ".join(m.operations) if m.operations else "filter"
            lines.append(
                f"  - {name} ({m.col_type}) | semantic={m.semantic_enabled} | ops=[{ops}] | sample={m.sample_values[:3]}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_column(self, df: pd.DataFrame, col: str) -> ColumnMeta:
        series = df[col]
        dtype_str = str(series.dtype)
        unique_count = int(series.nunique(dropna=True))
        null_pct = float(series.isna().mean())
        sample = [v for v in series.dropna().unique()[:5].tolist()]

        # ── Numeric ──────────────────────────────────────────────────
        if dtype_str in NUMERIC_TYPES or pd.api.types.is_numeric_dtype(series):
            return ColumnMeta(
                name=col,
                dtype=dtype_str,
                col_type="numeric",
                semantic_enabled=False,
                indexed=True,
                operations=["min", "max", "avg", "sum", "range", "sort"],
                unique_count=unique_count,
                sample_values=sample,
                null_pct=null_pct,
            )

        # ── Categorical / Text ────────────────────────────────────────
        use_semantic = unique_count <= SEMANTIC_MAX_UNIQUE
        col_type = "categorical" if unique_count <= HIGH_CARDINALITY_THRESHOLD else "text"

        return ColumnMeta(
            name=col,
            dtype=dtype_str,
            col_type=col_type,
            semantic_enabled=use_semantic,
            indexed=True,
            operations=["eq", "contains", "in", "semantic_match"],
            unique_count=unique_count,
            sample_values=[str(s) for s in sample],
            null_pct=null_pct,
        )
