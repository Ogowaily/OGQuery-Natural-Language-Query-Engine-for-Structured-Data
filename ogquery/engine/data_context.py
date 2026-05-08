 
from typing import Any, Optional, Dict, List


class DataContextBuilder:
    """
    Groq-ready dataset context builder.
    Provides schema + samples + API helpers.
    """

    def __init__(self, df, registry: Optional[Any] = None):
        self.df = df
        self.registry = registry

    # ------------------------------------------------------------------
    # CORE BUILD
    # ------------------------------------------------------------------

    def build(self) -> str:
        schema_lines: List[str] = []

        # Determine column source: prefer live DataFrame, fall back to registry
        if self.df is not None:
            columns = list(self.df.columns)
        elif self.registry and hasattr(self.registry, "columns") and self.registry.columns:
            columns = list(self.registry.columns.keys())
        else:
            raise ValueError(
                "DataContextBuilder has no DataFrame and no registry columns. "
                "Upload a dataset before querying."
            )

        for col in columns:
            # dtype from DataFrame if available, else from registry meta
            if self.df is not None:
                dtype = str(self.df[col].dtype)
            else:
                dtype = "unknown"

            meta = self.registry.columns[col] if self.registry and col in self.registry.columns else None

            col_type = getattr(meta, "col_type", dtype)
            semantic = getattr(meta, "semantic_enabled", False)
            ops = getattr(meta, "operations", [])
            sample_values = getattr(meta, "sample_values", [])

            ops_str = ",".join(ops) if isinstance(ops, list) else "filter"
            sample_str = sample_values[:3] if isinstance(sample_values, list) else []

            schema_lines.append(
                f"- {col} ({col_type}) | semantic={semantic} | ops=[{ops_str}] | sample={sample_str}"
            )

        return f"""
DATASET CONTEXT:

COLUMNS:
{chr(10).join(schema_lines)}

SAMPLE ROWS:
{self.get_sample_rows()}
""".strip()

    # ------------------------------------------------------------------
    # FIX 1: Missing method (your error)
    # ------------------------------------------------------------------

    def get_sample_rows(self) -> List[Dict]:
        """
        Returns top sample rows for ingestion logging + Groq grounding.
        Falls back to registry sample_values when DataFrame is not available
        (e.g. after a server restart restoring a persisted dataset).
        """
        if self.df is not None:
            return self.df.head(3).to_dict(orient="records")

        # Reconstruct sample rows from registry metadata
        if not self.registry or not hasattr(self.registry, "columns"):
            return []

        columns = list(self.registry.columns.keys())
        if not columns:
            return []

        # Zip first 3 sample_values across all columns into row dicts
        samples_per_col = {
            col: (getattr(self.registry.columns[col], "sample_values", []) or [])[:3]
            for col in columns
        }
        max_rows = max((len(v) for v in samples_per_col.values()), default=0)
        rows: List[Dict] = []
        for i in range(max_rows):
            row = {
                col: (samples_per_col[col][i] if i < len(samples_per_col[col]) else None)
                for col in columns
            }
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Optional helpers (useful later)
    # ------------------------------------------------------------------

    def get_column_names(self) -> List[str]:
        if self.df is not None:
            return list(self.df.columns)
        if self.registry and hasattr(self.registry, "columns"):
            return list(self.registry.columns.keys())
        return []

    def get_schema_summary(self) -> Dict[str, Any]:
        if not self.registry:
            return {}

        return self.registry.summary() if hasattr(self.registry, "summary") else {}