
"""
Dataset Ingestion Pipeline  ─  v4
====================================
Offline build phase: one-time execution that constructs all indexes
and wires a ready-to-use QueryEngine.

Pipeline order:
    1. Load DataFrame
    2. Schema analysis & column classification
    3. Row store (SQLite)
    4. Mapping store (inverted index)
    5. Numeric store (sorted index)
    6. Semantic store (FAISS, column-level embeddings)
    7. DataContextBuilder  ← schema + sample rows for Groq grounding
    8. Wire QueryEngine

BUG FIXES vs v3:
    - Removed stray `print(DataIngestionPipeline.__init__.__code__.co_varnames)`
      that executed on every import.
    - `DataIngestionPipeline` now always accepts `parser` in __init__
      so `main.py` and `helpers.py` both pass consistently.
    - `engine` property now returns the internally built engine instead of
      requiring callers to re-build it (avoids duplicate QueryEngine instances).
"""

from __future__ import annotations

from collections import defaultdict
import os
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from ogquery.engine.data_context import DataContextBuilder
from ogquery.engine.groq_parser import GroqParser
from ogquery.engine.planner import QueryPlanner
from ogquery.engine.query_engine import QueryEngine
from ogquery.embeddings.embedder import Embedder
from ogquery.schema.registry import SchemaRegistry
from ogquery.storage.mapping_store import MappingStore
from ogquery.storage.numeric_store import NumericStore
from ogquery.storage.row_store import RowStore
from ogquery.storage.semantic_store import SemanticStore
from ogquery.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestionPipeline:
    """
    Orchestrates full ingestion of a structured dataset.
    Exposes a ready-to-use QueryEngine after ingestion completes.
    """

    def __init__(
        self,
        row_store: RowStore,
        semantic_store: SemanticStore,
        numeric_store: NumericStore,
        mapping_store: MappingStore,
        registry: SchemaRegistry,
        embedder: Embedder,
        parser: GroqParser,          # required — no default to prevent silent misconfig
    ) -> None:
        self.row_store = row_store
        self.semantic_store = semantic_store
        self.numeric_store = numeric_store
        self.mapping_store = mapping_store
        self.registry = registry
        self.embedder = embedder
        self.parser = parser

        self.context_builder: Optional[DataContextBuilder] = None
        self._engine: Optional[QueryEngine] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, file_path: str, table_name: Optional[str] = None) -> dict:
        """
        Full ingestion pipeline for a CSV or Excel file.

        Returns an ingestion stats dict.
        After this call, self.engine holds a ready QueryEngine.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Sanitise table name: spaces → underscores, lowercase
        name = table_name or path.stem.replace(" ", "_").lower()

        # ── 1. Load ────────────────────────────────────────────────────
        logger.info(f"Loading {path.suffix}: {path.name}")
        df = self._load(path)
        logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")

        # ── 2. Schema Analysis ─────────────────────────────────────────
        logger.info("Analyzing schema...")
        self.registry.analyze(df, dataset_name=name)
        schema_summary = self.registry.summary()
        logger.info(f"Schema: {len(schema_summary['columns'])} columns classified")

        # ── 3. Row Store ───────────────────────────────────────────────
        logger.info("Building row store (SQLite)...")
        self.row_store.connect()
        self.row_store.ingest(df, table_name=name)
        logger.info(f"Row store: {self.row_store.total_rows()} rows")

        # ── 4. Mapping Store ───────────────────────────────────────────
        non_numeric = [
            col for col, meta in self.registry.columns.items()
            if meta.col_type != "numeric"
        ]
        logger.info(f"Building mapping index ({len(non_numeric)} columns)...")
        self.mapping_store.build(df, columns=non_numeric)

        # ── 5. Numeric Store ───────────────────────────────────────────
        numeric_cols = self.registry.numeric_columns()
        logger.info(f"Building numeric index ({len(numeric_cols)} columns)...")
        self.numeric_store.build(df, columns=numeric_cols)

        # ── 6. Semantic Store ──────────────────────────────────────────
        semantic_cols = self.registry.semantic_columns()
        logger.info(f"Building FAISS index ({len(semantic_cols)} columns)...")
        for col in semantic_cols:
            value_to_rowids: dict = defaultdict(set)
            for row_id, val in enumerate(df[col]):
                if pd.isna(val):
                    continue
                normalized = str(val).strip().lower()
                value_to_rowids[normalized].add(row_id)

            unique_values = list(value_to_rowids.keys())
            logger.info(
                f"  Embedding {len(unique_values)} unique values for '{col}'"
            )
            embeddings = self.embedder.embed_batch(unique_values)
            self.semantic_store.build_column(col, value_to_rowids, embeddings)

        # ── 7. Data Context Builder ────────────────────────────────────
        logger.info("Building DataContextBuilder (grounding layer)...")
        self.context_builder = DataContextBuilder(registry=self.registry, df=df)
        preview = self.context_builder.build()
        logger.info(
            f"Context block ready ({len(preview)} chars, "
            f"{len(self.context_builder.get_sample_rows())} sample rows)"
        )

        # ── 8. Wire QueryEngine ────────────────────────────────────────
        # Single engine built here; callers should use self.engine property.
        self._engine = QueryEngine(
            row_store=self.row_store,
            semantic_store=self.semantic_store,
            numeric_store=self.numeric_store,
            mapping_store=self.mapping_store,
            registry=self.registry,
            embedder=self.embedder,
            parser=self.parser,
            context_builder=self.context_builder,
        )

        logger.info("✅ Ingestion complete.")
        return {
            "dataset": name,
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_indexed": len(numeric_cols),
            "semantic_indexed": len(semantic_cols),
            "exact_indexed": len(non_numeric),
            "schema": schema_summary,
            "context_block_chars": len(preview),
            "embedder_type": (
                "sentence-transformers"
                if self.embedder.using_pretrained
                else "hash-fallback"
            ),
        }

    @property
    def engine(self) -> QueryEngine:
        if self._engine is None:
            raise RuntimeError("Call ingest() before accessing engine.")
        return self._engine

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> pd.DataFrame:
        ext = path.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(path, low_memory=False)
        elif ext in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
  
  
 
import os 
print("DB EXISTS:", os.path.exists("data.db"))     