"""
OGQuery  ─  Core Class
=======================
The single entry point for the ogquery library.

Usage
-----
    from ogquery import OGQuery

    engine = OGQuery(config={
        "data_dir": "./data",
        "api_keys": {"groq": "gsk_..."},
        "embedding_model": "all-MiniLM-L6-v2",  # optional
        "top_k": 3,                               # optional
    })

    # Upload once — compiles all indexes and saves to disk
    dataset_id = engine.upload("missions.csv", name="NASA Missions")

    # Query any time — loads from disk automatically on restart
    result = engine.query(dataset_id, "failed missions after 2015")

    # Optional: expose as HTTP API
    engine.serve(host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import uuid
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ogquery.engine.ingestion import DataIngestionPipeline
from ogquery.engine.query_engine import QueryEngine
from ogquery.engine.groq_parser import GroqParser
from ogquery.embeddings.embedder import Embedder
from ogquery.schema.registry import SchemaRegistry
from ogquery.storage.row_store import RowStore
from ogquery.storage.mapping_store import MappingStore
from ogquery.storage.numeric_store import NumericStore
from ogquery.storage.semantic_store import SemanticStore
from ogquery.storage.persistence import DatasetStore
from ogquery.engine.data_context import DataContextBuilder
from ogquery.engine.planner import QueryPlanner
from ogquery.engine.answer_generator import AnswerGenerator
from ogquery.utils.logger import get_logger

logger = get_logger("ogquery")


class OGQuery:
    """
    Compiled, persistent, multi-dataset natural language query engine.

    Lifecycle
    ---------
    1. upload(file)  → compile once, save all indexes to disk → returns dataset_id
    2. query(id, q)  → load from disk (if not in memory) → execute → return result
    3. serve(port)   → optional FastAPI HTTP server

    On every restart, previously compiled datasets are discovered automatically
    from data_dir — no re-ingestion ever needed.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._data_dir = config.get("data_dir", "./data")
        self._groq_key = config.get("api_keys", {}).get("groq", "")
        self._embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
        self._top_k = config.get("top_k", 3)

        # In-memory engine cache: dataset_id → QueryEngine
        self._engines: Dict[str, QueryEngine] = {}

        # Disk persistence layer
        self._store = DatasetStore(data_dir=self._data_dir)

        # Auto-discover previously compiled datasets on startup
        self._discover()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload(self, file_path: str, name: Optional[str] = None) -> str:
        """
        Compile a CSV or Excel file into a queryable dataset.

        This is a one-time operation. All indexes are built and saved to disk.
        On future restarts, this dataset loads instantly without re-ingestion.

        Parameters
        ----------
        file_path : str
            Path to a .csv, .xlsx, or .xls file.
        name : str, optional
            Human-readable name for the dataset. Defaults to filename stem.

        Returns
        -------
        str
            dataset_id — use this in all query() calls.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in {".csv", ".xlsx", ".xls"}:
            raise ValueError(f"Unsupported file type: {suffix}. Use .csv, .xlsx, or .xls")

        dataset_id = str(uuid.uuid4())
        dataset_name = name or path.stem

        # SQLite db path — persistent on disk
        db_dir = Path(self._data_dir) / "datasets" / dataset_id
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(db_dir / "row_store.db")

        logger.info(f"[upload] Compiling '{dataset_name}' → id={dataset_id}")

        # Build all components
        row_store      = RowStore(db_path=db_path)
        semantic_store = SemanticStore(dim=384)
        numeric_store  = NumericStore()
        mapping_store  = MappingStore()
        registry       = SchemaRegistry()
        embedder       = Embedder(dim=384, model_name=self._embedding_model)
        parser         = GroqParser(api_key=self._groq_key)

        pipeline = DataIngestionPipeline(
            row_store=row_store,
            semantic_store=semantic_store,
            numeric_store=numeric_store,
            mapping_store=mapping_store,
            registry=registry,
            embedder=embedder,
            parser=parser,
        )

        stats = pipeline.ingest(file_path, table_name=dataset_name)
        engine = pipeline.engine

        # Save everything to disk
        self._store.save(
            dataset_id=dataset_id,
            name=dataset_name,
            original_filename=path.name,
            stats=stats,
            row_store=row_store,
            mapping_store=mapping_store,
            numeric_store=numeric_store,
            semantic_store=semantic_store,
            registry=registry,
        )

        # Cache in memory
        self._engines[dataset_id] = engine

        logger.info(f"[upload] ✅ Saved → {self._data_dir}/datasets/{dataset_id}/")
        return dataset_id

    def query(self, dataset_id: str, question: str) -> Dict[str, Any]:
        """
        Run a natural language query against a compiled dataset.

        The dataset is loaded from disk on first access after a restart.
        Subsequent queries use the in-memory cache.

        Parameters
        ----------
        dataset_id : str
            ID returned from upload().
        question : str
            Natural language question.

        Returns
        -------
        dict with keys:
            query, results, summary (answer, total_matches, returned),
            execution_insight, elapsed_ms
        """
        engine = self._get_engine(dataset_id)
        raw = engine.query(question)

        return {
            "query":   raw["query"],
            "results": raw["results"],
            "summary": {
                "answer":        raw["answer"],
                "total_matches": raw["total_matches"],
                "returned":      raw["returned"],
            },
            "execution_insight": {
                "filters":         raw["execution_plan"].get("exact_filters", []),
                "numeric_filters": raw["execution_plan"].get("numeric_filters", []),
                "semantic":        raw["execution_plan"].get("semantic_filters", []),
                "sort_by":         raw["execution_plan"].get("sort_by"),
                "sort_direction":  "ASC" if raw["execution_plan"].get("sort_ascending", True) else "DESC",
                "objective":       raw["execution_plan"].get("objective"),
            },
            "elapsed_ms": raw["elapsed_ms"],
        }

    def datasets(self) -> List[Dict[str, Any]]:
        """
        List all compiled datasets (from disk — survives restarts).

        Returns
        -------
        List of dicts: dataset_id, name, original_filename, rows, columns, created_at
        """
        manifests = self._store.list_datasets()
        return [
            {
                "dataset_id":        m["dataset_id"],
                "name":              m["name"],
                "original_filename": m.get("original_filename", ""),
                "rows":              m["stats"].get("rows", 0),
                "columns":           m["stats"].get("columns", 0),
                "created_at":        m.get("created_at", ""),
            }
            for m in manifests
        ]

    def delete(self, dataset_id: str) -> bool:
        """
        Delete a compiled dataset from disk and memory.

        Returns True if deleted, False if not found.
        """
        self._engines.pop(dataset_id, None)
        deleted = self._store.delete(dataset_id)
        if deleted:
            logger.info(f"[delete] Dataset {dataset_id} removed.")
        return deleted

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """
        Start the built-in FastAPI HTTP server.

        Exposes:
            POST /upload
            POST /query
            GET  /datasets
            GET  /schema/{dataset_id}
            DELETE /datasets/{dataset_id}
            GET  /health

        Parameters
        ----------
        host : str
        port : int
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError("Install uvicorn to use serve(): pip install uvicorn")

        from ogquery.api import create_app
        app = create_app(self)
        logger.info(f"[serve] Starting OGQuery API on http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_engine(self, dataset_id: str) -> QueryEngine:
        """Return engine from memory cache, or load from disk."""
        if dataset_id in self._engines:
            return self._engines[dataset_id]

        # Try loading from disk
        loaded = self._store.load(dataset_id)
        if loaded is None:
            raise ValueError(
                f"Dataset '{dataset_id}' not found. "
                "Run upload() first or check your data_dir."
            )

        engine = self._rebuild_engine(loaded)
        self._engines[dataset_id] = engine
        logger.info(f"[load] Dataset {dataset_id} restored from disk.")
        return engine

    def _rebuild_engine(self, loaded: Dict[str, Any]) -> QueryEngine:
        """Reconstruct a QueryEngine from loaded disk artifacts."""
        embedder = Embedder(dim=384, model_name=self._embedding_model)
        parser   = GroqParser(api_key=self._groq_key)
        registry = loaded["registry"]

        context_builder = DataContextBuilder(
            registry=registry,
            df=self._dummy_df(registry),
        )

        return QueryEngine(
            row_store=loaded["row_store"],
            semantic_store=loaded["semantic_store"],
            numeric_store=loaded["numeric_store"],
            mapping_store=loaded["mapping_store"],
            registry=registry,
            embedder=embedder,
            parser=parser,
            context_builder=context_builder,
        )

    def _discover(self) -> None:
        """
        Auto-load previously compiled datasets from disk on startup.
        Engines are lazily loaded (only when queried), but we register
        which IDs are available so datasets() works immediately.
        """
        manifests = self._store.list_datasets()
        if manifests:
            ids = [m["dataset_id"] for m in manifests]
            logger.info(f"[startup] Found {len(ids)} compiled dataset(s): {ids}")
        else:
            logger.info("[startup] No compiled datasets found in data_dir.")

    @staticmethod
    def _dummy_df(registry: SchemaRegistry):
        """
        Build a minimal DataFrame from registry metadata.
        Used to reconstruct DataContextBuilder without re-reading the CSV.
        """
        import pandas as pd
        data = {}
        for col, meta in registry.columns.items():
            data[col] = meta.sample_values[:3] if meta.sample_values else [""]
        # Pad all columns to same length
        max_len = max((len(v) for v in data.values()), default=1)
        for col in data:
            while len(data[col]) < max_len:
                data[col].append(data[col][0] if data[col] else "")
        return pd.DataFrame(data)