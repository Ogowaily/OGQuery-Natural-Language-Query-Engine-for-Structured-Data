"""
Helpers & Application Factory
==============================
Wires all components into a ready-to-use QueryEngine via a single
DataIngestionPipeline call.

BUG FIXES vs v3:
    - No longer builds a second QueryEngine after ingestion.
      The pipeline already builds one internally (step 8 in ingest()).
      Returning ``pipeline.engine`` avoids store-sharing bugs and
      ensures the planner and context_builder are in sync.
    - Returns (engine, stats) tuple consistently with test.py expectations.
"""

from __future__ import annotations

from typing import Optional, Tuple

from schema.registry import SchemaRegistry
from storage.row_store import RowStore
from storage.semantic_store import SemanticStore
from storage.numeric_store import NumericStore
from storage.mapping_store import MappingStore
from embeddings.embedder import Embedder
from core.groq_parser import GroqParser
from core.ingestion import DataIngestionPipeline
from core.query_engine import QueryEngine


def build_engine(
    file_path: str,
    groq_api_key: Optional[str] = None,
    db_path: str = ":memory:",
    groq_model: str = "llama-3.3-70b-versatile",
    force_hash_embeddings: bool = False,
) -> Tuple[QueryEngine, dict]:
    """
    Full factory: ingest a CSV/Excel file and return (QueryEngine, stats).

    Parameters
    ----------
    file_path : str
        Path to the CSV or Excel dataset.
    groq_api_key : str, optional
        Groq API key. Falls back to GROQ_API_KEY env var.
    db_path : str
        SQLite database path. Default ":memory:" (in-process, fast).
    groq_model : str
        Groq model identifier. Updated default avoids deprecated models.
    force_hash_embeddings : bool
        If True, skip sentence-transformers and use hash embeddings.
        Useful for CI/offline environments.

    Usage
    -----
        engine, stats = build_engine("data/properties.csv", groq_api_key="gsk_...")
        result = engine.query("cheapest villa in Cairo")
    """
    row_store      = RowStore(db_path=db_path)
    semantic_store = SemanticStore(dim=384)
    numeric_store  = NumericStore()
    mapping_store  = MappingStore()
    registry       = SchemaRegistry()
    embedder       = Embedder(dim=384, force_hash=force_hash_embeddings)
    parser         = GroqParser(api_key=groq_api_key, model=groq_model)

    pipeline = DataIngestionPipeline(
        row_store=row_store,
        semantic_store=semantic_store,
        numeric_store=numeric_store,
        mapping_store=mapping_store,
        registry=registry,
        embedder=embedder,
        parser=parser,
    )
    stats = pipeline.ingest(file_path)

    # Use the engine built inside the pipeline — no duplicate construction.
    return pipeline.engine, stats