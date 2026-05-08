# """
# Dataset Persistence Layer
# ==========================
# Handles saving and loading all per-dataset artifacts to/from disk.

# Directory layout per dataset:
#     data/datasets/{dataset_id}/
#         meta.json               — name, stats, creation time
#         rows.db                 — SQLite row store
#         mapping.pkl             — MappingStore._index
#         numeric.pkl             — NumericStore._sorted + _values
#         registry.pkl            — SchemaRegistry.columns + dataset_name
#         semantic_{col}.index    — FAISS index per semantic column
#         semantic_{col}_meta.pkl — values + row_ids parallel arrays
#         original.{ext}          — copy of the uploaded source file

# All save/load operations are idempotent and atomic (write to tmp → rename).
# """

# from __future__ import annotations

# import json
# import os
# import pickle
# import shutil
# import time
# from pathlib import Path
# from typing import Any, Dict, Optional

# import faiss
# import numpy as np

# from schema.registry import SchemaRegistry, ColumnMeta
# from storage.mapping_store import MappingStore
# from storage.numeric_store import NumericStore
# from storage.row_store import RowStore
# from storage.semantic_store import SemanticStore

# DATASETS_ROOT = Path("data/datasets")


# # ─────────────────────────────────────────────────────────────────────────────
# # Directory helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def dataset_dir(dataset_id: str) -> Path:
#     return DATASETS_ROOT / dataset_id


# def ensure_dataset_dir(dataset_id: str) -> Path:
#     d = dataset_dir(dataset_id)
#     d.mkdir(parents=True, exist_ok=True)
#     return d


# def list_dataset_ids() -> list[str]:
#     """Return all persisted dataset IDs (directories that contain meta.json)."""
#     if not DATASETS_ROOT.exists():
#         return []
#     return [
#         p.name
#         for p in DATASETS_ROOT.iterdir()
#         if p.is_dir() and (p / "meta.json").exists()
#     ]


# # ─────────────────────────────────────────────────────────────────────────────
# # Meta
# # ─────────────────────────────────────────────────────────────────────────────

# def save_meta(dataset_id: str, stats: Dict[str, Any], filename: str) -> None:
#     d = ensure_dataset_dir(dataset_id)
#     payload = {
#         "dataset_id": dataset_id,
#         "filename": filename,
#         "created_at": time.time(),
#         "stats": stats,
#     }
#     _atomic_json(d / "meta.json", payload)


# def load_meta(dataset_id: str) -> Optional[Dict[str, Any]]:
#     p = dataset_dir(dataset_id) / "meta.json"
#     if not p.exists():
#         return None
#     with open(p) as f:
#         return json.load(f)


# # ─────────────────────────────────────────────────────────────────────────────
# # Source file
# # ─────────────────────────────────────────────────────────────────────────────

# def save_source_file(dataset_id: str, tmp_path: str, original_ext: str) -> Path:
#     """Copy the uploaded temp file into the dataset directory."""
#     d = ensure_dataset_dir(dataset_id)
#     dest = d / f"original{original_ext}"
#     shutil.copy2(tmp_path, dest)
#     return dest


# def source_file_path(dataset_id: str) -> Optional[Path]:
#     d = dataset_dir(dataset_id)
#     for ext in (".csv", ".xlsx", ".xls"):
#         p = d / f"original{ext}"
#         if p.exists():
#             return p
#     return None


# # ─────────────────────────────────────────────────────────────────────────────
# # RowStore  (SQLite — just point to a fixed path)
# # ─────────────────────────────────────────────────────────────────────────────

# def row_store_path(dataset_id: str) -> str:
#     d = ensure_dataset_dir(dataset_id)
#     return str(d / "rows.db")


# def load_row_store(dataset_id: str) -> Optional[RowStore]:
#     path = row_store_path(dataset_id)
#     if not Path(path).exists():
#         return None
#     rs = RowStore(db_path=path)
#     rs.connect()
#     # Recover table name from SQLite master
#     cur = rs._conn.execute(
#         "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 't_%' LIMIT 1"
#     )
#     row = cur.fetchone()
#     if row:
#         rs._table = row[0]
#         rs._raw_table = row[0]
#     return rs


# # ─────────────────────────────────────────────────────────────────────────────
# # MappingStore
# # ─────────────────────────────────────────────────────────────────────────────

# def save_mapping_store(dataset_id: str, store: MappingStore) -> None:
#     d = ensure_dataset_dir(dataset_id)
#     # Convert defaultdict → plain dict for pickle stability
#     plain = {col: dict(val_map) for col, val_map in store._index.items()}
#     _atomic_pickle(d / "mapping.pkl", plain)


# def load_mapping_store(dataset_id: str) -> Optional[MappingStore]:
#     p = dataset_dir(dataset_id) / "mapping.pkl"
#     if not p.exists():
#         return None
#     with open(p, "rb") as f:
#         plain = pickle.load(f)
#     store = MappingStore()
#     for col, val_map in plain.items():
#         for val, ids in val_map.items():
#             store._index[col][val] = ids
#     return store


# # ─────────────────────────────────────────────────────────────────────────────
# # NumericStore
# # ─────────────────────────────────────────────────────────────────────────────

# def save_numeric_store(dataset_id: str, store: NumericStore) -> None:
#     d = ensure_dataset_dir(dataset_id)
#     payload = {
#         "sorted": store._sorted,
#         "values": {col: arr.tolist() for col, arr in store._values.items()},
#     }
#     _atomic_pickle(d / "numeric.pkl", payload)


# def load_numeric_store(dataset_id: str) -> Optional[NumericStore]:
#     p = dataset_dir(dataset_id) / "numeric.pkl"
#     if not p.exists():
#         return None
#     with open(p, "rb") as f:
#         payload = pickle.load(f)
#     store = NumericStore()
#     store._sorted = payload["sorted"]
#     store._values = {col: np.array(v) for col, v in payload["values"].items()}
#     return store


# # ─────────────────────────────────────────────────────────────────────────────
# # SemanticStore  (FAISS index + metadata per column)
# # ─────────────────────────────────────────────────────────────────────────────

# def save_semantic_store(dataset_id: str, store: SemanticStore) -> None:
#     d = ensure_dataset_dir(dataset_id)
#     cols_saved = []
#     for col, data in store._stores.items():
#         safe_col = _safe_col(col)
#         # Save FAISS index
#         faiss.write_index(data["index"], str(d / f"semantic_{safe_col}.index"))
#         # Save parallel metadata (values + row_ids)
#         _atomic_pickle(d / f"semantic_{safe_col}_meta.pkl", {
#             "col": col,
#             "values": data["values"],
#             "row_ids": data["row_ids"],
#             "embeddings": data["embeddings"],
#         })
#         cols_saved.append(col)
#     # Save column manifest so we know which files belong to this store
#     _atomic_json(d / "semantic_cols.json", {"columns": cols_saved})


# def load_semantic_store(dataset_id: str, dim: int = 384) -> Optional[SemanticStore]:
#     d = dataset_dir(dataset_id)
#     manifest = d / "semantic_cols.json"
#     if not manifest.exists():
#         return None
#     with open(manifest) as f:
#         cols = json.load(f)["columns"]

#     store = SemanticStore(dim=dim)
#     for col in cols:
#         safe_col = _safe_col(col)
#         index_path = d / f"semantic_{safe_col}.index"
#         meta_path = d / f"semantic_{safe_col}_meta.pkl"
#         if not index_path.exists() or not meta_path.exists():
#             continue
#         idx = faiss.read_index(str(index_path))
#         with open(meta_path, "rb") as f:
#             meta = pickle.load(f)
#         store._stores[col] = {
#             "index": idx,
#             "values": meta["values"],
#             "row_ids": meta["row_ids"],
#             "embeddings": meta["embeddings"],
#         }
#     return store


# # ─────────────────────────────────────────────────────────────────────────────
# # SchemaRegistry
# # ─────────────────────────────────────────────────────────────────────────────

# def save_registry(dataset_id: str, registry: SchemaRegistry) -> None:
#     d = ensure_dataset_dir(dataset_id)
#     payload = {
#         "dataset_name": registry._dataset_name,
#         "columns": {k: v.to_dict() for k, v in registry.columns.items()},
#     }
#     _atomic_json(d / "registry.json", payload)


# def load_registry(dataset_id: str) -> Optional[SchemaRegistry]:
#     p = dataset_dir(dataset_id) / "registry.json"
#     if not p.exists():
#         return None
#     with open(p) as f:
#         payload = json.load(f)
#     reg = SchemaRegistry()
#     reg._dataset_name = payload["dataset_name"]
#     for col_name, col_dict in payload["columns"].items():
#         # ColumnMeta is a dataclass — reconstruct from dict
#         reg.columns[col_name] = ColumnMeta(**col_dict)
#     return reg


# # ─────────────────────────────────────────────────────────────────────────────
# # Full dataset save  (called once at the end of ingestion)
# # ─────────────────────────────────────────────────────────────────────────────

# def save_dataset(
#     dataset_id: str,
#     *,
#     row_store: RowStore,
#     mapping_store: MappingStore,
#     numeric_store: NumericStore,
#     semantic_store: SemanticStore,
#     registry: SchemaRegistry,
#     stats: Dict[str, Any],
#     tmp_source_path: str,
#     source_ext: str,
# ) -> None:
#     """Persist every component of a compiled dataset to disk."""
#     ensure_dataset_dir(dataset_id)
#     save_registry(dataset_id, registry)
#     save_mapping_store(dataset_id, mapping_store)
#     save_numeric_store(dataset_id, numeric_store)
#     save_semantic_store(dataset_id, semantic_store)
#     save_source_file(dataset_id, tmp_source_path, source_ext)
#     save_meta(dataset_id, stats, filename=f"original{source_ext}")
#     # RowStore is already on disk (rows.db) — nothing extra to flush


# # ─────────────────────────────────────────────────────────────────────────────
# # Full dataset load  (called on startup for each persisted dataset_id)
# # ─────────────────────────────────────────────────────────────────────────────

# def load_dataset_stores(dataset_id: str, dim: int = 384):
#     """
#     Load all stores from disk and return them as a tuple:
#         (row_store, mapping_store, numeric_store, semantic_store, registry, stats)
#     Returns None if the dataset directory is missing or incomplete.
#     """
#     meta = load_meta(dataset_id)
#     if meta is None:
#         return None

#     row_store = load_row_store(dataset_id)
#     mapping_store = load_mapping_store(dataset_id)
#     numeric_store = load_numeric_store(dataset_id)
#     semantic_store = load_semantic_store(dataset_id, dim=dim)
#     registry = load_registry(dataset_id)

#     if any(x is None for x in (row_store, mapping_store, numeric_store, semantic_store, registry)):
#         return None

#     return row_store, mapping_store, numeric_store, semantic_store, registry, meta["stats"]


# # ─────────────────────────────────────────────────────────────────────────────
# # Internal helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def _safe_col(col: str) -> str:
#     """Convert a column name to a filesystem-safe string."""
#     return "".join(c if c.isalnum() or c in "-_" else "_" for c in col)[:80]


# def _atomic_pickle(path: Path, obj: Any) -> None:
#     tmp = path.with_suffix(".tmp")
#     with open(tmp, "wb") as f:
#         pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
#     tmp.replace(path)


# def _atomic_json(path: Path, obj: Any) -> None:
#     tmp = path.with_suffix(".tmp")
#     with open(tmp, "w") as f:
#         json.dump(obj, f, indent=2, default=_json_default)
#     tmp.replace(path)


# def _json_default(o):
#     if isinstance(o, (set, frozenset)):
#         return list(o)
#     if isinstance(o, np.integer):
#         return int(o)
#     if isinstance(o, np.floating):
#         return float(o)
#     raise TypeError(f"Object of type {type(o)} is not JSON serializable")
"""
Persistence Layer
==================
Saves and loads a fully compiled dataset package to/from disk.

Disk layout per dataset:
    {data_dir}/datasets/{dataset_id}/
        manifest.json       ← id, name, original filename, stats, created_at
        row_store.db        ← SQLite (full rows)
        mapping.pkl         ← MappingStore inverted index
        numeric.pkl         ← NumericStore sorted index
        semantic.pkl        ← SemanticStore FAISS indexes + metadata
        registry.pkl        ← SchemaRegistry column metadata
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ogquery.storage.row_store import RowStore
from ogquery.storage.mapping_store import MappingStore
from ogquery.storage.numeric_store import NumericStore
from ogquery.storage.semantic_store import SemanticStore
from ogquery.schema.registry import SchemaRegistry


class DatasetStore:
    """
    Handles full save and load of a compiled dataset package.
    All indexing happens exactly once (during upload).
    On subsequent startups the engine is restored from disk instantly.
    """

    def __init__(self, data_dir: str) -> None:
        self.base = Path(data_dir) / "datasets"
        self.base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        dataset_id: str,
        name: str,
        original_filename: str,
        stats: Dict[str, Any],
        row_store: RowStore,
        mapping_store: MappingStore,
        numeric_store: NumericStore,
        semantic_store: SemanticStore,
        registry: SchemaRegistry,
    ) -> Path:
        """Persist all compiled artifacts for one dataset."""
        folder = self._folder(dataset_id)
        folder.mkdir(parents=True, exist_ok=True)

        # ── manifest ──────────────────────────────────────────────────
        manifest = {
            "dataset_id": dataset_id,
            "name": name,
            "original_filename": original_filename,
            "stats": stats,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )

        # ── row store (SQLite already on disk — just record path) ─────
        # RowStore writes to its own db_path; we store that path in manifest.
        manifest["row_store_db"] = str(row_store.db_path)
        (folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )

        # ── mapping store ─────────────────────────────────────────────
        with open(folder / "mapping.pkl", "wb") as f:
            pickle.dump({
                "index": dict(mapping_store._index),
            }, f)

        # ── numeric store ─────────────────────────────────────────────
        with open(folder / "numeric.pkl", "wb") as f:
            pickle.dump({
                "sorted": numeric_store._sorted,
                "values": {k: v.tolist() for k, v in numeric_store._values.items()},
            }, f)

        # ── semantic store ────────────────────────────────────────────
        semantic_store.save(str(folder / "semantic.pkl"))

        # ── schema registry ───────────────────────────────────────────
        with open(folder / "registry.pkl", "wb") as f:
            pickle.dump(registry, f)

        return folder

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        dataset_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Restore a full compiled dataset from disk.
        Returns dict with all stores + manifest, or None if not found.
        """
        folder = self._folder(dataset_id)
        if not folder.exists():
            return None

        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            return None

        manifest = json.loads(manifest_path.read_text())

        # ── registry ──────────────────────────────────────────────────
        with open(folder / "registry.pkl", "rb") as f:
            registry: SchemaRegistry = pickle.load(f)

        # ── row store ─────────────────────────────────────────────────
        db_path = manifest.get("row_store_db", str(folder / "row_store.db"))
        row_store = RowStore(db_path=db_path)
        row_store.connect()
        # Restore table name from manifest
        table_name = manifest["stats"].get("dataset", "dataset")
        row_store._raw_table = table_name
        row_store._table = row_store._sanitize_table_name(table_name)

        # ── mapping store ─────────────────────────────────────────────
        with open(folder / "mapping.pkl", "rb") as f:
            mapping_data = pickle.load(f)
        mapping_store = MappingStore()
        mapping_store._index = mapping_data["index"]

        # ── numeric store ─────────────────────────────────────────────
        import numpy as np
        with open(folder / "numeric.pkl", "rb") as f:
            numeric_data = pickle.load(f)
        numeric_store = NumericStore()
        numeric_store._sorted = numeric_data["sorted"]
        numeric_store._values = {
            k: np.array(v) for k, v in numeric_data["values"].items()
        }

        # ── semantic store ────────────────────────────────────────────
        semantic_store = SemanticStore.load(str(folder / "semantic.pkl"))

        return {
            "manifest": manifest,
            "registry": registry,
            "row_store": row_store,
            "mapping_store": mapping_store,
            "numeric_store": numeric_store,
            "semantic_store": semantic_store,
        }

    # ------------------------------------------------------------------
    # List / Delete
    # ------------------------------------------------------------------

    def list_datasets(self) -> list:
        """Return list of all manifest dicts for saved datasets."""
        result = []
        for folder in sorted(self.base.iterdir()):
            manifest_path = folder / "manifest.json"
            if manifest_path.exists():
                try:
                    result.append(json.loads(manifest_path.read_text()))
                except Exception:
                    pass
        return result

    def exists(self, dataset_id: str) -> bool:
        return (self._folder(dataset_id) / "manifest.json").exists()

    def delete(self, dataset_id: str) -> bool:
        """Remove all files for a dataset. Returns True if deleted."""
        import shutil
        folder = self._folder(dataset_id)
        if folder.exists():
            shutil.rmtree(folder)
            return True
        return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _folder(self, dataset_id: str) -> Path:
        return self.base / dataset_id