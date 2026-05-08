"""
Semantic Store
FAISS-backed per-column embedding store.
Maps unique categorical values → embedding vector → row_ids.

Only categorical/text columns with unique_count ≤ threshold get embedded.
Numeric columns are NEVER embedded.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
import faiss

import pickle
from pathlib import Path
class SemanticStore:
    """
    Per-column FAISS index.
    Index structure per column:
        - index (faiss.IndexFlatIP): normalized embedding vectors
        - values (list[str]): the unique string values
        - row_ids (list[set[int]]): row_ids for each unique value (parallel to values)
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        # col → dict with {index, values, row_ids}
        self._stores: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def build_column(
        self,
        col: str,
        value_to_rowids: Dict[str, Set[int]],
        embeddings: Dict[str, np.ndarray],
    ) -> None:
        """
        Build FAISS index for one column.
        value_to_rowids: { "villa": {1,3,9}, ... }
        embeddings:      { "villa": np.array([...]), ... }
        """
        values = list(value_to_rowids.keys())
        row_id_sets = [value_to_rowids[v] for v in values]

        vecs = np.array([embeddings[v] for v in values], dtype=np.float32)
        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        vecs = vecs / norms

        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)

        self._stores[col] = {
            "index": index,
            "values": values,
            "row_ids": row_id_sets,
            "embeddings": vecs,
        }

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def search(
        self,
        col: str,
        query_vec: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[str, float, Set[int]]]:
        """
        Return [(matched_value, score, row_ids), ...] sorted by score desc.
        Only results above threshold are returned.
        """
        store = self._stores.get(col)
        if store is None:
            return []

        vec = query_vec.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec) + 1e-10
        vec = vec / norm

        scores, idxs = store["index"].search(vec, min(top_k, len(store["values"])))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            if score < threshold:
                continue
            results.append((store["values"][idx], float(score), store["row_ids"][idx]))
        return results

    def get_all_row_ids(self, col: str, value: str) -> Set[int]:
        """Exact value lookup (for fallback)."""
        store = self._stores.get(col)
        if store is None:
            return set()
        try:
            idx = store["values"].index(value.lower())
            return store["row_ids"][idx]
        except ValueError:
            return set()

    def column_indexed(self, col: str) -> bool:
        return col in self._stores

    def indexed_columns(self) -> List[str]:
        return list(self._stores.keys())
    
    
    def save(self, path: str) -> None:
        data={
            "dim": self.dim,
            "stores": {}}
        for col, store in self._stores.items():
            data["stores"][col] = {
                "values": store["values"],
                "row_ids": store["row_ids"],
                "embeddings": store["embeddings"],
            }
        with open(path, "wb") as f:            pickle.dump(data, f)
        
    @classmethod
    def load(cls, path: str) -> SemanticStore:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(dim=data["dim"])
        for col, col_data in data["stores"].items():
            index = faiss.IndexFlatIP(obj.dim)
            index.add(col_data["embeddings"])
            obj._stores[col] = {
                "index": index,
                "values": col_data["values"],
                "row_ids": col_data["row_ids"],
                "embeddings": col_data["embeddings"],
            }
        return obj    
        
