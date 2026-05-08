 
"""
Embedder
=========
Two-tier embedding strategy:

  Tier 1 (default, production):
    sentence-transformers ``all-MiniLM-L6-v2`` → 384-dim dense vectors.
    Enables true semantic similarity: "villa" ≈ "mansion", synonyms, etc.

  Tier 2 (fallback, no heavy deps):
    Deterministic character n-gram hashing → 384-dim dense vectors.
    Used automatically when sentence-transformers is not installed or when
    ``force_hash=True`` is passed.

Both tiers produce 384-dim float32 vectors compatible with FAISS IndexFlatIP.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

import numpy as np

_DIM = 384


# ──────────────────────────────────────────────────────────────────────────────
# Hash-based fallback (no external deps)
# ──────────────────────────────────────────────────────────────────────────────

def _hash_embed(text: str, dim: int = _DIM) -> np.ndarray:
    """
    Deterministic 384-dim embedding via character n-gram hashing.
    Captures morphological similarity (villa/villas, cairo/cairo-city).
    """
    text = text.lower().strip()
    vec = np.zeros(dim, dtype=np.float64)

    for n in range(1, 4):
        for i in range(len(text) - n + 1):
            gram = text[i: i + n]
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            slot = h % dim
            sign = 1 if (h >> 128) & 1 else -1
            vec[slot] += sign

    norm = np.linalg.norm(vec) + 1e-10
    return (vec / norm).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Sentence-transformer loader (lazy, optional)
# ──────────────────────────────────────────────────────────────────────────────

def _try_load_st_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Attempt to load a sentence-transformers model.
    Returns the model on success, None if the package is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer(model_name)
    except ImportError:
        return None
    except Exception as exc:
        # Model download failure, CUDA OOM, etc. — degrade gracefully.
        print(f"[Embedder] Could not load sentence-transformer '{model_name}': {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Public Embedder class
# ──────────────────────────────────────────────────────────────────────────────

class Embedder:
    """
    Unified batch text embedding interface.

    Parameters
    ----------
    dim : int
        Target embedding dimension (default 384).
    model_name : str
        sentence-transformers model to attempt loading.
    force_hash : bool
        If True, skip sentence-transformers and use hash embeddings.
        Useful for testing or offline environments.
    """

    def __init__(
        self,
        dim: int = _DIM,
        model_name: str = "all-MiniLM-L6-v2",
        force_hash: bool = False,
    ) -> None:
        self.dim = dim
        self._cache: Dict[str, np.ndarray] = {}
        self._st_model = None

        if not force_hash:
            self._st_model = _try_load_st_model(model_name)

        if self._st_model is not None:
            print(f"[Embedder] Using sentence-transformers '{model_name}'")
        else:
            print("[Embedder] Using hash-based fallback embeddings")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string (cached)."""
        key = text.lower().strip()
        if key not in self._cache:
            self._cache[key] = self._embed_one(key)
        return self._cache[key]

    def embed_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Embed a list of strings efficiently.
        Sentence-transformers path uses batched inference for speed.
        Returns {original_text: vector}.
        """
        # Normalise keys
        keys = [t.lower().strip() for t in texts]
        missing = [k for k in keys if k not in self._cache]

        if missing:
            if self._st_model is not None:
                # Batch encode all missing strings at once
                vecs = self._st_model.encode(
                    missing,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=256,
                )
                for key, vec in zip(missing, vecs):
                    self._cache[key] = vec.astype(np.float32)
            else:
                for key in missing:
                    self._cache[key] = _hash_embed(key, self.dim)

        return {t: self._cache[t.lower().strip()] for t in texts}

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two strings (vectors are pre-normalised)."""
        va = self.embed(a)
        vb = self.embed(b)
        return float(np.dot(va, vb))

    @property
    def using_pretrained(self) -> bool:
        """True if a sentence-transformers model is active."""
        return self._st_model is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _embed_one(self, text: str) -> np.ndarray:
        if self._st_model is not None:
            vec = self._st_model.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )[0]
            return vec.astype(np.float32)
        return _hash_embed(text, self.dim)