"""TS and text embedding indices for retrieval.

TSIndex: cosine similarity on L2-normalised raw time series (baseline shape similarity).
TextIndex: cosine similarity on sentence embeddings of question text (skill/task similarity).

Both are pre-built once per pool split and reused across all (strategy, k) conditions.
"""

import numpy as np


class TSIndex:
    """Retrieves pool items by cosine similarity on L2-normalised time series.

    For two-series items, ts1 is used as the embedding (same 1024-dim as single-series).
    This keeps all vectors the same length without requiring special-casing.
    """

    def __init__(self, pool_items: list):
        self.pool_items = pool_items
        self._vecs = self._build_matrix(pool_items)  # (N, D) float32

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 0 else arr

    @staticmethod
    def _to_fixed(arr: np.ndarray, d: int) -> np.ndarray:
        """Truncate or zero-pad a 1-D array to exactly d elements."""
        if len(arr) >= d:
            return arr[:d]
        return np.pad(arr, (0, d - len(arr)))

    @staticmethod
    def _build_matrix(items: list, target_len: int = 1024) -> np.ndarray:
        vecs = []
        for item in items:
            ts = item.get_ts_for_embedding()
            arr = np.array(ts, dtype=np.float32)
            arr = TSIndex._to_fixed(arr, target_len)
            vecs.append(TSIndex._normalise(arr))
        if not vecs:
            return np.empty((0, target_len), dtype=np.float32)
        return np.stack(vecs, axis=0)

    def _query_vec(self, query_item) -> np.ndarray:
        ts = query_item.get_ts_for_embedding()
        arr = np.array(ts, dtype=np.float32)
        arr = self._to_fixed(arr, self._vecs.shape[1])
        return self._normalise(arr)

    def scores(self, query_item) -> np.ndarray:
        """Cosine similarity between query and all pool items. Shape: (N,)."""
        q = self._query_vec(query_item)
        return self._vecs @ q

    def top_k(self, query_item, k: int) -> list:
        sims = self.scores(query_item)
        idx = np.argsort(sims)[::-1][:k]
        return [self.pool_items[i] for i in idx]


class TextIndex:
    """Retrieves pool items by cosine similarity on sentence embeddings of question text.

    Uses sentence-transformers (all-MiniLM-L6-v2 by default) to embed questions.
    Embeddings are computed lazily on first use and cached.
    """

    def __init__(self, pool_items: list, model_name: str = "all-MiniLM-L6-v2"):
        self.pool_items = pool_items
        self.model_name = model_name
        self._encoder = None
        self._vecs = None  # (N, D) float32, unit-normalised

    def build(self):
        """Pre-build embeddings for all pool items. Call this once per split."""
        from sentence_transformers import SentenceTransformer
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.model_name)
        texts = [item.question for item in self.pool_items]
        self._vecs = self._encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
            convert_to_numpy=True,
        )

    def _embed_query(self, question: str) -> np.ndarray:
        if self._encoder is None or self._vecs is None:
            self.build()
        return self._encoder.encode(
            [question], normalize_embeddings=True, show_progress_bar=False,
            convert_to_numpy=True,
        )[0]

    def scores(self, query_item) -> np.ndarray:
        """Cosine similarity between query question and all pool items. Shape: (N,)."""
        if self._vecs is None:
            self.build()
        q = self._embed_query(query_item.question)
        return self._vecs @ q

    def top_k(self, query_item, k: int) -> list:
        sims = self.scores(query_item)
        idx = np.argsort(sims)[::-1][:k]
        return [self.pool_items[i] for i in idx]
