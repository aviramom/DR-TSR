from typing import Any, Dict, List

import numpy as np

from retrievers.base_retriever import BaseRetriever


class TextRetriever(BaseRetriever):
    """Cosine kNN retriever over sentence embeddings of the question text.

    Each pool item is embedded using the question field via a SentenceTransformer
    model. At retrieval time, the query question is embedded and the k most similar
    pool items by cosine similarity are returned.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._pool_items: List[Dict[str, Any]] = []
        self._pool_vecs: np.ndarray = np.empty((0,))
        self._id_to_row: Dict[Any, int] = {}

    def index(self, pool: List[Dict[str, Any]]) -> None:
        from sentence_transformers import SentenceTransformer

        print(f"[TextRetriever] loading {self._model_name}")
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._pool_items = list(pool)

        texts = [item["question"] for item in self._pool_items]
        vecs = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self._pool_vecs = np.array(vecs, dtype=np.float32)
        self._id_to_row = self._build_id_map(self._pool_items)
        self._offload_encoder()
        print(f"[TextRetriever] indexed {len(self._pool_items)} items  shape={self._pool_vecs.shape}")

    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        row = self._id_to_row.get(query.get("id"))
        if row is not None:
            q_vec = self._pool_vecs[row]
        else:
            q_vec = self._model.encode(
                [query["question"]],
                normalize_embeddings=True,
            )
            q_vec = np.array(q_vec[0], dtype=np.float32)
        return self._cosine_top_k(q_vec, self._pool_vecs, self._pool_items, k, query.get("id"), query.get("tid"))
