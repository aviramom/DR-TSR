from typing import Any, Dict, List

import numpy as np
import torch

from retrievers.base_retriever import BaseRetriever


def _zscore(series: List[float]) -> np.ndarray:
    arr = np.array(series, dtype=np.float32)
    std = arr.std()
    return (arr - arr.mean()) / std if std > 1e-8 else np.zeros_like(arr)


class TSRetriever(BaseRetriever):
    """Cosine kNN retriever over MOMENT embeddings of the time series.

    Each series in input_ts is embedded independently; multi-series items
    use the mean of the individual series embeddings.

    MOMENT truncates its input to seq_len (default 512). Our series are 1024
    points — they are truncated to 512 before encoding.
    """

    SEQ_LEN = 512  # MOMENT-1-large patch context length

    def __init__(
        self,
        model_name: str = "AutonLab/MOMENT-1-large",
        device: str = "cuda",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._pool_items: List[Dict[str, Any]] = []
        self._pool_vecs: np.ndarray = np.empty((0,))

    def _load_model(self):
        try:
            from momentfm import MOMENTPipeline  # installed as momentfm from GitHub
        except ImportError:
            from momentresearch import MOMENTPipeline  # alternate pip name
        model = MOMENTPipeline.from_pretrained(
            self._model_name,
            model_kwargs={"task_name": "embedding"},
        )
        model.init()
        model.to(self._device).eval()
        return model

    def _embed_series(self, series: List[float]) -> np.ndarray:
        """Embed a single time series → (D,) numpy vector."""
        arr = _zscore(series)
        if len(arr) >= self.SEQ_LEN:
            arr = arr[:self.SEQ_LEN]
        else:
            arr = np.pad(arr, (0, self.SEQ_LEN - len(arr)))
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self._device)
        mask = torch.ones(1, self.SEQ_LEN, dtype=torch.long).to(self._device)
        try:
            output = self._model(x_enc=x, input_mask=mask)
        except TypeError:
            output = self._model(x_enc=x)
        # Handle both .embeddings and .embedding attribute names
        if hasattr(output, "embeddings") and output.embeddings is not None:
            emb = output.embeddings
        elif hasattr(output, "embedding") and output.embedding is not None:
            emb = output.embedding
        else:
            emb = next(v for v in output.__dict__.values() if isinstance(v, torch.Tensor))
        return emb[0].detach().cpu().numpy()  # (D,)

    def _embed_item(self, input_ts: List[List[float]]) -> np.ndarray:
        """Mean-pool embeddings across all series in item["input_ts"]."""
        vecs = [self._embed_series(s) for s in input_ts]
        return np.mean(vecs, axis=0)

    def index(self, pool: List[Dict[str, Any]]) -> None:
        print(f"[TSRetriever] loading {self._model_name}")
        self._model = self._load_model()
        self._pool_items = list(pool)

        raw_vecs = []
        with torch.no_grad():
            for item in self._pool_items:
                raw_vecs.append(self._embed_item(item["input_ts"]))

        raw = np.stack(raw_vecs)  # (N, D)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self._pool_vecs = (raw / norms).astype(np.float32)
        print(f"[TSRetriever] indexed {len(self._pool_items)} items  shape={self._pool_vecs.shape}")

    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        with torch.no_grad():
            q_vec = self._embed_item(query["input_ts"])
        norm = np.linalg.norm(q_vec)
        q_vec = (q_vec / norm if norm > 1e-8 else q_vec).astype(np.float32)
        return self._cosine_top_k(q_vec, self._pool_vecs, self._pool_items, k, query.get("id"), query.get("tid"))
