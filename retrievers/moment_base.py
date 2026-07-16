"""Shared MOMENT plumbing for the long-series TS retriever variants.

MOMENT encodes a fixed window of SEQ_LEN=512 time steps (patch length 8 →
64 patch tokens), one channel at a time, and in embedding mode mean-pools
the patch tokens into a single D-dim vector per window. A series longer
than 512 therefore cannot be embedded whole, and no fixed-length vector can
losslessly represent an arbitrarily long series — each variant picks a
different strategy for *which* information to keep:

  A. TSCompressRetriever   — downsample the whole series to 512, embed once.
                             Keeps global shape/trend, loses local detail.
  B. TSMultiVecRetriever   — embed every 512-step window, index all vectors,
                             MaxSim at query time. Nothing lost at index time.
  C. TSWindowAggRetriever  — embed every window, length-weighted average into
                             one vector. Full resolution seen, cross-window blur.

All three left-pad short windows and mask the padding via MOMENT's
input_mask, so the encoder never attends to fabricated values (unlike the
original TSRetriever, which right-pads zeros under a full-ones mask and
truncates anything past 512).
"""

from typing import Any, Dict, List

import numpy as np
import torch

from retrievers.base_retriever import BaseRetriever


def _zscore(series: List[float]) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(1, dtype=np.float32)
    std = arr.std()
    return (arr - arr.mean()) / std if std > 1e-8 else np.zeros_like(arr)


def _split_windows(arr: np.ndarray, seq_len: int) -> List[np.ndarray]:
    """Non-overlapping seq_len windows covering the full series.

    The trailing partial window (if any) is returned as-is — _embed_window
    left-pads and masks it, so no timestep is dropped and none is invented.
    """
    return [arr[i:i + seq_len] for i in range(0, arr.size, seq_len)]


class MomentRetrieverBase(BaseRetriever):
    """Model loading + masked single-window embedding shared by all variants."""

    SEQ_LEN = 512  # MOMENT-1 patch context length

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
        self._id_to_row: Dict[Any, int] = {}

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

    def _embed_window(self, window: np.ndarray) -> np.ndarray:
        """Embed one window of <= SEQ_LEN steps → (D,) numpy vector.

        Windows shorter than SEQ_LEN are LEFT-padded with zeros and the padded
        region is masked out via input_mask (0 = padding, 1 = observed), so
        the encoder only pools over real observations.
        """
        n = min(window.size, self.SEQ_LEN)
        arr = np.zeros(self.SEQ_LEN, dtype=np.float32)
        mask = torch.zeros(1, self.SEQ_LEN, dtype=torch.long)
        if n > 0:
            arr[self.SEQ_LEN - n:] = window[:n]
            mask[:, self.SEQ_LEN - n:] = 1
        else:  # degenerate empty window — one masked-in zero step
            mask[:, -1] = 1

        x = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(self._device)
        mask = mask.to(self._device)
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


class MomentSingleVecRetriever(MomentRetrieverBase):
    """Variants that produce one vector per item (A and C) share this
    index/retrieve; only _embed_item differs between them."""

    def _embed_item(self, input_ts: List[List[float]]) -> np.ndarray:
        raise NotImplementedError

    def index(self, pool: List[Dict[str, Any]]) -> None:
        name = type(self).__name__
        print(f"[{name}] loading {self._model_name}")
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
        self._id_to_row = self._build_id_map(self._pool_items)
        self._offload_encoder()
        print(f"[{name}] indexed {len(self._pool_items)} items  shape={self._pool_vecs.shape}")

    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        row = self._id_to_row.get(query.get("id"))
        if row is not None:
            q_vec = self._pool_vecs[row]
        else:
            with torch.no_grad():
                q_vec = self._embed_item(query["input_ts"])
            norm = np.linalg.norm(q_vec)
            q_vec = (q_vec / norm if norm > 1e-8 else q_vec).astype(np.float32)
        return self._cosine_top_k(q_vec, self._pool_vecs, self._pool_items, k, query.get("id"), query.get("tid"))
