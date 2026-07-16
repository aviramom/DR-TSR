from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from retrievers.moment_base import MomentRetrieverBase, _split_windows, _zscore


class TSMultiVecRetriever(MomentRetrieverBase):
    """Strategy B — multi-vector: index every 512-step window separately.

    The series is split into non-overlapping 512-step windows (trailing
    partial window left-padded + masked), each window is embedded, and *all*
    window vectors are indexed — nothing is averaged away at index time, so
    this is the true "don't lose data" option: every timestep is seen by the
    encoder at full resolution.

    Scoring is late interaction (MaxSim): each query window is matched
    against a candidate's best-matching window, and the per-query-window
    maxima are averaged into the candidate score. A candidate containing one
    segment that strongly matches part of the query ranks high even if its
    remaining windows are unrelated.

    The cost is a multi-vector index: a flat (M, D) window matrix plus an
    owner map tracing each vector back to its parent item (the numpy
    equivalent of a FAISS index + sidecar id-map).

    Multi-series items contribute the windows of all their series to the
    same vector set.
    """

    def __init__(
        self,
        model_name: str = "AutonLab/MOMENT-1-large",
        device: str = "cuda",
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self._win_vecs: np.ndarray = np.empty((0,))   # (M, D) all window vectors
        self._win_owner: np.ndarray = np.empty((0,))  # (M,) window → pool row
        self._item_spans: List[Tuple[int, int]] = []  # pool row → [start, end) in _win_vecs

    def _item_window_vecs(self, input_ts: List[List[float]]) -> np.ndarray:
        """All window embeddings of one item, L2-normalized → (W, D)."""
        vecs = []
        for series in input_ts:
            arr = _zscore(series)  # z-score the full series, then window it
            for window in _split_windows(arr, self.SEQ_LEN):
                vecs.append(self._embed_window(window))
        raw = np.stack(vecs)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return (raw / norms).astype(np.float32)

    def index(self, pool: List[Dict[str, Any]]) -> None:
        print(f"[TSMultiVecRetriever] loading {self._model_name}")
        self._model = self._load_model()
        self._pool_items = list(pool)

        chunks: List[np.ndarray] = []
        owners: List[int] = []
        spans: List[Tuple[int, int]] = []
        start = 0
        with torch.no_grad():
            for row, item in enumerate(self._pool_items):
                vecs = self._item_window_vecs(item["input_ts"])
                chunks.append(vecs)
                owners.extend([row] * len(vecs))
                spans.append((start, start + len(vecs)))
                start += len(vecs)

        self._win_vecs = np.concatenate(chunks, axis=0)  # (M, D)
        self._win_owner = np.array(owners, dtype=np.int64)
        self._item_spans = spans
        self._id_to_row = self._build_id_map(self._pool_items)
        self._offload_encoder()
        print(
            f"[TSMultiVecRetriever] indexed {len(self._pool_items)} items  "
            f"{self._win_vecs.shape[0]} window vectors  shape={self._win_vecs.shape}"
        )

    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        row = self._id_to_row.get(query.get("id"))
        if row is not None:
            start, end = self._item_spans[row]
            q_wins = self._win_vecs[start:end]  # (Wq, D) precomputed
        else:
            with torch.no_grad():
                q_wins = self._item_window_vecs(query["input_ts"])

        sims = self._win_vecs @ q_wins.T  # (M, Wq) window-level cosine sims
        # MaxSim: per query window, each item's best-matching window ...
        per_item = np.full(
            (len(self._pool_items), q_wins.shape[0]), -np.inf, dtype=np.float32
        )
        np.maximum.at(per_item, self._win_owner, sims)
        # ... then average the per-query-window maxima into one item score.
        scores = per_item.mean(axis=1)
        return self._top_k_from_scores(
            scores, self._pool_items, k, query.get("id"), query.get("tid")
        )
