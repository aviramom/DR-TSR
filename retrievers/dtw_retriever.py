from typing import Any, Dict, List

import numpy as np
from tslearn.metrics import dtw

from retrievers.base_retriever import BaseRetriever


def _signature(input_ts: List[List[float]], target_len: int) -> np.ndarray:
    """Z-scored, fixed-length 1D signature for one item's series.

    Each channel is z-scored (so DTW compares shape, not amplitude — mirroring
    the z-scoring of the spectral / wavelet / vision retrievers) and resampled
    to target_len points, then all channels are concatenated into one sequence.
    DTW handles the resulting variable total length across items natively, and
    concatenation gives a uniform representation for any channel count.
    """
    sigs = []
    for series in input_ts:
        arr = np.asarray(series, dtype=np.float64)
        if arr.size == 0:
            sigs.append(np.zeros(target_len, dtype=np.float64))
            continue
        std = arr.std()
        z = (arr - arr.mean()) / std if std > 1e-12 else np.zeros_like(arr)
        if arr.size == target_len:
            sigs.append(z)
        else:
            src = np.linspace(0.0, 1.0, num=arr.size)
            dst = np.linspace(0.0, 1.0, num=target_len)
            sigs.append(np.interp(dst, src, z))
    return np.concatenate(sigs) if sigs else np.zeros(target_len, dtype=np.float64)


class DTWRetriever(BaseRetriever):
    """Dynamic Time Warping kNN retriever over the raw time series.

    The canonical shape-similarity baseline for time-series retrieval: each item
    is reduced to a z-scored, fixed-length signature and compared to every other
    item by DTW distance (Sakoe-Chiba banded, via tslearn). Pure numpy/tslearn —
    no pretrained encoder, no GPU.

    Leave-one-out: index() precomputes the full symmetric N×N DTW distance
    matrix once, so each query (already in the pool) is an O(1) row lookup.
    Multi-series items concatenate their per-channel z-scored signatures.
    """

    def __init__(
        self,
        device: str = "cpu",  # accepted for interface parity; unused (no encoder)
        target_len: int = 256,
        sakoe_chiba_radius: int = 25,
    ) -> None:
        self._target_len = target_len
        self._radius = sakoe_chiba_radius
        self._pool_items: List[Dict[str, Any]] = []
        self._pool_sigs: List[np.ndarray] = []
        self._dist: np.ndarray = np.empty((0, 0))
        self._id_to_row: Dict[Any, int] = {}

    def _dtw(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(
            dtw(a, b, global_constraint="sakoe_chiba", sakoe_chiba_radius=self._radius)
        )

    def index(self, pool: List[Dict[str, Any]]) -> None:
        self._pool_items = list(pool)
        self._pool_sigs = [
            _signature(item["input_ts"], self._target_len) for item in self._pool_items
        ]
        n = len(self._pool_sigs)
        dist = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                d = self._dtw(self._pool_sigs[i], self._pool_sigs[j])
                dist[i, j] = d
                dist[j, i] = d
        self._dist = dist
        self._id_to_row = self._build_id_map(self._pool_items)
        print(f"[DTWRetriever] indexed {n} items  shape={self._dist.shape}")

    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        row = self._id_to_row.get(query.get("id"))
        if row is not None:
            dists = self._dist[row]
        else:
            q_sig = _signature(query["input_ts"], self._target_len)
            dists = np.array([self._dtw(q_sig, s) for s in self._pool_sigs])
        # Negate so smaller DTW distance == higher score; fresh array keeps the
        # precomputed matrix immutable through _top_k_from_scores's masking.
        scores = -np.asarray(dists, dtype=np.float32)
        return self._top_k_from_scores(
            scores, self._pool_items, k, query.get("id"), query.get("tid")
        )
