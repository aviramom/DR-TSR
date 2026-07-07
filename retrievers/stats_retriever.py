import math
from typing import Any, Dict, List

import numpy as np
from scipy.stats import kurtosis as _scipy_kurtosis

from retrievers.base_retriever import BaseRetriever

# Feature layout (one value per name) — see _series_features.
FEATURE_NAMES = [
    "mean", "std", "trend_slope", "noise_level",
    "seasonality_strength", "kurtosis", "autocorr_lag1", "permutation_entropy",
]


def _autocorr(x: np.ndarray, lag: int) -> float:
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _permutation_entropy(x: np.ndarray, order: int = 3) -> float:
    """Normalized permutation entropy in [0, 1] (order-3 ordinal patterns)."""
    if x.size < order + 1:
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(x, order)
    patterns = np.argsort(windows, axis=1)
    # Encode each ordinal pattern as a single integer for fast counting.
    codes = patterns @ (order ** np.arange(order))
    counts = np.bincount(codes)
    probs = counts[counts > 0] / codes.size
    return float(-(probs * np.log(probs)).sum() / math.log(math.factorial(order)))


def _series_features(series: List[float], max_season_lag: int = 128) -> np.ndarray:
    """Interpretable feature vector for one series (see FEATURE_NAMES)."""
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < 4:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    mean, std = float(arr.mean()), float(arr.std())
    z = (arr - mean) / std if std > 1e-12 else np.zeros_like(arr)

    t = np.linspace(0.0, 1.0, num=arr.size)
    slope = float(np.polyfit(t, z, 1)[0])
    noise = float(np.std(np.diff(z)))

    # Seasonality strength: max autocorrelation of the detrended series over
    # lags >= 2 (lag 1 is short-term memory, reported separately).
    detrended = z - np.polyval(np.polyfit(t, z, 1), t)
    max_lag = min(arr.size // 2, max_season_lag)
    season = max(
        (_autocorr(detrended, lag) for lag in range(2, max_lag + 1)),
        default=0.0,
    )
    season = max(0.0, season)

    kurt = float(_scipy_kurtosis(z)) if std > 1e-12 else 0.0
    acf1 = _autocorr(z, 1)
    pent = _permutation_entropy(z)

    return np.array(
        [mean, std, slope, noise, season, kurt, acf1, pent],
        dtype=np.float32,
    )


class StatsRetriever(BaseRetriever):
    """Cosine kNN retriever over interpretable statistical features.

    Captures the statistical character of a series — trend, noise level,
    seasonality strength, spikiness, regularity — which several
    TimeSeriesExam categories test directly. Pure numpy/scipy: no
    pretrained encoder, no GPU.

    Features are on wildly different scales (raw mean vs. entropy in [0,1]),
    so each feature dimension is z-scored with pool-level statistics before
    the final L2 normalization — otherwise large-magnitude features would
    dominate the cosine similarity. Pool stats are reused for out-of-pool
    queries (same train-stats convention as DelayDINORetriever's min/max).

    Multi-series items use the mean of the per-series raw feature vectors.
    """

    def __init__(self, device: str = "cpu") -> None:  # device unused (no encoder)
        self._pool_items: List[Dict[str, Any]] = []
        self._pool_vecs: np.ndarray = np.empty((0,))
        self._id_to_row: Dict[Any, int] = {}
        self._feat_mu: np.ndarray = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        self._feat_sigma: np.ndarray = np.ones(len(FEATURE_NAMES), dtype=np.float32)

    @staticmethod
    def _raw_features(input_ts: List[List[float]]) -> np.ndarray:
        return np.mean([_series_features(s) for s in input_ts], axis=0)

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        vec = (raw - self._feat_mu) / self._feat_sigma
        norm = np.linalg.norm(vec)
        return (vec / norm if norm > 1e-12 else vec).astype(np.float32)

    def index(self, pool: List[Dict[str, Any]]) -> None:
        self._pool_items = list(pool)
        raw = np.stack([
            self._raw_features(item["input_ts"]) for item in self._pool_items
        ])
        self._feat_mu = raw.mean(axis=0)
        sigma = raw.std(axis=0)
        self._feat_sigma = np.where(sigma < 1e-8, 1.0, sigma)
        self._pool_vecs = np.stack([self._normalize(r) for r in raw])
        self._id_to_row = self._build_id_map(self._pool_items)
        print(f"[StatsRetriever] indexed {len(self._pool_items)} items  shape={self._pool_vecs.shape}")

    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        row = self._id_to_row.get(query.get("id"))
        if row is not None:
            q_vec = self._pool_vecs[row]
        else:
            q_vec = self._normalize(self._raw_features(query["input_ts"]))
        return self._cosine_top_k(q_vec, self._pool_vecs, self._pool_items, k, query.get("id"), query.get("tid"))
