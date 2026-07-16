from typing import List

import numpy as np

from retrievers.moment_base import MomentSingleVecRetriever, _zscore


class TSCompressRetriever(MomentSingleVecRetriever):
    """Strategy A — compress the whole series into one 512-step window.

    Series longer than SEQ_LEN are linearly interpolated down to exactly 512
    steps and embedded once; shorter series pass through unchanged (they are
    left-padded + masked inside _embed_window). One vector per series.

    Keeps:  global shape and trend of the *full* series — unlike the original
            TSRetriever, which simply truncated everything past step 512.
    Loses:  high-frequency detail — short spikes and brief anomalies get
            smoothed away by the downsampling.
    Use when questions are about overall shape/trend, not local events.

    Multi-series items use the mean of the per-series embeddings.
    """

    def _embed_series(self, series: List[float]) -> np.ndarray:
        arr = _zscore(series)
        if arr.size > self.SEQ_LEN:
            # Linear interpolation onto exactly SEQ_LEN points.
            src = np.linspace(0.0, 1.0, num=arr.size)
            dst = np.linspace(0.0, 1.0, num=self.SEQ_LEN)
            arr = np.interp(dst, src, arr).astype(np.float32)
        return self._embed_window(arr)

    def _embed_item(self, input_ts: List[List[float]]) -> np.ndarray:
        return np.mean([self._embed_series(s) for s in input_ts], axis=0)
