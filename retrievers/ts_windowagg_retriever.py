from typing import List

import numpy as np

from retrievers.moment_base import MomentSingleVecRetriever, _split_windows, _zscore


class TSWindowAggRetriever(MomentSingleVecRetriever):
    """Strategy C — window the series, embed every window, aggregate to one vector.

    The series is split into non-overlapping 512-step windows (the trailing
    partial window is left-padded + masked), each window is embedded at full
    resolution, and the window vectors are collapsed into a single vector by
    a length-weighted average — a window covering 512 real steps counts
    proportionally more than a short trailing remainder.

    Keeps:  every timestep is *seen* by the encoder (unlike compress-A, which
            downsamples), while the index stays one-vector-per-item (unlike
            multivec-B).
    Loses:  cross-window averaging blurs *which* window carried the signal —
            a localized anomaly's contribution is diluted by the other windows.
    The practical compromise between A and B.

    Multi-series items pool the windows of all their series into the same
    weighted average.
    """

    def _embed_item(self, input_ts: List[List[float]]) -> np.ndarray:
        vecs, weights = [], []
        for series in input_ts:
            arr = _zscore(series)  # z-score the full series, then window it
            for window in _split_windows(arr, self.SEQ_LEN):
                vecs.append(self._embed_window(window))
                weights.append(window.size)
        return np.average(np.stack(vecs), axis=0, weights=weights)
