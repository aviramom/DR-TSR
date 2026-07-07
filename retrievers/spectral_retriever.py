from typing import Any, Dict, List

import numpy as np

from retrievers.base_retriever import BaseRetriever


def _magnitude_spectrum(series: List[float], n_bins: int, top_k: int) -> np.ndarray:
    """FFT magnitude spectrum of one series on a fixed normalized-frequency grid.

    The rFFT magnitude (DC bin dropped — mean offset is not rhythm) is
    interpolated onto n_bins points over normalized frequency (0, 0.5], so
    series of different lengths become comparable vectors. All but the top_k
    largest components are zeroed to suppress broadband noise, and the result
    is normalized by total power (L2 norm == sqrt of power by Parseval), so
    amplitude scale differences don't dominate the cosine similarity.
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < 4:
        return np.zeros(n_bins, dtype=np.float32)

    mags = np.abs(np.fft.rfft(arr - arr.mean()))[1:]  # drop DC
    freqs = np.fft.rfftfreq(arr.size)[1:]             # (0, 0.5]

    grid = np.linspace(freqs[0], 0.5, num=n_bins)
    spec = np.interp(grid, freqs, mags)

    if top_k < n_bins:
        cutoff = np.partition(spec, -top_k)[-top_k]
        spec = np.where(spec >= cutoff, spec, 0.0)

    norm = np.linalg.norm(spec)
    if norm > 1e-12:
        spec = spec / norm
    return spec.astype(np.float32)


class SpectralRetriever(BaseRetriever):
    """Cosine kNN retriever over FFT magnitude spectra.

    Captures periodicity, dominant frequencies, and rhythmic structure —
    global frequency content that neither DINOv2 (visual appearance) nor
    MOMENT (learned shape features) reliably encodes. Pure numpy: no
    pretrained encoder, no GPU.

    Multi-series items use the mean of the per-series normalized spectra
    (re-normalized), mirroring the mean-pooling of the encoder retrievers.
    """

    def __init__(
        self,
        n_bins: int = 128,
        top_k: int = 32,
        device: str = "cpu",  # accepted for interface parity; unused (no encoder)
    ) -> None:
        self._n_bins = n_bins
        self._top_k = top_k
        self._pool_items: List[Dict[str, Any]] = []
        self._pool_vecs: np.ndarray = np.empty((0,))
        self._id_to_row: Dict[Any, int] = {}

    def _embed_item(self, input_ts: List[List[float]]) -> np.ndarray:
        vecs = [_magnitude_spectrum(s, self._n_bins, self._top_k) for s in input_ts]
        vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(vec)
        return (vec / norm if norm > 1e-12 else vec).astype(np.float32)

    def index(self, pool: List[Dict[str, Any]]) -> None:
        self._pool_items = list(pool)
        self._pool_vecs = np.stack([
            self._embed_item(item["input_ts"]) for item in self._pool_items
        ])
        self._id_to_row = self._build_id_map(self._pool_items)
        print(f"[SpectralRetriever] indexed {len(self._pool_items)} items  shape={self._pool_vecs.shape}")

    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        row = self._id_to_row.get(query.get("id"))
        q_vec = self._pool_vecs[row] if row is not None else self._embed_item(query["input_ts"])
        return self._cosine_top_k(q_vec, self._pool_vecs, self._pool_items, k, query.get("id"), query.get("tid"))
