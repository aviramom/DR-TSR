from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # must be before pyplot import
from matplotlib import colormaps
import numpy as np
import pywt
import torch
from PIL import Image

from retrievers.base_retriever import BaseRetriever


def _scalogram(series: List[float], num_scales: int) -> np.ndarray:
    """Morlet CWT magnitude scalogram of one series, normalized to [0, 1].

    Rows are log-spaced scales (small scale / high frequency at the top),
    columns are time — so the image encodes *when* each frequency is
    present, unlike the FFT which only encodes global frequency content.
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < 4:
        return np.zeros((num_scales, max(arr.size, 4)), dtype=np.float32)

    std = arr.std()
    z = (arr - arr.mean()) / std if std > 1e-12 else np.zeros_like(arr)

    scales = np.geomspace(1, max(2, arr.size // 2), num=num_scales)
    coeffs, _ = pywt.cwt(z, scales, "morl")
    mags = np.abs(coeffs)

    peak = mags.max()
    if peak > 1e-12:
        mags = mags / peak
    return mags.astype(np.float32)


class WaveletRetriever(BaseRetriever):
    """Cosine kNN retriever over DINOv3 CLS embeddings of CWT scalograms.

    Same frozen vision encoder as VisionTSRetriever, different input: the
    time series is transformed into a Morlet-wavelet scalogram (scales ×
    time magnitude heatmap) and rendered as an image. Where the FFT loses
    temporal localization, the scalogram keeps it — "periodicity throughout"
    and "periodicity only in the first half" produce different images. The
    vision_ts / vision_wavelet pair therefore isolates what time-frequency
    localization adds over raw visual appearance.

    Multivariate items tile the per-channel scalograms vertically into one
    image (resized to a square before encoding, so any channel count fits).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: str = "cuda",
        image_size: int = 224,
        num_scales: int = 64,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._image_size = image_size
        self._num_scales = num_scales
        self._cmap = colormaps["viridis"]
        self._processor = None
        self._model = None
        self._pool_items: List[Dict[str, Any]] = []
        self._pool_vecs: np.ndarray = np.empty((0,))
        self._id_to_row: Dict[Any, int] = {}

    def _to_image(self, input_ts: List[List[float]]) -> Image.Image:
        grids = [_scalogram(s, self._num_scales) for s in input_ts]
        # Tile per-channel scalograms vertically; pad widths to the max so
        # channels of different lengths stack cleanly.
        max_w = max(g.shape[1] for g in grids)
        grids = [
            np.pad(g, ((0, 0), (0, max_w - g.shape[1]))) if g.shape[1] < max_w else g
            for g in grids
        ]
        tiled = np.concatenate(grids, axis=0)
        rgb = (self._cmap(tiled)[..., :3] * 255.0).astype(np.uint8)
        return Image.fromarray(rgb).resize(
            (self._image_size, self._image_size), Image.LANCZOS
        )

    def _embed_item(self, input_ts: List[List[float]]) -> np.ndarray:
        img = self._to_image(input_ts)
        inputs = self._processor(images=img, return_tensors="pt").to(self._device)
        outputs = self._model(**inputs)
        # CLS token = first token of last_hidden_state
        return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

    def index(self, pool: List[Dict[str, Any]]) -> None:
        from transformers import AutoImageProcessor, AutoModel

        print(f"[WaveletRetriever] loading {self._model_name}")
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()
        self._pool_items = list(pool)

        raw_vecs = []
        with torch.no_grad():
            for item in self._pool_items:
                raw_vecs.append(self._embed_item(item["input_ts"]))

        raw = np.stack(raw_vecs)  # (N, 768)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self._pool_vecs = (raw / norms).astype(np.float32)
        self._id_to_row = self._build_id_map(self._pool_items)
        self._offload_encoder()
        print(f"[WaveletRetriever] indexed {len(self._pool_items)} items  shape={self._pool_vecs.shape}")

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
