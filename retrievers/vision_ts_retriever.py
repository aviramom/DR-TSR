import io
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from retrievers.base_retriever import BaseRetriever


def _zscore(series: List[float]) -> np.ndarray:
    arr = np.array(series, dtype=np.float32)
    std = arr.std()
    return (arr - arr.mean()) / std if std > 1e-8 else np.zeros_like(arr)


def _render_ts(series_list: List[List[float]], image_size: int = 224) -> Image.Image:
    """Render all series in one z-score normalized line plot → PIL RGB image."""
    colors = ["#1f77b4", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(image_size / 100, image_size / 100), dpi=100)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for i, s in enumerate(series_list):
        ax.plot(_zscore(s), color=colors[i % len(colors)], linewidth=1.5)
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((image_size, image_size), Image.LANCZOS)


class VisionTSRetriever(BaseRetriever):
    """Cosine kNN retriever over DINOv2 CLS embeddings of TS line plots.

    Each item's time series are rendered as a z-score normalized line plot
    (all series on one figure) and encoded with DINOv2. The CLS token
    (768-dim for dinov2-base) is used as the image embedding.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "cuda",
        image_size: int = 224,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._image_size = image_size
        self._processor = None
        self._model = None
        self._pool_items: List[Dict[str, Any]] = []
        self._pool_vecs: np.ndarray = np.empty((0,))

    def _embed_image(self, img: Image.Image) -> np.ndarray:
        inputs = self._processor(images=img, return_tensors="pt").to(self._device)
        outputs = self._model(**inputs)
        # CLS token = first token of last_hidden_state
        return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()  # (768,)

    def index(self, pool: List[Dict[str, Any]]) -> None:
        from transformers import AutoImageProcessor, AutoModel

        print(f"[VisionTSRetriever] loading {self._model_name}")
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()
        self._pool_items = list(pool)

        raw_vecs = []
        with torch.no_grad():
            for item in self._pool_items:
                img = _render_ts(item["input_ts"], self._image_size)
                raw_vecs.append(self._embed_image(img))

        raw = np.stack(raw_vecs)  # (N, 768)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self._pool_vecs = (raw / norms).astype(np.float32)
        print(f"[VisionTSRetriever] indexed {len(self._pool_items)} items  shape={self._pool_vecs.shape}")

    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        img = _render_ts(query["input_ts"], self._image_size)
        with torch.no_grad():
            q_vec = self._embed_image(img)
        norm = np.linalg.norm(q_vec)
        q_vec = (q_vec / norm if norm > 1e-8 else q_vec).astype(np.float32)
        return self._cosine_top_k(q_vec, self._pool_vecs, self._pool_items, k, query.get("id"), query.get("tid"))
