import math
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from retrievers.base_retriever import BaseRetriever


def _delay_embed_2d(
    x: np.ndarray,
    height: int,
    width: int,
    embed_ratio: float,
    embed_lmin: int,
    embed_lmax: int,
) -> np.ndarray:
    """Convert a normalized 1D series into a 2D delay-embedding image.

    A window of length l (embed_ratio * len(x), clamped to [lmin, lmax]) is
    slid across the series; each window becomes one image column, interpolated
    to the target height.
    """
    if x.size == 0:
        x = np.zeros(height, dtype=np.float32)

    length = int(x.size)
    raw_l = int(math.floor(embed_ratio * length))
    l = min(embed_lmax, max(embed_lmin, raw_l))
    l = max(1, min(l, length))

    max_start = max(0, length - l)
    delay = (max_start / float(width - 1)) if width > 1 else 0.0

    out = np.empty((height, width), dtype=np.float32)
    for col in range(width):
        start = min(int(round(col * delay)), max_start)
        window = x[start : start + l]
        if l == height:
            out[:, col] = window
        elif l == 1:
            out[:, col] = window[0]
        else:
            src = np.linspace(0.0, 1.0, num=l, dtype=np.float32)
            dst = np.linspace(0.0, 1.0, num=height, dtype=np.float32)
            out[:, col] = np.interp(dst, src, window).astype(np.float32)
    return out


class DelayDINORetriever(BaseRetriever):
    """Cosine kNN retriever over DINO embeddings of TS delay-embedding images.

    Each series is min-max rescaled to [0, 1] using pool-level min/max
    (computed once at index time and reused for queries, mirroring the
    train-stats-on-test convention), delay-embedded into a 2D grayscale
    image, stacked to RGB, and encoded with a DINO vision backbone. The
    CLS token is the image embedding. Multi-series items use the mean of
    the individual series embeddings.

    Defaults to DINOv3. These checkpoints are gated on Hugging Face — access
    must be requested and approved (per-account) on the model page before
    `from_pretrained` will succeed. To fall back to the ungated DINOv2, pass
    model_name="facebook/dinov2-base".
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: str = "cuda",
        base_height: int = 256,
        base_width: int = 256,
        embed_ratio: float = 0.6,
        embed_lmin: int = 48,
        embed_lmax: int = 192,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._base_height = base_height
        self._base_width = base_width
        self._embed_ratio = embed_ratio
        self._embed_lmin = embed_lmin
        self._embed_lmax = embed_lmax
        self._model = None
        self._transform = None
        self._pool_min: float = 0.0
        self._pool_max: float = 1.0
        self._pool_items: List[Dict[str, Any]] = []
        self._pool_vecs: np.ndarray = np.empty((0,))
        self._id_to_row: Dict[Any, int] = {}

    def _to_image(self, series: List[float]) -> Image.Image:
        arr = np.asarray(series, dtype=np.float32)
        if self._pool_max > self._pool_min:
            arr = (arr - self._pool_min) / (self._pool_max - self._pool_min)
        else:
            arr = np.zeros_like(arr)
        arr = np.clip(arr, 0.0, 1.0)

        x2d = _delay_embed_2d(
            arr, self._base_height, self._base_width,
            self._embed_ratio, self._embed_lmin, self._embed_lmax,
        )
        img = (x2d * 255.0).astype(np.uint8)
        rgb = np.stack([img] * 3, axis=-1)
        return Image.fromarray(rgb)

    def _embed_series(self, series: List[float]) -> np.ndarray:
        img = self._to_image(series)
        x = self._transform(img).unsqueeze(0).to(self._device)
        outputs = self._model(pixel_values=x)
        # CLS token = first token of last_hidden_state
        return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

    def _embed_item(self, input_ts: List[List[float]]) -> np.ndarray:
        """Mean-pool embeddings across all series in item["input_ts"]."""
        vecs = [self._embed_series(s) for s in input_ts]
        return np.mean(vecs, axis=0)

    def index(self, pool: List[Dict[str, Any]]) -> None:
        from torchvision import transforms as pth_transforms
        from transformers import AutoModel

        print(f"[DelayDINORetriever] loading {self._model_name}")
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()
        self._transform = pth_transforms.Compose([
            pth_transforms.Resize((224, 224), interpolation=pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self._pool_items = list(pool)

        all_values = np.concatenate([
            np.asarray(s, dtype=np.float32)
            for item in self._pool_items
            for s in item["input_ts"]
        ])
        self._pool_min = float(all_values.min())
        self._pool_max = float(all_values.max())

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
        print(f"[DelayDINORetriever] indexed {len(self._pool_items)} items  shape={self._pool_vecs.shape}")

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
