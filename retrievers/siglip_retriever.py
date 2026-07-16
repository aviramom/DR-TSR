from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from retrievers.base_retriever import BaseRetriever
from retrievers.delay_dino_retriever import _delay_embed_2d
from retrievers.vision_ts_retriever import _render_ts


def _l2(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 1e-8 else vec).astype(np.float32)


def _to_vec(feats) -> np.ndarray:
    """get_image_features/get_text_features return a raw tensor on
    transformers 4.x but a ModelOutput (with pooler_output) on 5.x."""
    if not torch.is_tensor(feats):
        feats = feats.pooler_output
    return feats.squeeze(0).cpu().numpy()


class SigLIPRetriever(BaseRetriever):
    """Cosine kNN retriever over fused SigLIP image+text embeddings.

    SigLIP maps images and text into one shared space, so — unlike the
    DINOv3 retrievers, which see only the series image — each item here is
    represented by a single fused vector pooling both modalities:

        fused = l2( mean( l2(image_emb), l2(text_emb) ) )

    where `image_emb` encodes the time series rendered as an image and
    `text_emb` encodes the question text (SigLIP's text tower truncates to
    its 64-token window, enough for TSE question stems).

    Two image representations are supported via `image_mode`, mirroring the
    existing DINOv3 pair so the encoder choice is the only difference:

      - "plot"  — z-score normalized line plot, all series on one figure
                  (the vision_ts rendering; one image per item).
      - "delay" — pool-level min/max normalized sliding-window delay
                  embedding (the delay_dino rendering; one image per
                  series, mean-pooled across series).

    Defaults to SigLIP2 (google/siglip2-base-patch16-224, ungated; needs
    transformers >= 4.49). Pass model_name="google/siglip-base-patch16-224"
    to fall back to SigLIP1.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        device: str = "cuda",
        image_mode: str = "plot",
        image_size: int = 224,
        base_height: int = 256,
        base_width: int = 256,
        embed_ratio: float = 0.6,
        embed_lmin: int = 48,
        embed_lmax: int = 192,
    ) -> None:
        if image_mode not in ("plot", "delay"):
            raise ValueError(f"image_mode must be 'plot' or 'delay', got {image_mode!r}")
        self._model_name = model_name
        self._device = device
        self._image_mode = image_mode
        self._image_size = image_size
        self._base_height = base_height
        self._base_width = base_width
        self._embed_ratio = embed_ratio
        self._embed_lmin = embed_lmin
        self._embed_lmax = embed_lmax
        self._processor = None
        self._model = None
        self._pool_min: float = 0.0
        self._pool_max: float = 1.0
        self._pool_items: List[Dict[str, Any]] = []
        self._pool_vecs: np.ndarray = np.empty((0,))
        self._id_to_row: Dict[Any, int] = {}

    def _delay_image(self, series: List[float]) -> Image.Image:
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
        return Image.fromarray(np.stack([img] * 3, axis=-1))

    def _embed_image(self, img: Image.Image) -> np.ndarray:
        inputs = self._processor(images=img, return_tensors="pt").to(self._device)
        return _to_vec(self._model.get_image_features(pixel_values=inputs["pixel_values"]))

    def _embed_ts(self, input_ts: List[List[float]]) -> np.ndarray:
        if self._image_mode == "plot":
            return self._embed_image(_render_ts(input_ts, self._image_size))
        vecs = [self._embed_image(self._delay_image(s)) for s in input_ts]
        return np.mean(vecs, axis=0)

    def _embed_text(self, text: str) -> np.ndarray:
        # SigLIP's text tower is trained with max_length padding (64 tokens).
        inputs = self._processor(
            text=[text],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)
        return _to_vec(self._model.get_text_features(input_ids=inputs["input_ids"]))

    def _embed_item(self, item: Dict[str, Any]) -> np.ndarray:
        ts_vec = _l2(self._embed_ts(item["input_ts"]))
        txt_vec = _l2(self._embed_text(item["question"]))
        return _l2((ts_vec + txt_vec) / 2.0)

    def index(self, pool: List[Dict[str, Any]]) -> None:
        from transformers import AutoModel, AutoProcessor

        print(f"[SigLIPRetriever/{self._image_mode}] loading {self._model_name}")
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()
        self._pool_items = list(pool)

        if self._image_mode == "delay":
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
                raw_vecs.append(self._embed_item(item))

        self._pool_vecs = np.stack(raw_vecs).astype(np.float32)  # rows already L2-normalized
        self._id_to_row = self._build_id_map(self._pool_items)
        self._offload_encoder()
        print(
            f"[SigLIPRetriever/{self._image_mode}] indexed {len(self._pool_items)} items"
            f"  shape={self._pool_vecs.shape}"
        )

    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        row = self._id_to_row.get(query.get("id"))
        if row is not None:
            q_vec = self._pool_vecs[row]
        else:
            with torch.no_grad():
                q_vec = self._embed_item(query)
        return self._cosine_top_k(q_vec, self._pool_vecs, self._pool_items, k, query.get("id"), query.get("tid"))
