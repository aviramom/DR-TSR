"""
QwenVLImageModel — Qwen3-VL-8B-Instruct with time series passed as matplotlib plots.

Prompt design: numeric TS arrays in `input_text` are replaced by inline images.
All surrounding text (ICL structure, labels, instructions) is preserved verbatim.
Uses HuggingFace transformers (AutoModelForVision2Seq) — no vLLM dependency.
"""

import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from models.base_model import BaseModelWrapper


def _plot_ts_to_pil(ts_values: Any) -> Image.Image:
    """Convert a 1D time series to a line-plot with axes, mean line, and ±1 std band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if isinstance(ts_values, torch.Tensor):
        ts_values = ts_values.detach().cpu().numpy()
    arr = np.asarray(ts_values, dtype=float).ravel()
    mask = ~np.isnan(arr)
    xs = np.where(mask)[0]
    ys = arr[mask]

    mean = float(np.mean(ys))
    std = float(np.std(ys))

    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=100)

    ax.axhspan(mean - std, mean + std, alpha=0.15, color="steelblue", linewidth=0)
    ax.axhline(mean, color="tomato", linewidth=1.0, linestyle="--")
    ax.plot(xs, ys, linewidth=1.5, color="steelblue")

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    ax.annotate(
        f"μ={mean:.2f}  σ={std:.2f}",
        xy=(0.99, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
    )

    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


_TS_PLACEHOLDER = "<ts><ts/>"


def _split_on_placeholders(text: str) -> Tuple[List[str], int]:
    """Split text on TS placeholders, return (text_parts, n_placeholders)."""
    parts = text.split(_TS_PLACEHOLDER)
    return parts, len(parts) - 1


class QwenVLImageModel(BaseModelWrapper):
    """Qwen3-VL-8B-Instruct image-TS variant: passes matplotlib plots instead of numeric arrays."""

    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(self, args: Any, device: str = "cuda"):
        self.args = args
        self.device = device
        self.method: Optional[str] = getattr(args, "method", None)
        self.cache_dir: Optional[str] = getattr(args, "cache_dir", None)
        self.model = None
        self.processor: Optional[AutoProcessor] = None

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        return {
            "model_type": "qwen_vl_image",
            "max_seq_length": 4096,
            "max_new_tokens": 50,
            "format": "chat",
            "input_mode": "separate",
        }

    def load_model(self, model_path: Optional[str] = None, cache_dir: Optional[str] = None):
        from transformers import AutoModelForVision2Seq

        path = model_path or self.method or self.MODEL_ID
        cache = cache_dir or self.cache_dir

        print(f"[QwenVLImageModel] Loading {path} via transformers")

        self.model = AutoModelForVision2Seq.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=cache,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            path,
            trust_remote_code=True,
            cache_dir=cache,
        )

        return self.model, self.processor

    def generate(
        self,
        batch: Dict[str, Any],
        max_new_tokens: int = 50,
        pred_only: bool = False,
        **generate_kwargs,
    ) -> List[str]:
        if self.model is None:
            self.load_model()

        questions: List[str] = batch["input_text"]
        raw_ts_batch: List[Any] = batch.get("input_ts", [])
        max_tok = getattr(self.args, "max_new_tokens", max_new_tokens)

        results = []
        for q, ts_item in zip(questions, raw_ts_batch):
            # ── 1. Split on placeholders ──
            text_parts, n_expected = _split_on_placeholders(q)

            # ── 2. Collect time series for this sample ──
            if isinstance(ts_item, (list, tuple)) and len(ts_item) > 0 and isinstance(ts_item[0], (list, np.ndarray, torch.Tensor)):
                series_list = list(ts_item)
            else:
                series_list = [ts_item]

            if len(series_list) != n_expected:
                if len(series_list) > n_expected:
                    series_list = series_list[:n_expected]
                else:
                    series_list = series_list + [series_list[-1]] * (n_expected - len(series_list))

            # ── 3. Plot each series ──
            pil_images = [_plot_ts_to_pil(ts) for ts in series_list]

            # ── 4. Build interleaved content list ──
            content: List[Dict] = []
            for idx, txt in enumerate(text_parts):
                if txt:
                    content.append({"type": "text", "text": txt})
                if idx < n_expected:
                    content.append({"type": "image", "image": pil_images[idx]})

            messages = [{"role": "user", "content": content}]

            # ── 5. Apply chat template ──
            try:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            # ── 6. Tokenize + encode images ──
            inputs = self.processor(
                text=[text],
                images=pil_images if pil_images else None,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            # ── 7. Generate ──
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tok,
                    do_sample=False,
                )

            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if "</think>" in response:
                response = response.split("</think>", 1)[1]
            results.append(response.strip())

        return results
