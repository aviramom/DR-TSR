"""
ImageInstructModel — Qwen3.6-27B (vision encoder) with time series passed as matplotlib plots.

The model ID "Qwen/Qwen3.6-27B-image-ts" is a registry alias: the `-image-ts` suffix is
stripped internally so the same cached checkpoint is used as the text-only variant.

Prompt design: numeric TS arrays in `input_text` are replaced by inline images.
All surrounding text (ICL structure, labels, instructions) is preserved verbatim.
"""

import io
import os
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


class ImageInstructModel(BaseModelWrapper):
    """Qwen3.6-27B image-TS variant: passes matplotlib plots instead of numeric arrays."""

    def __init__(self, args: Any, device: str = "cuda"):
        self.args = args
        self.device = device
        self.method: Optional[str] = getattr(args, "method", None)
        self.cache_dir: Optional[str] = getattr(args, "cache_dir", None)
        self._vllm_llm = None
        self.processor: Optional[AutoProcessor] = None

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        return {
            "model_type": "image_instruct",
            "max_seq_length": 4096,
            "max_new_tokens": 50,
            "format": "chat",
            "input_mode": "separate",
        }

    def load_model(self, model_path: Optional[str] = None, cache_dir: Optional[str] = None):
        from vllm import LLM

        path = model_path or self.method
        cache = cache_dir or self.cache_dir

        if path is None:
            raise ValueError("No model_path provided for ImageInstructModel")

        # Strip registry suffix — actual checkpoint has no such suffix
        hf_path = path.removesuffix("-image-ts")

        n_gpus = torch.cuda.device_count() or 1
        print(f"[ImageInstructModel] Loading {hf_path} via vLLM (tensor_parallel_size={n_gpus})")

        self._vllm_llm = LLM(
            model=hf_path,
            download_dir=cache,
            tensor_parallel_size=n_gpus,
            dtype="bfloat16",
            max_model_len=max(getattr(self.args, "max_seq_length", 4096), 16384),
            trust_remote_code=True,
            enforce_eager=True,
            disable_custom_all_reduce=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            hf_path, trust_remote_code=True, cache_dir=cache
        )
        # model attribute expected by base class not-None check in generate()
        self.model = self._vllm_llm
        return self._vllm_llm, self.processor

    def generate(
        self,
        batch: Dict[str, Any],
        max_new_tokens: int = 50,
        pred_only: bool = False,
        **generate_kwargs,
    ) -> List[str]:
        if self._vllm_llm is None:
            self.load_model()

        from vllm import SamplingParams

        questions: List[str] = batch["input_text"]
        raw_ts_batch: List[Any] = batch.get("input_ts", [])
        max_tok = getattr(self.args, "max_new_tokens", max_new_tokens)

        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tok)

        vllm_inputs = []
        for q, ts_item in zip(questions, raw_ts_batch):
            # ── 1. Split on placeholders to get surrounding text segments ──
            text_parts, n_expected = _split_on_placeholders(q)

            # ── 2. Collect time series for this sample ──
            # ts_item is a list of series (support examples + query)
            if isinstance(ts_item, (list, tuple)) and len(ts_item) > 0 and isinstance(ts_item[0], (list, np.ndarray, torch.Tensor)):
                series_list = list(ts_item)
            else:
                series_list = [ts_item]

            if len(series_list) != n_expected:
                # Mismatch — fall back gracefully by using whatever we have
                # (e.g., extra metadata arrays in input_text)
                # Trim or pad series_list to match n_expected
                if len(series_list) > n_expected:
                    series_list = series_list[:n_expected]
                else:
                    series_list = series_list + [series_list[-1]] * (n_expected - len(series_list))

            # ── 3. Plot each series ──
            pil_images = [_plot_ts_to_pil(ts) for ts in series_list]

            # ── 4. Build interleaved content list ──
            # text_parts: [before_ts1, between_ts1_ts2, ..., after_last_ts]
            # Interleave: text_parts[0], image0, text_parts[1], image1, ...
            content: List[Dict] = []
            for idx, txt in enumerate(text_parts):
                if txt:
                    content.append({"type": "text", "text": txt})
                if idx < n_expected:
                    content.append({"type": "image"})

            # ── 5. Apply chat template ──
            try:
                prompt = self.processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self.processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

            vllm_inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": pil_images},
                }
            )

        outputs = self._vllm_llm.generate(vllm_inputs, sampling_params)

        def _extract_answer(text: str) -> str:
            if "</think>" in text:
                text = text.split("</think>", 1)[1]
            return text.strip()

        return [_extract_answer(o.outputs[0].text) for o in outputs]
