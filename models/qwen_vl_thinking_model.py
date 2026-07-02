"""
QwenVLThinkingModel — Qwen3-VL-8B with thinking mode enabled.

Identical to QwenVLImageModel (time series as matplotlib plots) except:
  - enable_thinking=True in apply_chat_template()
  - max_new_tokens default is 2048 to accommodate <think>...</think> block
  - The </think> stripping is load-bearing here (fires on every inference call)

Model ID: "Qwen/Qwen3-VL-8B-Thinking" — uses that checkpoint if available on HF,
falls back to Qwen3-VL-8B-Instruct weights (which also support thinking mode).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoProcessor

from models.base_model import BaseModelWrapper
from models.qwen_vl_image_model import _plot_ts_to_pil, _split_on_placeholders


class QwenVLThinkingModel(BaseModelWrapper):
    """Qwen3-VL-8B with thinking enabled: passes matplotlib plots + reasons before answering."""

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
            "model_type": "qwen_vl_thinking",
            "max_seq_length": 2048,    # 1-shot input is ~700 tokens; 2048 gives safe headroom
            "max_new_tokens": 8192,    # thinking_budget=2048 + answer; 4096 was exhausted by thinking alone
            "thinking_budget": 2048,   # 512 was too small — model reopens reasoning in answer section
            "format": "chat",
            "input_mode": "separate",
        }

    def load_model(self, model_path: Optional[str] = None, cache_dir: Optional[str] = None):
        from transformers import AutoModelForVision2Seq

        path = model_path or self.method or self.MODEL_ID
        cache = cache_dir or self.cache_dir

        print(f"[QwenVLThinkingModel] Loading {path} via transformers (thinking enabled)")

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
        max_new_tokens: int = 4096,
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

            budget = getattr(self.args, "thinking_budget", 2048)
            messages = [
                {"role": "system", "content": f"You are a helpful assistant. Think for at most {budget} tokens before answering."},
                {"role": "user", "content": content},
            ]

            # ── 5. Apply chat template with thinking enabled ──
            try:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                    thinking_budget=budget,
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
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=20,
                )

            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            # skip_special_tokens=False so <think>/</think> survive decoding
            # (they are special tokens in Qwen3-VL and would otherwise be stripped silently)
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            # Strip thinking block and special tokens. Two failure modes handled:
            # - No </think>: model hit max_new_tokens mid-thought → use last 400 chars of thinking.
            # - </think> present but answer empty: EOS fired right after </think> → fall back
            #   to last 400 chars of the thinking block (conclusion is typically written there).
            _special = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")
            if "</think>" in response:
                think_block, answer = response.split("</think>", 1)
                for tok in _special:
                    answer = answer.replace(tok, "")
                answer = answer.strip()
                if answer:
                    response = answer
                else:
                    for tok in _special:
                        think_block = think_block.replace(tok, "")
                    response = think_block[-400:].strip()
            else:
                for tok in _special:
                    response = response.replace(tok, "")
                response = response[-400:].strip()

            results.append(response)

        return results
