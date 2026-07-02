"""
QwenVLThinkingVLLMModel — Qwen3-VL-8B with thinking mode, served via vLLM.

vLLM's continuous batching gives substantially better throughput than the HF
generate() loop in qwen_vl_thinking_model.py, making it the preferred backend
for full-scale experiments.

Registry key: "Qwen/Qwen3-VL-8B-Thinking-vllm"
The "-vllm" suffix is stripped internally; the checkpoint loaded is
"Qwen/Qwen3-VL-8B-Thinking" (or falls back to the Instruct weights if that
checkpoint is not yet published on HuggingFace).

Differences from ImageInstructModel (the 27B vLLM image model):
  - enable_thinking=True + thinking_budget in apply_chat_template()
  - skip_special_tokens=False in SamplingParams so </think> survives decoding
  - Manual cleanup of Qwen turn-end tokens after the </think> split
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoProcessor

from models.base_model import BaseModelWrapper
from models.qwen_vl_image_model import _plot_ts_to_pil, _split_on_placeholders

_QWEN_SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


class QwenVLThinkingVLLMModel(BaseModelWrapper):
    """Qwen3-VL-8B thinking model: matplotlib plots + vLLM backend."""

    def __init__(self, args: Any, device: str = "cuda"):
        self.args = args
        self.device = device
        self.method: Optional[str] = getattr(args, "method", None)
        self.cache_dir: Optional[str] = getattr(args, "cache_dir", None)
        self._vllm_llm = None
        self.model = None
        self.processor: Optional[AutoProcessor] = None

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        return {
            "model_type": "qwen_vl_thinking_vllm",
            "max_seq_length": 2048,    # 1-shot input is ~700 tokens; 2048 gives safe headroom
            "max_new_tokens": 8192,    # thinking_budget=2048 + answer; 4096 was exhausted by thinking alone
            "thinking_budget": 2048,   # soft hint; 512 is too small — model reopens reasoning after </think>
            "format": "chat",
            "input_mode": "separate",
        }

    def load_model(self, model_path: Optional[str] = None, cache_dir: Optional[str] = None):
        from vllm import LLM

        path = model_path or self.method
        cache = cache_dir or self.cache_dir

        if path is None:
            raise ValueError("No model_path provided for QwenVLThinkingVLLMModel")

        # Strip registry suffix to get the actual HF checkpoint path
        hf_path = path.removesuffix("-vllm")

        n_gpus = torch.cuda.device_count() or 1
        print(f"[QwenVLThinkingVLLMModel] Loading {hf_path} via vLLM "
              f"(tensor_parallel_size={n_gpus}, thinking enabled)")

        self._vllm_llm = LLM(
            model=hf_path,
            download_dir=cache,
            tensor_parallel_size=n_gpus,
            dtype="bfloat16",
            max_model_len=getattr(self.args, "max_seq_length", 4096) + getattr(self.args, "max_new_tokens", 4096),
            trust_remote_code=True,
            enforce_eager=True,
            disable_custom_all_reduce=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            hf_path, trust_remote_code=True, cache_dir=cache
        )
        self.model = self._vllm_llm
        return self._vllm_llm, self.processor

    def generate(
        self,
        batch: Dict[str, Any],
        max_new_tokens: int = 4096,
        pred_only: bool = False,
        **generate_kwargs,
    ) -> List[str]:
        if self._vllm_llm is None:
            self.load_model()

        from vllm import SamplingParams

        questions: List[str] = batch["input_text"]
        raw_ts_batch: List[Any] = batch.get("input_ts", [])
        max_tok = getattr(self.args, "max_new_tokens", max_new_tokens)
        budget = getattr(self.args, "thinking_budget", 512)

        # skip_special_tokens=False: <think>/<\think> are special tokens in Qwen3-VL
        # and would be silently stripped otherwise, making the </think> split impossible
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            max_tokens=max_tok,
            skip_special_tokens=False,
        )

        vllm_inputs = []
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
                    content.append({"type": "image"})

            # ── 5. Apply chat template with thinking enabled ──
            # Qwen3-VL accepts thinking_budget via apply_chat_template but silently ignores
            # it on some processor versions (no TypeError, budget just not injected).
            # We therefore also prepend a system message with the plaintext budget hint,
            # which the model was instruction-tuned to respect.
            messages = [
                {"role": "system", "content": f"You are a helpful assistant. Think for at most {budget} tokens before answering."},
                {"role": "user", "content": content},
            ]
            try:
                prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                    thinking_budget=budget,
                )
            except TypeError:
                try:
                    prompt = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                except TypeError:
                    prompt = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

            if len(vllm_inputs) == 0:
                budget_in_prompt = str(budget) in prompt or "think for at most" in prompt.lower()
                print(f"[QwenVLThinking] thinking_budget={budget} injected={budget_in_prompt} | prompt tail: {repr(prompt[-120:])}")
            vllm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": pil_images},
            })

        outputs = self._vllm_llm.generate(vllm_inputs, sampling_params)
        return [_extract_answer(o.outputs[0].text) for o in outputs]


def _extract_answer(text: str) -> str:
    """Strip thinking block and Qwen turn-end tokens, return just the label.

    Two failure modes are handled:
    - No </think>: model hit max_new_tokens mid-thought → return last 400 chars of thinking.
    - </think> present but empty answer: EOS fired immediately after </think> (model wrote
      its conclusion inside the thinking block then stopped) → fall back to last 400 chars
      of the thinking block, which typically contains the final reasoning step.
    """
    for tok in _QWEN_SPECIAL_TOKENS:
        text = text.replace(tok, "")

    if "</think>" not in text:
        return text[-400:].strip()

    think_block, answer = text.split("</think>", 1)
    answer = answer.strip()
    if answer:
        return answer
    return think_block[-400:].strip()
