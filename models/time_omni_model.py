"""
TimeOmniHFWrapper — anton-hugging/TimeOmni-1-7B (Qwen2.5-7B-Instruct base).

Time series are passed as numeric arrays embedded in the prompt text
(input_mode="combined"), identical to InstructModel.  The model generates
<think>...</think><answer>X</answer> output; _extract_answer strips the
reasoning block and returns only the answer content to the evaluator.
"""

import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base_model import BaseModelWrapper


class TimeOmniHFWrapper(BaseModelWrapper):

    def __init__(self, args: Any, device: str = "cuda"):
        self.args = args
        self.device = device
        self.method: Optional[str] = getattr(args, "method", None)
        self.cache_dir: Optional[str] = getattr(args, "cache_dir", None)
        self.model = None
        self.tokenizer: Optional[AutoTokenizer] = None

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        return {
            "model_type": "time_omni",
            "max_seq_length": 4096,
            "max_new_tokens": 512,
            "format": "chat",
            "input_mode": "combined",
        }

    def load_model(self, model_path: Optional[str] = None, cache_dir: Optional[str] = None):
        path = model_path or self.method
        cache = cache_dir or self.cache_dir

        print(f"[TimeOmniHFWrapper] Loading {path}")

        hf_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=cache,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(path, **hf_kwargs).to(self.device)
        except (ValueError, KeyError, OSError):
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(path, **hf_kwargs).to(self.device)

        self.model.eval()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, use_fast=False, trust_remote_code=True, cache_dir=cache,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True, cache_dir=cache,
            )

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.model, self.tokenizer

    @staticmethod
    def _extract_answer(text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return text.strip()

    def generate(
        self,
        batch: Dict[str, Any],
        max_new_tokens: int = 512,
        pred_only: bool = True,
        **generate_kwargs,
    ) -> List[str]:
        if self.model is None or self.tokenizer is None:
            self.load_model()

        prompts = batch["input_text"]

        def _apply_template(q: str) -> str:
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

        formatted = [_apply_template(q) for q in prompts]

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_seq_length,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs,
            )

        input_len = inputs["input_ids"].shape[1]
        results = []
        for i in range(output_ids.shape[0]):
            gen_ids = output_ids[i][input_len:]
            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(self._extract_answer(decoded))

        return results
