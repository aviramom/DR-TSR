"""
TimeOmniHFWrapper — anton-hugging/TimeOmni-1-7B (Qwen2.5-7B-Instruct base).

Time series are passed as numeric arrays embedded in the prompt text
(input_mode="combined"), identical to InstructModel.  Uses the original
training system prompt and hard-coded Qwen chat format (bypasses
apply_chat_template, which is unreliable for this checkpoint).
"""

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

    # Exact system prompt the model was trained with (AntonGuan/TimeOmni-1 repo).
    _SYSTEM_PROMPT = (
        "Output Format:\n"
        "<think>Your step-by-step reasoning process that justifies your answer</think>\n"
        "<answer>Your final answer"
        "(Note: Only output a single uppercase letter of the correct option)</answer>"
    )

    @staticmethod
    def _build_prompt(system: str, question: str) -> str:
        # Hard-coded Qwen chat format — mirrors build_legacy_prompt from the
        # TimeOmni-1 repo. More reliable than apply_chat_template for this
        # checkpoint because it bypasses any jinja2 template issues.
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @staticmethod
    def _extract_answer(text: str) -> str:
        import re
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

        formatted = [
            self._build_prompt(self._SYSTEM_PROMPT, q)
            for q in batch["input_text"]
        ]

        # add_special_tokens=False: the formatted string already contains all
        # Qwen special tokens (<|im_start|> etc); letting the tokenizer add more
        # would corrupt the prompt structure.
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_seq_length,
            add_special_tokens=False,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=True,
                temperature=0.1,
                top_p=0.001,
                top_k=20,
                repetition_penalty=1.05,
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
