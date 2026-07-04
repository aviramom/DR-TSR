"""
ChatTS vLLM wrapper.

Loads ChatTS via vLLM (PagedAttention, continuous batching) for better
throughput compared to the HF pipeline wrapper in chatts_model.py.

Key benefits over chatts_model.py:
- PagedAttention: near-zero KV cache memory waste vs. HF's static allocation.
- Continuous batching: all samples submitted at once; vLLM schedules them.
- Tensor parallelism: spread a large model across GPUs via tensor_parallel_size.
- Faster decoding: vLLM's optimized CUDA kernels.

NOTE on ts embedding injection:
  ChatTS merges time series patch embeddings into the token sequence inside
  _merge_input_ids_with_time_series_features during each forward pass.
  For vLLM to route the timeseries tensors through to that method, ChatTS
  must be registered as a vLLM multimodal model (SupportsMultiModal).
  This wrapper passes timeseries via multi_modal_data — if the registration
  is not present, vLLM will ignore it and generate from token IDs only.
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoProcessor
from vllm import LLM, SamplingParams
from models.base_model import BaseModelWrapper


def _clean_ts_for_chatts(ts: Any) -> np.ndarray:
    """Convert ts input to a clean 1D float32 numpy array (same logic as chatts_model.py)."""
    if isinstance(ts, torch.Tensor):
        ts = ts.detach().cpu().numpy()
    elif isinstance(ts, list):
        ts = np.array(ts)
    ts = np.asarray(ts)
    if ts.ndim > 1:
        if ts.shape[0] < ts.shape[1] and ts.shape[0] < 20:
            ts = ts[-1, :]
        else:
            ts = ts[:, -1]
    ts_flat = ts.reshape(-1)
    ts_flat = np.nan_to_num(ts_flat, nan=0.0)
    ts_flat = np.array([0.0 if x is None else x for x in ts_flat], dtype=np.float32)
    return ts_flat


class ChatTSVLLMWrapper(BaseModelWrapper):
    """ChatTS wrapper backed by vLLM instead of the HF pipeline.

    Drop-in replacement for ChatTSHFWrapper — same generate() signature.
    """

    def __init__(self, args: Any, device: str = "cuda"):
        self.args = args
        self.device = device
        self.method: Optional[str] = getattr(args, "method", None)
        self.cache_dir: Optional[str] = getattr(args, "cache_dir", None)
        self.llm: Optional[LLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.processor: Optional[Any] = None
        self.ts_place_holder: str = getattr(args, "ts_place_holder", "<ts><ts/>")

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        return {
            "model_type": "chatts_vllm",
            "max_seq_length": 4096,
            "max_new_tokens": 50,
            "chatts_system_prompt": "You are a helpful assistant.",
            "ts_place_holder": "<ts><ts/>",
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": 1,
        }

    def load_model(self, model_path: Optional[str] = None, cache_dir: Optional[str] = None):
        path = model_path or self.method
        cache = cache_dir or self.cache_dir
        if path is None:
            raise ValueError("No model_path provided for ChatTSVLLMWrapper.load_model")

        # Strip the "-vllm" registry suffix — the real checkpoint has no such suffix.
        hf_path = path.removesuffix("-vllm")

        print(f"[ChatTS-vLLM] Loading model from {hf_path} ...")
        # Force vLLM V0 engine: Qwen3TSForCausalLM has no native vLLM implementation
        # so vLLM falls back to the Transformers backend. The V1 engine's Transformers
        # backend has a strict weight-initialization check that fails for models where
        # lm_head.weight is tied to embed_tokens.weight and not saved separately.
        os.environ["VLLM_USE_V1"] = "0"
        self.llm = LLM(
            model=hf_path,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=getattr(self.args, "gpu_memory_utilization", 0.9),
            tensor_parallel_size=getattr(self.args, "tensor_parallel_size", 1),
            max_model_len=getattr(self.args, "max_seq_length", 4096),
            download_dir=cache,
        )

        # Processor loaded separately — used to tokenize text+ts before handing to vLLM.
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_path, trust_remote_code=True, cache_dir=cache
        )
        self.tokenizer.padding_side = "left"

        self.processor = AutoProcessor.from_pretrained(
            hf_path,
            trust_remote_code=True,
            tokenizer=self.tokenizer,
            cache_dir=cache,
        )
        return self.llm, self.tokenizer

    def _build_prompt(self, user_text: str) -> str:
        system = str(getattr(self.args, "chatts_system_prompt", "You are a helpful assistant."))
        if "<|im_start|>" in user_text:
            return user_text
        return (
            "<|im_start|>system\n"
            f"{system}<|im_end|><|im_start|>user\n"
            f"{user_text}<|im_end|><|im_start|>assistant\n"
        )

    def format_ts(self, ts_list: Any, q: str) -> str:
        """Prefix question with ts length metadata and placeholders (same as chatts_model.py)."""
        if ts_list is None:
            return q
        placeholder = self.ts_place_holder
        lengths = []
        if hasattr(ts_list, "shape"):
            if len(ts_list.shape) == 1:
                lengths = [ts_list.shape[0]]
            else:
                lengths = [ts_list.shape[1] for _ in range(ts_list.shape[0])]
        elif isinstance(ts_list, list):
            if not ts_list:
                lengths = []
            elif isinstance(ts_list[0], (list, np.ndarray, torch.Tensor)):
                for item in ts_list:
                    if hasattr(item, "shape"):
                        lengths.append(item.shape[0])
                    elif isinstance(item, list):
                        lengths.append(len(item))
                    else:
                        lengths.append(1)
            else:
                lengths = [len(ts_list)]
        else:
            return q
        if not lengths:
            return q
        n = len(lengths)
        parts = [f"Time series {i+1} is of length {L}: {placeholder};" for i, L in enumerate(lengths)]
        prefix = f"There are {n} time series: " + " ".join(parts)
        return f"{prefix} {q}"

    @torch.no_grad()
    def generate(
        self,
        batch: Dict[str, Any],
        max_new_tokens: int = 300,
        pred_only: bool = True,
        **kwargs,
    ) -> List[str]:
        if self.llm is None:
            self.load_model()

        questions = batch["input_text"]
        raw_ts_batch = batch.get("input_ts")
        if raw_ts_batch is None:
            raise ValueError("Batch missing 'input_ts'")

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            skip_special_tokens=True,
        )

        # Build one vLLM input dict per sample.
        # Each dict carries pre-tokenized token IDs (from the HF processor, which
        # expands <ts><ts/> placeholders to the right number of patch tokens) plus
        # the raw timeseries tensors as multi_modal_data so vLLM can route them to
        # the model's _merge_input_ids_with_time_series_features.
        vllm_inputs = []
        prompts_text = []  # kept only for pred_only=False reconstruction

        for q, ts_item in zip(questions, raw_ts_batch):
            if isinstance(q, (tuple, list)):
                q = q[0] if len(q) > 0 else ""
            prompt = self._build_prompt(q)
            prompts_text.append(prompt)

            # --- Flatten ts for this sample ---
            ts_inputs: List[np.ndarray] = []
            if (
                isinstance(ts_item, list)
                and len(ts_item) > 0
                and not isinstance(ts_item[0], (int, float))
            ):
                for ts in ts_item:
                    ts_inputs.append(_clean_ts_for_chatts(ts))
            else:
                ts_inputs.append(_clean_ts_for_chatts(ts_item))

            # --- Tokenize: HF processor expands <ts><ts/> to patch token IDs ---
            processed = self.processor(
                text=[prompt],
                timeseries=ts_inputs,
                padding=False,
                return_tensors="pt",
            )

            vllm_inputs.append(
                {
                    "prompt_token_ids": processed["input_ids"][0].tolist(),
                    # timeseries list is passed as multi_modal_data so vLLM routes
                    # it to the model's forward (requires ChatTS vLLM registration).
                    "multi_modal_data": {
                        "timeseries": [torch.tensor(ts) for ts in ts_inputs]
                    },
                }
            )

        # vLLM handles batching internally via continuous batching.
        outputs = self.llm.generate(vllm_inputs, sampling_params)

        # vLLM returns only the generated tokens by default (prompt is not echoed).
        res = []
        for i, output in enumerate(outputs):
            text = output.outputs[0].text.strip()
            if not pred_only:
                text = prompts_text[i] + text
            res.append(text)

        return res
