import os
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
import torch
from datasets import Dataset
from models.base_model import BaseModelWrapper
from utils.ts_serialize import fill_ts_placeholders


class InstructModel(BaseModelWrapper):
    """Wrapper for a multimodal QA model (HF causal LM) that implements the
    BaseModelWrapper interface used by the project.

    This class focuses on inference/generation and lazy-loading of the
    underlying HF model and tokenizer. It intentionally stays lightweight
    compared to training wrappers.
    """

    def __init__(self, args: Any, device: str = "cuda"):
        self.args = args

        self.device = device
        # model_path/config
        self.method: Optional[str] = getattr(args, "method", None)
        self.cache_dir: Optional[str] = getattr(args, "cache_dir", None)

    # (no quantization / low_cpu_mem_usage by default in this wrapper)

        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        #self.load_model()

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        return {
            "model_type": "instruct",
            "max_seq_length": 32768,
            "max_new_tokens": 10,
            "format": "chat",
        }


    def load_model(self, model_path: Optional[str] = None, cache_dir: Optional[str] = None) -> Any:
        """Load (or reuse) HF model and tokenizer. Returns (model, tokenizer).

        Uses bitsandbytes quantization when requested.
        """
        path = model_path or self.method
        cache = cache_dir or self.cache_dir

        if path is None:
            raise ValueError("No model_path provided for MQAModel.load_model")

        quantization = getattr(self.args, "quantization", "none")
        hf_kwargs = dict(
            trust_remote_code=True,
            cache_dir=cache,
            torch_dtype=torch.bfloat16,
        )
        try:
            import flash_attn  # noqa: F401
            hf_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            print("[InstructModel] flash_attn not installed, using default attention")
        if quantization == "8bit":
            hf_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            hf_kwargs.pop("torch_dtype")  # incompatible with bitsandbytes quantization
        elif quantization == "4bit":
            hf_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            hf_kwargs.pop("torch_dtype")

        # Load model and tokenizer
        if quantization != "none":
            self.model = AutoModelForCausalLM.from_pretrained(path, **hf_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, **hf_kwargs).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                      trust_remote_code=True,
                                                     return_tensors="pt",
                                                     max_length=self.args.max_seq_length,
                                                     padding=True, 
                                                     truncation=True,
                                                      cache_dir=cache)

        # Ensure padding token
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        pipe_kwargs = dict(model=self.model, tokenizer=self.tokenizer)
        if quantization != "none":
            pipe_kwargs["device_map"] = "auto"
        else:
            pipe_kwargs["device"] = self.device
        self.pipeline = transformers.pipeline("text-generation", **pipe_kwargs)


        return self.model, self.tokenizer

    def generate(self, batch: Optional[List[str]] = None, max_new_tokens: int = 50, pred_only: bool = False, **generate_kwargs) -> List[str]:
        """Concise generate: assume `batch` is a list of prompt strings.

        Tokenizes prompts, moves tensors to the model device, calls
        `model.generate(...)`, and returns decoded strings.

        Args:
            batch: A dict containing key "input_text" with a list of prompt strings.
            max_new_tokens: Max new tokens to generate (overridden by args if set).
            pred_only: If True, return only the generated continuation (without the prompt).
            **generate_kwargs: Additional generation kwargs forwarded to model.generate.

        Returns:
            List[str]: Decoded strings per input. If pred_only is True, only the continuation.
        """
        input_texts = batch["input_text"]
        input_ts_list = batch.get("input_ts", [[] for _ in input_texts])
        batch = [fill_ts_placeholders(t, ts) for t, ts in zip(input_texts, input_ts_list)]

        if self.model is None or self.tokenizer is None:
            self.load_model()


        #print("num conversations:", len(prompts))
        #print("first conversation:", prompts[0])
        def _extract_answer(text: str) -> str:
            # Qwen3/Qwen3.5 thinking models wrap reasoning in <think>...</think>;
            # strip that block so callers only see the final answer.
            if "</think>" in text:
                text = text.split("</think>", 1)[1]
            return text.strip()

        if self.args.format == "chat":
            def _apply_template(q):
                try:
                    return self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": q}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,  # disable Qwen3/3.6 thinking mode
                    )
                except TypeError:
                    return self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": q}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )

            prompts = [_apply_template(q) for q in batch]

            outputs = self.pipeline(
                prompts,
                max_new_tokens=self.args.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                return_full_text=False,
                batch_size=self.args.batch_size,
                do_sample=False,
                **generate_kwargs
            )
        else:
            outputs = []
            for q in batch:
                output = self.pipeline(
                    [{"role": "user", "content": q}],
                    max_new_tokens=self.args.max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_full_text=False,
                    batch_size=self.args.batch_size,
                    do_sample=False,
                    **generate_kwargs
                )
                outputs.append(output)
        texts = [_extract_answer(o[0]["generated_text"]) for o in outputs]

        return texts


class LargeInstructModel(InstructModel):
    """InstructModel variant for models that exceed single-GPU VRAM.

    Uses vLLM (tensor parallelism across all GPUs) when available — 5-10x faster
    than HF pipeline parallelism. Falls back to HF device_map='auto' otherwise.
    """

    def __init__(self, args: Any, device: str = "cuda"):
        super().__init__(args, device)
        self._use_vllm = False
        self._vllm_llm = None

    def load_model(self, model_path: Optional[str] = None, cache_dir: Optional[str] = None):
        path = model_path or self.method
        cache = cache_dir or self.cache_dir

        if path is None:
            raise ValueError("No model_path provided for LargeInstructModel.load_model")

        # "-vllm" is a routing suffix (selects this vLLM wrapper), not part of the
        # checkpoint name — strip it to get the real HF repo id.
        if path.endswith("-vllm"):
            path = path[: -len("-vllm")]

        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoConfig
            n_gpus = torch.cuda.device_count() or 1
            print(f"[LargeInstructModel] Loading via vLLM (tensor_parallel_size={n_gpus})")

            # vLLM refuses to start if max_model_len exceeds the checkpoint's trained
            # context window (max_position_embeddings) — going past it yields NaNs from
            # RoPE. Clamp our requested length to the model's real limit instead of
            # crashing (Qwen3-8B tops out at 40960, not the 50000 default).
            hf_cfg = AutoConfig.from_pretrained(
                path, trust_remote_code=True, cache_dir=cache,
            )
            model_ctx = getattr(hf_cfg, "max_position_embeddings", None) or 40960
            max_model_len = min(max(self.args.max_seq_length, 16384), model_ctx)
            print(f"[LargeInstructModel] max_model_len={max_model_len} "
                  f"(requested {self.args.max_seq_length}, model ctx {model_ctx})")

            self._vllm_llm = LLM(
                model=path,
                download_dir=cache,
                tensor_parallel_size=n_gpus,
                dtype="bfloat16",
                max_model_len=max_model_len,
                trust_remote_code=True,
                enforce_eager=True,  # skip CUDA graph compilation (30-60 min for 27B)
                disable_custom_all_reduce=True,  # required when GPUs lack direct P2P (non-adjacent PCIe)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True, cache_dir=cache,
            )
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self._use_vllm = True
            self.model = self._vllm_llm  # satisfy base-class not-None check
            return self.model, self.tokenizer
        except ImportError:
            print("[LargeInstructModel] vLLM not available, falling back to HF pipeline")

        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            cache_dir=cache,
            device_map="auto",
            torch_dtype="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            cache_dir=cache,
            return_tensors="pt",
            max_length=self.args.max_seq_length,
            padding=True,
            truncation=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        return self.model, self.tokenizer

    def generate(self, batch: Optional[List[str]] = None, max_new_tokens: int = 50,
                 pred_only: bool = False, **generate_kwargs) -> List[str]:
        if self.model is None or self.tokenizer is None:
            self.load_model()

        if not self._use_vllm:
            return super().generate(batch, max_new_tokens, pred_only, **generate_kwargs)

        from vllm import SamplingParams

        input_texts = batch["input_text"]
        input_ts_list = batch.get("input_ts", [[] for _ in input_texts])
        prompts_raw = [fill_ts_placeholders(t, ts) for t, ts in zip(input_texts, input_ts_list)]

        def _apply_template(q):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

        prompts = [_apply_template(q) for q in prompts_raw]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=getattr(self.args, "max_new_tokens", max_new_tokens),
        )
        outputs = self._vllm_llm.generate(prompts, sampling_params)

        def _extract_answer(text: str) -> str:
            if "</think>" in text:
                text = text.split("</think>", 1)[1]
            return text.strip()

        return [_extract_answer(o.outputs[0].text) for o in outputs]
