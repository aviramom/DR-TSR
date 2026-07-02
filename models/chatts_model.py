
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
# --- VERSION COMPATIBILITY PATCH START ---
# This patch fixes the crash between newer 'transformers' versions and the older ChatTS code.
from transformers.cache_utils import DynamicCache

# 1. Patch 'seen_tokens' (Access property)
if not hasattr(DynamicCache, "seen_tokens"):
    @property
    def seen_tokens(self):
        return self.get_seq_length()
    DynamicCache.seen_tokens = seen_tokens

# 2. Patch 'get_usable_length' (Method call)
if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, input_seq_len=None, layer_idx=0):
        # Fix: Ensure layer_idx is an integer (default to 0 if None)
        if layer_idx is None:
            layer_idx = 0
        return self.get_seq_length(layer_idx)
    
    DynamicCache.get_usable_length = get_usable_length
# --- VERSION COMPATIBILITY PATCH END ---

from models.base_model import BaseModelWrapper


def _clean_ts_for_chatts(ts: Any) -> np.ndarray:
    """
    CLEANS input but preserves RAW VALUES.
    Does NOT normalize (mean/std). 
    The Processor needs raw values to calculate metadata (max/min/offset).
    """
    # 1. Convert to numpy
    if isinstance(ts, torch.Tensor):
        ts = ts.detach().cpu().numpy()
    elif isinstance(ts, list):
        ts = np.array(ts)
    
    ts = np.asarray(ts)

    # 2. Handle Multivariate (Select last channel)
    # The model expects 1D arrays.
    if ts.ndim > 1:
        if ts.shape[0] < ts.shape[1] and ts.shape[0] < 20: 
            ts = ts[-1, :] 
        else:
            ts = ts[:, -1]

    # 3. Flatten but KEEP VALUES
    ts_flat = ts.reshape(-1)
    
    # 4. Remove NaNs (replace with 0 to prevent crash, but don't scale)
    ts_flat = np.nan_to_num(ts_flat, nan=0.0)
    # do the same for None
    ts_flat = np.array([0.0 if x is None else x for x in ts_flat]).astype(np.float32)

    return ts_flat    


class ChatTSHFWrapper(BaseModelWrapper):
    def __init__(self, args: Any, device: str = "cuda"):
        self.args = args
        self.device = device

        self.method: Optional[str] = getattr(args, "method", None)  # e.g. "bytedance-research/ChatTS-14B"
        self.cache_dir: Optional[str] = getattr(args, "cache_dir", None)

        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.processor: Optional[Any] = None  # AutoProcessor
        self.ts_place_holder = getattr(args, "ts_place_holder", "<ts><ts/>")

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        return {
            "model_type": "chatts_hf",
            "max_seq_length": 4096,
            "max_new_tokens": 50,
            "chatts_system_prompt": "You are a helpful assistant.",
            "input_mode": "separate",
            "ts_place_holder": "<ts><ts/>"
        }



    def load_model(self, model_path: Optional[str] = None, cache_dir: Optional[str] = None):
        path = model_path or self.method
        cache = cache_dir or self.cache_dir


        if path is None:
            raise ValueError("No model_path provided for ChatTSHFWrapper.load_model")

        print(f"[ChatTS] Loading model from {path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            device_map=self.args.device if self.args.device != "cpu" else None,
            torch_dtype=torch.float16,
            cache_dir=cache,
            # attn_implementation="eager",  # <--- ADD THIS LIN
           
           #  force_download=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            cache_dir=cache,
        )
        self.tokenizer.padding_side = "left"

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            path,
            trust_remote_code=True,
            tokenizer=self.tokenizer,
            cache_dir=cache,
        )
       

        return self.model, self.tokenizer

    def _build_prompt(self, user_text: str) -> str:
        """Wraps the user query in the ChatTS specific prompt template."""
        system = str(getattr(self.args, "chatts_system_prompt", "You are a helpful assistant."))
        
        # Check if template is already applied to avoid double wrapping
        if "<|im_start|>" in user_text:
            return user_text

        return (
            "<|im_start|>system\n"
            f"{system}<|im_end|><|im_start|>user\n"
            f"{user_text}<|im_end|><|im_start|>assistant\n"
        )

    def format_ts(self, ts_list: Any, q: str) -> str:
        """
        Prefix question text with a standardized time-series description.
        Calculates lengths strictly from metadata (shape/len) without 
        modifying, flattening, or copying the actual time series data.
        """
        if ts_list is None:
            return q

        placeholder = getattr(self, "ts_place_holder", "<ts><ts/>")
        lengths = []

        # 1. Handle Torch Tensor / Numpy Array
        if hasattr(ts_list, "shape"):
            # If 1D: It is one single series
            if len(ts_list.shape) == 1:
                lengths = [ts_list.shape[0]]
            # If 2D or more: Dimension 0 is the number of series, Dim 1 is length
            else:
                # We assume shape is [Num_Series, Length, ...]
                lengths = [ts_list.shape[1] for _ in range(ts_list.shape[0])]


        # 2. Handle Python Lists
        elif isinstance(ts_list, list):
            if not ts_list:
                lengths = []
            # Check content of the first item to see if it's a list-of-lists (Multivariate)
            elif isinstance(ts_list[0], (list, np.ndarray, torch.Tensor)):
                 for item in ts_list:
                    if hasattr(item, "shape"):
                         lengths.append(item.shape[0])
                    elif isinstance(item, list):
                         lengths.append(len(item))
                    else:
                         lengths.append(1) # Fallback for unknown iterables
            # Otherwise, it's a flat list representing a single series
            else:
                lengths = [len(ts_list)]
        
        # 3. Handle Unexpected Types (Fail gracefully)
        else:
            return q

        if not lengths:
            return q

        # 4. Construct the Prompt String
        n = len(lengths)
        parts = [
            f"Time series {i+1} is of length {L}: {placeholder};"
            for i, L in enumerate(lengths)
        ]
        prefix = f"There are {n} time series: " + " ".join(parts)

        return f"{prefix} {q}"

    


    @torch.no_grad()
    def generate(self, batch: Dict[str, Any], max_new_tokens: int = 300, pred_only: bool = True, **kwargs) -> List[str]:
        if self.model is None:
            self.load_model()

        questions = batch["input_text"]
        raw_ts_batch = batch.get("input_ts")
        
        if raw_ts_batch is None:
            raise ValueError("Batch missing 'input_ts'")

        prompts = []
        flat_ts_inputs = []

        # Loop through questions and data PAIRWISE
        for q, ts_item in zip(questions, raw_ts_batch):
            if isinstance(q, (tuple, list)):
                q = q[0] if len(q) > 0 else ""

            prompts.append(self._build_prompt(q))

            # --- 2. Data Processing (Your original flattening logic) ---
            # Case A: List of multiple series
            if isinstance(ts_item, list) and len(ts_item) > 0 and not isinstance(ts_item[0], (int, float)):
                for ts in ts_item:
                    flat_ts_inputs.append(_clean_ts_for_chatts(ts))
            
            # Case B: Single series
            else:
                flat_ts_inputs.append(_clean_ts_for_chatts(ts_item))
            # -----------------------------

        inputs = self.processor(
            text=prompts,
            timeseries=flat_ts_inputs, # Pass the flat list here
            padding=True,
            return_tensors="pt"
        )

        # 3. Move to GPU
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # 4. Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            **kwargs
        )

        # 5. Decode
        res = []
        for i in range(outputs.shape[0]):
            if pred_only:
                input_len = inputs["input_ids"][i].shape[0]
                gen_ids = outputs[i][input_len:]
            else:
                gen_ids = outputs[i]
            
            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            res.append(decoded.strip())

        return res