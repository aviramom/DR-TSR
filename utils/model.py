from typing import Dict

from models.base_model import BaseModelWrapper
from models.instruct_model import InstructModel, LargeInstructModel
from models.image_instruct_model import ImageInstructModel
from models.qwen_vl_image_model import QwenVLImageModel
from models.api_model import APIModelWrapper

try:
    from models.chatts_model import ChatTSHFWrapper
except ImportError:
    ChatTSHFWrapper = None
    print("ChatTS HF model unavailable — missing dependency")

try:
    from models.vllm_chatts_model import ChatTSVLLMWrapper
except ImportError:
    ChatTSVLLMWrapper = None
    print("ChatTS vLLM model unavailable — vllm not installed")

try:
    from models.time_omni_model import TimeOmniHFWrapper
except ImportError:
    TimeOmniHFWrapper = None
    print("TimeOmni model unavailable — missing dependency")

from models.baselines import (
    RandomBaseline
)

method_wrapper_dict: Dict[str, BaseModelWrapper] = {
    # --- Small text LLMs (HF, single RTX 4090) ---
    "Qwen/Qwen3-8B": InstructModel,
    "Qwen/Qwen3-4B": InstructModel,
    "Qwen/Qwen3-0.6B": InstructModel,
    "Qwen/Qwen2.5-7B-Instruct": InstructModel,
    "Qwen/Qwen2-7B-Instruct": InstructModel,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": InstructModel,
    "mistralai/Mistral-7B-Instruct-v0.3": InstructModel,

    # --- Large text LLMs (vLLM, 2× RTX 6000) ---
    "Qwen/Qwen3.6-27B": LargeInstructModel,
    "Qwen/Qwen3.6-27B-FP8": LargeInstructModel,

    # --- Vision LLMs (input_mode="separate": receive <ts><ts/> placeholders) ---
    "Qwen/Qwen3.6-27B-image-ts": ImageInstructModel,   # TS → matplotlib plot, vLLM
    "Qwen/Qwen3-VL-8B-Instruct": QwenVLImageModel,     # TS → matplotlib plot, HF

    # --- ChatTS (input_mode="separate": patch-embedding TS tokens) ---
    "bytedance-research/ChatTS-8B": ChatTSHFWrapper,
    "bytedance-research/ChatTS-14B": ChatTSHFWrapper,
    "bytedance-research/ChatTS-8B-vllm": ChatTSVLLMWrapper,
    "bytedance-research/ChatTS-14B-vllm": ChatTSVLLMWrapper,

    # --- TimeOmni (text + TS as numeric arrays, Qwen2.5 base) ---
    "anton-hugging/TimeOmni-1-7B": TimeOmniHFWrapper,

    # --- Cloud API models ---
    "openai": APIModelWrapper,
    "openai_o1": APIModelWrapper,
    "anthropic": APIModelWrapper,
    "gemini": APIModelWrapper,
    "deepseek_v3": APIModelWrapper,
    "ollama": APIModelWrapper,

    # --- Baselines (CPU / no weights) ---
    "random_baseline": RandomBaseline,

}
