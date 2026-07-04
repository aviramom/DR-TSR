import math
import re
import random
import numpy as np
import torch
import torch.nn.functional as F
from tslearn.metrics import dtw
from typing import Any, Dict, List
from PIL import Image
from torchvision import transforms as pth_transforms

from models.base_model import BaseModelWrapper
from models.instruct_model import InstructModel
from models.chatts_model import ChatTSHFWrapper
from utils.ts_serialize import fill_ts_placeholders


class RandomBaseline(BaseModelWrapper):
    """Predicts a uniformly random label drawn from the options listed in the prompt."""

    def __init__(self, args: Any, device: str = "cpu"):
        self.args = args
        self.rng = random.Random(getattr(args, "random_seed", None))

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        return {
            "model_type": "random_baseline",
            "max_seq_length": 4096,
            "max_new_tokens": 10,
            "format": "chat",
        }

    def load_model(self):
        pass

    def generate(self, batch, max_new_tokens: int = 10, **kwargs) -> List[str]:
        raw_texts = batch["input_text"]
        if isinstance(raw_texts, str):
            raw_texts = [raw_texts]
        input_ts_list = batch.get("input_ts", [[] for _ in raw_texts])
        prompts = [fill_ts_placeholders(t, ts) for t, ts in zip(raw_texts, input_ts_list)]
        batch_options = batch.get("options", [])
        results = []
        for i, prompt in enumerate(prompts):
            options = self._parse_options(prompt)
            if not options and batch_options and i < len(batch_options):
                options = list(batch_options[i]) if batch_options[i] else []
            chosen = self.rng.choice(options) if options else ""
            results.append(str(chosen))
        return results

    @staticmethod
    def _parse_options(prompt: str) -> List[str]:
        """Extract the label list from 'Return ONLY the label as one of: [a, b, ...]'."""
        match = re.search(r'Return ONLY the label as one of:\s*\[([^\]]+)\]', prompt)
        if not match:
            return []
        return [opt.strip() for opt in match.group(1).split(',')]


