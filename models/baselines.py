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
            "input_mode": "combined",
        }

    def load_model(self):
        pass

    def generate(self, batch, max_new_tokens: int = 10, **kwargs) -> List[str]:
        prompts = batch["input_text"]
        if isinstance(prompts, str):
            prompts = [prompts]
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


class KNNBaseline(BaseModelWrapper):
    """Predicts by 1-NN: finds the support example with the smallest L2 distance
    to the query time series and returns its label."""

    def __init__(self, args: Any, device: str = "cpu"):
        self.args = args

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        return {
            "model_type": "knn_baseline",
            "max_seq_length": 4096,
            "max_new_tokens": 10,
            "format": "chat",
            "input_mode": "combined",
        }

    def load_model(self):
        pass

    def generate(self, batch, max_new_tokens: int = 10, **kwargs) -> List[str]:
        prompts = batch["input_text"]
        if isinstance(prompts, str):
            prompts = [prompts]
        results = []
        for i, prompt in enumerate(prompts):
            # Labels of support examples, in the order they appear in the prompt
            support_labels = self._parse_support_labels(prompt)
            n_support = len(support_labels)

            ts_list = batch["input_ts"][i]   # [support_0, support_1, ..., query]

            if n_support == 0 or len(ts_list) < 2:
                results.append("")
                continue

            support_arrays = [self._to_array(ts_list[j]) for j in range(n_support)]
            query_array = self._to_array(ts_list[-1])

            dists = [dtw(query_array, s) for s in support_arrays]
            nearest = int(np.argmin(dists))
            results.append(str(support_labels[nearest]))

        return results

    @staticmethod
    def _parse_support_labels(prompt: str) -> List[str]:
        """Extract ordered support labels from 'Label: X' lines in the examples block."""
        return re.findall(r'Label:\s*(\S+)', prompt)

    @staticmethod
    def _to_array(ts) -> np.ndarray:
        if hasattr(ts, "numpy"):
            arr = ts.numpy().flatten().astype(float)
        else:
            arr = np.array(ts).flatten().astype(float)
        if np.isnan(arr).any():
            nans = np.isnan(arr)
            if nans.all():
                arr = np.zeros_like(arr)
            else:
                idxs = np.arange(len(arr))
                arr[nans] = np.interp(idxs[nans], idxs[~nans], arr[~nans])
        return arr


# Regex that matches a bracketed, comma-separated list of numbers in a prompt,
# i.e. the time series inserted by combine_ts_text().
_TS_ARRAY_RE = re.compile(r'\[[\d\s,.\-eE]+\]')

_TARGET_MARKER = "New Time Series:"
_OPTIONS_MARKER = "Return ONLY the label as one of:"


def _apply_to_examples_only(prompt: str, replace_fn) -> str:
    """Apply replace_fn to TS arrays only in the examples section (before the target)."""
    idx = prompt.find(_TARGET_MARKER)
    if idx == -1:
        return _TS_ARRAY_RE.sub(replace_fn, prompt)
    return _TS_ARRAY_RE.sub(replace_fn, prompt[:idx]) + prompt[idx:]


def _apply_to_examples_and_target(prompt: str, replace_fn) -> str:
    """Apply replace_fn to TS arrays in examples and target, but not the options line."""
    idx = prompt.rfind(_OPTIONS_MARKER)
    if idx == -1:
        return _TS_ARRAY_RE.sub(replace_fn, prompt)
    return _TS_ARRAY_RE.sub(replace_fn, prompt[:idx]) + prompt[idx:]


class ZeroedTSBaseline(InstructModel):
    """Qwen3-4B with the example time series replaced by all-zeros arrays.

    Substitutes every example TS array `[v1, v2, ...]` with an equal-length
    array of zeros, while leaving the target time series and options list
    untouched.  Any gap between this and the full model reveals how much the
    model uses the numerical values vs. relying on class priors / label text.
    """

    ZEROED_METHOD = "Qwen/Qwen3-4B-Instruct-2507"

    def __init__(self, args: Any, device: str = "cuda"):
        import copy
        zeroed_args = copy.copy(args)
        zeroed_args.method = self.ZEROED_METHOD
        super().__init__(zeroed_args, device=device)

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        d = InstructModel.get_args_dict()
        d["model_type"] = "zeroed_ts_baseline"
        return d

    @staticmethod
    def _zero_out_ts(prompt: str) -> str:
        """Replace example `[v1, ..., vN]` arrays with `[0, ..., 0]`; keep target and options."""
        def _replace(m):
            values = m.group(0)[1:-1].split(',')
            return '[' + ', '.join(['0'] * len(values)) + ']'
        return _apply_to_examples_only(prompt, _replace)

    def generate(self, batch, max_new_tokens: int = 50, **kwargs) -> List[str]:
        prompts = batch["input_text"]
        if isinstance(prompts, str):
            prompts = [prompts]
        zeroed_batch = dict(batch)
        zeroed_batch["input_text"] = [self._zero_out_ts(p) for p in prompts]
        return super().generate(zeroed_batch, max_new_tokens=max_new_tokens, **kwargs)


class EmptyTSBaseline(InstructModel):
    """Qwen3-4B with the example time series arrays removed from the prompt.

    Every example `[v1, v2, ...]` is replaced with `[]`; the target time series
    and options list are left untouched.  The model sees the task structure and
    label names but no numerical content from the support examples.
    """

    ZEROED_METHOD = "Qwen/Qwen3-4B-Instruct-2507"

    def __init__(self, args: Any, device: str = "cuda"):
        import copy
        empty_args = copy.copy(args)
        empty_args.method = self.ZEROED_METHOD
        super().__init__(empty_args, device=device)

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        d = InstructModel.get_args_dict()
        d["model_type"] = "empty_ts_baseline"
        return d

    @staticmethod
    def _empty_out_ts(prompt: str) -> str:
        """Replace example `[v1, ..., vN]` arrays with `[]`; keep target and options."""
        return _apply_to_examples_only(prompt, '[]')

    def generate(self, batch, max_new_tokens: int = 50, **kwargs) -> List[str]:
        prompts = batch["input_text"]
        if isinstance(prompts, str):
            prompts = [prompts]
        empty_batch = dict(batch)
        empty_batch["input_text"] = [self._empty_out_ts(p) for p in prompts]
        return super().generate(empty_batch, max_new_tokens=max_new_tokens, **kwargs)


class EmptyAllTSBaseline(InstructModel):
    """Qwen3-4B with both example and target time series removed.

    Replaces every TS array in examples and the target with `[]`, while keeping
    the options list intact.  The model sees only the task structure and label
    names — no numerical content whatsoever.
    """

    ZEROED_METHOD = "Qwen/Qwen3-4B-Instruct-2507"

    def __init__(self, args: Any, device: str = "cuda"):
        import copy
        empty_args = copy.copy(args)
        empty_args.method = self.ZEROED_METHOD
        super().__init__(empty_args, device=device)

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        d = InstructModel.get_args_dict()
        d["model_type"] = "empty_all_ts_baseline"
        return d

    @staticmethod
    def _empty_out_all_ts(prompt: str) -> str:
        """Replace all TS arrays (examples + target) with `[]`; keep options."""
        return _apply_to_examples_and_target(prompt, '[]')

    def generate(self, batch, max_new_tokens: int = 50, **kwargs) -> List[str]:
        prompts = batch["input_text"]
        if isinstance(prompts, str):
            prompts = [prompts]
        empty_batch = dict(batch)
        empty_batch["input_text"] = [self._empty_out_all_ts(p) for p in prompts]
        return super().generate(empty_batch, max_new_tokens=max_new_tokens, **kwargs)


class EmptyAllTSChatTSBaseline(ChatTSHFWrapper):
    """ChatTS with all time series (support + target) replaced by zero arrays.

    Keeps the prompt structure and placeholder positions intact so the processor
    receives valid input shapes, but every value is zero.  Reveals how much the
    model relies on actual TS content vs. task structure and label names.
    """

    CHATTS_METHOD = "bytedance-research/ChatTS-8B"

    def __init__(self, args: Any, device: str = "cuda"):
        import copy
        empty_args = copy.copy(args)
        empty_args.method = self.CHATTS_METHOD
        super().__init__(empty_args, device=device)

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        d = ChatTSHFWrapper.get_args_dict()
        d["model_type"] = "empty_all_ts_chatts_baseline"
        return d

    @staticmethod
    def _zero_ts_item(ts_item):
        """Return a zero array with the same shape as ts_item, or a list of such arrays."""
        if isinstance(ts_item, list) and len(ts_item) > 0 and not isinstance(ts_item[0], (int, float)):
            return [np.zeros_like(np.asarray(ts)) for ts in ts_item]
        arr = np.asarray(ts_item)
        return np.zeros_like(arr)

    def generate(self, batch, max_new_tokens: int = 50, **kwargs) -> List[str]:
        empty_batch = dict(batch)
        empty_batch["input_ts"] = [self._zero_ts_item(ts) for ts in batch["input_ts"]]
        return super().generate(empty_batch, max_new_tokens=max_new_tokens, **kwargs)


class DinoKNNBaseline(BaseModelWrapper):
    """1-NN classifier using DINOv2 embeddings of time series images.

    Each series is converted to a sliding-window trajectory image (224×224),
    encoded by DINOv2-Large into a 1024-dim CLS vector, then classified by
    nearest neighbour (Euclidean distance) against the support set embeddings.
    """

    def __init__(self, args: Any, device: str = "cuda"):
        self.args = args
        self.device = device
        self.encoder = None

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        return {
            "model_type": "dino_knn_baseline",
            "max_seq_length": 4096,
            "max_new_tokens": 10,
            "format": "chat",
            "input_mode": "combined",
        }

    def load_model(self):
        from models.ts_encoders.dino.dino_enc import DinoEncoder
        self.encoder = DinoEncoder(self.args, self.device)
        self.encoder.eval()

    def _embed(self, ts_1d) -> np.ndarray:
        """Embed a single 1-D time series into a 1024-dim DINOv2 CLS vector."""
        arr = KNNBaseline._to_array(ts_1d)                          # (L,)
        t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        mu = t.mean(dim=2, keepdim=True)
        sigma = t.std(dim=2, keepdim=True).clamp_min(1e-5)
        t = (t - mu) / sigma
        with torch.inference_mode():
            emb = self.encoder(t.to(self.device))  # (1, 1, 1024)
        return emb[0, 0].cpu().numpy()             # (1024,)

    def generate(self, batch, max_new_tokens: int = 10, **kwargs) -> List[str]:
        if self.encoder is None:
            self.load_model()

        prompts = batch["input_text"]
        if isinstance(prompts, str):
            prompts = [prompts]

        results = []
        for i, prompt in enumerate(prompts):
            support_labels = KNNBaseline._parse_support_labels(prompt)
            n_support = len(support_labels)
            ts_list = batch["input_ts"][i]  # [support_0, ..., support_n-1, query]

            if n_support == 0 or len(ts_list) < 2:
                results.append("")
                continue

            support_embs = [self._embed(ts_list[j]) for j in range(n_support)]
            query_emb = self._embed(ts_list[-1])

            dists = [np.linalg.norm(query_emb - s) for s in support_embs]
            nearest = int(np.argmin(dists))
            results.append(str(support_labels[nearest]))

        return results


# ── CLSA-DINOv2 helpers (ported from eval_ucr_clsa.py) ──────────────────────

def _delay_embed_2d_clsa(
    x: np.ndarray,
    height: int = 256,
    width: int = 256,
    embed_ratio: float = 0.6,
    embed_lmin: int = 48,
    embed_lmax: int = 192,
) -> np.ndarray:
    """Convert a 1D normalized [0,1] signal to a 2D delay-embedded image."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == 0:
        x = np.zeros(height, dtype=np.float32)
    length = int(x.size)
    raw_l = int(math.floor(embed_ratio * length))
    l = min(embed_lmax, max(embed_lmin, raw_l))
    l = max(1, min(l, length))
    max_start = max(0, length - l)
    delay = (max_start / float(width - 1)) if width > 1 else 0.0
    out = np.empty((height, width), dtype=np.float32)
    for col in range(width):
        start = int(round(col * delay))
        start = min(start, max_start)
        window = x[start : start + l]
        if l == height:
            out[:, col] = window
        elif l == 1:
            out[:, col] = window[0]
        else:
            src = np.linspace(0.0, 1.0, num=l, dtype=np.float32)
            dst = np.linspace(0.0, 1.0, num=height, dtype=np.float32)
            out[:, col] = np.interp(dst, src, window).astype(np.float32)
    return out


def _ts_to_pil_clsa(ts_1d, vmin: float, vmax: float, height: int = 256, width: int = 256) -> Image.Image:
    """Normalize TS with provided min/max stats then convert to RGB PIL image."""
    series = np.asarray(ts_1d, dtype=np.float32).ravel()
    if vmax > vmin:
        series = (series - vmin) / (vmax - vmin)
    else:
        series = np.zeros_like(series)
    series = series.clip(0.0, 1.0)
    img2d = _delay_embed_2d_clsa(series, height, width)
    img_u8 = (img2d * 255.0).astype(np.uint8)
    rgb = np.stack([img_u8] * 3, axis=-1)
    return Image.fromarray(rgb)


_CLSA_TRANSFORM = pth_transforms.Compose([
    pth_transforms.Resize((224, 224), interpolation=3),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class DinoKNNCLSABaseline(BaseModelWrapper):
    """1-NN classifier using DINOv2 embeddings of CLSA delay-embedded TS images.

    Follows the exact protocol from eval_ucr_clsa.py:
    - Min-max normalization using support-set (train) channel stats
    - 256×256 delay embedding → resize to 224×224 → DINOv2-Large
    - CLS token + avg patch tokens (2048-dim), L2-normalized
    - 1-NN Euclidean distance against support embeddings
    """

    def __init__(self, args: Any, device: str = "cuda"):
        self.args = args
        self.device = device
        self.model = None

    @staticmethod
    def get_args_dict() -> Dict[str, Any]:
        return {
            "model_type": "dino_knn_clsa",
            "max_seq_length": 4096,
            "max_new_tokens": 10,
            "format": "chat",
            "input_mode": "separate",
        }

    def load_model(self):
        from transformers import Dinov2Model
        cache = getattr(self.args, "cache_dir", None)
        self.model = Dinov2Model.from_pretrained(
            "facebook/dinov2-large",
            torch_dtype=torch.bfloat16,
            cache_dir=cache,
        ).to(self.device).eval()

    @torch.inference_mode()
    def _embed(self, ts_list, vmin: float, vmax: float) -> np.ndarray:
        """Embed a list of 1D series → (N, 2048) L2-normalized numpy array."""
        imgs = torch.stack([
            _CLSA_TRANSFORM(_ts_to_pil_clsa(ts, vmin, vmax))
            for ts in ts_list
        ]).to(self.device, dtype=torch.bfloat16)

        out = self.model(pixel_values=imgs)
        cls = out.last_hidden_state[:, 0, :].float()
        patches = out.last_hidden_state[:, 1:, :].mean(dim=1).float()
        feats = torch.cat([cls, patches], dim=1)
        feats = F.normalize(feats, dim=1)
        return feats.cpu().numpy()

    def generate(self, batch, max_new_tokens: int = 10, **kwargs) -> List[str]:
        if self.model is None:
            self.load_model()

        prompts = batch["input_text"]
        if isinstance(prompts, str):
            prompts = [prompts]

        results = []
        for i, prompt in enumerate(prompts):
            support_labels = KNNBaseline._parse_support_labels(prompt)
            n_support = len(support_labels)
            ts_list = batch["input_ts"][i]  # [support_0, ..., support_n-1, query]

            if n_support == 0 or len(ts_list) < 2:
                results.append("")
                continue

            # Compute min/max from support set (CLSA train-stats protocol)
            support_ts = [ts_list[j] for j in range(n_support)]
            all_vals = np.concatenate([np.asarray(ts).ravel() for ts in support_ts])
            vmin = float(all_vals.min())
            vmax = float(all_vals.max())

            support_embs = self._embed(support_ts, vmin, vmax)       # (k, 2048)
            query_emb = self._embed([ts_list[-1]], vmin, vmax)[0]    # (2048,)

            dists = np.linalg.norm(support_embs - query_emb, axis=1)
            nearest = int(np.argmin(dists))
            results.append(str(support_labels[nearest]))

        return results
