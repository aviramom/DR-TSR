"""Resolve --retriever names to instantiated retrievers.

Two kinds of names are accepted:

1. Plain names — a single retriever: "random", "text", "text_bge", "ts",
   "vision_ts", "delay_dino", "spectral", "stats", "vision_wavelet".
2. Composite specs — "rrf-<a>-<b>[-<c>...]" builds an RRFRetriever fusing the
   named base retrievers, e.g. "rrf-ts-delay_dino" or
   "rrf-ts-vision_ts-delay_dino". Any subset (>= 2) of the fusable retrievers
   below can be combined. Base names never contain "-", so the separator is
   unambiguous (and stays filename/W&B-tag friendly).

"rrf" (no components) is a legacy alias for "rrf-ts-text" — the combo the
original retriever_comparison_v1 runs were logged under.
"""

from functools import partial

from retrievers.random_retriever import RandomRetriever
from retrievers.spectral_retriever import SpectralRetriever
from retrievers.stats_retriever import StatsRetriever
from retrievers.text_retriever import TextRetriever

try:
    from retrievers.ts_retriever import TSRetriever
except ImportError:
    TSRetriever = None  # type: ignore[assignment,misc]

try:
    from retrievers.vision_ts_retriever import VisionTSRetriever
except ImportError:
    VisionTSRetriever = None  # type: ignore[assignment,misc]

try:
    from retrievers.delay_dino_retriever import DelayDINORetriever
except ImportError:
    DelayDINORetriever = None  # type: ignore[assignment,misc]

try:
    from retrievers.wavelet_retriever import WaveletRetriever
except ImportError:  # pywt not installed
    WaveletRetriever = None  # type: ignore[assignment,misc]

try:
    from retrievers.rrf_retriever import RRFRetriever
except ImportError:
    RRFRetriever = None  # type: ignore[assignment,misc]


# Encoder-backed retrievers that can appear alone or inside an rrf-... spec.
# (random is excluded: fusing a random ranking adds nothing but noise.)
FUSABLE_RETRIEVERS = {
    "text":           TextRetriever,
    "text_bge":       partial(TextRetriever, model_name="BAAI/bge-large-en-v1.5"),
    "ts":             TSRetriever,
    "vision_ts":      VisionTSRetriever,
    "delay_dino":     DelayDINORetriever,
    "spectral":       SpectralRetriever,
    "stats":          StatsRetriever,
    "vision_wavelet": WaveletRetriever,
}

RRF_PREFIX = "rrf-"
RRF_LEGACY_ALIAS = "rrf-ts-text"


def _make_base(name: str, device: str):
    cls = FUSABLE_RETRIEVERS.get(name)
    if cls is None:
        available = [k for k, v in FUSABLE_RETRIEVERS.items() if v is not None]
        raise ImportError(
            f"Retriever '{name}' is unavailable — check that its dependencies "
            f"are installed (see requirements.txt). Available: {available}"
        )
    return cls(device=device)


def build_retriever(name: str, device: str = "cpu"):
    """Instantiate the retriever named by a --retriever value.

    Args:
        name:   Plain name ("random", "text", ...), legacy alias ("rrf"),
                or composite spec ("rrf-ts-delay_dino").
        device: Device for encoder models; ignored by RandomRetriever.
    """
    if name == "random":
        return RandomRetriever()

    if name == "rrf":
        name = RRF_LEGACY_ALIAS

    if name.startswith(RRF_PREFIX):
        if RRFRetriever is None:
            raise ImportError("RRFRetriever is unavailable.")
        components = name[len(RRF_PREFIX):].split("-")
        if len(components) < 2:
            raise ValueError(
                f"RRF spec '{name}' needs at least two components, "
                f"e.g. rrf-ts-delay_dino."
            )
        if len(set(components)) != len(components):
            raise ValueError(f"RRF spec '{name}' repeats a component.")
        return RRFRetriever([_make_base(c, device) for c in components])

    return _make_base(name, device)
