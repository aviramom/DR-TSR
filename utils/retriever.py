"""Resolve --retriever names to instantiated retrievers.

Two kinds of names are accepted:

1. Plain names — a single retriever: "random", "text", "text_bge", "ts",
   "ts_compress", "ts_multivec", "ts_windowagg", "vision_ts", "delay_dino",
   "spectral", "stats", "vision_wavelet", "dtw", "siglip_plot", "siglip_delay".
2. Composite specs — "rrf-<a>-<b>[-<c>...]" builds an RRFRetriever fusing the
   named base retrievers, e.g. "rrf-ts-delay_dino" or
   "rrf-ts-vision_ts-delay_dino". Any subset (>= 2) of the fusable retrievers
   below can be combined. Base names never contain "-", so the separator is
   unambiguous (and stays filename/W&B-tag friendly).
3. Two-stage specs — "twostage-<a>-<b>" builds a TwoStageRetriever: stage <a>
   pulls a coarse candidate set by its signal, stage <b> re-ranks by its own
   (e.g. "twostage-delay_dino-text" or "twostage-vision_wavelet-text").
   Stage <b> must be a cosine-family retriever (it re-ranks via pool vectors).

"rrf" (no components) is a legacy alias for "rrf-ts-text" — the combo the
original retriever_comparison_v1 runs were logged under.
"""

from functools import partial

from retrievers.random_retriever import RandomRetriever
from retrievers.spectral_retriever import SpectralRetriever
from retrievers.stats_retriever import StatsRetriever
from retrievers.text_retriever import TextRetriever
from retrievers.two_stage_retriever import TwoStageRetriever

try:
    from retrievers.dtw_retriever import DTWRetriever
except ImportError:  # tslearn not installed
    DTWRetriever = None  # type: ignore[assignment,misc]

try:
    from retrievers.ts_retriever import TSRetriever
except ImportError:
    TSRetriever = None  # type: ignore[assignment,misc]

try:
    from retrievers.ts_compress_retriever import TSCompressRetriever
except ImportError:
    TSCompressRetriever = None  # type: ignore[assignment,misc]

try:
    from retrievers.ts_multivec_retriever import TSMultiVecRetriever
except ImportError:
    TSMultiVecRetriever = None  # type: ignore[assignment,misc]

try:
    from retrievers.ts_windowagg_retriever import TSWindowAggRetriever
except ImportError:
    TSWindowAggRetriever = None  # type: ignore[assignment,misc]

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
    from retrievers.siglip_retriever import SigLIPRetriever
except ImportError:
    SigLIPRetriever = None  # type: ignore[assignment,misc]

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
    "ts_compress":    TSCompressRetriever,
    "ts_multivec":    TSMultiVecRetriever,
    "ts_windowagg":   TSWindowAggRetriever,
    "vision_ts":      VisionTSRetriever,
    "delay_dino":     DelayDINORetriever,
    "spectral":       SpectralRetriever,
    "stats":          StatsRetriever,
    "vision_wavelet": WaveletRetriever,
    "dtw":            DTWRetriever,
    "siglip_plot":    partial(SigLIPRetriever, image_mode="plot") if SigLIPRetriever else None,
    "siglip_delay":   partial(SigLIPRetriever, image_mode="delay") if SigLIPRetriever else None,
}

RRF_PREFIX = "rrf-"
RRF_LEGACY_ALIAS = "rrf-ts-text"
TWOSTAGE_PREFIX = "twostage-"


def _make_base(name: str, device: str):
    cls = FUSABLE_RETRIEVERS.get(name)
    if cls is None:
        available = [k for k, v in FUSABLE_RETRIEVERS.items() if v is not None]
        raise ImportError(
            f"Retriever '{name}' is unavailable — check that its dependencies "
            f"are installed (see requirements.txt). Available: {available}"
        )
    return cls(device=device)


def build_retriever(
    name: str,
    device: str = "cpu",
    rrf_k: int = 60,
    stage1_candidates: int = 50,
):
    """Instantiate the retriever named by a --retriever value.

    Args:
        name:              Plain name ("random", "text", ...), legacy alias
                           ("rrf"), or composite spec ("rrf-ts-delay_dino",
                           "twostage-delay_dino-text").
        device:            Device for encoder models; ignored by RandomRetriever.
        rrf_k:             Smoothing constant for the RRF denominator; ignored
                           unless `name` is an rrf-... fusion spec.
        stage1_candidates: Coarse candidate-set size for the stage-1 retriever;
                           ignored unless `name` is a twostage-... spec.
    """
    if name == "random":
        return RandomRetriever()

    if name.startswith(TWOSTAGE_PREFIX):
        components = name[len(TWOSTAGE_PREFIX):].split("-")
        if len(components) != 2:
            raise ValueError(
                f"Two-stage spec '{name}' needs exactly two components, "
                f"e.g. twostage-delay_dino-text."
            )
        stage1 = _make_base(components[0], device)
        stage2 = _make_base(components[1], device)
        return TwoStageRetriever(stage1, stage2, n_candidates=stage1_candidates)

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
        return RRFRetriever([_make_base(c, device) for c in components], k_rrf=rrf_k)

    return _make_base(name, device)
