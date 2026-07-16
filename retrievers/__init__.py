from retrievers.base_retriever import BaseRetriever
from retrievers.random_retriever import RandomRetriever
from retrievers.rrf_retriever import RRFRetriever
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

__all__ = [
    "BaseRetriever",
    "DelayDINORetriever",
    "DTWRetriever",
    "RandomRetriever",
    "RRFRetriever",
    "SigLIPRetriever",
    "SpectralRetriever",
    "StatsRetriever",
    "TextRetriever",
    "TSCompressRetriever",
    "TSMultiVecRetriever",
    "TSRetriever",
    "TSWindowAggRetriever",
    "TwoStageRetriever",
    "VisionTSRetriever",
    "WaveletRetriever",
]
