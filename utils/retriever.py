"""Registry mapping --retriever names to retriever classes."""

from retrievers.random_retriever import RandomRetriever
from retrievers.text_retriever import TextRetriever

try:
    from retrievers.ts_retriever import TSRetriever
except ImportError:
    TSRetriever = None  # type: ignore[assignment,misc]

try:
    from retrievers.vision_ts_retriever import VisionTSRetriever
except ImportError:
    VisionTSRetriever = None  # type: ignore[assignment,misc]

retriever_dict = {
    "random":    RandomRetriever,
    "text":      TextRetriever,
    "ts":        TSRetriever,
    "vision_ts": VisionTSRetriever,
}
