import random
from typing import Any, Dict, List

from retrievers.base_retriever import BaseRetriever


class RandomRetriever(BaseRetriever):
    """Returns k uniformly random demonstrations from the indexed pool.

    Self-exclusion is performed by filtering on item["id"] == query["id"].
    """

    def __init__(self, seed: int = 2021) -> None:
        self._rng = random.Random(seed)
        self._pool: List[Dict[str, Any]] = []

    def index(self, pool: List[Dict[str, Any]]) -> None:
        self._pool = list(pool)

    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        exclude_id = query.get("id")
        exclude_tid = query.get("tid")
        candidates = [
            item for item in self._pool
            if item.get("id") != exclude_id and item.get("tid") != exclude_tid
        ]
        # Greedy template diversity: shuffle then pick at most 1 per tid.
        shuffled = self._rng.sample(candidates, len(candidates))
        selected = []
        seen_tids: set = set()
        for item in shuffled:
            if len(selected) >= k:
                break
            tid = item.get("tid")
            if tid is not None and tid in seen_tids:
                continue
            if tid is not None:
                seen_tids.add(tid)
            selected.append(item)
        return selected
