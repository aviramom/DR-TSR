from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseRetriever(ABC):
    """Abstract base class for all demonstration retrievers.

    Subclasses implement two methods:
      - index(pool)    — build the retrieval index once from a pool of items
      - retrieve(query, k) — return the k best demonstrations for a query

    All items (pool and query) follow the standard batch-contract dict
    format documented in retrievers/CLAUDE.md.
    """

    @abstractmethod
    def index(self, pool: List[Dict[str, Any]]) -> None:
        """Build the retrieval index from a pool of demonstration items.

        Called once before any retrieve() calls. The pool must contain only
        items that are safe to use as demonstrations (no test-set leakage).

        Args:
            pool: List of batch-contract dicts, each with at minimum:
                  input_text, input_ts, output_text, task_id, options.
        """

    @abstractmethod
    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """Return up to k demonstration items ranked by relevance to query.

        Args:
            query: A single batch-contract dict representing the test item.
            k:     Maximum number of demonstrations to return.

        Returns:
            List of at most k items from the indexed pool, best first.
            Must never include the query item itself.
        """

    @staticmethod
    def _cosine_top_k(
        query_vec: np.ndarray,
        pool_vecs: np.ndarray,
        pool_items: List[Dict[str, Any]],
        k: int,
        exclude_id: Optional[int] = None,
        exclude_tid: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-k pool items by cosine similarity to query_vec.

        Same-template items (matching exclude_tid) are masked out entirely so
        they are never returned. Among the remaining candidates, at most one
        item per unique tid is selected (greedy template diversity), so the k
        returned demonstrations cover k distinct question templates.

        Args:
            query_vec:   Shape (D,), already L2-normalized.
            pool_vecs:   Shape (N, D), already L2-normalized (rows).
            pool_items:  Parallel list of item dicts.
            k:           Number of items to return.
            exclude_id:  Value of item["id"] to exclude (the query itself).
            exclude_tid: Value of item["tid"] to exclude (the query's template).

        Returns:
            Up to k items from pool_items, highest-similarity first, with at
            most one item per tid.
        """
        scores = pool_vecs @ query_vec  # (N,) cosine similarity
        for i, item in enumerate(pool_items):
            if exclude_id is not None and item.get("id") == exclude_id:
                scores[i] = -np.inf
            elif exclude_tid is not None and item.get("tid") == exclude_tid:
                scores[i] = -np.inf

        sorted_idx = np.argsort(scores)[::-1]
        selected: List[Dict[str, Any]] = []
        seen_tids: set = set()
        for i in sorted_idx:
            if len(selected) >= k:
                break
            if scores[i] == -np.inf:
                break  # all remaining are masked
            item = pool_items[i]
            tid = item.get("tid")
            if tid is not None and tid in seen_tids:
                continue
            if tid is not None:
                seen_tids.add(tid)
            selected.append(item)
        return selected
