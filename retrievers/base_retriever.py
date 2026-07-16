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
    def _build_id_map(pool_items: List[Dict[str, Any]]) -> Dict[Any, int]:
        """Map item["id"] → row index in the pool embedding matrix.

        Leave-one-out evaluation indexes the full dataset, so a query's
        embedding usually already exists in the pool matrix. retrieve() can
        look it up by id instead of re-encoding — saving a forward pass per
        query and keeping query/pool vectors numerically identical.
        """
        return {
            item["id"]: i
            for i, item in enumerate(pool_items)
            if item.get("id") is not None
        }

    def _offload_encoder(self) -> None:
        """Move the encoder (self._model) to CPU and free its GPU memory.

        Called at the end of index(): with id-based query lookup the encoder
        is only needed for queries absent from the pool, so it does not have
        to stay on the GPU competing with the LLM for VRAM. Relies on the
        subclass conventions self._model / self._device.
        """
        if getattr(self, "_model", None) is None or self._device == "cpu":
            return
        import torch

        self._model.to("cpu")
        self._device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def pool_similarity(
        self,
        query: Dict[str, Any],
        items: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Cosine similarity of each item in `items` to `query`.

        Read straight from the retriever's indexed, L2-normalized pool vectors
        (`self._pool_vecs` / `self._id_to_row`) — no re-encoding. Used by
        TwoStageRetriever to re-rank a coarse candidate set with a second signal.

        Relies on the repo's leave-one-out indexing, so the query and every
        candidate already have a row in the pool matrix. Items absent from the
        pool (and the whole array, if the query itself is absent) score -inf, so
        the caller can detect that case and fall back to the stage-1 order.

        Only defined for cosine-family retrievers (text, ts*, vision_ts,
        delay_dino, vision_wavelet, spectral, stats) that expose those two
        attributes; other retrievers raise TypeError.
        """
        pool_vecs = getattr(self, "_pool_vecs", None)
        id_to_row = getattr(self, "_id_to_row", None)
        if pool_vecs is None or id_to_row is None:
            raise TypeError(
                f"{type(self).__name__} cannot rank a candidate subset "
                "(it exposes no indexed pool vectors)."
            )
        out = np.full(len(items), -np.inf, dtype=np.float32)
        q_row = id_to_row.get(query.get("id"))
        if q_row is None:
            return out
        q_vec = pool_vecs[q_row]
        for i, item in enumerate(items):
            r = id_to_row.get(item.get("id"))
            if r is not None:
                out[i] = float(pool_vecs[r] @ q_vec)
        return out

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
        return BaseRetriever._top_k_from_scores(
            scores, pool_items, k, exclude_id, exclude_tid
        )

    @staticmethod
    def _top_k_from_scores(
        scores: np.ndarray,
        pool_items: List[Dict[str, Any]],
        k: int,
        exclude_id: Optional[int] = None,
        exclude_tid: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Select top-k items from a precomputed per-item score array.

        Shared selection logic behind _cosine_top_k, exposed separately for
        retrievers whose score is not a single dot product per item (e.g.
        multi-vector MaxSim). Applies the same exclusion and greedy
        template-diversity rules. `scores` is mutated (masked entries are
        set to -inf).
        """
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
