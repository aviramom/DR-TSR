from typing import Any, Dict, List, Optional

from retrievers.base_retriever import BaseRetriever


class RRFRetriever(BaseRetriever):
    """Reciprocal Rank Fusion over N sub-retrievers.

    Runs every sub-retriever independently on the same pool, then fuses their
    ranked lists using the RRF score:

        score(d) = sum_i 1 / (k + rank_i(d))

    Items that appear in only some lists still receive a partial RRF score;
    items appearing in more lists accumulate more contributions.

    Template diversity and same-query exclusion (tid / id) are re-enforced
    on the fused ranking before the final k are returned.
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        k_rrf: int = 60,
        n_candidates: Optional[int] = None,
    ) -> None:
        """
        Args:
            retrievers:    Sub-retrievers to fuse (e.g. [TSRetriever(), DelayDINORetriever()]).
                           Any number >= 2 is supported.
            k_rrf:         Smoothing constant in the RRF denominator (default 60,
                           the value from the original paper).
            n_candidates:  How many candidates to fetch from each sub-retriever
                           before fusion. Defaults to the full pool size, which
                           gives RRF the most information. Can be lowered (e.g.
                           4 * k) to trade recall for speed.
        """
        self._retrievers = retrievers
        self._k_rrf = k_rrf
        self._n_candidates = n_candidates
        self._pool_size = 0

    def index(self, pool: List[Dict[str, Any]]) -> None:
        self._pool_size = len(pool)
        for retriever in self._retrievers:
            retriever.index(pool)

    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        n = self._n_candidates if self._n_candidates is not None else self._pool_size

        # Accumulate RRF scores keyed by Python object identity.
        # Every sub-retriever stores the same pool dicts (shallow list copies),
        # so id(item) is a stable, collision-free key across all lists.
        rrf_scores: Dict[int, float] = {}
        items_by_oid: Dict[int, Dict[str, Any]] = {}

        for retriever in self._retrievers:
            ranked_list = retriever.retrieve(query, n)
            for rank, item in enumerate(ranked_list):
                oid = id(item)
                rrf_scores[oid] = rrf_scores.get(oid, 0.0) + 1.0 / (self._k_rrf + rank + 1)
                items_by_oid[oid] = item

        sorted_oids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        # Apply same exclusion + template-diversity rules as _cosine_top_k.
        exclude_id = query.get("id")
        exclude_tid = query.get("tid")
        selected: List[Dict[str, Any]] = []
        seen_tids: set = set()

        for oid in sorted_oids:
            if len(selected) >= k:
                break
            item = items_by_oid[oid]
            if exclude_id is not None and item.get("id") == exclude_id:
                continue
            if exclude_tid is not None and item.get("tid") == exclude_tid:
                continue
            tid = item.get("tid")
            if tid is not None and tid in seen_tids:
                continue
            if tid is not None:
                seen_tids.add(tid)
            selected.append(item)

        return selected
