from typing import Any, Dict, List, Optional

from retrievers.base_retriever import BaseRetriever


class RRFRetriever(BaseRetriever):
    """Reciprocal Rank Fusion over two sub-retrievers.

    Runs retriever_a and retriever_b independently on the same pool, then
    fuses their ranked lists using the RRF score:

        score(d) = 1 / (k_rrf + rank_a(d)) + 1 / (k_rrf + rank_b(d))

    Items that appear in only one list still receive a partial RRF score;
    items appearing in both lists get contributions from both ranks.

    Template diversity and same-query exclusion (tid / id) are re-enforced
    on the fused ranking before the final k are returned.
    """

    def __init__(
        self,
        retriever_a: BaseRetriever,
        retriever_b: BaseRetriever,
        k_rrf: int = 60,
        n_candidates: Optional[int] = None,
    ) -> None:
        """
        Args:
            retriever_a:   First sub-retriever (e.g. TSRetriever).
            retriever_b:   Second sub-retriever (e.g. TextRetriever).
            k_rrf:         Smoothing constant in the RRF denominator (default 60,
                           the value from the original paper).
            n_candidates:  How many candidates to fetch from each sub-retriever
                           before fusion. Defaults to the full pool size, which
                           gives RRF the most information. Can be lowered (e.g.
                           4 * k) to trade recall for speed.
        """
        self._retriever_a = retriever_a
        self._retriever_b = retriever_b
        self._k_rrf = k_rrf
        self._n_candidates = n_candidates
        self._pool_size = 0

    def index(self, pool: List[Dict[str, Any]]) -> None:
        self._pool_size = len(pool)
        self._retriever_a.index(pool)
        self._retriever_b.index(pool)

    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        n = self._n_candidates if self._n_candidates is not None else self._pool_size

        list_a = self._retriever_a.retrieve(query, n)
        list_b = self._retriever_b.retrieve(query, n)

        # Accumulate RRF scores keyed by Python object identity.
        # Both sub-retrievers store shallow copies of the same pool dicts,
        # so id(item) is a stable, collision-free key across both lists.
        rrf_scores: Dict[int, float] = {}
        items_by_oid: Dict[int, Dict[str, Any]] = {}

        for rank, item in enumerate(list_a):
            oid = id(item)
            rrf_scores[oid] = rrf_scores.get(oid, 0.0) + 1.0 / (self._k_rrf + rank + 1)
            items_by_oid[oid] = item

        for rank, item in enumerate(list_b):
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
