from typing import Any, Dict, List

import numpy as np

from retrievers.base_retriever import BaseRetriever


class TwoStageRetriever(BaseRetriever):
    """Coarse-to-fine retriever: TS-similar candidates, then re-ranked by text.

    Stage 1 pulls a coarse candidate set (default 50) by time-series similarity
    (e.g. delay_dino or vision_wavelet). Stage 2 re-ranks those candidates by a
    second signal (e.g. text similarity of the question) and returns the top k.

    Stage 2 must be a cosine-family retriever — re-ranking reuses its indexed
    pool vectors via BaseRetriever.pool_similarity (no re-encoding). Both stages
    index the same pool; template-diversity and query exclusion are enforced by
    stage 1's retrieve() and re-enforced on the final selection.
    """

    def __init__(
        self,
        stage1: BaseRetriever,
        stage2: BaseRetriever,
        n_candidates: int = 50,
    ) -> None:
        self._stage1 = stage1
        self._stage2 = stage2
        self._n_candidates = n_candidates

    def index(self, pool: List[Dict[str, Any]]) -> None:
        self._stage1.index(pool)
        self._stage2.index(pool)

    def retrieve(self, query: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        candidates = self._stage1.retrieve(query, self._n_candidates)
        if not candidates:
            return []

        scores = self._stage2.pool_similarity(query, candidates)
        if not np.any(np.isfinite(scores)):
            # Query absent from the pool (non leave-one-out): stage 2 cannot
            # rank, so fall back to stage 1's ordering.
            return candidates[:k]

        return self._top_k_from_scores(
            np.asarray(scores, dtype=np.float32),
            candidates,
            k,
            query.get("id"),
            query.get("tid"),
        )
