"""Retrieval strategy implementations for Experiment 1.

Each strategy selects k demonstration items from the pool for a given query.
All strategies share the same interface so they can be swept uniformly.
"""

import random as _random
import numpy as np


ALL_STRATEGIES = ["zero_shot", "random", "ts_only", "text_only", "fusion", "oracle"]


def retrieve(
    query_item,
    pool_items: list,
    strategy: str,
    k: int,
    ts_index=None,
    text_index=None,
    alpha: float = 0.5,
    seed: int = None,
) -> list:
    """Select k demonstrations from pool for a given query item.

    Args:
        strategy: one of ALL_STRATEGIES
        k: number of demonstrations (0 for zero_shot)
        alpha: TS weight for fusion (0 = text only, 1 = TS only)
        seed: random seed for non-deterministic strategies (random, oracle)

    Returns:
        List of k TSEItem demonstrations (empty list when k=0 or zero_shot).
    """
    if strategy == "zero_shot" or k == 0:
        return []

    k = min(k, len(pool_items))

    if strategy == "random":
        rng = _random.Random(seed)
        return rng.sample(pool_items, k)

    if strategy == "ts_only":
        return ts_index.top_k(query_item, k)

    if strategy == "text_only":
        return text_index.top_k(query_item, k)

    if strategy == "fusion":
        ts_scores = ts_index.scores(query_item)
        text_scores = text_index.scores(query_item)

        def _minmax(v):
            mn, mx = v.min(), v.max()
            return (v - mn) / (mx - mn) if mx > mn else np.zeros_like(v)

        combined = alpha * _minmax(ts_scores) + (1 - alpha) * _minmax(text_scores)
        idx = np.argsort(combined)[::-1][:k]
        return [pool_items[i] for i in idx]

    if strategy == "oracle":
        same_cat = [item for item in pool_items if item.category == query_item.category]
        rng = _random.Random(seed)
        return rng.sample(same_cat, min(k, len(same_cat)))

    raise ValueError(f"Unknown retrieval strategy: {strategy!r}. Choose from {ALL_STRATEGIES}")
