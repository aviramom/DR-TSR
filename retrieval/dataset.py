"""TSE dataset loading and cross-template splitting for Experiment 1.

Each item in qa_dataset.json (the official 746-question benchmark) is loaded as a
TSEItem. Cross-template splitting holds out entire templates for the test/query set,
so the pool never contains another instance of the same question template as the query.
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class TSEItem:
    """A single question-answer pair from TimeSeriesExam."""
    tid: int
    item_id: int
    question: str
    options: list          # display texts: ["Linear", "Exponential", "No Trend"]
    answer_letter: str     # "A", "B", "C", ...
    answer_text: str       # full option text of the correct answer
    category: str
    subcategory: str
    difficulty: str
    format_hint: str
    ts: Optional[list]     # 1024 floats, None for two-series items
    ts1: Optional[list]    # 1024 floats, None for single-series items
    ts2: Optional[list]    # 1024 floats, None for single-series items

    @property
    def is_two_series(self) -> bool:
        return self.ts is None and self.ts1 is not None

    @property
    def option_letters(self) -> list:
        return [chr(ord("A") + i) for i in range(len(self.options))]

    def get_ts_for_embedding(self) -> list:
        """Return a single flat TS for index embedding.

        For two-series items, ts1 is used (same length as single-series ts).
        """
        return self.ts if self.ts is not None else self.ts1


def load_tse_items(path: str = "qa_dataset.json") -> list:
    """Load all items from qa_dataset.json as TSEItem objects."""
    with open(path) as f:
        raw = json.load(f)

    items = []
    for rec in raw:
        options = rec["options"]
        answer_text = rec["answer"]
        try:
            answer_idx = options.index(answer_text)
            answer_letter = chr(ord("A") + answer_idx)
        except ValueError:
            answer_letter = "A"

        items.append(TSEItem(
            tid=rec["tid"],
            item_id=rec["id"],
            question=rec["question"],
            options=options,
            answer_letter=answer_letter,
            answer_text=answer_text,
            category=rec["category"],
            subcategory=rec.get("subcategory", ""),
            difficulty=rec.get("difficulty", ""),
            format_hint=rec.get("format_hint", ""),
            ts=rec.get("ts"),
            ts1=rec.get("ts1"),
            ts2=rec.get("ts2"),
        ))

    return items


def cross_template_split(
    items: list,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Split items into pool and query sets by template (tid), stratified by category.

    No template appears in both pool and query — this prevents the model from finding
    another instance of the exact same question template in the pool.

    Returns:
        (pool_items, query_items)
    """
    rng = random.Random(seed)

    tid_to_items = defaultdict(list)
    for item in items:
        tid_to_items[item.tid].append(item)

    cat_to_tids = defaultdict(list)
    for tid, tid_items in tid_to_items.items():
        cat_to_tids[tid_items[0].category].append(tid)

    pool_tids, test_tids = set(), set()
    for cat, tids in cat_to_tids.items():
        shuffled = tids[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_fraction))
        test_tids.update(shuffled[:n_test])
        pool_tids.update(shuffled[n_test:])

    pool_items = [item for item in items if item.tid in pool_tids]
    query_items = [item for item in items if item.tid in test_tids]
    return pool_items, query_items
