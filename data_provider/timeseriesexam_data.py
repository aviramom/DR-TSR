"""TimeSeriesExam dataset wrapper for the DR-TSR benchmark framework."""

import json
from typing import Any, Dict, Iterator, List, Optional

_OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_TS_PLACEHOLDER = "<ts><ts/>"


def _letter(idx: int) -> str:
    return _OPTION_LETTERS[idx]


def _build_prompt(
    question: str,
    options: List[str],
    ts: Optional[List[float]],
    ts1: Optional[List[float]],
    ts2: Optional[List[float]],
) -> str:
    """Construct the full prompt for a single TSE item.

    TS values are always represented as '<ts><ts/>' placeholders. Models
    that consume numeric text call fill_ts_placeholders() in their generate()
    to substitute the raw arrays before inference.
    """
    lines: List[str] = []

    if ts is not None:
        lines.append(f"Time Series: {_TS_PLACEHOLDER}")
    elif ts1 is not None:
        lines.append(f"Time Series 1: {_TS_PLACEHOLDER}")
        lines.append(f"Time Series 2: {_TS_PLACEHOLDER}")

    lines.append("")
    lines.append(f"Question: {question}")

    letters = [_letter(i) for i in range(len(options))]
    for letter, opt in zip(letters, options):
        lines.append(f"{letter}) {opt}")

    lines.append("")
    lines.append(f"Return ONLY the label as one of: [{', '.join(letters)}]")

    return "\n".join(lines)


class TimeSeriesExamDataset:
    """Wraps qa_dataset.json and exposes each sample in the batch format
    expected by BaseModelWrapper subclasses.

    Each item dict contains:

    Model-facing fields (consumed by generate()):
        input_text  str        Full prompt. TS values are serialized as
                               numeric text in combined mode; each series is
                               replaced by a '<ts><ts/>' placeholder in
                               separate mode.
        input_ts    list       List of raw float arrays, one per TS in the
                               sample — always populated regardless of
                               input_mode, so models that read raw arrays
                               (e.g. KNNBaseline, ChatTS) work in both modes.
                               Single-series: [[v, ...]].
                               Two-series:    [[v, ...], [u, ...]].
                               Order matches the placeholder/serialization
                               order in input_text.
        output_text str        Correct option letter (e.g. 'A').
        task_id     str        'TimeSeriesExam'.
        options     list[str]  Valid option letters (e.g. ['A', 'B', 'C']).

    Metadata fields (used for retrieval / splitting):
        answer_text str        Full text of the correct answer.
        question    str        Raw question text (without prompt scaffolding).
        category    str        TSE category (e.g. 'Pattern Recognition').
        subcategory str        TSE subcategory (e.g. 'Trend Recognition').
        tid         int        Template ID — used for cross-template splits.
        id          int        Unique item ID.
        difficulty  str        'easy' / 'medium' / 'hard'.

    Args:
        data_path:   Path to qa_dataset.json.
        num_samples: Cap on total samples loaded (None = all).
    """

    TASK_ID = "TimeSeriesExam"

    def __init__(
        self,
        data_path: str = "qa_dataset.json",
        num_samples: Optional[int] = None,
        category: Optional[str] = None,
    ):
        self.data_path = data_path

        with open(data_path) as f:
            raw: List[Dict[str, Any]] = json.load(f)

        if category is not None:
            raw = [r for r in raw if r.get("category") == category]

        if num_samples is not None:
            raw = raw[:num_samples]

        self._items: List[Dict[str, Any]] = [self._process(r) for r in raw]

    def _process(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = raw.get("ts")    # None for two-series questions
        ts1 = raw.get("ts1")  # None for single-series questions
        ts2 = raw.get("ts2")  # None for single-series questions

        options: List[str] = raw["options"]
        answer_text: str = raw["answer"]

        letters = [_letter(i) for i in range(len(options))]
        answer_letter = _letter(options.index(answer_text))

        prompt = _build_prompt(
            question=raw["question"],
            options=options,
            ts=ts,
            ts1=ts1,
            ts2=ts2,
        )

        # input_ts: list of raw float arrays in the same order as the
        # placeholders / serialized blocks in input_text.
        if ts is not None:
            raw_arrays: List[List[float]] = [ts]
        elif ts1 is not None:
            raw_arrays = [ts1, ts2]
        else:
            raw_arrays = []

        return {
            # model-facing
            "input_text": prompt,
            "input_ts": raw_arrays,
            "output_text": answer_letter,
            "task_id": self.TASK_ID,
            "options": letters,
            # metadata
            "answer_text": answer_text,
            "question": raw["question"],
            "category": raw["category"],
            "subcategory": raw["subcategory"],
            "tid": raw["tid"],
            "id": raw["id"],
            "difficulty": raw.get("difficulty", ""),
        }

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._items[idx]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._items)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def as_batch(self, indices: Optional[List[int]] = None) -> Dict[str, List[Any]]:
        """Collate items into a batch dict for direct use with model.generate().

        Args:
            indices: Optional subset of item indices. None = all items.

        Returns:
            Dict whose keys are the item field names and whose values are
            lists aligned by sample position.
        """
        items = [self._items[i] for i in indices] if indices is not None else self._items
        if not items:
            return {k: [] for k in ("input_text", "input_ts", "output_text", "task_id", "options")}
        return {k: [item[k] for item in items] for k in items[0]}

    def get_field(self, field: str) -> List[Any]:
        """Return a single field across all items as a flat list.

        Convenience for retrieval code that needs e.g. all questions or all
        tid values without constructing a full batch dict.
        """
        return [item[field] for item in self._items]
