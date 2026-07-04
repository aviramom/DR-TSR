"""Evaluation loop for the TimeSeriesExam benchmark.

Entry point: ``evaluate_tse(model, dataset, ...)``.

The function is retriever-aware: pass a retriever object to enable k-shot
in-context learning.  Without one it runs zero-shot.

Retriever contract (for future use):
    retriever.retrieve(query_item: dict, k: int) -> List[dict]
    Returns k demo items in the same format as TimeSeriesExamDataset items.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

INVALID_PREDICTION = "INVALID_PREDICTION"


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def _extract_label(response: str, options: List[str]) -> str:
    """Return the option letter that best matches the model's response.

    Strategy: collect every position in the string where an option is
    recognisably mentioned under any pattern, then return the option whose
    *last* match is latest — that is the model's final answer rather than
    a label it cited in reasoning.

    Options here are single capital letters (A–D), so patterns are tight
    to avoid spurious matches on common words.
    """
    if not response:
        return INVALID_PREDICTION

    # Fast path: exact single-token match
    if response.strip() in options:
        return response.strip()

    patterns = [
        # "Answer: A" or "Answer is A"
        lambda o: [(m.start(), o) for m in re.finditer(
            r'\bAnswer\s*:?\s*' + re.escape(o) + r'\b', response, re.IGNORECASE)],
        # "the answer is A" / "the label is A"
        lambda o: [(m.start(), o) for m in re.finditer(
            r'\b(?:answer|label)\s+is\s+' + re.escape(o) + r'\b', response, re.IGNORECASE)],
        # "A) Exponential" style — letter followed immediately by closing paren
        lambda o: [(m.start(), o) for m in re.finditer(
            r'(?<!\w)' + re.escape(o) + r'\)', response)],
        # "Return ONLY the label ... [A]" — letter inside brackets
        lambda o: [(m.start(), o) for m in re.finditer(
            r'\[' + re.escape(o) + r'\]', response)],
        # Bare letter at a word boundary (lowest priority)
        lambda o: [(m.start(), o) for m in re.finditer(
            r'(?<!\w)' + re.escape(o) + r'(?!\w)', response)],
    ]

    all_hits: List[Tuple[int, str]] = []
    for pat in patterns:
        for opt in options:
            all_hits.extend(pat(opt))

    if all_hits:
        _, best = max(all_hits, key=lambda x: x[0])
        return best

    # Last-resort: scan the final 100 characters only
    tail = response[-100:]
    for opt in options:
        if re.search(r'(?<!\w)' + re.escape(opt) + r'(?!\w)', tail):
            return opt

    return INVALID_PREDICTION


# ---------------------------------------------------------------------------
# k-shot prompt building
# ---------------------------------------------------------------------------

def build_icl_prompt(
    query_item: Dict[str, Any],
    demo_items: List[Dict[str, Any]],
) -> Tuple[str, List[List[float]]]:
    """Prepend demo_items to the query prompt to form a k-shot input.

    Each demo is shown with its answer; the query keeps its original
    "Return ONLY the label…" instruction.

    Works for both combined and separate input modes — in separate mode the
    ``<ts><ts/>`` placeholders accumulate in order so image/ChatTS models
    receive the right number of raw arrays.

    Returns:
        augmented_input_text: str — the k-shot prompt
        augmented_input_ts:   List[List[float]] — demo arrays + query arrays
                              in placeholder order
    """
    parts: List[str] = []
    aug_ts: List[List[float]] = []

    for i, demo in enumerate(demo_items):
        # Strip "Return ONLY…" tail from the demo, then append its answer
        demo_body = demo["input_text"].rsplit("\nReturn ONLY", 1)[0]
        parts.append(
            f"--- Example {i + 1} ---\n{demo_body}\n\nAnswer: {demo['output_text']}"
        )
        aug_ts.extend(demo["input_ts"])

    parts.append(f"--- Query ---\n{query_item['input_text']}")
    aug_ts.extend(query_item["input_ts"])

    return "\n\n".join(parts), aug_ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_accuracy(
    golds: List[str],
    preds: List[str],
    groups: List[str],
) -> Dict[str, float]:
    correct: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    for g, p, gr in zip(golds, preds, groups):
        total[gr] += 1
        correct[gr] += int(g == p)
    return {k: correct[k] / total[k] for k in sorted(total)}


def _batched(items: List[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _collate(items: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    return {k: [item[k] for item in items] for k in items[0]}


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_tse(
    model: Any,
    dataset: Any,
    batch_size: int = 1,
    num_shots: int = 0,
    retriever: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the TimeSeriesExam evaluation loop.

    Args:
        model:      Any BaseModelWrapper subclass with a ``generate(batch)``
                    method.
        dataset:    A ``TimeSeriesExamDataset`` (or any iterable of item dicts
                    with the standard batch contract fields).
        batch_size: Number of samples per model call.
        num_shots:  How many demonstrations to prepend to each query.
                    Ignored when ``retriever`` is None.
        retriever:  Optional retriever object.  Must implement
                    ``retrieve(query_item: dict, k: int) -> List[dict]``.
                    When None, the evaluation is zero-shot regardless of
                    ``num_shots``.

    Returns:
        metrics:    Dict of scalar / nested metrics for logging.
        artifacts:  Dict of per-sample lists for debugging and analysis.
    """
    all_items: List[Dict[str, Any]] = list(dataset)

    # ---- Retriever augmentation (k-shot) -----------------------------------
    if retriever is not None and num_shots > 0:
        augmented: List[Dict[str, Any]] = []
        for item in tqdm(all_items, desc="Retrieving demonstrations"):
            demo_items = retriever.retrieve(item, k=num_shots)
            aug_text, aug_ts = build_icl_prompt(item, demo_items)
            augmented.append({**item, "input_text": aug_text, "input_ts": aug_ts})
        all_items = augmented
        effective_shots = num_shots
    else:
        effective_shots = 0

    # ---- Inference loop -----------------------------------------------------
    gold_answers: List[str] = []
    predicted_answers: List[str] = []
    generated_texts: List[str] = []
    categories: List[str] = []
    subcategories: List[str] = []
    difficulties: List[str] = []
    item_ids: List[int] = []
    input_prompts: List[str] = []
    input_ts_list: List[List[List[float]]] = []
    questions: List[str] = []

    for chunk in tqdm(list(_batched(all_items, batch_size)), desc="Evaluating TSE"):
        batch = _collate(chunk)
        gen_out = model.generate(batch)

        for i, item in enumerate(chunk):
            response = str(gen_out[i]).strip()
            gold = item["output_text"]
            predicted = _extract_label(response, item["options"])

            gold_answers.append(gold)
            predicted_answers.append(predicted)
            generated_texts.append(response)
            categories.append(item["category"])
            subcategories.append(item["subcategory"])
            difficulties.append(item["difficulty"])
            item_ids.append(item["id"])
            input_prompts.append(item["input_text"])
            input_ts_list.append(item["input_ts"])
            questions.append(item.get("question", ""))

    # ---- Metrics ------------------------------------------------------------
    n = len(gold_answers)
    correct = [int(g == p) for g, p in zip(gold_answers, predicted_answers)]
    n_invalid = sum(1 for p in predicted_answers if p == INVALID_PREDICTION)

    metrics: Dict[str, Any] = {
        "accuracy": sum(correct) / n if n else 0.0,
        "invalid_rate": n_invalid / n if n else 0.0,
        "total_samples": n,
        "num_shots": effective_shots,
        # Per-group breakdowns (nested dicts — logger can flatten as needed)
        "accuracy_by_category": _group_accuracy(gold_answers, predicted_answers, categories),
        "accuracy_by_subcategory": _group_accuracy(gold_answers, predicted_answers, subcategories),
        "accuracy_by_difficulty": _group_accuracy(gold_answers, predicted_answers, difficulties),
    }

    if _SKLEARN:
        metrics["balanced_accuracy"] = balanced_accuracy_score(gold_answers, predicted_answers)
        metrics["f1_macro"] = f1_score(gold_answers, predicted_answers, average="macro", zero_division=0)
        metrics["f1_weighted"] = f1_score(gold_answers, predicted_answers, average="weighted", zero_division=0)
        metrics["precision_macro"] = precision_score(gold_answers, predicted_answers, average="macro", zero_division=0)
        metrics["precision_weighted"] = precision_score(gold_answers, predicted_answers, average="weighted", zero_division=0)
        metrics["recall_macro"] = recall_score(gold_answers, predicted_answers, average="macro", zero_division=0)
        metrics["recall_weighted"] = recall_score(gold_answers, predicted_answers, average="weighted", zero_division=0)

    # ---- Artifacts ----------------------------------------------------------
    artifacts: Dict[str, Any] = {
        "item_ids": item_ids,
        "questions": questions,
        "input_prompts": input_prompts,
        "generated_texts": generated_texts,
        "predicted_answers": predicted_answers,
        "gold_answers": gold_answers,
        "correct": correct,
        "categories": categories,
        "subcategories": subcategories,
        "difficulties": difficulties,
        "input_ts": input_ts_list,
    }

    return metrics, artifacts
