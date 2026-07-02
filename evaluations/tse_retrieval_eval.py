"""MCQ accuracy evaluation for Experiment 1 retrieval conditions.

Reuses _extract_predicted_label() from icl_ucr_eval.py — same pattern matching works
for the A/B/C/D option letters used in the retrieval prompt format.
"""

from tqdm import tqdm

from evaluations.icl_ucr_eval import _extract_predicted_label
from retrieval.strategies import retrieve
from retrieval.prompt import build_retrieval_prompt


def run_evaluation_retrieval(
    model,
    query_items: list,
    pool_items: list,
    strategy: str,
    k: int,
    ts_index=None,
    text_index=None,
    alpha: float = 0.5,
    seed: int = 42,
    batch_size: int = 1,
    ts_max_len: int = 128,
    display_samples: int = 0,
) -> dict:
    """Evaluate a single (strategy, k) condition across all query items.

    Args:
        model: BaseModelWrapper — must implement generate(batch) -> List[str]
        query_items: TSEItems from held-out test templates
        pool_items: TSEItems from pool templates (used for retrieval)
        strategy: retrieval condition name
        k: demonstration budget
        ts_index: TSIndex instance (required for ts_only / fusion)
        text_index: TextIndex instance (required for text_only / fusion)
        alpha: TS weight for fusion strategy
        seed: random seed for random / oracle strategies
        batch_size: inference batch size
        ts_max_len: TS points to include in the prompt (subsampled from 1024)
        display_samples: number of sample predictions to print

    Returns:
        dict with keys: accuracy, n_correct, n_total, n_invalid, predictions
    """
    gold_answers = []
    prompts = []
    options_list = []

    for query_item in query_items:
        demos = retrieve(
            query_item, pool_items, strategy=strategy, k=k,
            ts_index=ts_index, text_index=text_index, alpha=alpha, seed=seed,
        )
        prompt, options = build_retrieval_prompt(query_item, demos, ts_max_len=ts_max_len)
        prompts.append(prompt)
        options_list.append(options)
        gold_answers.append(query_item.answer_letter)

    predicted_answers = []
    raw_outputs = []

    for start in tqdm(range(0, len(prompts), batch_size), desc=f"{strategy} k={k}", leave=False):
        batch_prompts = prompts[start:start + batch_size]
        batch_options = options_list[start:start + batch_size]

        batch = {
            "input_text": batch_prompts,
            "input_ts": [[] for _ in batch_prompts],
            "output_text": [""] * len(batch_prompts),
            "options": batch_options,
            "task_id": ["retrieval"] * len(batch_prompts),
        }
        responses = model.generate(batch)

        for resp, opts in zip(responses, batch_options):
            predicted = _extract_predicted_label(resp.strip(), opts)
            predicted_answers.append(predicted)
            raw_outputs.append(resp)

    n_correct = sum(p == g for p, g in zip(predicted_answers, gold_answers))
    n_invalid = sum(p == "INVALID_PREDICTION" for p in predicted_answers)
    n_total = len(gold_answers)

    if display_samples > 0:
        print(f"\n  Sample predictions ({strategy} k={k}):")
        for i in range(min(display_samples, n_total)):
            status = "✓" if predicted_answers[i] == gold_answers[i] else "✗"
            print(f"    [{i+1}] {status} gold={gold_answers[i]} pred={predicted_answers[i]}")
            print(f"         raw: {raw_outputs[i][:120]}")

    return {
        "accuracy": n_correct / n_total if n_total > 0 else 0.0,
        "n_correct": n_correct,
        "n_total": n_total,
        "n_invalid": n_invalid,
        "predictions": [
            {"gold": g, "predicted": p, "raw_output": r}
            for g, p, r in zip(gold_answers, predicted_answers, raw_outputs)
        ],
    }
