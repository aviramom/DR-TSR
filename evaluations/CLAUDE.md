# evaluations/

> **Legacy note**: `icl_ucr_eval.py` was designed for the UCR/TSE classification benchmark phase.
> For Experiment 1 (retrieval baseline study), a new `tse_retrieval_eval.py` module is planned
> (see below). `_extract_predicted_label()` is shared between both.

Contains the evaluation loop and label-extraction logic for ICL experiments.
Called from `run_icl.py` (legacy) and the planned retrieval runner after model inference.

## Active Files

### `icl_ucr_eval.py`

**`run_evaluation_icl_ucr(model, dataloader, args)`** — main evaluation loop.
1. Iterates over batches from the DataLoader.
2. Calls `model.generate(batch)` to get free-text responses.
3. Scores each response against the gold label using 6 regex patterns (see below).
4. Extracts a validated predicted label via `_extract_predicted_label()` for sklearn metrics.
5. Returns two dicts:
   - `results` — scalar metrics logged to W&B / saved to JSON.
   - `input_output` — `questions`, `generated_texts`, `gold_answers`, `input_ts` for display.

**`_extract_predicted_label(response, options)`** — maps a free-text response to one of the
known class labels. Collects **all** match positions across six patterns for all options,
then returns the option whose **last** match appears latest in the string. This handles
thinking-mode models that mention example labels during reasoning before stating the final
prediction at the end.

Patterns checked (all case-insensitive):
1. Exact match (`response == opt`) — fast path
2. `"The class is X"` / `"The class is <X>"`
3. `"Predicted Label: X"` (regex)
4. `"Predicted: X"`
5. `"Label: X"`
6. `"(correct) label is X"`
7. Fallback: bare option token in the last 300 chars of the response

Returns `"INVALID_PREDICTION"` if no pattern matches.

**`_parse_options(prompt)`** — extracts the valid label list from the prompt string.
Looks for the literal substring `"Return ONLY the label as one of: [a, b, ...]"`.

## Metrics Returned

| Key | Description |
|-----|-------------|
| `balanced_accuracy` | Primary metric — macro recall (sklearn). Handles class imbalance. |
| `f1_macro` | Macro F1 |
| `f1_weighted` | Weighted F1 |
| `precision_macro` / `recall_macro` | Macro precision/recall |
| `precision_weighted` / `recall_weighted` | Weighted variants |
| `num_of_classes` | Number of distinct gold classes |
| `total_test_size` | Samples scored |
| `accuracy_scores` | Per-sample binary list — **not logged** (filtered in `run_icl.py`) |

Note: `recall_macro` == `balanced_accuracy` mathematically (both are macro-averaged recall).

---

## `tse_aggregate_results.py`

Post-hoc analysis script for TSE experiments. Reads per-template JSON result files from
`outputs/` and aggregates by metadata from `qa_dataset_augmented.json`.

```bash
python evaluations/tse_aggregate_results.py \
  --results_dir outputs/ \
  --augmented_path qa_dataset_augmented.json \
  --method Qwen/Qwen3-4B-Instruct-2507
```

Reports mean balanced_accuracy broken down by `difficulty` (easy/medium/hard) and
`category` (Anomaly Detection, Pattern Recognition, etc.). This is an offline script
and does not modify any run state.

---

---

## Planned: `tse_retrieval_eval.py` — Retrieval Experiment Evaluation

New evaluation module for Experiment 1. Different from `icl_ucr_eval.py` in two ways:
- **Metric**: MCQ accuracy (fraction correct), not balanced accuracy. TSE test sets are class-balanced, so accuracy and balanced accuracy coincide — but accuracy is the correct framing for MCQ.
- **Loop structure**: iterates over retrieval conditions and k values, not just a single (method, task) pair.

**Planned interface**:

```python
run_evaluation_tse_retrieval(
    model,           # BaseModelWrapper
    query_items,     # list of (question_text, ts, correct_option) from test templates
    pool_items,      # list of (question_text, ts, correct_option) from pool templates
    retrieval_fn,    # callable(query, pool, k) → list[pool_item]
    k_values,        # e.g. [0, 1, 2, 3, 5, 8]
    args,
) → dict[k, MCQ_accuracy]
```

Reuses `_extract_predicted_label()` from `icl_ucr_eval.py` for option matching (same A/B/C/D patterns).

---

## Known Issues

1. **`_parse_options` fails silently** — if the prompt does not contain the exact string
   `"Return ONLY the label as one of: [...]"`, it returns `[]`, causing
   `_extract_predicted_label` to always return `"INVALID_PREDICTION"` and all sklearn
   metrics to be meaningless. Depends on `utils/formatting.py` always injecting that line.

2. **task_id guard** — `run_evaluation_icl_ucr()` raises `ValueError` if `task_id` contains
   neither `"ucr"` nor `"tse"`. TSE tasks (`icl_tse_*`) pass this check. Any future
   benchmark needs its prefix added to the guard at `evaluations/icl_ucr_eval.py:36`.
