# utils/

Shared utilities for `run_icl.py`. Three active files; nothing here does model inference or data loading.

---

## Files

### `args.py` — CLI argument definitions

**`get_parser()`** — returns the `ArgumentParser` used by `run_icl.py`.

```python
parser = get_parser()
args, _ = parser.parse_known_args()
```

**`create_parser(notebook=False)`** — convenience wrapper for notebook usage. Calls `get_parser()`,
parses args, then post-processes: converts `quantization="none"` → `None` and `cache_dir=""` → `None`.
Not used by `run_icl.py` directly.

Key argument groups:

| Group | Key args |
|-------|----------|
| Experiment | `--exp_id`, `--random_seed` |
| Logging | `--use_wandb`, `--project`, `--override_run`, `--keys_to_match` |
| Model | `--method`, `--cache_dir`, `--quantization`, `--device` |
| Data | `--data_path`, `--num_samples` |
| Task | `--task_id` (`icl_ucr_<Name>` for UCR, `icl_tse_<tid>` for TSE) |
| ICL | `--num_shots`, `--picking_strategy`, `--use_label_desc`, `--desc_dir` |
| Inference | `--batch_size`, `--display_samples` |
| TSE | `--tse_data_path` (path to `qa_dataset_augmented.json`, default: `qa_dataset_augmented.json`), `--tse_test_fraction` (train/test split ratio, default: `0.3`) |

---

### `model.py` — model registry

**`method_wrapper_dict`** — maps `--method` string → `BaseModelWrapper` subclass.

```python
wrapper_class = method_wrapper_dict[args.method]
args = wrapper_class.get_relevant_args(args, parser)
model = wrapper_class(args, device=...)
```

Optional deps (`ChatTSHFWrapper`, `ChatTSVLLMWrapper`) are wrapped in `try/except ImportError`
so missing packages don't crash the registry at import time.

---

### `formatting.py` — prompt builder

**`icl_classification_format(desc, examples, target, options)`** — assembles the final ICL prompt
string for a single test sample. Called by `data_provider/icl_dataset.py`.

Output structure:
```
Time Series Classification.
<desc>

<few-shot examples>

<query>
Return ONLY the label as one of: [class_a, class_b, ...] without any explanation
```

The literal string `Return ONLY the label as one of: [...]` is load-bearing — `evaluations/icl_ucr_eval.py:_parse_options()` parses it with a regex to recover the valid label list. Do not change the phrasing.

**Options string format** — options are serialized as `[A, B, C]` (no quotes around items).
This is intentional: Python's default `str(list)` produces `['A', 'B', 'C']` with embedded
quotes, which causes `_parse_options` to return `["'A'", " 'B'", ...]` (with extra quote chars),
breaking exact-match against gold labels `"A"`, `"B"`, etc. The fix is in `formatting.py:3`:
```python
options_str = "[" + ", ".join(str(o) for o in options) + "]"
```
Do not revert this to `{options}`.

---

## Planned Retrieval Args (to be added to `args.py`)

When the `retrieval/` module is built, these args will be added to `get_parser()`:

| Arg | Description |
|-----|-------------|
| `--pool_path` | Path to demonstration pool JSON |
| `--retrieval_strategy` | `random` \| `ts_only` \| `text_only` \| `fusion` \| `oracle` |
| `--retrieval_k` | Number of demonstrations to retrieve (list, e.g. `1 2 3 5 8`) |
| `--fusion_alpha` | Weight for TS similarity in fusion score (0.0–1.0) |
| `--n_splits` | Number of cross-template splits for averaging |

---

## Fit with `run_icl.py`

```
run_icl.py
  ├─ get_parser()                     ← utils/args.py
  ├─ method_wrapper_dict[args.method] ← utils/model.py
  └─ (icl_dataset.py calls)
       └─ icl_classification_format() ← utils/formatting.py
```

`utils/` has no other active files. `cloud_logger.py` and `metrics.py` were removed —
they were legacy code from a prior project with no callers.
