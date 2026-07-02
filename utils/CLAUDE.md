# utils/

Shared infrastructure used by `run_exp.py` and model wrappers.

---

## Files

### `args.py`

Defines the global CLI argument namespace via `get_parser()` / `create_parser()`.

```python
parser = get_parser()          # returns ArgumentParser
args   = create_parser()       # parses sys.argv and post-processes defaults
```

**Key args:**

| Arg | Default | Description |
|-----|---------|-------------|
| `--method` | `random_baseline` | Key into `method_wrapper_dict` in `model.py` |
| `--task_id` | `TimeSeriesExam` | Dataset to evaluate; data path resolved from `configs/data_paths.yaml` |
| `--num_shots` | `1` | k-shot demonstrations (0 = zero-shot) |
| `--batch_size` | `1` | Samples per `model.generate()` call |
| `--num_samples` | `None` | Cap on dataset size for smoke tests |
| `--display_samples` | `3` | Debug samples printed at end of run |
| `--exp_id` | `"1"` | Experiment group label (used in W&B deduplication) |
| `--project` | `aviramom-/DR-TSR` | W&B project (`entity/project`) |
| `--use_wandb` | `0` | Enable W&B logging |
| `--override_run` | `1` | Re-run even if a matching finished W&B run exists |
| `--device` | `cuda` | PyTorch device |
| `--quantization` | `none` | `4bit` / `8bit` / `none` |
| `--cache_dir` | `""` → `None` | HuggingFace model cache directory |

**Important constraint**: `BaseModelWrapper.get_relevant_args()` raises `ValueError` if any key
in a model's `get_args_dict()` already exists in the global namespace. Never add model-specific
keys (`input_mode`, `max_new_tokens`, `model_type`, etc.) to `get_parser()`.

---

### `model.py`

Contains `method_wrapper_dict` — the single registry that maps `--method` string IDs to
`BaseModelWrapper` subclasses.

```python
from utils.model import method_wrapper_dict

model_cls = method_wrapper_dict["random_baseline"]   # → RandomBaseline
model_cls = method_wrapper_dict["Qwen/Qwen3-4B-Instruct-2507"]  # → InstructModel
```

Adding a new model: import the class and add one or more `"method-id": ModelClass` entries.
See `models/CLAUDE.md` for the full wrapper overview and batch contract.
