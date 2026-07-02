# loggers/

Logging abstraction used by `run_icl.py`. Only W&B is used in practice;
the rest of the files support the fallback print path and the composite pattern.

## Active Files

| File | Purpose |
|------|---------|
| `base_logger.py` | Abstract `BaseLogger` — defines the interface all loggers implement |
| `print_logger.py` | `TqdmLogger` (always active), `PrintLogger`, `LoggerL` — console output |
| `wandb_logger.py` | `WandbLogger` — logs hparams and metrics to W&B; handles run deduplication |
| `composite_logger.py` | `CompositeLogger` — fans out calls to a list of loggers |
| `__init__.py` | `setup_logger(args)` factory — entry point used by `run_icl.py` |

## How setup_logger works

```
setup_logger(args)
    ├─ always: TqdmLogger          ← prints metrics to terminal via tqdm
    └─ if args.use_wandb:
           WandbLogger(project, configs=args)
               ├─ get_matching_run()  ← deduplication: skip if a finished run with
               │                        the same (exp_id, task_id, method, seed) exists
               └─ wandb.init()        ← start a new run otherwise
    → CompositeLogger([TqdmLogger, WandbLogger])   (or just TqdmLogger if W&B off)
```

## WandbLogger details

- **Auth**: reads API key from env (`WANDB_API_KEY`) or from `wandb_logger/token.txt` /
  `logger/token.txt`. If neither exists, raises `FileNotFoundError` and falls back to
  `TqdmLogger` only (see `setup_logger`).
- **Project**: passed via `--project aviramom-/ts-icl`. Entity is the part before `/`.
- **Run deduplication**: `get_matching_run()` queries the W&B API for a finished run
  whose config matches on `args.keys_to_match`. If found and `--override_run 0`,
  `logger.is_completed()` returns `True` and `run_icl.py` exits early.
- **Run completion marker**: `stop()` sets `run.summary["finished"] = True` — this is
  what `get_matching_run()` filters on to identify completed runs.
- **Rank guard**: all logging is no-op when `rank != 0` (multi-GPU safety).

## Retrieval Experiments

The same logging infrastructure is used for retrieval experiments. New W&B metric keys for
retrieval runs will include `retrieval_strategy`, `fusion_alpha`, `k`, and `split_idx` so
results per condition and per k can be filtered and plotted from W&B.

---

## Key CLI args

| Arg | Effect |
|-----|--------|
| `--use_wandb 1` | Enable W&B logging |
| `--project aviramom-/ts-icl` | W&B project (`entity/project`) |
| `--exp_id <name>` | Experiment group name, included in deduplication keys |
| `--override_run 1` | Re-run even if a matching finished run already exists (default: 1) |
