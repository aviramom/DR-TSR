"""DR-TSR experiment runner.

Usage:
    python run_exp.py --method random_baseline --task_id TimeSeriesExam \
        [--use_wandb 1] [--num_shots 0]

Data paths are resolved automatically from configs/data_paths.yaml.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from utils.args import get_parser
from utils.model import method_wrapper_dict
from data_provider import TimeSeriesExamDataset
from evaluations import evaluate_tse
from loggers import setup_logger

_DATA_PATHS_CONFIG = Path(__file__).parent / "configs" / "data_paths.yaml"


def _load_data_paths() -> Dict[str, str]:
    with open(_DATA_PATHS_CONFIG) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset / eval dispatch
# ---------------------------------------------------------------------------

def _build_dataset(args, data_paths: Dict[str, str]):
    task = args.task_id
    if task not in data_paths:
        raise ValueError(
            f"No data path configured for task_id {task!r}. "
            f"Add an entry to {_DATA_PATHS_CONFIG}."
        )
    path = data_paths[task]
    if task == "TimeSeriesExam":
        return TimeSeriesExamDataset(
            data_path=path,
            num_samples=args.num_samples,
            category=getattr(args, "category", None),
        )
    raise ValueError(f"Unknown task_id: {task!r}")


def _get_eval_fn(task_id: str):
    if task_id == "TimeSeriesExam":
        return evaluate_tse
    raise ValueError(f"No eval function for task_id: {task_id!r}")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _flatten_metrics(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Recursively flatten nested metric dicts with '/' separators."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_metrics(v, prefix=key))
        else:
            flat[key] = v
    return flat


def _print_metrics(metrics: Dict[str, Any]) -> None:
    flat = _flatten_metrics(metrics)
    width = max(len(k) for k in flat)
    for k in sorted(flat):
        v = flat[k]
        print(f"  {k:<{width}}  {v:.4f}" if isinstance(v, float) else f"  {k:<{width}}  {v}")


def _save_results(args, artifacts: Dict[str, Any]) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    retriever_tag = getattr(args, "retriever", "none")
    slug = re.sub(
        r"[^a-zA-Z0-9_\-]", "_",
        f"{args.exp_id}__{args.method}__{args.task_id}"
        f"__seed{args.random_seed}__{retriever_tag}__{args.num_shots}shot",
    )
    out_path = results_dir / f"{slug}.json"

    # Run-level metadata written into every row so the file is self-contained
    # (no need to cross-reference filenames or W&B configs during analysis).
    run_meta = {
        "exp_id":     args.exp_id,
        "method":     args.method,
        "task_id":    args.task_id,
        "seed":       args.random_seed,
        "retriever":  retriever_tag,
        "num_shots":  args.num_shots,
        "rrf_k":      args.rrf_k,
        "stage1_candidates": args.stage1_candidates,
    }

    rows = [
        {
            **run_meta,
            "id":                  artifacts["item_ids"][i],
            "tid":                 artifacts["tids"][i],
            "category":            artifacts["categories"][i],
            "subcategory":         artifacts["subcategories"][i],
            "difficulty":          artifacts["difficulties"][i],
            "question":            artifacts["questions"][i],
            "gold_answer":         artifacts["gold_answers"][i],
            "predicted_answer":    artifacts["predicted_answers"][i],
            "generated_text":      artifacts["generated_texts"][i],
            "correct":             bool(artifacts["correct"][i]),
            "input_prompt":        artifacts["input_prompts"][i],
            "retrieved_demo_ids":  artifacts["retrieved_demo_ids"][i],
            "retrieved_demo_tids": artifacts["retrieved_demo_tids"][i],
        }
        for i in range(len(artifacts["item_ids"]))
    ]

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"[run_exp] Results saved → {out_path}  ({len(rows)} rows)")


def _print_samples(artifacts: Dict[str, Any], n: int) -> None:
    if n <= 0:
        return
    ids     = artifacts["item_ids"]
    prompts = artifacts["input_prompts"]
    golds   = artifacts["gold_answers"]
    preds   = artifacts["predicted_answers"]
    gens    = artifacts["generated_texts"]
    cats    = artifacts["categories"]

    count = min(n, len(ids))
    print(f"\n{'=' * 70}")
    print(f"Debug samples (first {count})")
    print("=" * 70)
    for i in range(count):
        mark = "OK" if golds[i] == preds[i] else "X"
        print(f"\n[{i}] id={ids[i]}  cat={cats[i]}  [{mark}]")
        print(f"  gold={golds[i]}  pred={preds[i]}")
        print(f"  response: {gens[i][:150]!r}")
        snippet = prompts[i][:300].replace("\n", " | ")
        print(f"  prompt:   {snippet}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load data path config
    data_paths = _load_data_paths()

    # 2. Base arg parse — need --method before we can load model-specific args
    parser = get_parser()
    args = parser.parse_args()

    if args.method not in method_wrapper_dict:
        raise ValueError(
            f"Unknown --method {args.method!r}. "
            f"Available: {sorted(method_wrapper_dict)}"
        )

    # 3. Extend parser with model-specific defaults and re-parse
    model_cls = method_wrapper_dict[args.method]
    args = model_cls.get_relevant_args(args, parser)

    # 4. Logger — before heavy work so early-exit is cheap
    logger = setup_logger(args)
    if logger.is_completed():
        print("Run already completed — skipping. Pass --override_run 1 to force.")
        sys.exit(0)
    logger.log_hparams(vars(args))

    print(f"\n[run_exp] method={args.method}  task={args.task_id}  shots={args.num_shots}")
    print(f"[run_exp] batch_size={args.batch_size}  num_samples={args.num_samples}")

    # 5. Dataset
    dataset = _build_dataset(args, data_paths=data_paths)
    print(f"[run_exp] dataset={args.task_id}  n={len(dataset)}")

    # 6. Retriever — leave-one-out: index full dataset, exclusion per query.
    #    Indexed BEFORE the LLM loads so encoders can use the GPU (vLLM
    #    preallocates most of VRAM at load time); encoders offload themselves
    #    to CPU at the end of index().
    from utils.retriever import build_retriever

    if args.retriever == "none" or args.num_shots == 0:
        retriever = None
    else:
        print(f"[run_exp] retriever={args.retriever}  rrf_k={args.rrf_k}  indexing {len(dataset)} items...")
        retriever = build_retriever(
            args.retriever,
            device=args.retriever_device,
            rrf_k=args.rrf_k,
            stage1_candidates=args.stage1_candidates,
        )
        retriever.index(list(dataset))

    # 7. Model
    model = model_cls(args, device=args.device)
    model.load_model()

    # 8. Evaluate (full dataset — retriever excludes query item internally)
    eval_fn = _get_eval_fn(args.task_id)
    metrics, artifacts = eval_fn(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        num_shots=args.num_shots,
        retriever=retriever,
    )

    # 9. Log and print metrics
    print("\n[run_exp] Metrics:")
    _print_metrics(metrics)
    logger.log_metrics(_flatten_metrics(metrics))

    # 10. Save per-sample predictions
    _save_results(args, artifacts)

    # 11. Debug samples
    _print_samples(artifacts, n=args.display_samples)

    logger.stop()
    print("\n[run_exp] Done.")


if __name__ == "__main__":
    main()
