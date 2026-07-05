#!/usr/bin/env python3
"""
Pull all retriever_comparison_v1 runs from W&B and save a flat CSV summary.

Each row is one completed run (method × retriever × num_shots × seed).
Accuracy-by-group columns (category, subcategory, difficulty) are flattened
into individual columns so the notebook can slice them directly.

Usage:
    python scripts/fetch_wandb_results.py
    python scripts/fetch_wandb_results.py --output analysis/wandb_runs.csv
    python scripts/fetch_wandb_results.py --include_unfinished  # include crashed runs too

Requires: pip install wandb pandas
"""

import argparse
from pathlib import Path

import pandas as pd

ENTITY = "aviramom-"
PROJECT = "DR-TSR"
EXP_ID = "retriever_comparison_v1"

# Config keys stored in W&B (live in run._attrs['config'], not run.config)
CONFIG_KEYS = [
    "method", "task_id", "exp_id",
    "random_seed", "retriever",
    "batch_size", "num_samples",
]

# Keys that the eval loop logs into summary (not config)
SUMMARY_CONFIG_KEYS = ["num_shots"]

# Top-level summary scalars
SUMMARY_SCALARS = [
    "accuracy", "balanced_accuracy",
    "f1_macro", "f1_weighted",
    "precision_macro", "precision_weighted",
    "recall_macro", "recall_weighted",
    "invalid_rate", "total_samples",
]

# Summary prefixes whose keys get flattened into individual columns
SUMMARY_PREFIXES = [
    "accuracy_by_category",
    "accuracy_by_subcategory",
    "accuracy_by_difficulty",
    "accuracy_by_tid",
]


def _extract_row(run) -> dict:
    # api.runs() returns lightweight objects without full config.
    # _attrs['config'] is populated only when the run is loaded individually.
    cfg = run._attrs.get("config") or {}
    summary = dict(run.summary or {})

    row: dict = {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
    }

    for key in CONFIG_KEYS:
        row[key] = cfg.get(key)

    # num_shots is logged to summary, not config
    for key in SUMMARY_CONFIG_KEYS:
        row[key] = summary.get(key)

    for key in SUMMARY_SCALARS:
        row[key] = summary.get(key)

    # All accuracy_by_* keys are already flat in summary (e.g. "accuracy_by_category/Noise Understanding")
    for s_key, s_val in summary.items():
        if any(s_key.startswith(p + "/") for p in SUMMARY_PREFIXES):
            row[s_key] = s_val

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="analysis/wandb_runs.csv",
        help="Destination CSV path (default: analysis/wandb_runs.csv)",
    )
    parser.add_argument(
        "--exp_id", default=EXP_ID,
        help=f"W&B exp_id to filter on (default: {EXP_ID})",
    )
    parser.add_argument(
        "--include_unfinished", action="store_true",
        help="Include runs that did not log summary.finished=True",
    )
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb not installed — run: pip install wandb")

    api = wandb.Api()
    path = f"{ENTITY}/{PROJECT}"

    filters: dict = {"config.exp_id": args.exp_id}
    if not args.include_unfinished:
        filters["summary_metrics.finished"] = True

    print(f"Querying {path}  exp_id={args.exp_id}  finished_only={not args.include_unfinished}")
    # First pass: collect run IDs (api.runs returns lightweight objects without config)
    run_ids = [r.id for r in api.runs(path=path, filters=filters)]
    print(f"Found {len(run_ids)} runs — loading each one for full config...")

    rows = []
    for i, run_id in enumerate(run_ids, 1):
        run = api.run(f"{path}/{run_id}")
        rows.append(_extract_row(run))
        if i % 10 == 0 or i == len(run_ids):
            print(f"  {i}/{len(run_ids)}")

    if not rows:
        print("No runs found — check ENTITY/PROJECT or try --include_unfinished")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["method", "retriever", "num_shots", "random_seed"],
                        ignore_index=True, na_position="last")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nSaved {len(df)} runs → {out}")
    display_cols = [c for c in ["method", "retriever", "num_shots", "random_seed", "accuracy"] if c in df.columns]
    print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
