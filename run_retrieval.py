#!/usr/bin/env python3
"""Experiment 1: cross-template retrieval baseline study on TimeSeriesExam.

Compares six retrieval conditions (zero_shot, random, ts_only, text_only, fusion, oracle)
across k ∈ {0,1,2,3,5,8} demonstration budgets, averaged over multiple cross-template splits.
Results saved to retrieval_results/ and optionally logged to W&B.

Usage:
    python run_retrieval.py --method random_baseline --n_splits 2 --retrieval_strategies zero_shot random
    python run_retrieval.py --method Qwen/Qwen3-4B-Instruct-2507 --n_splits 5 --use_wandb 1
"""

import os
import json
import random

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from utils.args import get_retrieval_parser
from utils.model import method_wrapper_dict
from retrieval.dataset import load_tse_items, cross_template_split
from retrieval.indices import TSIndex, TextIndex
from retrieval.strategies import ALL_STRATEGIES
from evaluations.tse_retrieval_eval import run_evaluation_retrieval
from loggers import setup_logger


def main():
    parser = get_retrieval_parser()
    args, _ = parser.parse_known_args()

    wrapper_class = method_wrapper_dict[args.method]
    args = wrapper_class.get_relevant_args(args, parser)

    if hasattr(args, "quantization") and args.quantization == "none":
        args.quantization = None
    if hasattr(args, "cache_dir") and args.cache_dir == "":
        args.cache_dir = None

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    logger = setup_logger(args)
    if logger.is_completed():
        print("Run already completed, exiting.")
        return
    logger.log_hparams(vars(args))

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"Loading TSE items from {args.qa_dataset_path} ...")
    items = load_tse_items(args.qa_dataset_path)
    n_tids = len(set(i.tid for i in items))
    print(f"  {len(items)} items across {n_tids} templates")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {args.method}")
    model = wrapper_class(args, device=getattr(args, "device", "cuda"))
    if hasattr(model, "setup"):
        model.setup(args)

    # ------------------------------------------------------------------
    # Resolve strategies
    # ------------------------------------------------------------------
    strategies = (
        ALL_STRATEGIES
        if args.retrieval_strategies == ["all"]
        else args.retrieval_strategies
    )
    k_values = [k for k in args.retrieval_k if k > 0]  # positive k for non-zero-shot
    alpha_values = args.fusion_alpha

    print(f"Strategies: {strategies}")
    print(f"k values: {k_values}  |  fusion alphas: {alpha_values}")
    print(f"Splits: {args.n_splits}  |  test_fraction: {args.test_fraction}")

    # ------------------------------------------------------------------
    # Run n_splits
    # ------------------------------------------------------------------
    # all_results: condition_key -> [accuracy per split]
    all_results = {}

    for split_idx in range(args.n_splits):
        split_seed = args.random_seed + split_idx
        print(f"\n=== Split {split_idx + 1}/{args.n_splits} (seed={split_seed}) ===")

        pool_items, query_items = cross_template_split(
            items, test_fraction=args.test_fraction, seed=split_seed,
        )
        print(f"  Pool : {len(pool_items)} items "
              f"({len(set(i.tid for i in pool_items))} templates)")
        print(f"  Query: {len(query_items)} items "
              f"({len(set(i.tid for i in query_items))} templates)")

        # Build indices (once per split, reused across all conditions)
        needs_ts   = any(s in strategies for s in ["ts_only", "fusion"])
        needs_text = any(s in strategies for s in ["text_only", "fusion"])

        ts_index, text_index = None, None
        if needs_ts:
            print("  Building TS index ...")
            ts_index = TSIndex(pool_items)

        if needs_text:
            print(f"  Building text index ({args.text_encoder}) ...")
            text_index = TextIndex(pool_items, model_name=args.text_encoder)
            text_index.build()

        # Evaluate each (strategy, k[, alpha]) combination
        for strategy in strategies:
            ks = [0] if strategy == "zero_shot" else k_values
            alphas = alpha_values if strategy == "fusion" else [None]

            for k in ks:
                for alpha in alphas:
                    cond_key = f"{strategy}_k{k}"
                    if alpha is not None:
                        cond_key += f"_a{alpha:.2f}"

                    result = run_evaluation_retrieval(
                        model=model,
                        query_items=query_items,
                        pool_items=pool_items,
                        strategy=strategy,
                        k=k,
                        ts_index=ts_index,
                        text_index=text_index,
                        alpha=alpha if alpha is not None else 0.5,
                        seed=split_seed,
                        batch_size=args.batch_size,
                        ts_max_len=args.ts_max_len,
                        display_samples=args.display_samples if split_idx == 0 else 0,
                    )

                    all_results.setdefault(cond_key, []).append(result["accuracy"])

                    print(f"  {cond_key:35s}: acc={result['accuracy']:.4f} "
                          f"({result['n_correct']}/{result['n_total']}, "
                          f"{result['n_invalid']} invalid)")

    # ------------------------------------------------------------------
    # Aggregate and report
    # ------------------------------------------------------------------
    print("\n==== Final Results (mean ± std across splits) ====")
    summary = {}
    for cond_key, accs in sorted(all_results.items()):
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        summary[cond_key] = {"mean": mean_acc, "std": std_acc, "per_split": accs}
        print(f"  {cond_key:35s}: {mean_acc:.4f} ± {std_acc:.4f}")

    # Log to W&B
    loggable = {k: v["mean"] for k, v in summary.items()}
    logger.log_metrics(loggable)

    # Save to disk
    os.makedirs("retrieval_results", exist_ok=True)
    model_tag = args.method.replace("/", "_").replace(".", "v")
    out_path = os.path.join("retrieval_results", f"retrieval_{model_tag}_{args.exp_id}.json")
    with open(out_path, "w") as f:
        json.dump(
            {"args": vars(args), "summary": summary},
            f, indent=2, default=str,
        )
    print(f"\nSaved results to {out_path}")

    logger.close()


if __name__ == "__main__":
    main()
