"""Accuracy comparison tables across the four retriever head-to-head sweeps.

Reads `analysis/wandb_runs.csv` (see scripts/fetch_wandb_results.py — pulls every
run's logged `accuracy` summary scalar directly from W&B for all five exp_ids
below) and builds, for each of the three non-baseline models, a table of
accuracy by retriever x num_shots plus a column averaging each retriever across
shots. A final table aggregates across all models and shots.

Reading from W&B instead of local results/*.json matters here: a run can be
"finished" (accuracy logged) on W&B without its local per-sample JSON having
made it to disk, so parsing results/ undercounts completed runs. Refresh the
CSV first with:
    /home/aviramom/.conda/envs/multits/bin/python3 scripts/fetch_wandb_results.py

Retriever groups, each tied to the exp_id of the script that produced it:
    0-shot / random  <- retriever_comparison_v1  (scripts/submit_tse_exp.sh)
    ts_compress / ts_multivec / ts_windowagg  <- ts_encoder_variants_v2  (scripts/submit_tse_ts_variants.sh)
    vision_ts / delay_dino  <- dino3_vision_variants_v1  (scripts/submit_tse_dino3_variants.sh)
    text / text_bge  <- text_encoder_variants_v1  (scripts/submit_tse_text_variants.sh)
    spectral / stats / vision_wavelet  <- freqstat_variants_v1  (scripts/submit_tse_freqstat_variants.sh)

After the tables above (identical to the original head-to-head sweeps), the same
set of tables is printed again with more retrievers appended:
  - the top-3 MR-RRF fusion of the best-performing retriever from each sweep above
    (`vision_ts` + `delay_dino` + `vision_wavelet`, spec
    `rrf-vision_ts-delay_dino-vision_wavelet`), swept over 3 RRF smoothing
    constants (`rrf_k` in {10, 60, 100} — see scripts/submit_tse_top3_mrrf_variants.sh);
  - `dtw` — the classic DTW-on-the-series shape baseline (scripts/submit_tse_dtw.sh);
  - `twostage-delay_dino-text` / `twostage-vision_wavelet-text` — coarse-to-fine
    retrievers that pull TS-similar candidates then re-rank by text
    (scripts/submit_tse_twostage.sh);
  - `siglip_plot` / `siglip_delay` — SigLIP2 fused image+text retrievers (series
    image and question text pooled into one shared vector; plot vs.
    delay-embedding image, scripts/submit_tse_siglip_variants.sh).
Any jobs that haven't run yet show "—" until the CSV is refreshed after they finish.

Usage:
    /home/aviramom/.conda/envs/multits/bin/python3 analysis/retriever_variants_summary.py
"""

import os

import pandas as pd

WANDB_CSV = os.path.join(os.path.dirname(__file__), "wandb_runs.csv")
SHOTS_KSHOT = [1, 2, 3, 5, 8]

MODELS = [
    ("Qwen/Qwen3-VL-8B-Instruct", "Qwen3-VL-8B"),
    ("bytedance-research/ChatTS-8B", "ChatTS-8B"),
    ("Qwen/Qwen3-8B-vllm", "Qwen3-8B (vLLM)"),
]

# (retriever_id, exp_id, display label, group label)
RETRIEVER_SPECS = [
    ("none", "retriever_comparison_v1", "0-shot (no retrieval)", "Baseline"),
    ("random", "retriever_comparison_v1", "random", "Baseline"),
    ("ts_compress", "ts_encoder_variants_v2", "ts_compress", "MOMENT (ts_variants)"),
    ("ts_multivec", "ts_encoder_variants_v2", "ts_multivec", "MOMENT (ts_variants)"),
    ("ts_windowagg", "ts_encoder_variants_v2", "ts_windowagg", "MOMENT (ts_variants)"),
    ("vision_ts", "dino3_vision_variants_v1", "vision_ts (plot embed)", "Vision (dino3_variants)"),
    ("delay_dino", "dino3_vision_variants_v1", "delay_dino (delay embed)", "Vision (dino3_variants)"),
    ("text", "text_encoder_variants_v1", "text (MiniLM)", "Text (text_variants)"),
    ("text_bge", "text_encoder_variants_v1", "text_bge (BGE-large)", "Text (text_variants)"),
    ("spectral", "freqstat_variants_v1", "spectral", "Freq/Stats (freqstat_variants)"),
    ("stats", "freqstat_variants_v1", "stats", "Freq/Stats (freqstat_variants)"),
    ("vision_wavelet", "freqstat_variants_v1", "vision_wavelet", "Freq/Stats (freqstat_variants)"),
]

# Top-3 MR-RRF fusion (vision_ts + delay_dino + vision_wavelet, the best retriever
# from each sweep above) swept over 3 RRF smoothing constants — one exp_id per k
# (see scripts/submit_tse_top3_mrrf_variants.sh). Appended as its own group so the
# original tables above stay unchanged and this shows up as an extra block below.
TOP3_MRRF_RETRIEVER = "rrf-vision_ts-delay_dino-vision_wavelet"
NEW_RETRIEVER_SPECS = [
    (TOP3_MRRF_RETRIEVER, "top3_mrrf_k10_v1", "top3 MR-RRF (rrf_k=10)", "Top-3 MR-RRF (rrf_k sweep)"),
    (TOP3_MRRF_RETRIEVER, "top3_mrrf_k60_v1", "top3 MR-RRF (rrf_k=60)", "Top-3 MR-RRF (rrf_k sweep)"),
    (TOP3_MRRF_RETRIEVER, "top3_mrrf_k100_v1", "top3 MR-RRF (rrf_k=100)", "Top-3 MR-RRF (rrf_k sweep)"),
]

# DTW shape baseline (scripts/submit_tse_dtw.sh) and the two coarse-to-fine
# two-stage retrievers (scripts/submit_tse_twostage.sh): TS-similar candidates
# re-ranked by text. Appended as their own groups, same convention as above.
DTW_TWOSTAGE_SPECS = [
    ("dtw", "dtw_v1", "dtw (DTW shape)", "DTW baseline"),
    ("twostage-delay_dino-text", "twostage_v1", "twostage delay_dino→text", "Two-stage (TS→text)"),
    ("twostage-vision_wavelet-text", "twostage_v1", "twostage vision_wavelet→text", "Two-stage (TS→text)"),
]

# SigLIP2 fused image+text retrievers (scripts/submit_tse_siglip_variants.sh):
# one shared vector per item = pooled (series-image embedding, question-text
# embedding) from the same SigLIP2 encoder. The plot/delay pair mirrors the
# DINOv3 vision_ts/delay_dino pair, isolating the encoder + text fusion.
SIGLIP_SPECS = [
    ("siglip_plot", "siglip_variants_v1", "siglip_plot (plot + text)", "SigLIP2 fused (siglip_variants)"),
    ("siglip_delay", "siglip_variants_v1", "siglip_delay (delay embed + text)", "SigLIP2 fused (siglip_variants)"),
]

EXTENDED_RETRIEVER_SPECS = RETRIEVER_SPECS + NEW_RETRIEVER_SPECS + DTW_TWOSTAGE_SPECS + SIGLIP_SPECS


def load_runs() -> pd.DataFrame:
    df = pd.read_csv(WANDB_CSV)
    df = df.rename(columns={"random_seed": "seed", "accuracy": "acc"})
    return df[["exp_id", "method", "retriever", "num_shots", "seed", "acc"]]


def cell(df: pd.DataFrame, exp_id: str, method: str, retriever: str, shot: int):
    """Mean accuracy across seeds for one (exp_id, method, retriever, shot); None if absent."""
    sub = df[
        (df.exp_id == exp_id)
        & (df.method == method)
        & (df.retriever == retriever)
        & (df.num_shots == shot)
    ]
    if sub.empty:
        return None
    return sub["acc"].mean()


def fmt(v):
    return f"{v:.3f}" if v is not None else "—"


def per_model_table(df: pd.DataFrame, method: str, specs=RETRIEVER_SPECS) -> str:
    header = ["Retriever", "0-shot"] + [f"k={s}" for s in SHOTS_KSHOT] + ["Avg (k-shot)"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    last_group = None
    for retriever, exp_id, label, group in specs:
        if group != last_group:
            lines.append(f"| *{group}* | " + " | ".join([""] * (len(header) - 1)) + " |")
            last_group = group

        zero = cell(df, exp_id, method, retriever, 0) if retriever == "none" else None
        kshot_vals = (
            [] if retriever == "none" else [cell(df, exp_id, method, retriever, s) for s in SHOTS_KSHOT]
        )
        avg = None
        if kshot_vals:
            present = [v for v in kshot_vals if v is not None]
            avg = sum(present) / len(present) if present else None

        row = [label, fmt(zero)]
        if retriever == "none":
            row += ["—"] * len(SHOTS_KSHOT)
        else:
            row += [fmt(v) for v in kshot_vals]
        row.append(fmt(avg))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def summary_table(df: pd.DataFrame, specs=RETRIEVER_SPECS) -> str:
    header = ["Retriever", "Group", "Overall avg accuracy", "Coverage (model x shot cells)"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for retriever, exp_id, label, group in specs:
        shots = [0] if retriever == "none" else SHOTS_KSHOT
        expected = len(MODELS) * len(shots)
        vals = []
        for method, _ in MODELS:
            for s in shots:
                v = cell(df, exp_id, method, retriever, s)
                if v is not None:
                    vals.append(v)
        avg = sum(vals) / len(vals) if vals else None
        lines.append(
            f"| {label} | {group} | {fmt(avg)} | {len(vals)}/{expected} |"
        )
    return "\n".join(lines)


def main():
    df = load_runs()

    for method, label in MODELS:
        print(f"\n### {label}\n")
        print(per_model_table(df, method))

    print("\n### Overall summary (aggregated across all models and shots)\n")
    print(summary_table(df))

    print("\n\n## Same tables, with top-3 MR-RRF, DTW and two-stage retrievers added\n")
    print(
        "(top-3 MR-RRF: vision_ts + delay_dino + vision_wavelet over rrf_k in "
        "{10, 60, 100}, scripts/submit_tse_top3_mrrf_variants.sh; dtw, "
        "scripts/submit_tse_dtw.sh; twostage-<ts>-text, "
        "scripts/submit_tse_twostage.sh — cells show '—' until those jobs finish "
        "and the CSV is refreshed)"
    )

    for method, label in MODELS:
        print(f"\n### {label} (incl. top-3 MR-RRF, DTW, two-stage)\n")
        print(per_model_table(df, method, specs=EXTENDED_RETRIEVER_SPECS))

    print("\n### Overall summary (incl. top-3 MR-RRF, DTW, two-stage)\n")
    print(summary_table(df, specs=EXTENDED_RETRIEVER_SPECS))


if __name__ == "__main__":
    main()
