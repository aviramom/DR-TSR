#!/bin/bash

################################################################################################
# Compare the three MOMENT long-series indexing strategies on TimeSeriesExam.
#
# The original `ts` retriever truncates every series to MOMENT's 512-step window,
# throwing away half of each 1024-point series. These three variants each keep the
# full series a different way (see retrievers/moment_base.py):
#   ts_compress   (A) — downsample whole series to 512, embed once
#   ts_multivec   (B) — embed every 512-step window, index all vectors, MaxSim
#   ts_windowagg  (C) — embed every window, length-weighted average to one vector
#
# Usage:
#   bash scripts/submit_tse_ts_variants.sh
#
# Job breakdown (45 total): 3 models × 3 retrievers × 5 shot values × 1 seed.
# No random baseline, no seed sweep — this is a head-to-head of the retrievers,
# and retrieval itself is deterministic.
#
# Runner routing:
#   Qwen3-8B-vllm              → run_single_gpu_vllm.sh   (RTX 4090, multits_large)
#   Qwen3-VL-8B, ChatTS-8B     → run_single_gpu.sh        (RTX 4090, multits)
#
# Device split: MOMENT indexes the pool on the GPU before the LLM loads
# (--retriever_device cuda), then offloads to CPU — safe alongside vLLM.
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/DR-TSR"
exp_id="ts_encoder_variants_v2"
task_id="TimeSeriesExam"
batch_size=1
num_samples=""          # set to e.g. 50 for smoke tests; leave empty for full eval

seed=2021               # single seed — retrieval is deterministic per retriever
shots=(1 2 3 5 8)                # single k; matches the k=3 runs of retriever_comparison_v1
                        # so results are comparable against the truncating `ts`

retrievers=(ts_compress ts_multivec ts_windowagg)

# HF-served GPU models (run_single_gpu.sh, multits env).
gpu_models=(
    "Qwen/Qwen3-VL-8B-Instruct"
    "bytedance-research/ChatTS-8B"
)

# vLLM-served GPU models (run_single_gpu_vllm.sh, multits_large env).
vllm_models=(
    "Qwen/Qwen3-8B-vllm"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Args shared by every job.
common_args() {
    local method="$1" retriever="$2" shot="$3"
    local args=(
        --cache_dir        "$cache_dir"
        --method           "$method"
        --task_id          "$task_id"
        --exp_id           "$exp_id"
        --project          "$project"
        --batch_size       "$batch_size"
        --num_shots        "$shot"
        --random_seed      "$seed"
        --retriever        "$retriever"
        --device           cuda
        --retriever_device cuda
        --use_wandb        1
        --display_samples  3
    )
    [ -n "$num_samples" ] && args+=(--num_samples "$num_samples")
    echo "${args[@]}"
}

total=0

# ---------------------------------------------------------------------------
# One job per (model × retriever)
# ---------------------------------------------------------------------------

echo "--- ts-variant jobs ---"

# GPU: LLMs (HF runner)
for method in "${gpu_models[@]}"; do
    for retriever in "${retrievers[@]}"; do
        for shot in "${shots[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_gpu.sh" \
                $(common_args "$method" "$retriever" "$shot")
            ((total++))
        done
    done
done

# GPU: LLMs (vLLM runner)
for method in "${vllm_models[@]}"; do
    for retriever in "${retrievers[@]}"; do
        for shot in "${shots[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_gpu_vllm.sh" \
                $(common_args "$method" "$retriever" "$shot")
            ((total++))
        done
    done
done

# ---------------------------------------------------------------------------
echo ""
echo "Submitted $total jobs"
echo "  task=${task_id}  exp_id=${exp_id}"
echo "  seed: ${seed}  shots: ${shots[*]}  retrievers: ${retrievers[*]}"
echo "  gpu models (HF):   ${gpu_models[*]}"
echo "  gpu models (vLLM): ${vllm_models[*]}"
