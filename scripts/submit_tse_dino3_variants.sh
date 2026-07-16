#!/bin/bash

################################################################################################
# Compare the two DINOv3-backed vision retrievers on TimeSeriesExam.
#
# Both VisionTSRetriever and DelayDINORetriever now default to DINOv3
# (facebook/dinov3-vitb16-pretrain-lvd1689m) instead of the previously ungated
# DINOv2 backbone — see retrievers/vision_ts_retriever.py and
# retrievers/delay_dino_retriever.py. This is a head-to-head of the two image
# representations (line plot vs. delay-embedding) on the same encoder:
#   vision_ts    — z-score normalized line plot of the series
#   delay_dino   — pool-normalized sliding-window delay-embedding image
#
# Usage:
#   bash scripts/submit_tse_dino3_variants.sh
#
# Job breakdown (30 total): 3 models × 2 retrievers × 5 shot values × 1 seed.
# No random baseline, no seed sweep — this is a head-to-head of the retrievers,
# and retrieval itself is deterministic.
#
# Runner routing:
#   Qwen3-8B-vllm              → run_single_gpu_vllm.sh   (RTX 4090, multits_large)
#   Qwen3-VL-8B, ChatTS-8B     → run_single_gpu.sh        (RTX 4090, multits)
#
# Device split: DINOv3 indexes the pool on the GPU before the LLM loads
# (--retriever_device cuda), then offloads to CPU — safe alongside vLLM.
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/DR-TSR"
exp_id="dino3_vision_variants_v1"
task_id="TimeSeriesExam"
batch_size=1
num_samples=""          # set to e.g. 50 for smoke tests; leave empty for full eval

seed=2021               # single seed — retrieval is deterministic per retriever
shots=(1 2 3 5 8)                # matches the shot sweep of submit_tse_ts_variants.sh
                        # so results are comparable across encoder/representation choices

retrievers=(delay_dino vision_ts)

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

echo "--- dino3-vision-variant jobs ---"

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
