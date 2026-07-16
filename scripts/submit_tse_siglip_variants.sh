#!/bin/bash

################################################################################################
# Compare the two SigLIP2-backed multimodal retrievers on TimeSeriesExam.
#
# Both variants use the same SigLIP2 encoder (google/siglip2-base-patch16-224),
# which maps images and text into one shared space. Each item is represented by
# a single fused vector: the L2-normalized mean of the series-image embedding
# and the question-text embedding (see retrievers/siglip_retriever.py). The two
# variants differ only in the image representation, mirroring the DINOv3 pair
# from submit_tse_dino3_variants.sh so the encoder is the isolated variable:
#   siglip_plot    — z-score normalized line plot of the series (vision_ts rendering)
#   siglip_delay   — pool-normalized sliding-window delay-embedding image
#                    (delay_dino rendering)
#
# Usage:
#   bash scripts/submit_tse_siglip_variants.sh
#
# Job breakdown (30 total): 3 models × 2 retrievers × 5 shot values × 1 seed.
# No random baseline, no seed sweep — this is a head-to-head of the retrievers,
# and retrieval itself is deterministic.
#
# Runner routing:
#   Qwen3-8B-vllm              → run_single_gpu_vllm.sh   (RTX 4090, multits_large)
#   Qwen3-VL-8B, ChatTS-8B     → run_single_gpu.sh        (RTX 4090, multits)
#
# Device split: SigLIP2 indexes the pool on the GPU before the LLM loads
# (--retriever_device cuda), then offloads to CPU — safe alongside vLLM.
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/DR-TSR"
exp_id="siglip_variants_v1"
task_id="TimeSeriesExam"
batch_size=1
num_samples=""          # set to e.g. 50 for smoke tests; leave empty for full eval

seed=2021               # single seed — retrieval is deterministic per retriever
shots=(1 2 3 5 8)       # matches the shot sweep of the other head-to-head scripts
                        # so results are comparable across encoder/representation choices

retrievers=(siglip_plot siglip_delay)

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
# One job per (model × retriever × shot)
# ---------------------------------------------------------------------------

echo "--- siglip-variant jobs ---"

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
