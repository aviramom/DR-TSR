#!/bin/bash

################################################################################################
# Submit TimeSeriesExam jobs for the top-3 retriever RRF fusion, swept over 3 RRF k values.
#
# Per analysis/retriever_variants_summary.py, the three best-performing retrievers across
# the head-to-head sweeps are:
#   vision_wavelet   DINOv3 embedding of the Morlet CWT scalogram (freqstat_variants_v1)
#   delay_dino       DINOv3 embedding of the delay-embedding image (dino3_vision_variants_v1)
#   vision_ts        DINOv3 embedding of the rendered line plot   (dino3_vision_variants_v1)
#
# This script fuses all three via Reciprocal Rank Fusion (retriever spec
# "rrf-vision_ts-delay_dino-vision_wavelet", see MRRF_METHOD.md / retrievers/rrf_retriever.py)
# and sweeps the RRF smoothing constant k (score = sum 1/(k + rank)) over 3 choices to see
# how sensitive the fusion is to it. Each k gets its own exp_id so runs land in distinct
# W&B groups / results files (the retriever spec string is identical across all three).
#
# Usage:
#   bash scripts/submit_tse_top3_mrrf_variants.sh
#
# Job breakdown (135 total at default settings):
#   3 models × 3 rrf_k values × 5 shots × 3 seeds = 135 jobs  (no 0-shot — retrieval needs k>=1)
#
# Runner routing (same as submit_tse_mrrf_full.sh):
#   Qwen3-8B-vllm                → run_single_gpu_vllm.sh   (RTX 4090, multits_large, batch_size=4)
#   Qwen3-VL-8B, ChatTS-8B       → run_single_gpu.sh        (RTX 4090, multits, batch_size=1)
#
# Device split:
#   LLM                          → cuda  (--device cuda)
#   Retriever encoders           → cuda  (--retriever_device cuda)
#   All three retrievers here share the same DINOv3 encoder; each indexes the pool on the
#   GPU sequentially before the LLM loads, then offloads to CPU.
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/DR-TSR"
task_id="TimeSeriesExam"
num_samples=""          # set to e.g. 50 for smoke tests; leave empty for full eval

seeds=(2021 0 1)
shots_kshot=(1 2 3 5 8)
retriever="rrf-vision_ts-delay_dino-vision_wavelet"

# RRF smoothing constant k -> exp_id (one W&B group per k so results stay distinguishable).
rrf_ks=(10 60 100)
exp_id_for_k() {
    case "$1" in
        10)  echo "top3_mrrf_k10_v1"  ;;
        60)  echo "top3_mrrf_k60_v1"  ;;
        100) echo "top3_mrrf_k100_v1" ;;
        *)   echo "top3_mrrf_k$1_v1"  ;;
    esac
}

# HF-served GPU models (run_single_gpu.sh, multits env, batch_size=1).
gpu_models=(
    "Qwen/Qwen3-VL-8B-Instruct"
    "bytedance-research/ChatTS-8B"
)

# vLLM-served GPU model (run_single_gpu_vllm.sh, multits_large env, batch_size=4).
vllm_model="Qwen/Qwen3-8B-vllm"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Args shared by every job.
common_args() {
    local method="$1" seed="$2" shot="$3" batch_size="$4" exp_id="$5"
    local args=(
        --cache_dir       "$cache_dir"
        --method          "$method"
        --task_id         "$task_id"
        --exp_id          "$exp_id"
        --project         "$project"
        --batch_size      "$batch_size"
        --num_shots       "$shot"
        --random_seed     "$seed"
        --use_wandb       1
        --display_samples 3
    )
    [ -n "$num_samples" ] && args+=(--num_samples "$num_samples")
    echo "${args[@]}"
}

total=0

echo "--- k-shot jobs (top-3 MR-RRF: vision_ts + delay_dino + vision_wavelet) ---"

for rrf_k in "${rrf_ks[@]}"; do
    exp_id="$(exp_id_for_k "$rrf_k")"
    echo "  rrf_k=${rrf_k}  exp_id=${exp_id}"

    # GPU: HF-served LLMs  (encoders index on cuda before the LLM loads, then offload)
    for method in "${gpu_models[@]}"; do
        for seed in "${seeds[@]}"; do
            for shot in "${shots_kshot[@]}"; do
                sbatch "$SCRIPT_DIR/run_single_gpu.sh" \
                    $(common_args "$method" "$seed" "$shot" 1 "$exp_id") \
                    --retriever        "$retriever" \
                    --rrf_k            "$rrf_k" \
                    --device           cuda \
                    --retriever_device cuda
                ((total++))
            done
        done
    done

    # GPU: vLLM-served LLM  (encoders index on cuda before vLLM preallocates VRAM)
    for seed in "${seeds[@]}"; do
        for shot in "${shots_kshot[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_gpu_vllm.sh" \
                $(common_args "$vllm_model" "$seed" "$shot" 4 "$exp_id") \
                --retriever        "$retriever" \
                --rrf_k             "$rrf_k" \
                --device           cuda \
                --retriever_device cuda
            ((total++))
        done
    done
done

# ---------------------------------------------------------------------------
echo ""
echo "Submitted $total jobs"
echo "  task=${task_id}"
echo "  seeds: ${seeds[*]}"
echo "  k-shot values: ${shots_kshot[*]}"
echo "  retriever: ${retriever}"
echo "  rrf_k values: ${rrf_ks[*]}"
echo "  gpu models (HF):   ${gpu_models[*]}"
echo "  gpu model (vLLM):  ${vllm_model}"
