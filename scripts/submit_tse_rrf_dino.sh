#!/bin/bash

################################################################################################
# Submit TimeSeriesExam jobs for the rrf and delay_dino retrievers — the two retrievers
# missing from analysis/wandb_runs.csv (random/text/ts/vision_ts are already covered).
#
# Usage:
#   bash scripts/submit_tse_rrf_dino.sh
#
# Job breakdown (90 total at default settings):
#   3 methods × 2 retrievers × 5 shots × 3 seeds = 90 jobs  (no 0-shot — already logged)
#
# Runner routing (same as submit_tse_exp.sh):
#   Qwen3-8B-vllm                → run_single_gpu_vllm.sh   (RTX 4090, multits_large, batch_size=4)
#   Qwen3-VL-8B, ChatTS-8B       → run_single_gpu.sh        (RTX 4090, multits, batch_size=1)
#
# Device split:
#   LLM                         → cuda  (--device cuda)
#   Retriever encoders           → cuda  (--retriever_device cuda)
#   Encoders index the pool on the GPU before the LLM loads, then offload to
#   CPU; leave-one-out queries reuse their indexed embeddings (no re-encoding).
#
# Same exp_id/seeds as submit_tse_exp.sh so these runs land in the same W&B comparison group.
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/DR-TSR"
exp_id="retriever_comparison_v1"
task_id="TimeSeriesExam"
num_samples=""          # set to e.g. 50 for smoke tests; leave empty for full eval

seeds=(2021 0 1)
shots_kshot=(1 2 3 5 8)
retrievers=(rrf delay_dino)

# HF-served GPU models (run_single_gpu.sh, multits env, batch_size=1).
gpu_models=(
    "Qwen/Qwen3-VL-8B-Instruct"
    "bytedance-research/ChatTS-8B"
)

# vLLM-served GPU model (run_single_gpu_vllm.sh, multits_large env, batch_size=4 —
# matches the batch_size already logged for this method in wandb_runs.csv).
vllm_model="Qwen/Qwen3-8B-vllm"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Args shared by every job.
common_args() {
    local method="$1" seed="$2" shot="$3" batch_size="$4"
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

echo "--- k-shot jobs (rrf, delay_dino) ---"

# GPU: HF-served LLMs  (encoders index on cuda before the LLM loads, then offload)
for method in "${gpu_models[@]}"; do
    for seed in "${seeds[@]}"; do
        for shot in "${shots_kshot[@]}"; do
            for retriever in "${retrievers[@]}"; do
                sbatch "$SCRIPT_DIR/run_single_gpu.sh" \
                    $(common_args "$method" "$seed" "$shot" 1) \
                    --retriever        "$retriever" \
                    --device           cuda \
                    --retriever_device cuda
                ((total++))
            done
        done
    done
done

# GPU: vLLM-served LLM  (encoders index on cuda before vLLM preallocates VRAM)
for seed in "${seeds[@]}"; do
    for shot in "${shots_kshot[@]}"; do
        for retriever in "${retrievers[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_gpu_vllm.sh" \
                $(common_args "$vllm_model" "$seed" "$shot" 4) \
                --retriever        "$retriever" \
                --device           cuda \
                --retriever_device cuda
            ((total++))
        done
    done
done

# ---------------------------------------------------------------------------
echo ""
echo "Submitted $total jobs"
echo "  task=${task_id}  exp_id=${exp_id}"
echo "  seeds: ${seeds[*]}"
echo "  k-shot values: ${shots_kshot[*]}  retrievers: ${retrievers[*]}"
echo "  gpu models (HF):   ${gpu_models[*]}"
echo "  gpu model (vLLM):  ${vllm_model}"
