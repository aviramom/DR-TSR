#!/bin/bash

################################################################################################
# Submit TimeSeriesExam jobs for Qwen/Qwen3-8B only (text-only model).
#
# Served via vLLM ("Qwen/Qwen3-8B-vllm" → LargeInstructModel) on a single RTX 4090
# using run_single_gpu_vllm.sh (multits_large env). vLLM's paged KV cache + chunked
# prefill fit the long k-shot ICL prompts in bf16 — the HF pipeline OOM'd on 3-shot.
#
# Usage:
#   bash scripts/submit_tse_qwen3_text.sh
#
# Job breakdown (39 total at default settings):
#   0-shot  : 1 model × 3 seeds              =   3 jobs
#   k-shot  : 1 model × 4 retrievers × 3 shots × 3 seeds = 36 jobs
#   Total   : 39
################################################################################################

SCRIPT_DIR="$(dirname "$0")"

cache_dir="/cs/azencot_fsas/aviramom"
project="aviramom-/DR-TSR"
exp_id="retriever_comparison_v1"
task_id="TimeSeriesExam"
batch_size=4
num_samples=""

seeds=(2021 0 1)
shots_kshot=(1 2 3)
retrievers=(random text vision_ts ts)
method="Qwen/Qwen3-8B-vllm"   # served via vLLM (bf16, paged KV) on a single 4090

common_args() {
    local seed="$1" shot="$2"
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

# echo "--- 0-shot jobs ---"
# for seed in "${seeds[@]}"; do
#     sbatch "$SCRIPT_DIR/run_single_gpu_vllm.sh" \
#         $(common_args "$seed" 0) \
#         --device cuda \
#         --retriever_device cpu
#     ((total++))
# done

echo "--- k-shot jobs ---"
for seed in "${seeds[@]}"; do
    for shot in "${shots_kshot[@]}"; do
        for retriever in "${retrievers[@]}"; do
            sbatch "$SCRIPT_DIR/run_single_gpu_vllm.sh" \
                $(common_args "$seed" "$shot") \
                --retriever        "$retriever" \
                --device           cuda \
                --retriever_device cpu
            ((total++))
        done
    done
done

echo ""
echo "Submitted $total jobs"
echo "  model=${method}  task=${task_id}  exp_id=${exp_id}"
echo "  seeds: ${seeds[*]}"
echo "  k-shot values: ${shots_kshot[*]}  retrievers: ${retrievers[*]}"
