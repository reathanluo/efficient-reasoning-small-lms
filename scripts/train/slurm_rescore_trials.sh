#!/usr/bin/env bash
# Reuse saved adapters; no training. Also score an untrained, matched chat baseline.
#SBATCH --job-name=b-rescore
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load anaconda
source activate sc4001-g
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
score_out="runs/trial_rescore_${SLURM_JOB_ID}"
# Fail before generating anything if an expected adapter is unavailable.
for trial_id in 70327 70328 70464 70465; do
    test -f "ckpt/human_05B_s0_trial_${trial_id}/adapter_config.json" || {
        echo "Missing adapter for trial ${trial_id}" >&2
        exit 1
    }
done
for trial_id in base 70327 70328 70464 70465; do
    lora_args=()
    if [[ "$trial_id" != base ]]; then
        lora_args=(--lora "ckpt/human_05B_s0_trial_${trial_id}")
    fi
    echo "Scoring ${trial_id} on all 500 validation questions"
    python -m reasoning.eval.generate --model Qwen/Qwen2.5-0.5B \
        "${lora_args[@]}" --prompt chat --split gsm8k_val --dtype float16 \
        --max-new-tokens 512 --seed 0 --n 1 --temperature 0 \
        --out "$score_out/${trial_id}.jsonl"
    python -m reasoning.eval.evaluate "$score_out/${trial_id}.jsonl"
done
echo "Completed: ${score_out} (raw generations, scores, metadata and summaries)"
