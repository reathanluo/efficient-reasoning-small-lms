#!/usr/bin/env bash
# No training: compare three validation modes on one adapter, then run C's vLLM path.
#SBATCH --job-name=b-val-check
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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
check_adapter="${1:-ckpt/human_05B_s0_full_e1}"
check_limit="${2:-500}"
check_out="runs/validation_check_${SLURM_JOB_ID}"
python -m reasoning.train.check_validation --adapter "$check_adapter" \
    --limit "$check_limit" --out "$check_out"
# The first Python process exits and releases GPU memory before vLLM starts.
python -m reasoning.eval.generate --model Qwen/Qwen2.5-0.5B --lora "$check_adapter" \
    --prompt chat --split gsm8k_val --limit "$check_limit" --dtype float16 \
    --max-new-tokens 512 --out "$check_out/vllm.jsonl"
python -m reasoning.eval.evaluate "$check_out/vllm.jsonl"
cat "$check_out/summary.json"
