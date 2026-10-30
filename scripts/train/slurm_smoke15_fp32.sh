#!/usr/bin/env bash
# Diagnostic only: same tiny 1.5B run as 70517, changing training precision to FP32.
#SBATCH --job-name=b-smoke15-fp32
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
nvidia-smi
python -m reasoning.train.sft --model Qwen/Qwen2.5-1.5B --precision float32 \
    --limit 64 --val-limit 16 --epochs 1 --val-batch-size 1 \
    --out "ckpt/human_15B_s0_fp32_smoke_${SLURM_JOB_ID}"
