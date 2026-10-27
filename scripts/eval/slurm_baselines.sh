#!/usr/bin/env bash
# SLURM job for the baseline matrix on TC1 (QoS "normal", see MyTCinfo:
# 1 V100 32GB, <=20 CPUs, <=64G, 6 h per job, one GPU per user at a time).
#
#   mkdir -p logs   # once: SLURM writes the log files there
#   sbatch scripts/eval/slurm_baselines.sh
#
#SBATCH --job-name=g-baselines
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                   # one GPU is plenty for <=1.5B models
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00                # TC1 MaxWall
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# The uv environment from README.md (pinned for V100).
module load uv
source .venv/bin/activate
export DTYPE=float16   # V100 has no bf16

# Models are downloaded beforehand on the login node (compute nodes may have no
# internet); point HF at that shared cache and forbid downloads here.
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=1

nvidia-smi
bash scripts/eval/run_baselines.sh gsm8k_val
