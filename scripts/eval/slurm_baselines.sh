#!/usr/bin/env bash
# SLURM job for the baseline matrix on the NTU CCDS TC1 cluster.
# Submit it from the repo root on the TC1 head node:
#
#   sbatch scripts/eval/slurm_baselines.sh
#
# Values follow the TC1 user guide (Feb 2026): partition UGGPU-TC1, QoS
# "normal" = at most 1 GPU, 20 CPUs, 64G memory, 2 jobs and 6 hours per user.
# Run `MyTCinfo` on the head node to confirm your own QoS before submitting.
# The GPUs are NVIDIA V100 32GB, which have no bfloat16, so DTYPE=float16 below.
#
#SBATCH --job-name=g-baselines
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# A uv .venv in the repo (README "Setup on the TC1 cluster"), or the older conda env.
if [[ -f .venv/bin/activate ]]; then
  module load uv
  source .venv/bin/activate
else
  module load anaconda
  source activate sc4001-g
fi

# Models are downloaded beforehand on the head node; read them from that cache
# and never download inside the job.
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=1
export DTYPE=float16

nvidia-smi
bash scripts/eval/run_baselines.sh gsm8k_val
