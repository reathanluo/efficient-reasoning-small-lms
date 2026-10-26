#!/usr/bin/env bash
# SLURM job for the baseline matrix. The #SBATCH values are placeholders:
# check the NTU cluster guide for the right partition, QoS, GPU type and time
# limit, and edit the lines marked ADAPT.
#
#   sbatch scripts/eval/slurm_baselines.sh
#
#SBATCH --job-name=g-baselines
#SBATCH --partition=CHANGE_ME          # ADAPT
#SBATCH --qos=CHANGE_ME                # ADAPT (delete if unused)
#SBATCH --gres=gpu:1                   # one GPU is plenty for <=1.5B models
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00                # ADAPT to the cluster's limit
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ADAPT: activate the environment created per README.md.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sc4001-g

# Models are downloaded beforehand on the login node (compute nodes may have no
# internet); point HF at that shared cache and forbid downloads here.
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=1

nvidia-smi
bash scripts/eval/run_baselines.sh gsm8k_val
