#!/usr/bin/env bash
# One explicit run per job; never auto-submits the full matrix.
# Submit: sbatch scripts/train/slurm_final.sh STYLE 05B|15B SEED
# Recovery: add --resume-from-checkpoint ckpt/RUN/checkpoint-N after SEED.
#SBATCH --job-name=b-final
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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
python -m reasoning.train.run_final "$@"
