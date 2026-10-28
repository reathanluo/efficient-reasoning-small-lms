#!/usr/bin/env bash
# One independent tuning trial per job. Submit from the repository root:
# sbatch scripts/train/slurm_trial.sh 0.0001 8 1
# Arguments: learning rate, rank, epochs. Defaults are provisional.
#SBATCH --job-name=b-sft-trial
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
trial_lr="${1:-0.0001}"
trial_rank="${2:-8}"
trial_epochs="${3:-1}"
nvidia-smi
python -m reasoning.train.sft --learning-rate "$trial_lr" --rank "$trial_rank" \
    --epochs "$trial_epochs" --val-batch-size "${VAL_BATCH_SIZE:-8}" \
    --out "ckpt/human_05B_s0_trial_${SLURM_JOB_ID}"
