#!/usr/bin/env bash
# Full 1.5B/long/seed-0 pilot. Not a frozen final configuration.
# Submit: sbatch scripts/train/slurm_teacher_pilot.sh
# Resume (same output): sbatch scripts/train/slurm_teacher_pilot.sh OUT CHECKPOINT
#SBATCH --job-name=b-teacher-pilot
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
pilot_out="${1:-ckpt/long_15B_s0_pilot_${SLURM_JOB_ID}}"
resume_args=()
if [[ -n "${2:-}" ]]; then
    resume_args=(--resume-from-checkpoint "$2")
fi
nvidia-smi
python -m reasoning.train.sft --model Qwen/Qwen2.5-1.5B \
    --train-file data/sft/long.jsonl --precision float32 --max-length 4096 \
    --learning-rate 0.0001 --rank 8 --epochs 1 --seed 0 \
    --batch-size 1 --grad-accum 8 --val-batch-size 1 --save-steps 100 \
    --out "$pilot_out" "${resume_args[@]}"
