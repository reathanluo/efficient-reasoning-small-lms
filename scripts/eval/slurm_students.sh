#!/usr/bin/env bash
# SLURM job that evaluates trained LoRA students on TC1. Arguments are passed to
# run_students.sh. Submit from the repo root on the TC1 head node:
#
#   sbatch scripts/eval/slurm_students.sh gsm8k_val                        # all of ckpt/
#   sbatch scripts/eval/slurm_students.sh gsm8k_val ckpt/long_05B_s0
#   RUNS=greedy sbatch --export=ALL scripts/eval/slurm_students.sh gsm8k_val
#
# Long-style students write about 1,800 tokens per answer, so 16 samples x 500
# questions takes about 2.5 hours per adapter. If a job hits the 6-hour limit,
# resubmit the same command: finished runs are skipped, and the unfinished run
# starts again, because a run only saves when it ends. On gsm8k_test, 16 long
# samples can't finish in one job, so use RUNS="greedy sc8" for long adapters: see
# scripts/eval/README.md. Same QoS limits as slurm_baselines.sh.
#
#SBATCH --job-name=g-students
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

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=1
export DTYPE=float16

nvidia-smi
bash scripts/eval/run_students.sh "$@"
