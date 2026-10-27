#!/usr/bin/env bash
# SLURM job for one teacher step on TC1. Everything after the script name is
# passed to `python -m`, so the same script runs generate, compress or build:
#
#   sbatch -J t-std-test scripts/teacher/slurm_teacher.sh reasoning.teacher.generate \
#       --style standard --limit 50 --out data/sft/raw/test_standard.jsonl
#
# TC1 QoS "normal" (MyTCinfo): 1 GPU (V100 32GB), <=20 CPUs, <=64G, 6 h per job.
# V100 has no bf16, so --dtype float16 is added for the vLLM steps below.
#
#SBATCH --job-name=teacher
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# The team's conda env from README.md, or a uv .venv in the repo if there is one.
if [[ -f .venv/bin/activate ]]; then
  module load uv
  source .venv/bin/activate
else
  module load anaconda
  source activate sc4001-g
fi

# Models are downloaded beforehand on the head node (compute nodes may have no
# internet); point HF at that cache and forbid downloads here.
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE=1

nvidia-smi
python -m pytest tests -q
case "$1" in
  reasoning.teacher.generate|reasoning.teacher.compress) python -m "$@" --dtype float16 ;;
  *) python -m "$@" ;;
esac
