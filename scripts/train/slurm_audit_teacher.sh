#!/usr/bin/env bash
# Run independently of tuning; checks files and tokenizers, loads no model weights.
# Uses the documented partition/QoS; a CPU-only partition can be used if assigned.
#SBATCH --job-name=b-data-audit
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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
python -m reasoning.train.audit_teacher --out "runs/teacher_audit_${SLURM_JOB_ID}.json"
