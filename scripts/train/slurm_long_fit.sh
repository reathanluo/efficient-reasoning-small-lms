#!/usr/bin/env bash
# Diagnostic only: test both FP32 students on eight long examples, including max length.
#SBATCH --job-name=b-long-fit
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
for probe_size in 1.5B 0.5B; do
    probe_dir="runs/long_fit_${SLURM_JOB_ID}/${probe_size}"
    python -m reasoning.train.prepare_long_probe --model "Qwen/Qwen2.5-${probe_size}" \
        --out "$probe_dir/subset.jsonl"
    python -m reasoning.train.sft --model "Qwen/Qwen2.5-${probe_size}" \
        --train-file "$probe_dir/subset.jsonl" --precision float32 \
        --max-length 4096 --learning-rate 0.0001 --rank 8 --epochs 1 \
        --batch-size 1 --grad-accum 8 --val-limit 4 --val-batch-size 1 \
        --out "ckpt/long_fit_${probe_size}_${SLURM_JOB_ID}"
done
