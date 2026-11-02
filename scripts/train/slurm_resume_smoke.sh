#!/usr/bin/env bash
# Verify Trainer/PEFT recovery on TC1 before relying on periodic checkpoints.
#SBATCH --job-name=b-resume-check
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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
resume_out="ckpt/resume_check_${SLURM_JOB_ID}"
common_args=(--model Qwen/Qwen2.5-0.5B --precision float32 --learning-rate 0.0001
    --rank 8 --epochs 1 --limit 64 --val-limit 4 --val-batch-size 1 --save-steps 4
    --out "$resume_out")
python -m reasoning.train.sft "${common_args[@]}"
cp "$resume_out/adapter_model.safetensors" "$resume_out/reference.safetensors"
python -m reasoning.train.sft "${common_args[@]}" \
    --resume-from-checkpoint "$resume_out/checkpoint-4"
python - "$resume_out" <<'PY'
import json
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file
root = Path(sys.argv[1])
expected = load_file(str(root / 'reference.safetensors'))
actual = load_file(str(root / 'adapter_model.safetensors'))
assert expected.keys() == actual.keys(), 'Adapter keys changed'
maximum = max((expected[key] - actual[key]).abs().max().item() for key in expected)
assert all(torch.allclose(expected[key], actual[key], atol=1e-6, rtol=1e-5)
           for key in expected), f'Resumed weights differ; max absolute difference={maximum}'
log = json.loads((root / 'train_log.json').read_text())
assert log['status'] == 'complete' and log['resume_events'][-1]['global_step'] == 4
assert log['epochs'][-1]['global_step'] == 8
print(f'RESUME CHECK PASSED: recovered step 4, completed step 8; max weight difference={maximum}')
PY
