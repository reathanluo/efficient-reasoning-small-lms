#!/usr/bin/env bash
# Integration check: save without HF validation, exit training, run vLLM, attach scores.
#SBATCH --job-name=b-final-smoke
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
smoke_out="ckpt/deferred_smoke_${SLURM_JOB_ID}"
smoke_raw="runs/deferred_smoke_${SLURM_JOB_ID}.jsonl"
python -m reasoning.train.sft --precision float32 --learning-rate 0.0001 --rank 8 \
    --epochs 1 --limit 64 --max-length 4096 --save-steps 4 --defer-validation --out "$smoke_out"
python - "$smoke_out" <<'PY'
import json, sys
from pathlib import Path
log = json.loads((Path(sys.argv[1]) / 'train_log.json').read_text())
assert log['status'] == 'trained_pending_validation'
assert log['epochs'][0]['gsm8k_val_accuracy'] is None
assert log['n_train'] == 64
print('Deferred save passed; starting external vLLM validation')
PY
python -m reasoning.eval.generate --model Qwen/Qwen2.5-0.5B --lora "$smoke_out" \
    --prompt chat --split gsm8k_val --dtype float16 --max-model-len 8192 \
    --max-new-tokens 512 --out "$smoke_raw"
python -m reasoning.eval.evaluate "$smoke_raw"
python - "$smoke_out" "$smoke_raw" <<'PY'
import json, sys
from pathlib import Path
from reasoning.train.sft import read_rows
from reasoning.train.run_final import attach_validation, atomic_json
out, raw = map(Path, sys.argv[1:])
protocol = json.loads(Path('configs/sft-final.json').read_text())
protocol['protocol_id'] = 'smoke-only-not-final'
log = json.loads((out / 'train_log.json').read_text())
log = attach_validation(log, read_rows(raw), read_rows(raw.with_suffix('.scored.jsonl')),
                        json.loads(raw.with_suffix('.summary.json').read_text()), protocol, raw)
atomic_json(out / 'train_log.json', log)
assert log['status'] == 'complete' and log['epochs'][0]['validation_backend'] == 'vllm'
print('FINAL PIPELINE SMOKE PASSED (diagnostic only, not one of the 24 adapters)')
PY
