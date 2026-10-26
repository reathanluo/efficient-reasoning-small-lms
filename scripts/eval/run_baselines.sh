#!/usr/bin/env bash
# Baseline matrix on the GSM8K validation split (week 1 milestone).
#
#   bash scripts/eval/run_baselines.sh [split]      # default split: gsm8k_val
#
# For each size (0.5B, 1.5B):
#   base     + 8-shot   : greedy, and 16 samples at T=0.7 for self-consistency
#   instruct + chat     : greedy, and 16 samples at T=0.7
#   base     + chat     : greedy only (the untrained starting point of every SFT student)
# Then scores all runs. Re-running skips generations that already exist.
set -euo pipefail

SPLIT="${1:-gsm8k_val}"
OUT="runs/baselines/${SPLIT}"
MAXTOK=1024
DTYPE="${DTYPE:-auto}"   # set DTYPE=float16 on V100-class GPUs
mkdir -p "$OUT"

gen() {  # gen <name> <model> <prompt> <n> <temperature>
  local out="$OUT/$1.jsonl"
  if [[ -f "$out" ]]; then echo "skip $out"; return; fi
  python -m reasoning.eval.generate --model "$2" --prompt "$3" --n "$4" --temperature "$5" \
    --split "$SPLIT" --max-new-tokens "$MAXTOK" --dtype "$DTYPE" --out "$out"
}

for SIZE in 0.5B 1.5B; do
  tag="${SIZE/./}"   # 0.5B -> 05B
  gen "base${tag}_fewshot_greedy" "Qwen/Qwen2.5-${SIZE}"          fewshot 1  0.0
  gen "base${tag}_fewshot_sc16"   "Qwen/Qwen2.5-${SIZE}"          fewshot 16 0.7
  gen "inst${tag}_chat_greedy"    "Qwen/Qwen2.5-${SIZE}-Instruct" chat    1  0.0
  gen "inst${tag}_chat_sc16"      "Qwen/Qwen2.5-${SIZE}-Instruct" chat    16 0.7
  gen "base${tag}_chat_greedy"    "Qwen/Qwen2.5-${SIZE}"          chat    1  0.0
done

runs=()
for f in "$OUT"/*.jsonl; do [[ "$f" == *.scored.jsonl ]] || runs+=("$f"); done
python -m reasoning.eval.evaluate "${runs[@]}"
