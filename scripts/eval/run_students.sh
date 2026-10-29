#!/usr/bin/env bash
# Evaluate the trained LoRA students (Person B's adapters) on one split.
#
#   bash scripts/eval/run_students.sh [split] [adapter dirs...]
#   bash scripts/eval/run_students.sh gsm8k_val                       # every adapter in ckpt/
#   bash scripts/eval/run_students.sh gsm8k_val ckpt/concise_05B_s*   # just these
#
# For each adapter ckpt/{style}_{size}_s{seed}/: greedy, and 16 samples at T=0.7
# for self-consistency. Choose with RUNS="greedy", RUNS="sc16" or both (default).
#
# Length cap: the long-style teacher traces average ~1,200 tokens and reach
# ~4,000 (data/sft/stats.json), so students get 4,096 new tokens. With the
# baselines' 1,024 cap, long-style students would be cut off by our setting
# rather than by their own rambling, which would bias H3.
#
# Re-running skips generations that already exist, so a job that hits the 6-hour
# limit can simply be resubmitted. Then scores every run in the output folder.
set -euo pipefail

SPLIT="${1:-gsm8k_val}"
shift || true
ADAPTERS=("$@")
if [[ ${#ADAPTERS[@]} -eq 0 ]]; then
  shopt -s nullglob
  ADAPTERS=(ckpt/*_s*/)
fi
if [[ ${#ADAPTERS[@]} -eq 0 ]]; then
  echo "no adapters found in ckpt/" >&2
  exit 1
fi

OUT="runs/students/${SPLIT}"
MAXTOK="${MAXTOK:-4096}"
MAXLEN="${MAXLEN:-8192}"   # prompt (~100 tokens) + 4,096 new tokens must fit
DTYPE="${DTYPE:-auto}"     # TC1 V100: DTYPE=float16
RUNS="${RUNS:-greedy sc16}"
mkdir -p "$OUT"

base_model() {  # the base model an adapter was trained on, from its PEFT config
  python -c 'import json, sys; print(json.load(open(sys.argv[1]))["base_model_name_or_path"])' \
    "$1/adapter_config.json"
}

gen() {  # gen <adapter dir> <run name> <n> <temperature>
  local name out
  name="$(basename "$1")"
  out="$OUT/${name}_$2.jsonl"
  if [[ -f "$out" ]]; then echo "skip $out"; return; fi
  # BACKEND=hf LIMIT=2 MAXTOK=64 runs a tiny local check without a GPU.
  python -m reasoning.eval.generate --model "$(base_model "$1")" --lora "$1" --prompt chat \
    --split "$SPLIT" --n "$3" --temperature "$4" \
    --max-new-tokens "$MAXTOK" --max-model-len "$MAXLEN" --dtype "$DTYPE" \
    --backend "${BACKEND:-vllm}" ${LIMIT:+--limit "$LIMIT"} --out "$out"
}

for a in "${ADAPTERS[@]}"; do
  a="${a%/}"
  for r in $RUNS; do
    case "$r" in
      greedy) gen "$a" greedy 1 0.0 ;;
      sc16)   gen "$a" sc16 16 0.7 ;;
      *) echo "unknown run '$r' (use greedy or sc16)" >&2; exit 1 ;;
    esac
  done
done

runs=()
for f in "$OUT"/*.jsonl; do [[ "$f" == *.scored.jsonl ]] || runs+=("$f"); done
python -m reasoning.eval.evaluate "${runs[@]}"
