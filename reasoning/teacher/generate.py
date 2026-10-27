"""Teacher generation: sample standard or long rationales for gsm8k_train.

Raw output only: this script samples and records text, token counts and finish
reasons. `reasoning.teacher.filter` checks the answers afterwards, so a parser or
cleaning fix never needs a GPU rerun.

Examples
--------
Standard rationales (cluster, vLLM), 4 attempts per problem:
    python -m reasoning.teacher.generate --style standard --n 4 --temperature 0.7 \
        --out data/sft/raw/standard.jsonl

Long rationales from the reasoning model:
    python -m reasoning.teacher.generate --style long --n 1 \
        --out data/sft/raw/long.jsonl

Split a long run into jobs that each fit the cluster's 6-hour limit:
    python -m reasoning.teacher.generate --style long --start 0 --limit 2400 \
        --out data/sft/raw/long.part0.jsonl

Retry only the problems with no correct attempt yet:
    python -m reasoning.teacher.generate --style long --n 4 \
        --only-missing data/sft/raw/long.jsonl --out data/sft/raw/long.retry1.jsonl

Local debugging without a GPU (a small stand-in model, 3 problems):
    python -m reasoning.teacher.generate --style standard --backend hf \
        --model Qwen/Qwen2.5-0.5B-Instruct --limit 3 --max-new-tokens 256 \
        --out data/sft/raw/debug_standard.jsonl
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Set

from ..data import load_split, read_jsonl
from ..prompts import chat_messages
from ..eval.generate import generate_hf, generate_vllm
from .filter import pick

# One teacher per style. "concise" is rewritten from "long" by compress.py, and
# "answer_only" needs no teacher, so only these two styles are sampled here.
STYLES: Dict[str, Dict] = {
    "standard": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        # The same neutral instruction the students see, so "standard" is the
        # Instruct model's natural working length.
        "temperature": 0.7,
        "top_p": 0.8,
        "max_new_tokens": 1024,
    },
    "long": {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        # DeepSeek's recommended settings for R1-Distill: no system prompt,
        # T = 0.6, top_p = 0.95, and the answer asked for in \boxed{}.
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 4096,
    },
}

LONG_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def teacher_messages(style: str, question: str) -> List[Dict[str, str]]:
    if style == "long":
        return [{"role": "user", "content": f"{question}\n{LONG_INSTRUCTION}"}]
    return chat_messages(question)


def solved_ids(raw_path: str, style: str) -> Set[str]:
    """Problem ids in a raw file that already have an attempt filter.py would keep."""
    return {r["id"] for r in read_jsonl(Path(raw_path)) if pick(r, style)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", choices=sorted(STYLES), required=True)
    ap.add_argument("--model", default=None, help="override the style's teacher (e.g. a small model for debugging)")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1, help="attempts per problem")
    ap.add_argument("--temperature", type=float, default=None, help="default: the style's setting")
    ap.add_argument("--top-p", type=float, default=None, help="default: the style's setting")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default: the style's setting")
    ap.add_argument("--only-missing", nargs="*", default=[], help="raw files; skip problems already solved in them")
    ap.add_argument("--start", type=int, default=0, help="skip the first N problems (to split a long run into jobs)")
    ap.add_argument("--limit", type=int, default=None, help="at most N problems from --start")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--dtype", default="auto", help="use float16 on GPUs without bfloat16 (e.g. V100)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    args = ap.parse_args()

    cfg = STYLES[args.style]
    args.model = args.model or cfg["model"]
    args.temperature = cfg["temperature"] if args.temperature is None else args.temperature
    args.top_p = cfg["top_p"] if args.top_p is None else args.top_p
    args.max_new_tokens = args.max_new_tokens or cfg["max_new_tokens"]
    # Settings the shared eval generators expect: chat prompts (no few-shot stop
    # strings) and no LoRA adapter.
    args.prompt, args.lora, args.max_lora_rank = "chat", None, 64

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Teacher data comes from gsm8k_train only; val/test never enter training.
    rows = load_split("gsm8k_train", args.splits_dir)
    if args.only_missing:
        done = set().union(*(solved_ids(p, args.style) for p in args.only_missing))
        rows = [r for r in rows if r["id"] not in done]
        print(f"{len(done)} problems already solved; generating for {len(rows)}", flush=True)
    rows = rows[args.start:][: args.limit]

    prompts = [
        tokenizer.apply_chat_template(teacher_messages(args.style, r["question"]), tokenize=False, add_generation_prompt=True)
        for r in rows
    ]
    # R1-Distill's template already starts with the BOS token, and the generators
    # add one when they tokenize the prompt, so drop it here to avoid a double BOS.
    bos = tokenizer.bos_token
    if bos:
        prompts = [p[len(bos):] if p.startswith(bos) else p for p in prompts]

    t0 = time.time()
    if args.backend == "vllm":
        samples = generate_vllm(args, prompts)
    else:
        samples = generate_hf(args, prompts, tokenizer)
    wall = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r, s in zip(rows, samples):
            f.write(json.dumps({"id": r["id"], "question": r["question"], "gold": r["gold"],
                                "style": args.style, "model": args.model, "samples": s}) + "\n")

    meta = {
        **vars(args),
        "n_problems": len(rows),
        "wall_seconds": round(wall, 2),
        "host": platform.node(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(rows)} problems x {args.n} attempts to {out} in {wall:.1f}s")


if __name__ == "__main__":
    main()
