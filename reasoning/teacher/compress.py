"""Concise style: rewrite each kept long rationale down to its key steps.

    python -m reasoning.teacher.compress data/sft/raw/long.filtered.jsonl \
        --n 4 --out data/sft/raw/concise.filtered.jsonl

The concise rationale is a compression of the *same* long rationale, so the two
styles share their reasoning and differ in length only. A rewrite is kept only
if it finished on its own, gives its answer as '#### N', that answer is still
`gold`, and it is shorter than the long rationale. Of the passing rewrites the
shortest is kept. Problems with no passing rewrite are dropped (build.py then
drops them from every style).

Local debugging without a GPU (any filtered file works as input):
    python -m reasoning.teacher.compress data/sft/raw/debug_standard.filtered.jsonl \
        --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --limit 2 \
        --out data/sft/raw/debug_concise.filtered.jsonl
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from ..answer import extract_pred, is_correct
from ..data import read_jsonl
from ..eval.generate import generate_hf, generate_vllm
from .filter import clean_standard

REWRITER = "Qwen/Qwen2.5-7B-Instruct"

REWRITE_INSTRUCTION = (
    "Below is a math problem and a long but correct solution. Rewrite the solution "
    "in as few words as possible, but keep every calculation needed to reach the "
    "answer, written as short equations. No explanations, no repeated checks, no LaTeX.\n"
    "Example of the style: '48/2=24. 48+24=72.'\n"
    "Write the equations first, then the final answer on the last line in the form '#### <number>'."
)


def rewrite_messages(question: str, long_rationale: str) -> List[Dict[str, str]]:
    content = f"{REWRITE_INSTRUCTION}\n\nProblem:\n{question}\n\nLong solution:\n{long_rationale}"
    return [{"role": "user", "content": content}]


def pick(row: Dict, samples: List[Dict], n_tokens, reasons: Counter) -> Optional[Dict]:
    """The shortest rewrite that passes every check, cleaned, or None."""
    long_len = n_tokens(row["rationale"])
    best = None
    for attempt, s in enumerate(samples):
        pred, method = extract_pred(s["text"])
        rationale = clean_standard(s["text"])
        if s["finish_reason"] != "stop":
            reasons["truncated"] += 1
        elif method != "hash":
            reasons["no_answer_format"] += 1
        elif not is_correct(pred, row["gold"]):
            reasons["answer_changed"] += 1
        elif not rationale:
            reasons["no_working"] += 1  # answer only: that is the answer_only style, not concise
        elif n_tokens(rationale) >= long_len:
            reasons["not_shorter"] += 1
        else:
            reasons["passed"] += 1
            if best is None or n_tokens(rationale) < n_tokens(best["rationale"]):
                best = {**row, "rationale": rationale, "style": "concise",
                        "model": REWRITER, "attempt": attempt, "teacher_tokens": s["gen_tokens"]}
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("long", help="filtered long rationales from reasoning.teacher.filter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=REWRITER, help="override the rewriter (e.g. a small model for debugging)")
    ap.add_argument("--n", type=int, default=4, help="rewrites per problem")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="first N problems only (debugging)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--dtype", default="auto", help="use float16 on GPUs without bfloat16 (e.g. V100)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    args = ap.parse_args()
    # Settings the shared eval generators expect: chat prompts, no LoRA adapter.
    args.prompt, args.lora, args.max_lora_rank = "chat", None, 64

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def n_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    rows = read_jsonl(Path(args.long))[: args.limit]
    prompts = [
        tokenizer.apply_chat_template(rewrite_messages(r["question"], r["rationale"]), tokenize=False, add_generation_prompt=True)
        for r in rows
    ]

    t0 = time.time()
    if args.backend == "vllm":
        samples = generate_vllm(args, prompts)
    else:
        samples = generate_hf(args, prompts, tokenizer)
    wall = time.time() - t0

    reasons = Counter()
    kept = [c for r, s in zip(rows, samples) if (c := pick(r, s, n_tokens, reasons))]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for c in kept:
            f.write(json.dumps(c) + "\n")

    meta = {**vars(args), "n_problems": len(rows), "n_kept": len(kept), "attempts": dict(reasons),
            "wall_seconds": round(wall, 2), "host": platform.node(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"concise: kept {len(kept)} / {len(rows)} problems ({len(kept) / max(1, len(rows)):.1%})")
    print(f"attempts: {dict(reasons)}")


if __name__ == "__main__":
    main()
