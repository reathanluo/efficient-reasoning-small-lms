"""Rejection sampling: keep one correct, cleaned rationale per problem.

    python -m reasoning.teacher.filter --style standard \
        data/sft/raw/standard.jsonl data/sft/raw/standard.retry1.jsonl \
        --out data/sft/raw/standard.filtered.jsonl

An attempt is kept only if
  * it finished on its own (finish_reason "stop", not cut off at the length limit),
  * the shared parser finds its answer in the requested format ("####" or
    \\boxed{}), not by falling back to the last number in the text, and
  * that answer equals `gold` (answer.is_correct).
For each problem the first passing attempt is kept, taking the raw files in the
order given. The cleaned rationale is the working only: format_target appends
the "#### answer" line at training time.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from ..answer import extract_pred, is_correct
from ..data import read_jsonl

# Extraction methods that show the teacher actually followed the answer format.
FORMAT_METHODS = ("hash", "boxed")

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def clean_standard(text: str) -> str:
    """Drop the '#### N' line and anything after it; unwrap any \\boxed{N}."""
    body = text.split("####")[0]
    return _BOXED_RE.sub(r"\1", body).strip()


def clean_long(text: str) -> Optional[str]:
    """Keep the reasoning inside <think> ... </think>.

    R1-Distill's chat template opens the <think> block in the prompt, so the
    output usually starts mid-thought and contains only the closing tag. The
    short summary after </think> repeats the answer and is dropped. Returns None
    if the thinking never closed, which means it was cut off.
    """
    if "</think>" not in text:
        return None
    thought = text.split("</think>", 1)[0]
    thought = thought.split("<think>", 1)[-1]
    return _BOXED_RE.sub(r"\1", thought).strip() or None


CLEANERS = {"standard": clean_standard, "long": clean_long}


def pick(row: Dict, style: str) -> Optional[Dict]:
    """The first attempt that passes every check, cleaned, or None."""
    for attempt, s in enumerate(row["samples"]):
        if s["finish_reason"] != "stop":
            continue
        pred, method = extract_pred(s["text"])
        if method not in FORMAT_METHODS or not is_correct(pred, row["gold"]):
            continue
        rationale = CLEANERS[style](s["text"])
        if not rationale:
            continue
        return {"id": row["id"], "question": row["question"], "gold": row["gold"],
                "rationale": rationale, "style": style, "model": row["model"],
                "attempt": attempt, "teacher_tokens": s["gen_tokens"]}
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", nargs="+", help="raw files from reasoning.teacher.generate, first run first")
    ap.add_argument("--style", choices=sorted(CLEANERS), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kept: Dict[str, Dict] = {}
    seen: List[str] = []
    reasons = Counter()
    for path in args.raw:
        for row in read_jsonl(Path(path)):
            if row["id"] not in seen:
                seen.append(row["id"])
            if row["id"] in kept:
                continue
            chosen = pick(row, args.style)
            if chosen:
                kept[row["id"]] = chosen
            for s in row["samples"]:
                pred, method = extract_pred(s["text"])
                if s["finish_reason"] != "stop":
                    reasons["truncated"] += 1
                elif method not in FORMAT_METHODS:
                    reasons["no_answer_format"] += 1
                elif not is_correct(pred, row["gold"]):
                    reasons["wrong_answer"] += 1
                else:
                    reasons["passed"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for pid in seen:  # keep the gsm8k_train order
            if pid in kept:
                f.write(json.dumps(kept[pid]) + "\n")

    print(f"{args.style}: kept {len(kept)} / {len(seen)} problems ({len(kept) / max(1, len(seen)):.1%})")
    print(f"attempts: {dict(reasons)}")


if __name__ == "__main__":
    main()
