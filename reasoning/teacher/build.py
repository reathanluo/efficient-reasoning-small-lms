"""Write the four SFT datasets and stats.json (the A -> B handoff).

    python -m reasoning.teacher.build \
        --standard data/sft/raw/standard.filtered.jsonl \
        --long data/sft/raw/long.filtered.jsonl \
        --concise data/sft/raw/concise.filtered.jsonl \
        --out data/sft

Only problems with a kept rationale in *every* style go into the datasets, so all
four files hold exactly the same ids in the same (gsm8k_train) order, and any
difference between the trained students comes from the style alone. answer_only
uses the same ids with rationale = "".

stats.json records, per style, the count and the mean, median and 90th-percentile
rationale length in Qwen2.5 tokens, plus how many problems survived each stage.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

from ..data import load_split, read_jsonl, write_jsonl

STYLES = ("long", "standard", "concise", "answer_only")
# The students' tokenizer: lengths are reported in the tokens the students write.
STUDENT_TOKENIZER = "Qwen/Qwen2.5-0.5B"


def percentile(values: List[int], q: float) -> float:
    s = sorted(values)
    return float(s[min(len(s) - 1, int(q * len(s)))]) if s else 0.0


def length_stats(lengths: List[int]) -> Dict:
    if not lengths:
        return {"count": 0}
    return {"count": len(lengths),
            "mean_tokens": round(statistics.mean(lengths), 1),
            "median_tokens": statistics.median(lengths),
            "p90_tokens": percentile(lengths, 0.9)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standard", required=True)
    ap.add_argument("--long", required=True)
    ap.add_argument("--concise", required=True)
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out", default="data/sft")
    args = ap.parse_args()

    train = load_split("gsm8k_train", args.splits_dir)
    val_questions = {r["question"] for r in load_split("gsm8k_val", args.splits_dir)}

    by_style = {s: {r["id"]: r for r in read_jsonl(Path(getattr(args, s)))}
                for s in ("standard", "long", "concise")}
    common = set.intersection(*(set(rows) for rows in by_style.values()))
    rows = [r for r in train if r["id"] in common]  # keep the gsm8k_train order
    # Teacher data must come from gsm8k_train only.
    assert not {r["question"] for r in rows} & val_questions, "a validation question leaked into SFT data"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_TOKENIZER)

    def n_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0

    stats: Dict = {
        "funnel": {"gsm8k_train": len(train),
                   **{f"kept_{s}": len(by_style[s]) for s in by_style},
                   "kept_all_styles": len(rows)},
        "tokenizer": STUDENT_TOKENIZER,
        "styles": {},
    }
    out = Path(args.out)
    for style in STYLES:
        data = [{"id": r["id"], "question": r["question"], "gold": r["gold"],
                 "rationale": "" if style == "answer_only" else by_style[style][r["id"]]["rationale"],
                 "style": style}
                for r in rows]
        write_jsonl(out / f"{style}.jsonl", data)
        stats["styles"][style] = length_stats([n_tokens(d["rationale"]) for d in data])
        print(f"{style:12s} {len(data):5d} -> {out}/{style}.jsonl  {stats['styles'][style]}")

    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"funnel: {stats['funnel']}")


if __name__ == "__main__":
    main()
