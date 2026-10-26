"""Dataset loading and the fixed train/val/test splits.

Run once to materialise the splits (everyone on the team uses the same files):

    python -m reasoning.data --out data/splits

Split policy (keep test data out of model development):
  * gsm8k_train : GSM8K train minus the validation set; used for few-shot
                  exemplars, teacher generation and SFT.
  * gsm8k_val   : 500 GSM8K train problems held out; used for all tuning and
                  debugging.
  * gsm8k_test  : the official 1,319 test problems; used only for final runs.
  * svamp       : all 1,000 SVAMP problems; out-of-distribution, eval only.
  * gsm_symbolic: 5,000 templated variants of 100 GSM8K test problems (Apple);
                  a robustness/contamination check, eval only.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List

from .answer import extract_gold, normalize

VAL_SIZE = 500
SPLIT_SEED = 0
N_EXEMPLARS = 8

_CALC_RE = re.compile(r"<<[^>]*>>")


def clean_solution(gsm8k_answer_field: str) -> str:
    """GSM8K rationale without the '<<48/2=24>>' calculator annotations and the '####' line."""
    body = gsm8k_answer_field.split("####")[0]
    return _CALC_RE.sub("", body).strip()


def _gsm8k_records(rows, prefix: str) -> List[Dict]:
    return [
        {
            "id": f"{prefix}-{i}",
            "question": r["question"].strip(),
            "gold": extract_gold(r["answer"]),
            "solution": clean_solution(r["answer"]),
        }
        for i, r in enumerate(rows)
    ]


def build_splits() -> Dict[str, List[Dict]]:
    from datasets import load_dataset  # imported lazily so tests run without it

    gsm = load_dataset("openai/gsm8k", "main")
    train_all = _gsm8k_records(gsm["train"], "gsm8k-train")
    test = _gsm8k_records(gsm["test"], "gsm8k-test")

    order = list(range(len(train_all)))
    random.Random(SPLIT_SEED).shuffle(order)
    val = [train_all[i] for i in sorted(order[:VAL_SIZE])]
    train = [train_all[i] for i in sorted(order[VAL_SIZE:])]

    svamp_ds = load_dataset("ChilleD/SVAMP")
    svamp = [
        {
            "id": f"svamp-{r['ID']}",
            "question": r["question_concat"].strip(),
            "gold": normalize(str(r["Answer"])),
        }
        for split in ("train", "test")  # never trained on, so all 1,000 are eval data
        for r in svamp_ds[split]
    ]

    sym = load_dataset("apple/GSM-Symbolic", "main")["test"]
    gsm_symbolic = [
        {
            "id": f"gsmsym-{r['original_id']}-{r['instance']}",
            "question": r["question"].strip(),
            "gold": extract_gold(r["answer"]),
            "original_id": f"gsm8k-test-{r['original_id']}",
        }
        for r in sym
    ]

    # Few-shot exemplars come from the training split only.
    exemplars = random.Random(SPLIT_SEED + 1).sample(train, N_EXEMPLARS)

    return {
        "gsm8k_train": train,
        "gsm8k_val": val,
        "gsm8k_test": test,
        "svamp": svamp,
        "gsm_symbolic": gsm_symbolic,
        "exemplars": exemplars,
    }


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(name: str, splits_dir: str = "data/splits") -> List[Dict]:
    return read_jsonl(Path(splits_dir) / f"{name}.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/splits")
    args = ap.parse_args()

    splits = build_splits()
    for name, rows in splits.items():
        write_jsonl(Path(args.out) / f"{name}.jsonl", rows)
        print(f"{name:13s} {len(rows):5d} -> {args.out}/{name}.jsonl")

    # Guard against leakage between splits.
    val_q = {r["question"] for r in splits["gsm8k_val"]}
    assert not val_q & {r["question"] for r in splits["gsm8k_train"]}
    assert not val_q & {r["question"] for r in splits["exemplars"]}


if __name__ == "__main__":
    main()
