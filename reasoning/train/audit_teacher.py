"""Check A's SFT handoff and measure full training sequences without training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from ..prompts import chat_prompt, format_target
from .sft import check_data, read_rows

STYLES = ("long", "standard", "concise", "answer_only")


def load_handoff(root):
    datasets = {style: read_rows(Path(root) / f"{style}.jsonl") for style in STYLES}
    reference = datasets["long"]
    train = {row["id"]: row for row in read_rows("data/splits/gsm8k_train.jsonl")}
    val = read_rows("data/splits/gsm8k_val.jsonl")
    for style, rows in datasets.items():
        check_data(rows, val)
        if [row["id"] for row in rows] != [row["id"] for row in reference]:
            raise ValueError(f"{style}: IDs/order differ from long")
        for row in rows:
            original = train[row["id"]]
            if row["question"] != original["question"] or row["gold"] != original["gold"]:
                raise ValueError(f"{style}: question/gold mismatch for {row['id']}")
            if row.get("style") != style or not isinstance(row.get("rationale"), str):
                raise ValueError(f"{style}: invalid style/rationale for {row['id']}")
            if (style == "answer_only") != (not row["rationale"].strip()):
                raise ValueError(f"{style}: unexpected empty/nonempty rationale for {row['id']}")
    return datasets


def length_summary(lengths, limits):
    ordered = sorted(lengths)
    return {"count": len(lengths), "mean": round(sum(lengths) / len(lengths), 2),
            "median": ordered[len(ordered) // 2],
            "p90": ordered[math.ceil(0.9 * len(ordered)) - 1], "max": ordered[-1],
            "over_limit": {str(limit): sum(n > limit for n in lengths) for limit in limits}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/sft")
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"])
    parser.add_argument("--limits", nargs="+", type=int, default=[2048, 4096, 8192, 16384])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if any(limit <= 0 for limit in args.limits):
        parser.error("Limits must be positive")
    datasets = load_handoff(args.data_dir)
    report = {"status": "running", "checks": "matching ordered IDs, train-only IDs, questions/gold, schema",
              "files": {style: {"count": len(rows), "sha256": hashlib.sha256(
                  (Path(args.data_dir) / f"{style}.jsonl").read_bytes()).hexdigest()}
                  for style, rows in datasets.items()}, "models": {}}
    out = Path(args.out)
    if out.exists():
        parser.error("Output already exists; choose a new --out")
    out.parent.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    for model_name in args.models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not tokenizer.chat_template or tokenizer.eos_token_id is None:
            raise ValueError(f"{model_name}: missing chat template or EOS")
        results = {}
        for style, rows in datasets.items():
            lengths = []
            for i, row in enumerate(rows, 1):
                # Match sft.encode_row exactly: prompt + target + one EOS.
                prompt = tokenizer(chat_prompt(tokenizer, row["question"]), add_special_tokens=False)["input_ids"]
                target = tokenizer(format_target(row["rationale"], row["gold"]), add_special_tokens=False)["input_ids"]
                lengths.append(len(prompt) + len(target) + 1)
                if i % 1000 == 0:
                    print(f"{model_name} {style}: {i}/{len(rows)}", flush=True)
            results[style] = length_summary(lengths, args.limits)
            results[style]["longest_example_id"] = rows[max(range(len(rows)), key=lengths.__getitem__)]["id"]
            print(f"{model_name} {style}: {json.dumps(results[style])}", flush=True)
        report["models"][model_name] = results
        out.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "complete"
    report["note"] = "Lengths include prompt, rationale, final answer and EOS. This is not a GPU memory-fit test."
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audit saved to {out}", flush=True)


if __name__ == "__main__":
    main()
