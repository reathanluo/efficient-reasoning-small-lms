"""Select eight long-style examples spanning lengths, including the longest.

The subset is for memory/stability testing only. Never use it as a final dataset.
"""
import argparse
import json
from pathlib import Path

from .sft import encode_row, read_rows


def select_indices(lengths):
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    # Start with the longest so it is present even in a shortened diagnostic.
    quantiles = (1, 0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
    return list(dict.fromkeys(order[round(q * (len(order) - 1))] for q in quantiles))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        parser.error("Output exists; choose a new path")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = read_rows("data/sft/long.jsonl")
    lengths = []
    for i, row in enumerate(rows, 1):
        lengths.append(len(encode_row(row, tokenizer, 1_000_000)["input_ids"]))
        if i % 1000 == 0:
            print(f"Measured {i}/{len(rows)} examples", flush=True)
    indices = select_indices(lengths)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(rows[i]) + "\n" for i in indices))
    report = {"diagnostic_only": True, "model": args.model,
              "source": "data/sft/long.jsonl", "global_max_tokens": max(lengths),
              "examples": [{"id": rows[i]["id"], "tokens": lengths[i]} for i in indices]}
    out.with_suffix(".selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
