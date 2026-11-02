"""Collect *.summary.json files into one CSV table for the report.

    python -m reasoning.eval.table results/tables/*.summary.json \
        --out results/tables/baselines_gsm8k_val.csv

One row per run: accuracy at k = 1 and the largest k (with 95% CIs), generated
tokens per answer and per question at that k, prompt tokens, truncation rate,
answer diversity, and how often the answer came in the requested format.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLUMNS = [
    "run", "model", "lora", "prompt", "split", "n_questions", "n_samples", "temperature",
    "acc_k1", "acc_k1_lo", "acc_k1_hi", "k_max", "acc_kmax", "acc_kmax_lo", "acc_kmax_hi",
    "gen_tokens_per_answer", "gen_tokens_kmax", "prompt_tokens", "truncation_rate",
    "mean_unique_answers", "format_hash", "format_boxed", "format_fallback", "wall_seconds",
]


def row(summary: dict) -> dict:
    curve = {p["k"]: p for p in summary["curve"]}
    k1, kmax = curve[1], curve[max(curve)]
    fmt = summary["extraction_methods"]
    return {
        "run": summary["run"], "model": summary["model"], "lora": summary["lora"] or "",
        "prompt": summary["prompt"], "split": summary["split"],
        "n_questions": summary["n_questions"], "n_samples": summary["n_samples"],
        "temperature": summary["temperature"],
        "acc_k1": round(k1["acc"], 4), "acc_k1_lo": round(k1["ci95"][0], 4), "acc_k1_hi": round(k1["ci95"][1], 4),
        "k_max": kmax["k"], "acc_kmax": round(kmax["acc"], 4),
        "acc_kmax_lo": round(kmax["ci95"][0], 4), "acc_kmax_hi": round(kmax["ci95"][1], 4),
        "gen_tokens_per_answer": round(summary["mean_gen_tokens_per_sample"], 1),
        "gen_tokens_kmax": round(kmax["gen_tokens"], 1),
        "prompt_tokens": round(summary["mean_prompt_tokens"], 1),
        "truncation_rate": round(summary["truncation_rate"], 4),
        "mean_unique_answers": round(summary["mean_unique_answers"], 2),
        "format_hash": round(fmt.get("hash", 0.0), 4),
        "format_boxed": round(fmt.get("boxed", 0.0), 4),
        # answered without the requested format: parser fell back or found nothing
        "format_fallback": round(sum(fmt.get(m, 0.0) for m in ("answer_is", "last_number", "none")), 4),
        "wall_seconds": summary["wall_seconds"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summaries", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = sorted((row(json.loads(p.read_text())) for p in args.summaries), key=lambda r: r["run"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
