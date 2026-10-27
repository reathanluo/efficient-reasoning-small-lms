"""Score a generation file: accuracy, self-consistency curve, cost and diagnostics.

    python -m reasoning.eval.evaluate runs/base05_fewshot_greedy.jsonl [more.jsonl ...]

For each run it writes
  <run>.scored.jsonl : per-question parsed answers and scores (for paired tests)
  <run>.summary.json : aggregate metrics (one row of the results table)

Cost accounting (state this in the report):
  * gen_tokens@k  = expected generated tokens per question when voting over k samples.
  * flops@k       ~ 2 * N_params * (prompt_tokens + gen_tokens@k). The prompt is
                    counted once because the n samples of one vLLM request share
                    its prefill. Attention FLOPs are ignored (small at these lengths).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from ..answer import extract_pred, is_correct
from ..data import read_jsonl
from ..stats import bootstrap_ci

MAX_EXACT_SUBSETS = 256  # enumerate all size-k subsets up to this many
N_SUBSETS = 128          # otherwise, random size-k subsets per question


def majority(preds: List[Optional[str]]) -> Optional[str]:
    """Most common parsed answer; ties go to the answer that appeared first."""
    valid = [p for p in preds if p is not None]
    if not valid:
        return None
    counts = Counter(valid)
    best = max(counts.values())
    return next(p for p in valid if counts[p] == best)


def maj_at_k(preds: List[Optional[str]], gold: str, k: int, rng: random.Random) -> float:
    """Expected correctness of majority voting over k of the n samples.

    Exact when the number of size-k subsets is small, Monte Carlo otherwise.
    """
    n = len(preds)
    if k == 1:
        return sum(is_correct(p, gold) for p in preds) / n
    if math.comb(n, k) <= MAX_EXACT_SUBSETS:
        subsets = list(itertools.combinations(preds, k))
    else:
        subsets = [rng.sample(preds, k) for _ in range(N_SUBSETS)]
    return sum(is_correct(majority(list(s)), gold) for s in subsets) / len(subsets)


def score_run(path: Path) -> Dict:
    rows = read_jsonl(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    n_params = meta.get("n_params")
    n = len(rows[0]["samples"])
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= n]
    rng = random.Random(0)

    scored, methods = [], Counter()
    for r in rows:
        parsed = [extract_pred(s["text"]) for s in r["samples"]]
        preds = [p for p, _ in parsed]
        methods.update(m for _, m in parsed)
        gen = [s["gen_tokens"] for s in r["samples"]]
        scored.append({
            "id": r["id"],
            "gold": r["gold"],
            "preds": preds,
            "correct": [is_correct(p, r["gold"]) for p in preds],
            "maj": {str(k): maj_at_k(preds, r["gold"], k, rng) for k in ks},
            "mean_gen_tokens": sum(gen) / n,
            "prompt_tokens": r["prompt_tokens"],
            "truncated": sum(s["finish_reason"] == "length" for s in r["samples"]) / n,
            "n_unique": len({p for p in preds if p is not None}),
        })

    q = len(scored)
    mean_gen = sum(s["mean_gen_tokens"] for s in scored) / q
    mean_prompt = sum(s["prompt_tokens"] for s in scored) / q
    curve = []
    for k in ks:
        vals = [s["maj"][str(k)] for s in scored]
        acc = sum(vals) / q
        lo, hi = bootstrap_ci(vals)
        point = {"k": k, "acc": acc, "ci95": [lo, hi], "gen_tokens": k * mean_gen}
        if n_params:
            point["flops"] = 2 * n_params * (mean_prompt + k * mean_gen)
        curve.append(point)

    total_samples = q * n
    summary = {
        "run": path.stem,
        "model": meta.get("model"),
        "lora": meta.get("lora"),
        "prompt": meta.get("prompt"),
        "split": meta.get("split"),
        "n_questions": q,
        "n_samples": n,
        "temperature": meta.get("temperature"),
        "n_params": n_params,
        "sample_acc": sum(sum(s["correct"]) / n for s in scored) / q,  # = greedy acc when n=1
        "curve": curve,
        "mean_gen_tokens_per_sample": mean_gen,
        "mean_prompt_tokens": mean_prompt,
        "truncation_rate": sum(s["truncated"] for s in scored) / q,
        "extraction_methods": {m: methods[m] / total_samples for m in methods},
        "mean_unique_answers": sum(s["n_unique"] for s in scored) / q,
        "wall_seconds": meta.get("wall_seconds"),
    }

    with path.with_suffix(".scored.jsonl").open("w") as f:
        for s in scored:
            f.write(json.dumps(s) + "\n")
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def print_summary(s: Dict) -> None:
    print(f"\n== {s['run']}  ({s['n_questions']} questions x {s['n_samples']} samples)")
    print(f"   sample acc {s['sample_acc']:.3f} | gen tokens/sample {s['mean_gen_tokens_per_sample']:.1f} "
          f"| prompt tokens {s['mean_prompt_tokens']:.0f} | truncated {s['truncation_rate']:.1%}")
    print(f"   extraction: " + ", ".join(f"{m} {v:.1%}" for m, v in sorted(s["extraction_methods"].items())))
    for p in s["curve"]:
        flops = f" | {p['flops']:.2e} FLOPs" if "flops" in p else ""
        print(f"   maj@{p['k']:<2d} acc {p['acc']:.3f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] "
              f"| {p['gen_tokens']:.0f} gen tokens{flops}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    for path in args.runs:
        print_summary(score_run(path))


if __name__ == "__main__":
    main()
