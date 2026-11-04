"""Test H1-H3 from docs/research-question-card.md on scored student runs.

    python -m reasoning.eval.hypotheses results/scored/gsm8k_test

Every comparison is paired over the same questions. "Pooled" averages each
question's score over the three training seeds first; "per seed" pairs seed i
of one style with seed i of the other. CIs are paired-bootstrap 95% intervals.

H1: concise at the largest k that fits the 512-token budget, against standard
    with one greedy answer (which uses fewer tokens than that).
H2: unique answers among the first 8 samples, and the self-consistency gain
    from k = 1 to k = 8, concise against standard.
H3: long against concise, one greedy answer each, plus truncation rates.
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

from ..data import read_jsonl
from ..stats import mcnemar_exact, paired_bootstrap

SIZES = ("05B", "15B")
BUDGET = 512


def load(folder: Path, style: str, size: str, run: str) -> list[list[dict]]:
    """One list of records per seed, sorted by seed and then by question id."""
    paths = sorted(folder.glob(f"{style}_{size}_s*_{run}.scored.jsonl"),
                   key=lambda p: re.search(r"_s(\d)", p.name).group(1))
    seeds = [sorted(read_jsonl(p), key=lambda r: r["id"]) for p in paths]
    assert len(seeds) == 3, f"expected 3 seeds of {style}_{size}_{run}, found {len(seeds)}"
    return seeds


def votes_run(folder: Path, style: str, size: str) -> str:
    return "sc16" if any(folder.glob(f"{style}_{size}_s*_sc16.scored.jsonl")) else "sc8"


def pooled(seeds: list[list[dict]], value) -> list[float]:
    """Each question's value, averaged over seeds."""
    return [statistics.mean(value(r) for r in rows) for rows in zip(*seeds)]


def tokens(seeds: list[list[dict]]) -> float:
    """Mean generated tokens per answer."""
    return statistics.mean(r["mean_gen_tokens"] for rows in seeds for r in rows)


def compare(name: str, a: list[float], b: list[float], scale: float = 100) -> None:
    mean, (lo, hi), p = paired_bootstrap(a, b)
    print(f"  {name}: {scale * statistics.mean(a):.1f} vs {scale * statistics.mean(b):.1f}, "
          f"diff {scale * mean:+.1f} [{scale * lo:+.1f}, {scale * hi:+.1f}], p={p:.3f}")


def h1(folder: Path, size: str) -> None:
    std = load(folder, "standard", size, "greedy")
    con = load(folder, "concise", size, votes_run(folder, "concise", size))
    per_answer = tokens(con)
    k = max(k for k in (1, 2, 4, 8, 16) if k * per_answer <= BUDGET)
    print(f"H1 {size}: concise maj@{k} ({k * per_answer:.0f} tokens) vs standard greedy ({tokens(std):.0f} tokens)")
    acc = lambda key: lambda r: r["maj"][key]  # noqa: E731
    compare("pooled, concise - standard", pooled(con, acc(str(k))), pooled(std, acc("1")))
    for i, (c, s) in enumerate(zip(con, std)):
        compare(f"seed {i}", [r["maj"][str(k)] for r in c], [r["maj"]["1"] for r in s])


def h2(folder: Path, size: str) -> None:
    con = load(folder, "concise", size, votes_run(folder, "concise", size))
    std = load(folder, "standard", size, votes_run(folder, "standard", size))
    unique = lambda r: len({p for p in r["preds"][:8] if p is not None})  # noqa: E731
    gain = lambda r: r["maj"]["8"] - r["maj"]["1"]  # noqa: E731
    print(f"H2 {size}: concise vs standard")
    compare("unique answers among 8, concise - standard", pooled(con, unique), pooled(std, unique), scale=1)
    compare("gain k=1 to 8 (points), concise - standard", pooled(con, gain), pooled(std, gain))


def h3(folder: Path, size: str) -> None:
    long_, con = load(folder, "long", size, "greedy"), load(folder, "concise", size, "greedy")
    trunc = lambda seeds: 100 * statistics.mean(r["truncated"] for rows in seeds for r in rows)  # noqa: E731
    print(f"H3 {size}: long vs concise, greedy ({tokens(long_):.0f} vs {tokens(con):.0f} tokens, "
          f"truncated {trunc(long_):.1f}% vs {trunc(con):.1f}%)")
    correct = lambda r: float(r["correct"][0])  # noqa: E731
    compare("pooled, long - concise", pooled(long_, correct), pooled(con, correct))
    for i, (a, b) in enumerate(zip(long_, con)):
        only_a, only_b, p = mcnemar_exact([r["correct"][0] for r in a], [r["correct"][0] for r in b])
        print(f"  seed {i}: long-only correct {only_a}, concise-only correct {only_b}, McNemar p={p:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    args = ap.parse_args()
    for test in (h1, h2, h3):
        for size in SIZES:
            test(args.folder, size)
        print()


if __name__ == "__main__":
    main()
