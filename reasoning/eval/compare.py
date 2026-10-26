"""Paired comparison of two scored runs on the same questions.

    python -m reasoning.eval.compare runs/A.scored.jsonl runs/B.scored.jsonl --k 1

Reports the accuracy difference A - B with a paired-bootstrap 95% CI, and the
exact McNemar test when both runs are single-sample (k=1, n=1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..data import read_jsonl
from ..stats import mcnemar_exact, paired_bootstrap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--k", type=int, default=1, help="compare maj@k")
    args = ap.parse_args()

    a = {r["id"]: r for r in read_jsonl(args.a)}
    b = {r["id"]: r for r in read_jsonl(args.b)}
    ids = sorted(a.keys() & b.keys())
    if len(ids) != len(a) or len(ids) != len(b):
        print(f"WARNING: only {len(ids)} shared questions ({len(a)} vs {len(b)})")

    key = str(args.k)
    va = [a[i]["maj"][key] for i in ids]
    vb = [b[i]["maj"][key] for i in ids]
    mean, (lo, hi), p = paired_bootstrap(va, vb)
    print(f"maj@{args.k}: A {sum(va) / len(ids):.3f}  B {sum(vb) / len(ids):.3f}  "
          f"diff {mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  bootstrap p={p:.3f}")

    if all(len(a[i]["correct"]) == 1 and len(b[i]["correct"]) == 1 for i in ids):
        only_a, only_b, p_mc = mcnemar_exact([a[i]["correct"][0] for i in ids], [b[i]["correct"][0] for i in ids])
        print(f"McNemar: A-only correct {only_a}, B-only correct {only_b}, exact p={p_mc:.4f}")


if __name__ == "__main__":
    main()
