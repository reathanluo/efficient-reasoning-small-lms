"""Small statistics helpers: bootstrap CIs and paired tests on per-question scores."""
from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling questions."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0) -> Tuple[float, Tuple[float, float], float]:
    """Mean difference (a - b) over the same questions, its 95% CI, and a two-sided p-value."""
    assert len(a) == len(b), "runs must cover the same questions in the same order"
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    mean = sum(diffs) / n
    p = 2 * min(sum(d <= 0 for d in boots), sum(d >= 0 for d in boots)) / n_boot
    return mean, (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]), min(1.0, p)


def mcnemar_exact(a: List[bool], b: List[bool]) -> Tuple[int, int, float]:
    """Exact McNemar test on paired correctness. Returns (a-only, b-only, p)."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n, k = only_a + only_b, min(only_a, only_b)
    if n == 0:
        return only_a, only_b, 1.0
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return only_a, only_b, min(1.0, p)
