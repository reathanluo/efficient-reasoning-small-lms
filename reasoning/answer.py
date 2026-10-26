"""Answer extraction and comparison for GSM8K-style numeric answers.

Every condition in the project (baselines, SFT students, self-consistency) is
scored by this one module, so a parsing bug can never favour one condition.
Each extraction also reports *how* the answer was found, which lets us measure
format compliance separately from correctness.
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional, Tuple

# A number: optional sign, digits with optional thousands separators,
# optional decimal part, optional "/denominator". Also accepts ".5".
_NUM = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?|-?\.\d+"
NUM_RE = re.compile(_NUM)

_HASH_RE = re.compile(r"####\s*\$?\s*(" + _NUM + ")")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_FRAC_RE = re.compile(r"\\[dt]?frac\{(-?\d+)\}\{(\d+)\}")
_ANSWER_IS_RE = re.compile(
    r"answer\s*(?:is|:)\s*[:=]?\s*\$?\s*(" + _NUM + ")", re.IGNORECASE
)

# Text after which a base model has started inventing the next few-shot item.
_CONTINUATION_MARKERS = ("\n\nQuestion:", "\nQuestion:")

# Extraction methods, from most to least format-compliant.
METHODS = ("hash", "boxed", "answer_is", "last_number", "none")


def to_fraction(s: str) -> Optional[Fraction]:
    """Parse a number string such as '1,234.50', '$18', '3/4' or '-2' exactly."""
    s = s.strip().replace(",", "").replace("$", "").rstrip(".").rstrip("%")
    if not s:
        return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def canonical(x: Fraction) -> str:
    """Canonical string used as the vote key in self-consistency."""
    if x.denominator == 1:
        return str(x.numerator)
    return f"{float(x):.6f}".rstrip("0").rstrip(".")


def normalize(s: str) -> Optional[str]:
    f = to_fraction(s)
    return None if f is None else canonical(f)


def extract_gold(gsm8k_answer_field: str) -> str:
    """GSM8K gold answers end with '#### <number>'."""
    tail = gsm8k_answer_field.split("####")[-1]
    value = normalize(tail)
    if value is None:
        raise ValueError(f"Unparseable gold answer: {gsm8k_answer_field!r}")
    return value


def extract_pred(text: str) -> Tuple[Optional[str], str]:
    """Return (canonical answer or None, extraction method).

    Priority: the first '#### N' (the format every prompt asks for), then the
    last \\boxed{N}, then the last 'the answer is N', then the last number in
    the text as a fallback.
    """
    for marker in _CONTINUATION_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0]

    m = _HASH_RE.search(text)
    if m and (v := normalize(m.group(1))) is not None:
        return v, "hash"

    boxed = _BOXED_RE.findall(text)
    if boxed:
        inner = _FRAC_RE.sub(lambda f: f"{f.group(1)}/{f.group(2)}", boxed[-1])
        nums = NUM_RE.findall(inner)
        if nums and (v := normalize(nums[-1])) is not None:
            return v, "boxed"

    ans = _ANSWER_IS_RE.findall(text)
    if ans and (v := normalize(ans[-1])) is not None:
        return v, "answer_is"

    nums = NUM_RE.findall(text)
    if nums and (v := normalize(nums[-1])) is not None:
        return v, "last_number"

    return None, "none"


def is_correct(pred: Optional[str], gold: str, rel_tol: float = 1e-6) -> bool:
    """Numeric equality with a tiny relative tolerance for rounded decimals."""
    if pred is None:
        return False
    p, g = to_fraction(pred), to_fraction(gold)
    if p is None or g is None:
        return False
    return abs(float(p) - float(g)) <= rel_tol * max(1.0, abs(float(g)))
