"""Tests for answer extraction. Run: python -m pytest tests"""
import random

import pytest

from reasoning.answer import extract_gold, extract_pred, is_correct, normalize
from reasoning.data import clean_solution
from reasoning.eval.evaluate import maj_at_k, majority
from reasoning.stats import mcnemar_exact, paired_bootstrap


@pytest.mark.parametrize("raw,expected", [
    ("72", "72"), ("1,234", "1234"), ("$18", "18"), ("5.00", "5"), ("-3", "-3"),
    ("0.5", "0.5"), ("3/4", "0.75"), ("18.", "18"), ("25%", "25"), (".5", "0.5"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_gold_and_clean_solution():
    field = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nTotal 48+24 = <<48+24=72>>72.\n#### 72"
    assert extract_gold(field) == "72"
    assert clean_solution(field) == "Natalia sold 48/2 = 24 clips in May.\nTotal 48+24 = 72."


@pytest.mark.parametrize("text,value,method", [
    ("She has 3 + 4 = 7 apples.\n#### 7", "7", "hash"),
    ("#### $1,200 dollars", "1200", "hash"),
    ("#### 5\nQuestion: next one has 99", "5", "hash"),
    ("so the total is \\boxed{42}.", "42", "boxed"),
    ("\\boxed{\\frac{3}{4}}", "0.75", "boxed"),
    ("Therefore, the answer is 18.", "18", "answer_is"),
    ("The answer: $2,500.", "2500", "answer_is"),
    ("We get 10 then 12 so 22 total", "22", "last_number"),
    ("I cannot solve this.", None, "none"),
    # a base model continuing into a hallucinated next few-shot item
    ("2 + 2 = 4\n#### 4\n\nQuestion: Tom has 9 cats\nAnswer: #### 9", "4", "hash"),
    ("x is 4\n\nQuestion: Tom has 9 cats", "4", "last_number"),
])
def test_extract_pred(text, value, method):
    assert extract_pred(text) == (value, method)


def test_is_correct():
    assert is_correct("72", "72")
    assert is_correct("72.0000001", "72")
    assert not is_correct("71", "72")
    assert not is_correct(None, "72")
    assert is_correct("0.333333", "0.333333")


@pytest.mark.parametrize("number", [
    "1" * 400,                 # integer too large for a float
    "1" * 400 + ".5",          # non-integer too large for a float
    "1" * 400 + "/3",          # fraction too large for a float
    "3/" + "1" * 400,          # tiny fraction
    "1" * 5000,                # past Python's integer parsing limit
])
def test_runaway_numbers_are_wrong_not_a_crash(number):
    pred, _ = extract_pred(f"so the total is {number}")
    assert not is_correct(pred, "42")


def test_majority_and_ties():
    assert majority(["5", "6", "5", None]) == "5"
    assert majority(["6", "5"]) == "6"           # tie -> first seen
    assert majority([None, None]) is None


def test_maj_at_k():
    rng = random.Random(0)
    preds = ["7"] * 3 + ["8"]
    assert maj_at_k(preds, "7", 4, rng) == 1.0
    assert maj_at_k(preds, "7", 1, rng) == 0.75
    # pairs from [7,7,7,8]: 3 of 6 are (7,7); the 3 (7,8) ties go to whichever came first -> 7
    assert maj_at_k(preds, "7", 2, rng) == 1.0
    assert maj_at_k(["8", "7", "7", "7"], "7", 2, rng) == 0.5
    assert maj_at_k(["8"] * 4, "7", 2, rng) == 0.0


def test_stats():
    a = [1.0] * 60 + [0.0] * 40
    b = [1.0] * 40 + [0.0] * 60
    mean, (lo, hi), p = paired_bootstrap(a, b)
    assert mean == pytest.approx(0.2) and lo > 0 and p < 0.05
    assert mcnemar_exact([True, True, False], [True, False, False]) == (1, 0, 1.0)
