"""HotPotQA-compatible answer metrics and deterministic aggregate helpers."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable


def normalize_answer(value: str) -> str:
    """Apply the standard lower/punctuation/article/whitespace normalization."""

    lowered = value.lower()
    no_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def answer_exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def answer_token_f1(prediction: str, gold: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def mean(values: Iterable[float]) -> float:
    items = tuple(values)
    return sum(items) / len(items) if items else 0.0


def percentile(values: Iterable[float], percentile_value: float) -> float:
    """Linear n-minus-one percentile, matching the retrieval evaluator."""

    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
