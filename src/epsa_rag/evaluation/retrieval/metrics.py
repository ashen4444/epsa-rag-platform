"""Pure document-retrieval metrics with explicit denominators and rank handling."""

from __future__ import annotations

import math
from collections.abc import Sequence


def retrieval_metrics(
    retrieved: Sequence[str], gold: frozenset[str], cutoffs: tuple[int, ...]
) -> dict[str, float]:
    """Score ranked identities; duplicates consume a rank but never earn credit twice.

    Recall is the fraction of distinct gold documents retrieved, not an any-hit indicator.
    Binary nDCG uses all gold documents to construct the ideal ranking, including missed gold.
    """

    if not gold:
        raise ValueError("gold documents must not be empty")
    if not cutoffs or any(k < 1 for k in cutoffs) or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("cutoffs must be unique positive integers")
    seen: set[str] = set()
    relevance: list[int] = []
    for identity in retrieved:
        relevance.append(int(identity in gold and identity not in seen))
        seen.add(identity)
    scores = {"top1_supporting_document_hit_rate": float(bool(relevance and relevance[0]))}
    for k in cutoffs:
        hits = sum(relevance[:k])
        recall = hits / len(gold)
        first = next((rank for rank, hit in enumerate(relevance[:k], 1) if hit), None)
        dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(relevance[:k], 1))
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
        scores.update(
            {
                f"recall@{k}": recall,
                f"mrr@{k}": 0.0 if first is None else 1 / first,
                f"ndcg@{k}": dcg / ideal,
                f"all_supporting_documents_found@{k}": float(hits == len(gold)),
                f"missing_gold_document_rate@{k}": 1 - recall,
                f"any_gold_missing@{k}": float(hits < len(gold)),
            }
        )
        # This metric has a conditional denominator: questions with exactly two gold documents.
        if len(gold) == 2:
            scores[f"both_supporting_documents_found@{k}"] = float(hits == 2)
    return scores


def mean_metrics(rows: Sequence[dict[str, float]]) -> tuple[dict[str, float], dict[str, int]]:
    """Macro-average each metric and return its explicit eligible-question denominator."""

    names = sorted({name for row in rows for name in row})
    counts = {name: sum(name in row for row in rows) for name in names}
    return (
        {
            name: math.fsum(row[name] for row in rows if name in row) / counts[name]
            for name in names
        },
        counts,
    )


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Linear interpolation at (n - 1) * fraction; empty samples are unavailable."""

    if not 0 <= fraction <= 1:
        raise ValueError("percentile fraction must be between zero and one")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latencies must be finite and nonnegative")
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
