import math

import pytest
from pydantic import ValidationError

from epsa_rag.evaluation.retrieval.metrics import mean_metrics, percentile, retrieval_metrics
from epsa_rag.evaluation.retrieval.models import EvaluationConfig, RunMetadata


def test_hand_calculated_ranking() -> None:
    scores = retrieval_metrics(["noise", "a", "other", "b"], frozenset({"a", "b"}), (1, 5, 10))
    assert scores["recall@1"] == 0
    assert scores["recall@5"] == 1
    assert scores["mrr@10"] == 0.5
    assert scores["ndcg@10"] == pytest.approx(
        (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    )
    assert scores["top1_supporting_document_hit_rate"] == 0
    assert scores["both_supporting_documents_found@5"] == 1


def test_recall_is_not_hit_rate_and_missing_gold_stays_in_ideal() -> None:
    scores = retrieval_metrics(["a"], frozenset({"a", "b"}), (1, 5, 10))
    assert scores["top1_supporting_document_hit_rate"] == 1
    assert scores["recall@1"] == 0.5
    assert scores["recall@10"] == 0.5
    assert scores["ndcg@10"] == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert scores["missing_gold_document_rate@10"] == 0.5
    assert scores["any_gold_missing@10"] == 1
    assert scores["both_supporting_documents_found@10"] == 0


@pytest.mark.parametrize("ranking", [[], ["x"], ["x", "y", "z"]])
def test_no_hits(ranking: list[str]) -> None:
    scores = retrieval_metrics(ranking, frozenset({"a", "b"}), (1, 10))
    assert scores["recall@10"] == scores["mrr@10"] == scores["ndcg@10"] == 0
    assert scores["missing_gold_document_rate@10"] == 1


def test_duplicates_cannot_inflate_metrics_or_shift_later_ranks() -> None:
    scores = retrieval_metrics(["a", "a", "b"], frozenset({"a", "b"}), (1, 2, 3))
    assert scores["recall@2"] == 0.5
    assert scores["ndcg@2"] <= 1
    assert scores["ndcg@3"] == pytest.approx(1.5 / (1 + 1 / math.log2(3)))


def test_macro_averages_record_the_conditional_both_documents_denominator() -> None:
    rows = [
        retrieval_metrics(["a"], frozenset({"a"}), (10,)),
        retrieval_metrics(["a"], frozenset({"a", "b"}), (10,)),
    ]
    scores, counts = mean_metrics(rows)
    assert scores["recall@10"] == 0.75
    assert counts["recall@10"] == 2
    assert counts["both_supporting_documents_found@10"] == 1
    assert not any(name.startswith("all_supporting_documents_found") for name in scores)
    assert mean_metrics([]) == ({}, {})


@pytest.mark.parametrize(
    ("gold", "cutoffs"),
    [
        (frozenset(), (1,)),
        (frozenset({"a"}), ()),
        (frozenset({"a"}), (0,)),
        (frozenset({"a"}), (1, 1)),
    ],
)
def test_invalid_metric_inputs(gold: frozenset[str], cutoffs: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="must"):
        retrieval_metrics([], gold, cutoffs)


def test_latency_interpolation_and_empty_sample() -> None:
    assert percentile([], 0.95) is None
    assert percentile([20], 0.95) == 20
    assert percentile([40, 10, 30, 20], 0.5) == 25
    assert percentile([40, 10, 30, 20], 0.95) == pytest.approx(38.5)
    assert percentile([10, 20], 0) == 10
    assert percentile([10, 20], 1) == 20


@pytest.mark.parametrize(
    ("values", "fraction"), [([1], 2), ([-1], 0.5), ([float("nan")], 0.5), ([float("inf")], 0.5)]
)
def test_invalid_latency_inputs(values: list[float], fraction: float) -> None:
    with pytest.raises(ValueError, match="must"):
        percentile(values, fraction)


@pytest.mark.parametrize(
    "change",
    [
        {"cutoffs": ()},
        {"cutoffs": (5, 1)},
        {"cutoffs": (1, 1)},
        {"cutoffs": (0, 1)},
        {"cutoffs": (1, 11)},
        {"relevance": "title"},
        {"question_limit": 0},
        {"openai_timeout_seconds": float("inf")},
    ],
)
def test_invalid_config(change: dict) -> None:
    with pytest.raises(ValidationError):
        EvaluationConfig(**change)


@pytest.mark.parametrize("run_id", ["../escape", "x/y", "x\\y", "a:b", "", "has space"])
def test_run_id_cannot_escape_export_root(evaluation_metadata: RunMetadata, run_id: str) -> None:
    data = evaluation_metadata.model_dump()
    data["run_id"] = run_id
    with pytest.raises(ValidationError):
        RunMetadata.model_validate(data)
