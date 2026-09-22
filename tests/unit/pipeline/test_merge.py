"""Tests for neutral exact Hop-1/Hop-2 chunk merging."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from epsa_rag.core.exceptions import ContractError
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, RetrievalQuery, Sentence
from epsa_rag.pipeline import (
    HopMergeDiagnostics,
    HopMergeResult,
    HybridRetrieverProtocol,
    MergedChunkProvenance,
    RetrievalOccurrence,
    merge_retrieval_hops,
)
from epsa_rag.retrieval.models import RetrievalResult


def _chunk(chunk_id: str) -> ParagraphChunk:
    text = f"{chunk_id} text."
    return ParagraphChunk(
        chunk_id=chunk_id,
        title=f"Title {chunk_id}",
        paragraph_text=text,
        sentences=(Sentence(index=0, text=text),),
    )


def _hit(chunk_id: str, rank: int, score: float) -> RankedParagraphChunk:
    return RankedParagraphChunk(
        chunk=_chunk(chunk_id),
        rank=rank,
        score=score,
        source_scores={"dense": score},
        source_ranks={"dense": rank},
    )


def _result(
    query: str,
    hits: tuple[RankedParagraphChunk, ...],
    *,
    version: str = "hybrid-retriever-v2",
    question_id: str = "q1",
) -> RetrievalResult:
    return RetrievalResult(
        query=RetrievalQuery(text=query, question_id=question_id),
        retriever_version=version,
        results=hits,
    )


def _occurrence(
    chunk_id: str,
    hop: Literal[1, 2],
    *,
    rank: int = 1,
) -> RetrievalOccurrence:
    return RetrievalOccurrence(
        chunk_id=chunk_id,
        hop=hop,
        query=RetrievalQuery(text=f"query {hop}", question_id="q1"),
        rank=rank,
        score=0.5,
        source_scores={"dense": 0.5},
        source_ranks={"dense": rank},
    )


def test_merge_preserves_hop1_order_and_appends_only_unseen_hop2_chunks() -> None:
    hop1 = _result("original question", (_hit("a", 1, 0.9), _hit("b", 2, 0.8)))
    hop2 = _result("bridge relation", (_hit("b", 1, 0.95), _hit("c", 2, 0.7)))

    merged = merge_retrieval_hops(hop1, hop2)

    assert [hit.chunk.chunk_id for hit in merged.results] == ["a", "b", "c"]
    assert [hit.rank for hit in merged.results] == [1, 2, 3]
    assert merged.results[1].score == 0.8
    assert [item.chunk_id for item in merged.provenance] == ["a", "b", "c"]
    assert [occurrence.hop for occurrence in merged.provenance[1].occurrences] == [1, 2]
    assert [occurrence.rank for occurrence in merged.provenance[1].occurrences] == [2, 1]
    assert merged.diagnostics == HopMergeDiagnostics(
        hop1_input_chunks=2,
        hop2_input_chunks=2,
        total_input_chunks=4,
        unique_chunks=3,
        duplicate_occurrences_removed=1,
    )


def test_merge_supports_hop1_only_and_exact_duplicates_within_one_hop() -> None:
    hop1 = _result("original question", (_hit("a", 1, 0.9), _hit("a", 2, 0.8)))

    merged = merge_retrieval_hops(hop1)

    assert len(merged.results) == 1
    assert len(merged.provenance[0].occurrences) == 2
    assert merged.diagnostics.duplicate_occurrences_removed == 1


@pytest.mark.parametrize(
    ("hop2", "message"),
    [
        (
            _result("next", (_hit("b", 1, 0.8),), version="hybrid-retriever-v1"),
            "same retriever version",
        ),
        (
            _result("next", (_hit("b", 1, 0.8),), question_id="q2"),
            "same original question",
        ),
    ],
)
def test_merge_rejects_incompatible_hops(
    hop2: RetrievalResult,
    message: str,
) -> None:
    hop1 = _result("original question", (_hit("a", 1, 0.9),))

    with pytest.raises(ContractError, match=message):
        merge_retrieval_hops(hop1, hop2)


def test_merge_rejects_noncanonical_input_ranks() -> None:
    hop1 = _result("original question", (_hit("a", 2, 0.9),))

    with pytest.raises(ContractError, match="contiguous"):
        merge_retrieval_hops(hop1)


def test_merge_models_reject_inconsistent_diagnostics() -> None:
    with pytest.raises(ValidationError, match="account for every input chunk"):
        HopMergeDiagnostics(
            hop1_input_chunks=2,
            hop2_input_chunks=1,
            total_input_chunks=3,
            unique_chunks=1,
            duplicate_occurrences_removed=1,
        )

    with pytest.raises(ValidationError, match="equal the two hop input counts"):
        HopMergeDiagnostics(
            hop1_input_chunks=1,
            hop2_input_chunks=1,
            total_input_chunks=1,
            unique_chunks=1,
            duplicate_occurrences_removed=0,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("score", float("inf"), "score must be finite"),
        ("source_scores", {"dense": float("nan")}, "source scores must be finite"),
    ],
)
def test_retrieval_occurrence_rejects_nonfinite_scores(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _occurrence("a", 1).model_dump()
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        RetrievalOccurrence.model_validate(payload)


def test_retrieval_occurrence_detaches_score_and_rank_maps() -> None:
    scores = {"dense": 0.5}
    ranks = {"dense": 1}
    occurrence = RetrievalOccurrence(
        chunk_id="a",
        hop=1,
        query=RetrievalQuery(text="question", question_id="q1"),
        rank=1,
        score=0.5,
        source_scores=scores,
        source_ranks=ranks,
    )

    scores["dense"] = 0.1
    ranks["dense"] = 2
    assert occurrence.source_scores == {"dense": 0.5}
    assert occurrence.source_ranks == {"dense": 1}


@pytest.mark.parametrize(
    ("chunk_id", "first_seen_hop", "occurrences", "message"),
    [
        ("a", 1, (_occurrence("b", 1),), "same chunk"),
        ("a", 2, (_occurrence("a", 1),), "first retained occurrence"),
        (
            "a",
            2,
            (_occurrence("a", 2), _occurrence("a", 1)),
            "preserve hop order",
        ),
    ],
)
def test_merged_chunk_provenance_rejects_inconsistent_state(
    chunk_id: str,
    first_seen_hop: Literal[1, 2],
    occurrences: tuple[RetrievalOccurrence, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        MergedChunkProvenance(
            chunk_id=chunk_id,
            merged_rank=1,
            first_seen_hop=first_seen_hop,
            occurrences=occurrences,
        )


def test_hop_merge_result_rejects_misaligned_rank_and_diagnostics() -> None:
    valid = merge_retrieval_hops(_result("question", (_hit("a", 1, 0.5),)))
    payload = valid.model_dump()
    payload["results"][0]["rank"] = 2
    with pytest.raises(ValidationError, match="result ranks"):
        HopMergeResult.model_validate(payload)

    payload = valid.model_dump()
    payload["diagnostics"] = HopMergeDiagnostics(
        hop1_input_chunks=0,
        hop2_input_chunks=0,
        total_input_chunks=0,
        unique_chunks=0,
        duplicate_occurrences_removed=0,
    ).model_dump()
    with pytest.raises(ValidationError, match="merged result count"):
        HopMergeResult.model_validate(payload)


def test_hybrid_retriever_protocol_is_runtime_checkable() -> None:
    class StubRetriever:
        def retrieve(
            self,
            query: RetrievalQuery,
            *,
            top_k: int | None = None,
        ) -> RetrievalResult:
            del top_k
            return _result(query.text, (), question_id=query.question_id or "q1")

    assert isinstance(StubRetriever(), HybridRetrieverProtocol)
