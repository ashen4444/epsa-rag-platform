"""Neutral exact chunk merge shared by adaptive evaluation pipelines."""

from __future__ import annotations

from epsa_rag.core.exceptions import ContractError
from epsa_rag.core.models import RankedParagraphChunk
from epsa_rag.pipeline.models import (
    HopMergeDiagnostics,
    HopMergeResult,
    MergedChunkProvenance,
    RetrievalOccurrence,
)
from epsa_rag.retrieval.models import RetrievalResult


def merge_retrieval_hops(
    hop1: RetrievalResult,
    hop2: RetrievalResult | None = None,
) -> HopMergeResult:
    """Append unseen Hop-2 chunks after Hop-1 while retaining every appearance."""

    _validate_retrieval_result(hop1, label="Hop-1")
    if hop2 is not None:
        _validate_retrieval_result(hop2, label="Hop-2")
        _validate_compatible_hops(hop1, hop2)

    ordered_hits: list[RankedParagraphChunk] = []
    occurrences: dict[str, list[RetrievalOccurrence]] = {}
    for hop_number, retrieval in ((1, hop1), (2, hop2)):
        if retrieval is None:
            continue
        for hit in retrieval.results:
            chunk_id = hit.chunk.chunk_id
            occurrence = RetrievalOccurrence(
                chunk_id=chunk_id,
                hop=hop_number,
                query=retrieval.query,
                rank=hit.rank,
                score=hit.score,
                source_scores=hit.source_scores,
                source_ranks=hit.source_ranks,
            )
            if chunk_id not in occurrences:
                occurrences[chunk_id] = []
                ordered_hits.append(hit)
            occurrences[chunk_id].append(occurrence)

    merged_results = tuple(
        hit.model_copy(update={"rank": merged_rank})
        for merged_rank, hit in enumerate(ordered_hits, start=1)
    )
    provenance = tuple(
        MergedChunkProvenance(
            chunk_id=hit.chunk.chunk_id,
            merged_rank=merged_rank,
            first_seen_hop=occurrences[hit.chunk.chunk_id][0].hop,
            occurrences=tuple(occurrences[hit.chunk.chunk_id]),
        )
        for merged_rank, hit in enumerate(ordered_hits, start=1)
    )
    hop1_count = len(hop1.results)
    hop2_count = len(hop2.results) if hop2 is not None else 0
    total_count = hop1_count + hop2_count
    return HopMergeResult(
        retriever_version=hop1.retriever_version,
        results=merged_results,
        provenance=provenance,
        diagnostics=HopMergeDiagnostics(
            hop1_input_chunks=hop1_count,
            hop2_input_chunks=hop2_count,
            total_input_chunks=total_count,
            unique_chunks=len(merged_results),
            duplicate_occurrences_removed=total_count - len(merged_results),
        ),
    )


def _validate_retrieval_result(result: RetrievalResult, *, label: str) -> None:
    expected_ranks = tuple(range(1, len(result.results) + 1))
    actual_ranks = tuple(hit.rank for hit in result.results)
    if actual_ranks != expected_ranks:
        raise ContractError(f"{label} retrieval ranks must be contiguous and one-based")


def _validate_compatible_hops(hop1: RetrievalResult, hop2: RetrievalResult) -> None:
    if hop1.retriever_version != hop2.retriever_version:
        raise ContractError("Hop-1 and Hop-2 must use the same retriever version")
    hop1_question_id = hop1.query.question_id
    hop2_question_id = hop2.query.question_id
    if (
        hop1_question_id is not None
        and hop2_question_id is not None
        and hop1_question_id != hop2_question_id
    ):
        raise ContractError("Hop-1 and Hop-2 must belong to the same original question")
