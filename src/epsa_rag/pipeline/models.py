"""Immutable contracts for deterministic multi-hop retrieval orchestration."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel, RankedParagraphChunk, RetrievalQuery


class RetrievalOccurrence(ContractModel):
    """One appearance of a canonical chunk in a ranked retrieval hop."""

    chunk_id: Identifier
    hop: Literal[1, 2]
    query: RetrievalQuery
    rank: Annotated[int, Field(ge=1)]
    score: float
    source_scores: dict[Identifier, float] = Field(default_factory=dict)
    source_ranks: dict[Identifier, Annotated[int, Field(ge=1)]] = Field(
        default_factory=dict
    )

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("retrieval occurrence score must be finite")
        return value

    @field_validator("source_scores")
    @classmethod
    def validate_source_scores(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(score) for score in value.values()):
            raise ValueError("retrieval occurrence source scores must be finite")
        return value.copy()

    @field_validator("source_ranks")
    @classmethod
    def detach_source_ranks(cls, value: dict[str, int]) -> dict[str, int]:
        return value.copy()


class MergedChunkProvenance(ContractModel):
    """All ranked appearances retained for one exact-deduplicated chunk."""

    chunk_id: Identifier
    merged_rank: Annotated[int, Field(ge=1)]
    first_seen_hop: Literal[1, 2]
    occurrences: Annotated[tuple[RetrievalOccurrence, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_consistent_occurrences(self) -> MergedChunkProvenance:
        if any(item.chunk_id != self.chunk_id for item in self.occurrences):
            raise ValueError("merged provenance occurrences must reference the same chunk")
        if self.first_seen_hop != self.occurrences[0].hop:
            raise ValueError("first_seen_hop must match the first retained occurrence")
        hop_order = tuple(item.hop for item in self.occurrences)
        if hop_order != tuple(sorted(hop_order)):
            raise ValueError("retrieval occurrences must preserve hop order")
        return self


class HopMergeDiagnostics(ContractModel):
    """Inspectable counts for one exact Hop-1/Hop-2 merge."""

    hop1_input_chunks: Annotated[int, Field(ge=0)]
    hop2_input_chunks: Annotated[int, Field(ge=0)]
    total_input_chunks: Annotated[int, Field(ge=0)]
    unique_chunks: Annotated[int, Field(ge=0)]
    duplicate_occurrences_removed: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def require_consistent_counts(self) -> HopMergeDiagnostics:
        if self.total_input_chunks != self.hop1_input_chunks + self.hop2_input_chunks:
            raise ValueError("total input chunks must equal the two hop input counts")
        if self.unique_chunks + self.duplicate_occurrences_removed != self.total_input_chunks:
            raise ValueError("unique and duplicate counts must account for every input chunk")
        return self


class HopMergeResult(ContractModel):
    """Canonical merged chunks plus complete retrieval provenance."""

    retriever_version: Identifier
    results: tuple[RankedParagraphChunk, ...]
    provenance: tuple[MergedChunkProvenance, ...]
    diagnostics: HopMergeDiagnostics

    @model_validator(mode="after")
    def require_aligned_results_and_provenance(self) -> HopMergeResult:
        expected_ranks = tuple(range(1, len(self.results) + 1))
        result_ranks = tuple(result.rank for result in self.results)
        if result_ranks != expected_ranks:
            raise ValueError("merged result ranks must be contiguous and one-based")
        provenance_ranks = tuple(item.merged_rank for item in self.provenance)
        if provenance_ranks != expected_ranks:
            raise ValueError("merged provenance ranks must align with merged results")
        result_ids = tuple(result.chunk.chunk_id for result in self.results)
        provenance_ids = tuple(item.chunk_id for item in self.provenance)
        if result_ids != provenance_ids:
            raise ValueError("merged results and provenance must have identical chunk order")
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("merged results must contain unique chunk IDs")
        if self.diagnostics.unique_chunks != len(self.results):
            raise ValueError("merge diagnostics must match the merged result count")
        return self
