"""Strict inference-safe sentence evidence contracts."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from epsa_rag.core.ids import Identifier, make_evidence_unit_id
from epsa_rag.core.models import ContractModel, NonEmptyText
from epsa_rag.epsa.question_analysis.models import AnswerType

Offset = Annotated[int, Field(ge=0)]
UnitScore = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class ResolvedSentence(ContractModel):
    original_text: str
    resolved_text: str
    method: Literal["unchanged", "title_pronoun", "ambiguous_context", "provider"]
    changed: bool
    reason: str
    provider_version: str

    @model_validator(mode="after")
    def consistent_change(self) -> ResolvedSentence:
        if self.changed != (self.original_text != self.resolved_text):
            raise ValueError("resolution change flag does not match text")
        return self


class SentenceEntity(ContractModel):
    text: NonEmptyText
    normalized: NonEmptyText
    source: Literal["sentence_text", "doc_title"]
    start_char: Offset | None = None
    end_char: Offset | None = None

    @model_validator(mode="after")
    def valid_span(self) -> SentenceEntity:
        if self.source == "doc_title":
            if self.start_char is not None or self.end_char is not None:
                raise ValueError("document title cannot claim a sentence offset")
        elif self.start_char is None or self.end_char is None or self.end_char <= self.start_char:
            raise ValueError("sentence entity requires a nonempty source span")
        return self


class StructuredAnswerCandidate(ContractModel):
    text: NonEmptyText
    normalized: NonEmptyText
    answer_type: AnswerType
    source: str
    span_scope: Literal["sentence_text", "doc_title"]
    start_char: Offset | None = None
    end_char: Offset | None = None
    rule_score: UnitScore

    @model_validator(mode="after")
    def valid_span(self) -> StructuredAnswerCandidate:
        if self.span_scope == "doc_title":
            if self.start_char is not None or self.end_char is not None:
                raise ValueError("title candidate cannot claim a sentence offset")
        elif self.start_char is None or self.end_char is None or self.end_char <= self.start_char:
            raise ValueError("sentence candidate requires a nonempty source span")
        return self


class EvidenceUnitMetadata(ContractModel):
    version: Literal["rule_based_v1", "rule_based_v2"]
    configuration_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    sentence_source: Literal["metadata", "paragraph_fallback", "segmented"]
    resolution: ResolvedSentence
    provider_versions: tuple[str, ...]
    fallbacks_used: tuple[str, ...]


class EvidenceUnit(ContractModel):
    """Candidate features only; gold support labels are excluded from inference."""

    evidence_unit_id: Identifier
    chunk_id: Identifier
    doc_title: str
    paragraph_index: Offset | None
    sentence_id: Offset
    sentence_text: str
    resolved_text: str
    entities: tuple[str, ...]
    entity_features: tuple[SentenceEntity, ...]
    relation_hints: tuple[str, ...]
    answer_type_candidates: tuple[AnswerType, ...]
    structured_answer_candidates: tuple[StructuredAnswerCandidate, ...]
    question_entity_overlap: tuple[str, ...]
    question_token_overlap: UnitScore
    retrieval_rank: Annotated[int, Field(ge=1)] | None
    retrieval_score: float | None
    source_question_id: Identifier | None
    start_char: Offset | None
    end_char: Offset | None
    metadata: EvidenceUnitMetadata

    @field_validator("retrieval_score")
    @classmethod
    def finite_retrieval_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("retrieval score must be finite")
        return value

    @model_validator(mode="after")
    def consistent_identity(self) -> EvidenceUnit:
        if self.evidence_unit_id != make_evidence_unit_id(self.chunk_id, self.sentence_id):
            raise ValueError("evidence unit ID must derive from chunk and sentence IDs")
        if self.start_char is None or self.end_char is None:
            if self.start_char is not None or self.end_char is not None:
                raise ValueError("source offsets must both be present or absent")
        elif self.end_char <= self.start_char:
            raise ValueError("source offsets must identify nonempty text")
        if self.metadata.resolution.original_text != self.sentence_text:
            raise ValueError("resolution must refer to the original sentence")
        if self.metadata.resolution.resolved_text != self.resolved_text:
            raise ValueError("resolution must refer to the derived sentence")
        return self
