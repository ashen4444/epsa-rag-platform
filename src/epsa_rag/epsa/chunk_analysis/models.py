"""Immutable Component 02 inputs and outputs."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel, NonEmptyText, Sentence
from epsa_rag.epsa.question_analysis.models import AnswerType

RuleScore = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Offset = Annotated[int, Field(ge=0)]


class CanonicalRetrievedChunk(ContractModel):
    """Backend-neutral, inference-only paragraph and retrieval provenance."""

    chunk_id: Identifier
    doc_id: Identifier | None = None
    doc_title: str
    paragraph_index: Annotated[int, Field(ge=0)] | None = None
    paragraph_text: str
    chunk_text: str
    sentences: tuple[Sentence, ...] = ()
    retrieval_rank: Annotated[int, Field(ge=1)] | None = None
    retrieval_score: float | None = None
    source_question_id: Identifier | None = None

    @field_validator("retrieval_score")
    @classmethod
    def finite_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("retrieval score must be finite")
        return value


class ChunkEntityMention(ContractModel):
    text: NonEmptyText
    normalized: NonEmptyText
    source: Literal["doc_title", "chunk"]
    start_char: Offset | None = None
    end_char: Offset | None = None
    confidence: RuleScore


class ChunkEntityMentionV2(ChunkEntityMention):
    """Body offsets use paragraph_text; title identity has no paragraph offset."""

    span_scope: Literal["paragraph_text", "doc_title"]

    @model_validator(mode="after")
    def valid_span(self) -> ChunkEntityMentionV2:
        if self.span_scope == "doc_title":
            if (
                self.source != "doc_title" or self.start_char is not None
                or self.end_char is not None
            ):
                raise ValueError("title entity cannot claim a paragraph span")
        elif (
            self.source != "chunk" or self.start_char is None or self.end_char is None
            or self.end_char <= self.start_char
        ):
            raise ValueError("body entity requires a nonempty paragraph span")
        return self


class ChunkRelationHint(ContractModel):
    relation: NonEmptyText
    matched_text: NonEmptyText
    source: Literal["chunk"] = "chunk"
    start_char: Offset
    end_char: Offset
    confidence: RuleScore


class GroundedChunkRelationHint(ChunkRelationHint):
    """A contextual candidate, not a proven fact or directed graph edge."""

    sentence_index: Annotated[int, Field(ge=0)] | None = None
    subject_entity: str | None = None
    object_entity: str | None = None
    span_scope: Literal["paragraph_text"]
    grounding_type: Literal[
        "entity_pair", "subject_only", "object_only", "lexical_hint"
    ] = "lexical_hint"

    @model_validator(mode="before")
    @classmethod
    def fill_grounding_type(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        subject = result.get("subject_entity")
        object_ = result.get("object_entity")
        derived = (
            "entity_pair" if subject and object_ else
            "subject_only" if subject else
            "object_only" if object_ else "lexical_hint"
        )
        if "grounding_type" in result and result["grounding_type"] != derived:
            raise ValueError("grounding type must match relation entity fields")
        result["grounding_type"] = derived
        return result


class ChunkAnswerCandidate(ContractModel):
    answer_type: AnswerType
    text: NonEmptyText
    source: Literal["chunk_pattern", "chunk_entity_like_span"]
    start_char: Offset
    end_char: Offset
    confidence: RuleScore


class ChunkAnswerCandidateV2(ContractModel):
    """Typed candidate with an explicit source coordinate namespace."""

    answer_type: AnswerType
    text: NonEmptyText
    source: Literal["chunk_pattern", "chunk_entity_like_span"]
    start_char: Offset | None = None
    end_char: Offset | None = None
    span_scope: Literal["paragraph_text", "doc_title"]
    confidence: RuleScore

    @model_validator(mode="after")
    def valid_span(self) -> ChunkAnswerCandidateV2:
        if self.span_scope == "doc_title":
            if self.start_char is not None or self.end_char is not None:
                raise ValueError("title answer cannot claim a paragraph span")
        elif (
            self.start_char is None or self.end_char is None
            or self.end_char <= self.start_char
        ):
            raise ValueError("body answer requires a nonempty paragraph span")
        return self


class BridgeCandidateDecision(ContractModel):
    entity: NonEmptyText
    accepted: bool
    reason: Literal["candidate", "document_title", "question_entity", "too_short"]


class BridgeCandidateDecisionV2(ContractModel):
    entity: NonEmptyText
    accepted: bool
    reason: Literal[
        "candidate",
        "document_title",
        "question_entity",
        "non_specific",
        "answer_role",
        "not_seed_side",
        "no_required_relation",
        "no_cross_chunk_link",
        "no_context",
    ]
    relation: str | None = None
    linked_chunk_ids: tuple[Identifier, ...] = ()
    relation_basis: Literal["entity_pair", "sentence_window"] | None = None
    link_basis: Literal["title", "title_alias", "body_mention"] | None = None
    matches_question_relation: bool = False


class ChunkAnalysisMetadata(ContractModel):
    analyzer: Literal["RuleBasedCandidateChunkEvidenceAnalyzer"] = (
        "RuleBasedCandidateChunkEvidenceAnalyzer"
    )
    version: Literal["rule_based_v1"] = "rule_based_v1"
    configuration_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    bridge_decisions: tuple[BridgeCandidateDecision, ...] = ()
    provider_versions: tuple[str, ...] = ()
    fallbacks_used: tuple[str, ...] = ()


class ChunkAnalysisMetadataV2(ContractModel):
    analyzer: Literal["RuleBasedV2CandidateChunkEvidenceAnalyzer"] = (
        "RuleBasedV2CandidateChunkEvidenceAnalyzer"
    )
    version: Literal["rule_based_v2"] = "rule_based_v2"
    configuration_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    bridge_decisions: tuple[BridgeCandidateDecisionV2, ...] = ()
    provider_versions: tuple[str, ...] = ("question-entity-rules-v1",)
    fallbacks_used: tuple[str, ...] = ()


class CandidateChunkEvidence(ContractModel):
    """Question-conditioned candidate features, with no sufficiency conclusion."""

    chunk_id: Identifier
    doc_title: str
    paragraph_index: Annotated[int, Field(ge=0)] | None
    retrieval_rank: Annotated[int, Field(ge=1)] | None
    retrieval_score: float | None
    entities: tuple[ChunkEntityMention | ChunkEntityMentionV2, ...]
    relation_hints: tuple[ChunkRelationHint | GroundedChunkRelationHint, ...]
    answer_type_candidates: tuple[ChunkAnswerCandidate | ChunkAnswerCandidateV2, ...]
    potential_bridge_entities: tuple[ChunkEntityMention | ChunkEntityMentionV2, ...]
    question_entity_overlap: tuple[str, ...]
    question_token_overlap: tuple[str, ...]
    question_token_overlap_score: RuleScore
    is_title_match: bool
    chunk_text: str
    paragraph_text: str
    source_question_id: Identifier | None
    sentences: tuple[Sentence, ...]
    metadata: ChunkAnalysisMetadata | ChunkAnalysisMetadataV2

    @field_validator("retrieval_score")
    @classmethod
    def finite_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("retrieval score must be finite")
        return value
