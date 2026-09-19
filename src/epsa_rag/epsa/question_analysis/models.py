"""Immutable, serializable Component 01 domain contracts."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator

from epsa_rag.core.models import ContractModel, NonEmptyText


class QuestionType(StrEnum):
    """Operational EPSA reasoning categories, not a linguistic taxonomy."""

    BRIDGE = "bridge"
    COMPARISON = "comparison"
    YES_NO = "yes_no"
    FACTOID = "factoid"


class AnswerType(StrEnum):
    """Coarse answer constraints used by later evidence-path components."""

    PERSON = "PERSON"
    LOCATION = "LOCATION"
    DATE = "DATE"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    TITLE_OR_WORK = "TITLE_OR_WORK"
    ORGANIZATION = "ORGANIZATION"
    ENTITY = "ENTITY"
    UNKNOWN = "UNKNOWN"


RuleScore = Annotated[float, Field(ge=0, le=1)]
CharacterOffset = Annotated[int, Field(ge=0)]


class EntityMention(ContractModel):
    """A deterministic candidate anchor with a half-open normalized-question span."""

    text: NonEmptyText
    normalized: NonEmptyText
    source: Literal[
        "quoted_string",
        "capitalized_phrase",
        "title_like_token",
        "comparison_which_of",
        "comparison_between",
    ]
    start: CharacterOffset
    end: CharacterOffset
    confidence: RuleScore

    @field_validator("confidence")
    @classmethod
    def require_finite_confidence(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence must be finite")
        return value

    @field_validator("normalized")
    @classmethod
    def require_normalized_value(cls, value: str) -> str:
        if value != value.casefold() or "  " in value:
            raise ValueError("normalized entity text must be casefolded and whitespace-collapsed")
        return value

    def model_post_init(self, __context: object) -> None:
        if self.end <= self.start:
            raise ValueError("entity span end must be greater than start")


class RelationHint(ContractModel):
    """A lexical relation hypothesis, not evidence that the relation holds."""

    relation: NonEmptyText
    matched_text: NonEmptyText
    source: Literal["normalized_question"] = "normalized_question"
    start: CharacterOffset
    end: CharacterOffset
    confidence: RuleScore

    @field_validator("confidence")
    @classmethod
    def require_finite_confidence(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence must be finite")
        return value

    def model_post_init(self, __context: object) -> None:
        if self.end <= self.start:
            raise ValueError("relation span end must be greater than start")


class AnswerTypeCandidate(ContractModel):
    """The single v1 answer-type rule result, exposed as an extensible candidate list."""

    answer_type: AnswerType
    text: NonEmptyText
    confidence: RuleScore

    @field_validator("confidence")
    @classmethod
    def require_finite_confidence(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence must be finite")
        return value


class AnalysisMetadata(ContractModel):
    """Version/configuration identity required to reproduce a deterministic result."""

    analyzer: Literal["RuleBasedQuestionAnalyzer"] = "RuleBasedQuestionAnalyzer"
    version: Literal["rule_based_v1"] = "rule_based_v1"
    configuration_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    span_source: Literal["normalized_question"] = "normalized_question"


class QuestionAnalysis(ContractModel):
    """Stable immutable output of EPSA Component 01."""

    raw_question: NonEmptyText
    normalized_question: NonEmptyText
    question_type: QuestionType
    expected_answer_type: AnswerType
    seed_entities: tuple[EntityMention, ...]
    required_relation_hints: tuple[RelationHint, ...]
    comparison_targets: tuple[EntityMention, ...]
    answer_type_candidates: tuple[AnswerTypeCandidate, ...]
    metadata: AnalysisMetadata

    def model_post_init(self, __context: object) -> None:
        if len(self.answer_type_candidates) != 1:
            raise ValueError("rule_based_v1 requires exactly one answer-type candidate")
        if self.answer_type_candidates[0].answer_type != self.expected_answer_type:
            raise ValueError("expected answer type must match the v1 answer-type candidate")
        if self.question_type is not QuestionType.COMPARISON and self.comparison_targets:
            raise ValueError("only comparison questions may have comparison targets")
