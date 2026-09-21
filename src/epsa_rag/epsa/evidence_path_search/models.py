"""Immutable, provenance-preserving contracts for EPSA Component 06."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel, NonEmptyText
from epsa_rag.epsa.evidence_graph.models import EvidenceGraphMetadata
from epsa_rag.epsa.evidence_scoring.models import ScoredEvidenceUnit
from epsa_rag.epsa.question_analysis.models import AnswerType, QuestionType

PathScore = Annotated[float, Field(ge=0, allow_inf_nan=False)]
type PathKind = Literal[
    "bridge_candidate",
    "factoid_candidate",
    "comparison_target_partial",
    "yes_no_evidence_connection",
]


class PathScoreBreakdown(ContractModel):
    """Inspectable, uncalibrated research-v1 path-score features."""

    average_evidence_score: PathScore
    expected_answer_type_match: PathScore
    relation_match_score: PathScore
    path_length_score: PathScore
    bridge_entity_quality: PathScore
    answer_specificity_score: PathScore
    retrieval_quality_score: PathScore

    @field_validator("average_evidence_score", "expected_answer_type_match", "relation_match_score",
                     "path_length_score", "bridge_entity_quality", "answer_specificity_score",
                     "retrieval_quality_score")
    @classmethod
    def require_rounded_finite_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("path score features must be finite")
        if value != round(value, 6):
            raise ValueError("path score features must be rounded to six decimals")
        return value


class EvidencePathMetadata(ContractModel):
    """Component identity, graph identity, and explicit non-decision markers."""

    schema_version: Literal["evidence-path-v1"] = "evidence-path-v1"
    searcher: Literal["EvidencePathSearcherV1"] = "EvidencePathSearcherV1"
    version: Literal["research_v1"] = "research_v1"
    configuration_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    path_kind: PathKind
    source_graph: EvidenceGraphMetadata
    makes_sufficiency_decision: Literal[False] = False
    does_not_compare_values_yet: bool = False
    does_not_decide_yes_no: bool = False
    bridge_entity: str | None = None
    comparison_target: str | None = None

    @model_validator(mode="after")
    def require_kind_specific_metadata(self) -> EvidencePathMetadata:
        if self.path_kind == "bridge_candidate" and not self.bridge_entity:
            raise ValueError("bridge paths require bridge entity metadata")
        if self.path_kind == "comparison_target_partial":
            if not self.does_not_compare_values_yet or not self.comparison_target:
                raise ValueError("comparison partial paths must declare their incomplete target")
        if self.path_kind == "yes_no_evidence_connection" and not self.does_not_decide_yes_no:
            raise ValueError("yes/no paths must declare that polarity is undecided")
        return self


class EvidencePath(ContractModel):
    """One ranked candidate path; never a sufficiency or answer decision."""

    path_id: Identifier
    question_type: QuestionType
    # Component 05 owns graph identity. Its stable surface-based edge IDs can legitimately exceed
    # the portable external-ID length limit, so preserve them verbatim rather than re-validating
    # them as Component 06 identifiers.
    node_ids: tuple[NonEmptyText, ...]
    edge_ids: tuple[NonEmptyText, ...]
    evidence_unit_ids: tuple[Identifier, ...]
    entity_chain: tuple[str, ...]
    relation_chain: tuple[str, ...]
    answer_candidate: NonEmptyText | None
    answer_type: AnswerType
    score: PathScore
    score_breakdown: PathScoreBreakdown
    scored_evidence_units: tuple[ScoredEvidenceUnit, ...]
    metadata: EvidencePathMetadata

    @field_validator("score")
    @classmethod
    def require_rounded_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("path score must be finite")
        if value != round(value, 6):
            raise ValueError("path score must be rounded to six decimals")
        return value

    @model_validator(mode="after")
    def require_complete_ordered_provenance(self) -> EvidencePath:
        if not self.node_ids or not self.edge_ids or not self.evidence_unit_ids:
            raise ValueError("candidate paths require nodes, edges, and evidence-unit provenance")
        scored_ids = tuple(
            unit.evidence_unit.evidence_unit_id for unit in self.scored_evidence_units
        )
        if self.evidence_unit_ids != scored_ids:
            raise ValueError("path evidence IDs must match retained scored-evidence provenance")
        if self.question_type is QuestionType.YES_NO and self.answer_candidate is not None:
            raise ValueError("yes/no evidence connections cannot select an answer candidate")
        return self
