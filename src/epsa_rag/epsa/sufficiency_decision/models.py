"""Immutable, provenance-preserving contracts for EPSA Component 07."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel, NonEmptyText
from epsa_rag.epsa.evidence_graph.models import EvidenceGraphMetadata
from epsa_rag.epsa.evidence_path_search.models import EvidencePath
from epsa_rag.epsa.question_analysis.models import AnswerType, QuestionType


class DecisionReasonCode(StrEnum):
    SUFFICIENT_BRIDGE = "sufficient_bridge"
    SUFFICIENT_FACTOID = "sufficient_factoid"
    SUFFICIENT_YES_NO_EVIDENCE = "sufficient_yes_no_evidence"
    NO_CANDIDATE_PATHS = "no_candidate_paths"
    BRIDGE_RULES_UNSATISFIED = "bridge_rules_unsatisfied"
    FACTOID_RULES_UNSATISFIED = "factoid_rules_unsatisfied"
    COMPARISON_UNSUPPORTED_RESEARCH_V1 = "comparison_unsupported_research_v1"
    YES_NO_RULES_UNSATISFIED = "yes_no_rules_unsatisfied"


class GuardCode(StrEnum):
    PATH_AVAILABLE = "path_available"
    CONTROLLER_PARTIAL_PATH = "controller_partial_path"
    EVIDENCE_UNITS = "evidence_units"
    SEED_CONNECTION = "seed_connection"
    BRIDGE_ENTITY = "bridge_entity"
    BRIDGE_SPECIFICITY = "bridge_specificity"
    BRIDGE_ANSWER_SIDE_GROUNDING = "bridge_answer_side_grounding"
    ANSWER_CANDIDATE = "answer_candidate"
    ANSWER_SURFACE_TYPE = "answer_surface_type"
    GRAPH_PATH_ANSWER_TYPE = "graph_path_answer_type"
    RELATION_COVERAGE = "relation_coverage"
    QUESTION_ANCHOR_COVERAGE = "question_anchor_coverage"
    ROLE_COVERAGE = "role_coverage"
    INDEPENDENT_EVIDENCE_COVERAGE = "independent_evidence_coverage"
    GENERIC_FACTOID_COVERAGE = "generic_factoid_coverage"
    MULTI_FACT_COVERAGE = "multi_fact_coverage"
    COMPARISON_RESOLUTION = "comparison_resolution"
    YES_NO_POLARITY = "yes_no_polarity"


class DecisionTraceEntry(ContractModel):
    rule_code: GuardCode
    passed: bool
    message: NonEmptyText
    observed: dict[str, JsonValue] = Field(default_factory=dict)
    required: dict[str, JsonValue] = Field(default_factory=dict)
    evidence_unit_ids: tuple[Identifier, ...] = ()


class SufficiencyDecisionMetadata(ContractModel):
    schema_version: Literal["sufficiency-decision-v1"] = "sufficiency-decision-v1"
    engine: Literal["RuleBasedSufficiencyEngineV1"] = "RuleBasedSufficiencyEngineV1"
    version: Literal["research_v1"] = "research_v1"
    configuration_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_graph: EvidenceGraphMetadata
    candidate_path_ids: tuple[Identifier, ...]
    makes_sufficiency_decision: Literal[True] = True
    makes_next_query: Literal[False] = False
    confidence_kind: Literal["uncalibrated_heuristic"] = "uncalibrated_heuristic"
    does_not_generate_final_answer: Literal[True] = True
    does_not_generate_yes_no_polarity: bool = False


HeuristicConfidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class SufficiencyDecision(ContractModel):
    """A deterministic Component 07 decision, never a generated final answer."""

    sufficient: bool
    confidence: HeuristicConfidence
    question_type: QuestionType
    best_path: EvidencePath | None
    candidate_paths_considered: tuple[EvidencePath, ...]
    selected_evidence_unit_ids: tuple[Identifier, ...]
    selected_chunk_ids: tuple[Identifier, ...]
    answer_candidate: NonEmptyText | None
    answer_type: AnswerType
    missing_evidence: str | None
    decision_reason: DecisionReasonCode
    rule_trace: tuple[DecisionTraceEntry, ...]
    metadata: SufficiencyDecisionMetadata

    @field_validator("confidence")
    @classmethod
    def require_rounded_confidence(cls, value: float) -> float:
        if not math.isfinite(value) or value != round(value, 6):
            raise ValueError("heuristic confidence must be finite and rounded to six decimals")
        return value

    @model_validator(mode="after")
    def require_consistent_provenance(self) -> SufficiencyDecision:
        candidate_ids = tuple(path.path_id for path in self.candidate_paths_considered)
        if candidate_ids != self.metadata.candidate_path_ids:
            raise ValueError("metadata candidate path IDs must retain considered path provenance")
        if self.best_path is not None and self.best_path.path_id not in candidate_ids:
            raise ValueError("best path must be one of the considered candidate paths")
        if self.sufficient and self.best_path is None:
            raise ValueError("sufficient decisions require a best path")
        if self.sufficient and self.missing_evidence is not None:
            raise ValueError("sufficient decisions cannot report missing evidence")
        if self.question_type is QuestionType.YES_NO and self.answer_candidate is not None:
            raise ValueError("Component 07 does not select a yes/no polarity")
        return self
