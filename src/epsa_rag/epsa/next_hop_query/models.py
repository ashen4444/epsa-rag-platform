"""Immutable, provenance-preserving contracts for EPSA Component 09."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel, NonEmptyText
from epsa_rag.epsa.evidence_graph.models import EvidenceGraphMetadata
from epsa_rag.epsa.question_analysis.models import AnalysisMetadata, AnswerType
from epsa_rag.epsa.sufficiency_decision.models import SufficiencyDecisionMetadata


class NextHopQueryType(StrEnum):
    """Stable classifications for reconstructed and recovered Component 09 output."""

    NO_QUERY = "no_query"
    BRIDGE_ENTITY_RELATION = "bridge_entity_relation"
    COMPARISON_TARGET_RELATION = "comparison_target_relation"
    SEED_ENTITY_RELATION = "seed_entity_relation"
    BRIDGE_COMPLETION = "bridge_completion"
    RELATION_COMPLETION = "relation_completion"
    ANSWER_TYPE_COMPLETION = "answer_type_completion"
    FACTOID_COMPLETION = "factoid_completion"
    COMPARISON_TARGET_COMPLETION = "comparison_target_completion"
    YES_NO_RELATION_CHECK = "yes_no_relation_check"


class NextHopQuerySource(StrEnum):
    """The inference-visible origin of a reconstructed query target."""

    NO_QUERY = "no_query"
    PATH_BRIDGE_ENTITY = "path_bridge_entity"
    PATH_COMPARISON_TARGET = "path_comparison_target"
    QUESTION_SEED = "question_seed"
    SUFFICIENCY_DECISION = "sufficiency_decision"
    QUESTION_ANALYSIS_AND_SUFFICIENCY_DECISION = "question_analysis+sufficiency_decision"


class QueryReasonCode(StrEnum):
    """Inspectable reconstructed-policy reasons, never historical parity claims."""

    SUFFICIENT_DECISION = "sufficient_decision"
    NO_CANDIDATE_PATHS = "no_candidate_paths"
    NO_RELATION_HINT = "no_relation_hint"
    NO_GROUNDED_TARGET = "no_grounded_target"
    BRIDGE_ENTITY_RELATION = "reconstructed_bridge_entity_relation"
    COMPARISON_TARGET_RELATION = "reconstructed_comparison_target_relation"
    SEED_ENTITY_RELATION = "reconstructed_seed_entity_relation"
    HISTORICAL_BRIDGE_COMPLETION = "historical_bridge_completion"
    HISTORICAL_RELATION_COMPLETION = "historical_relation_completion"
    HISTORICAL_ANSWER_TYPE_COMPLETION = "historical_answer_type_completion"
    HISTORICAL_FACTOID_COMPLETION = "historical_factoid_completion"
    HISTORICAL_COMPARISON_TARGET_COMPLETION = "historical_comparison_target_completion"
    HISTORICAL_YES_NO_RELATION_CHECK = "historical_yes_no_relation_check"
    HISTORICAL_NO_QUERY = "historical_no_query"


HeuristicConfidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class NextHopQueryMetadata(ContractModel):
    """Version identity and complete upstream provenance for one proposal."""

    schema_version: Literal[
        "next-hop-query-v1-reconstructed", "next-hop-query-v1-historical-adapted"
    ] = "next-hop-query-v1-reconstructed"
    generator: Literal[
        "RuleBasedNextHopQueryGeneratorReconstructedV1",
        "RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1",
    ] = "RuleBasedNextHopQueryGeneratorReconstructedV1"
    version: Literal["research_v1_reconstructed", "research_v1_historical_adapted"] = (
        "research_v1_reconstructed"
    )
    historical_rules_status: Literal[
        "unrecovered_reconstructed_policy",
        "recovered_historical_policy_adapted_to_current_contracts",
    ] = "unrecovered_reconstructed_policy"
    configuration_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_question_analysis: AnalysisMetadata
    source_sufficiency_decision: SufficiencyDecisionMetadata
    source_graph: EvidenceGraphMetadata
    candidate_path_ids: tuple[Identifier, ...]
    selected_path_id: Identifier | None
    reason_code: QueryReasonCode
    makes_sufficiency_decision: Literal[False] = False
    retrieves_documents: Literal[False] = False
    reranks_documents: Literal[False] = False
    scores_evidence: Literal[False] = False
    alters_evidence_paths: Literal[False] = False
    calls_llm: Literal[False] = False
    generates_final_answer: Literal[False] = False

    @model_validator(mode="after")
    def require_consistent_graph_provenance(self) -> NextHopQueryMetadata:
        if self.source_sufficiency_decision.source_graph != self.source_graph:
            raise ValueError("sufficiency-decision and query graph provenance must match")
        if len(self.candidate_path_ids) != len(set(self.candidate_path_ids)):
            raise ValueError("candidate path provenance must not contain duplicates")
        if (
            self.selected_path_id is not None
            and self.selected_path_id not in self.candidate_path_ids
        ):
            raise ValueError("selected path must be part of candidate path provenance")
        return self


class NextHopQuery(ContractModel):
    """One retrieval proposal, or an explicit explanation that none is usable."""

    query: NonEmptyText | None
    query_type: NextHopQueryType
    source: NextHopQuerySource
    target_entity: NonEmptyText | None = None
    missing_relation: NonEmptyText | None = None
    expected_answer_type: AnswerType | None = None
    reason: NonEmptyText
    confidence: HeuristicConfidence
    metadata: NextHopQueryMetadata

    @field_validator("confidence")
    @classmethod
    def require_rounded_confidence(cls, value: float) -> float:
        if not math.isfinite(value) or value != round(value, 6):
            raise ValueError("heuristic confidence must be finite and rounded to six decimals")
        return value

    @model_validator(mode="after")
    def require_consistent_query_state(self) -> NextHopQuery:
        if self.query is None:
            if self.metadata.version == "research_v1_historical_adapted":
                if (
                    self.query_type is not NextHopQueryType.NO_QUERY
                    or self.target_entity is not None
                ):
                    raise ValueError("historical no-query results must use no-query state")
                return self
            if (
                self.query_type is not NextHopQueryType.NO_QUERY
                or self.source is not NextHopQuerySource.NO_QUERY
                or self.target_entity is not None
                or self.missing_relation is not None
                or self.expected_answer_type is not None
                or self.confidence != 0.0
                or self.metadata.selected_path_id is not None
            ):
                raise ValueError("no-query results must contain only no-query state")
            return self
        if self.metadata.version == "research_v1_historical_adapted":
            return self
        if (
            self.query_type is NextHopQueryType.NO_QUERY
            or self.source is NextHopQuerySource.NO_QUERY
            or self.target_entity is None
            or self.missing_relation is None
            or self.expected_answer_type is None
        ):
            raise ValueError("usable queries require target, relation, type, and source")
        if self.metadata.version == "research_v1_reconstructed":
            expected_query = f"{self.target_entity} {self.missing_relation}"
            if self.query != expected_query:
                raise ValueError("reconstructed queries must use the target-relation template")
        mappings = {
            NextHopQueryType.BRIDGE_ENTITY_RELATION: NextHopQuerySource.PATH_BRIDGE_ENTITY,
            NextHopQueryType.COMPARISON_TARGET_RELATION: NextHopQuerySource.PATH_COMPARISON_TARGET,
            NextHopQueryType.SEED_ENTITY_RELATION: NextHopQuerySource.QUESTION_SEED,
        }
        if mappings.get(self.query_type) is not self.source:
            raise ValueError("query type and source must use the same reconstructed policy")
        return self
