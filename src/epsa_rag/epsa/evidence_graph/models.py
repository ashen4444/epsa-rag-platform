"""Immutable, serializable contracts for EPSA Component 05 evidence graphs."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel, NonEmptyText
from epsa_rag.epsa.evidence_scoring.models import ScoredEvidenceUnit
from epsa_rag.epsa.question_analysis.models import AnswerType, QuestionType

GraphWeight = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class GraphNodeType(StrEnum):
    """The frozen research-v1 graph node categories."""

    ENTITY = "entity"
    CHUNK = "chunk"
    TITLE = "title"
    SENTENCE = "sentence"
    RELATION = "relation"
    ANSWER_TYPE = "answer_type"


class GraphEdgeType(StrEnum):
    """The frozen research-v1 graph edge categories."""

    CHUNK_TO_SENTENCE = "chunk_to_sentence"
    TITLE_TO_SENTENCE = "title_to_sentence"
    TITLE_TO_ENTITY = "title_to_entity"
    SENTENCE_IN_DOCUMENT_ABOUT_ENTITY = "sentence_in_document_about_entity"
    SENTENCE_MENTIONS_ENTITY = "sentence_mentions_entity"
    SENTENCE_HAS_RELATION = "sentence_has_relation"
    SENTENCE_HAS_ANSWER_TYPE = "sentence_has_answer_type"
    ENTITY_COOCCURS_WITH_ENTITY = "entity_cooccurs_with_entity"
    SEED_ENTITY_TO_SENTENCE = "seed_entity_to_sentence"
    POSSIBLE_BRIDGE = "possible_bridge"
    POSSIBLE_ANSWER_CANDIDATE = "possible_answer_candidate"


class GraphNode(ContractModel):
    """One stable graph node with optional sentence-level scored evidence."""

    node_id: NonEmptyText
    node_type: GraphNodeType
    label: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    scored_evidence: ScoredEvidenceUnit | None = None

    @field_validator("metadata")
    @classmethod
    def detach_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return value.copy()

    @model_validator(mode="after")
    def sentence_evidence_is_consistent(self) -> GraphNode:
        if self.node_type is GraphNodeType.SENTENCE and self.scored_evidence is None:
            raise ValueError("sentence nodes require scored evidence provenance")
        if self.node_type is not GraphNodeType.SENTENCE and self.scored_evidence is not None:
            raise ValueError("only sentence nodes may retain scored evidence provenance")
        return self


class GraphEdge(ContractModel):
    """A weighted, evidence-traceable connection between two graph nodes."""

    edge_id: NonEmptyText
    source_id: NonEmptyText
    target_id: NonEmptyText
    edge_type: GraphEdgeType
    weight: GraphWeight
    evidence_unit_id: Identifier | None = None
    relation: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def detach_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return value.copy()

    @field_validator("weight")
    @classmethod
    def require_finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("graph edge weight must be finite")
        return value


class EvidenceGraphMetadata(ContractModel):
    """Version and question-level context for a pre-sufficiency graph."""

    schema_version: Literal["evidence-graph-v1"] = "evidence-graph-v1"
    builder: Literal["EvidenceGraphBuilderV1"] = "EvidenceGraphBuilderV1"
    version: Literal["research_v1"] = "research_v1"
    configuration_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    expected_answer_type: AnswerType
    required_relation_hints: tuple[str, ...]
    num_scored_evidence_units: Annotated[int, Field(ge=0)]
    makes_sufficiency_decision: Literal[False] = False
    identity_strategy: Literal["surface_normalized_ascii_v1"] = "surface_normalized_ascii_v1"
    inference_gold_labels_present: Literal[False] = False


class EvidenceGraph(ContractModel):
    """Deterministic, pre-sufficiency graph built from scored evidence units."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    question_type: QuestionType
    seed_entity_node_ids: tuple[NonEmptyText, ...]
    evidence_unit_node_ids: tuple[NonEmptyText, ...]
    entity_node_ids: tuple[NonEmptyText, ...]
    metadata: EvidenceGraphMetadata

    @model_validator(mode="after")
    def validate_graph_identity_and_order(self) -> EvidenceGraph:
        node_ids = tuple(node.node_id for node in self.nodes)
        edge_ids = tuple(edge.edge_id for edge in self.edges)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph node IDs must be unique")
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("graph edge IDs must be unique")
        if node_ids != tuple(sorted(node_ids)):
            raise ValueError("graph nodes must be ordered by stable node ID")
        if edge_ids != tuple(sorted(edge_ids)):
            raise ValueError("graph edges must be ordered by stable edge ID")
        known_nodes = set(node_ids)
        if any(
            edge.source_id not in known_nodes or edge.target_id not in known_nodes
            for edge in self.edges
        ):
            raise ValueError("graph edges must reference graph nodes")
        if any(node_id not in known_nodes for node_id in self.seed_entity_node_ids):
            raise ValueError("seed nodes must exist in graph")
        if any(node_id not in known_nodes for node_id in self.evidence_unit_node_ids):
            raise ValueError("evidence sentence nodes must exist in graph")
        if any(node_id not in known_nodes for node_id in self.entity_node_ids):
            raise ValueError("entity nodes must exist in graph")
        return self

    def node_by_id(self, node_id: str) -> GraphNode:
        """Return a graph node by stable identifier without exposing mutable state."""

        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)
