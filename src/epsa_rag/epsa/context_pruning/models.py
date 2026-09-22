"""Immutable, provenance-preserving contracts for EPSA Component 08."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel
from epsa_rag.epsa.evidence_scoring.models import ScoredEvidenceUnit
from epsa_rag.epsa.sufficiency_decision.models import SufficiencyDecisionMetadata


class PruningStrategy(StrEnum):
    """Diagnostic classification of a completed research-v1 pruning operation."""

    SUFFICIENT_BRIDGE_NEIGHBOR_SENTENCE = "sufficient_bridge_neighbor_sentence_pruning"
    SUFFICIENT_PATH_SENTENCE = "sufficient_path_sentence_pruning"
    PARTIAL_EVIDENCE_SENTENCE = "partial_evidence_sentence_pruning"
    EMPTY_EVIDENCE = "empty_evidence_pruning"


class PruningDiagnostics(ContractModel):
    """Inference-safe selection and expansion diagnostics."""

    requested_evidence_unit_ids: tuple[Identifier, ...]
    missing_requested_evidence_unit_ids: tuple[Identifier, ...]
    input_evidence_unit_count: Annotated[int, Field(ge=0)]
    mandatory_evidence_unit_count: Annotated[int, Field(ge=0)]
    neighbor_evidence_unit_count: Annotated[int, Field(ge=0)]


class PrunedContextMetadata(ContractModel):
    """Component identity and immutable Component 07 provenance."""

    schema_version: Literal["pruned-context-v1"] = "pruned-context-v1"
    pruner: Literal["ResearchContextPrunerV1"] = "ResearchContextPrunerV1"
    version: Literal["research_v1"] = "research_v1"
    configuration_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_sufficiency_decision: SufficiencyDecisionMetadata
    makes_sufficiency_decision: Literal[False] = False
    retrieves_documents: Literal[False] = False
    makes_next_query: Literal[False] = False
    calls_llm: Literal[False] = False
    generates_final_answer: Literal[False] = False


class PrunedContext(ContractModel):
    """A typed, immutable context diagnostic produced without changing evidence authority."""

    selected_chunk_ids: tuple[Identifier, ...]
    selected_evidence_unit_ids: tuple[Identifier, ...]
    selected_evidence_units: tuple[ScoredEvidenceUnit, ...]
    selected_sentences: tuple[str, ...]
    selected_context_text: str
    estimated_context_tokens: Annotated[int, Field(ge=0)]
    pruning_strategy: PruningStrategy
    removed_evidence_unit_ids: tuple[Identifier, ...]
    diagnostics: PruningDiagnostics
    metadata: PrunedContextMetadata

    @model_validator(mode="after")
    def require_consistent_selected_provenance(self) -> PrunedContext:
        selected_ids = tuple(
            unit.evidence_unit.evidence_unit_id for unit in self.selected_evidence_units
        )
        if selected_ids != self.selected_evidence_unit_ids:
            raise ValueError("selected evidence IDs must match selected evidence provenance")
        expected_sentences = tuple(
            unit.evidence_unit.resolved_text or unit.evidence_unit.sentence_text
            for unit in self.selected_evidence_units
        )
        if expected_sentences != self.selected_sentences:
            raise ValueError("selected sentences must match selected evidence provenance")
        if not self.selected_evidence_units and self.selected_context_text:
            raise ValueError("empty selected evidence cannot render context text")
        if self.pruning_strategy is PruningStrategy.EMPTY_EVIDENCE and self.selected_evidence_units:
            raise ValueError("empty evidence strategy requires no selected evidence")
        return self
