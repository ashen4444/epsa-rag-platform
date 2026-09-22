"""Typed, provenance-preserving contracts for EPSA system orchestration."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel
from epsa_rag.epsa.chunk_analysis.models import CandidateChunkEvidence
from epsa_rag.epsa.context_pruning.models import PrunedContext
from epsa_rag.epsa.evidence_graph.models import EvidenceGraph
from epsa_rag.epsa.evidence_path_search.models import EvidencePath
from epsa_rag.epsa.evidence_scoring.models import ScoredEvidenceUnit
from epsa_rag.epsa.evidence_units.models import EvidenceUnit
from epsa_rag.epsa.next_hop_query.models import NextHopQuery
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.epsa.sufficiency_decision.models import SufficiencyDecision
from epsa_rag.pipeline.models import (
    AnswerGenerationRequest,
    FinalAnswer,
    HopMergeResult,
)
from epsa_rag.retrieval.models import RetrievalResult


class EPSATerminalState(StrEnum):
    """Research-visible terminal state for one bounded EPSA execution."""

    SUFFICIENT_HOP1 = "sufficient_hop1"
    SUFFICIENT_HOP2 = "sufficient_hop2"
    INSUFFICIENT_NO_QUERY = "insufficient_no_query"
    INSUFFICIENT_AFTER_HOP2 = "insufficient_after_hop2"


class EPSAPassResult(ContractModel):
    """Complete output of one Components 01-08 pass and optional Component 09 call."""

    controller_version: Literal["epsa-controller-research-v1"] = (
        "epsa-controller-research-v1"
    )
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    pass_number: Literal[1, 2]
    input_chunk_ids: tuple[Identifier, ...]
    question_analysis: QuestionAnalysis
    chunk_evidence: tuple[CandidateChunkEvidence, ...]
    evidence_units: tuple[EvidenceUnit, ...]
    scored_evidence_units: tuple[ScoredEvidenceUnit, ...]
    evidence_graph: EvidenceGraph
    candidate_paths: tuple[EvidencePath, ...]
    sufficiency_decision: SufficiencyDecision
    pruned_context: PrunedContext
    next_hop_query: NextHopQuery | None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_component_provenance_alignment(self) -> EPSAPassResult:
        if len(self.input_chunk_ids) != len(set(self.input_chunk_ids)):
            raise ValueError("EPSA pass input chunks must be unique")
        if tuple(item.chunk_id for item in self.chunk_evidence) != self.input_chunk_ids:
            raise ValueError("chunk evidence must align with EPSA pass input order")
        known_chunks = set(self.input_chunk_ids)
        if any(unit.chunk_id not in known_chunks for unit in self.evidence_units):
            raise ValueError("evidence units must originate from EPSA pass chunks")
        scored_ids = tuple(
            item.evidence_unit.evidence_unit_id for item in self.scored_evidence_units
        )
        if scored_ids != tuple(unit.evidence_unit_id for unit in self.evidence_units):
            raise ValueError("scored evidence must align with extracted evidence units")
        if self.evidence_graph.metadata.num_scored_evidence_units != len(
            self.scored_evidence_units
        ):
            raise ValueError("evidence graph must account for every scored evidence unit")
        if self.sufficiency_decision.metadata.source_graph != self.evidence_graph.metadata:
            raise ValueError("sufficiency decision must originate from the pass graph")
        if (
            self.pruned_context.metadata.source_sufficiency_decision
            != self.sufficiency_decision.metadata
        ):
            raise ValueError("pruned context must originate from the pass decision")
        if self.next_hop_query is not None:
            metadata = self.next_hop_query.metadata
            if (
                metadata.source_question_analysis != self.question_analysis.metadata
                or metadata.source_sufficiency_decision
                != self.sufficiency_decision.metadata
                or metadata.source_graph != self.evidence_graph.metadata
            ):
                raise ValueError("next-hop query must originate from this EPSA pass")
        return self


class EPSAPipelineTrace(ContractModel):
    """Complete two-hop EPSA execution and strict final-context provenance."""

    pipeline_version: Literal["epsa-two-hop-rag-v1"] = "epsa-two-hop-rag-v1"
    question_id: Identifier
    top_k: Annotated[int, Field(ge=1)]
    insufficient_context_policy: Literal["strict_partial_pruned_sentences"] = (
        "strict_partial_pruned_sentences"
    )
    hop1: RetrievalResult
    epsa_pass1: EPSAPassResult
    hop2: RetrievalResult | None
    merged_retrieval: HopMergeResult
    epsa_pass2: EPSAPassResult | None
    terminal_state: EPSATerminalState
    final_answer_request: AnswerGenerationRequest
    final_answer: FinalAnswer
    latency_ms: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_consistent_epsa_flow(self) -> EPSAPipelineTrace:
        if self.hop1.query.question_id != self.question_id:
            raise ValueError("EPSA trace question ID must match Hop-1")
        if len(self.hop1.results) > self.top_k:
            raise ValueError("Hop-1 result count cannot exceed configured top_k")
        hop1_ids = tuple(item.chunk.chunk_id for item in self.hop1.results)
        if self.epsa_pass1.pass_number != 1 or self.epsa_pass1.input_chunk_ids != hop1_ids:
            raise ValueError("EPSA pass 1 must consume the complete Hop-1 ranking")
        if self.epsa_pass1.question_analysis.raw_question != self.hop1.query.text:
            raise ValueError("EPSA pass 1 must analyze the original question")

        proposal = self.epsa_pass1.next_hop_query
        query_text = proposal.query if proposal is not None else None
        if self.epsa_pass1.sufficiency_decision.sufficient and query_text is not None:
            raise ValueError("sufficient Hop-1 evidence cannot propose Hop-2")
        if self.hop2 is None:
            if self.epsa_pass2 is not None:
                raise ValueError("EPSA pass 2 requires Hop-2 retrieval")
            if not self.epsa_pass1.sufficiency_decision.sufficient and query_text is not None:
                raise ValueError("a usable EPSA query requires Hop-2 retrieval")
            if self.merged_retrieval.diagnostics.hop2_input_chunks != 0:
                raise ValueError("one-hop EPSA trace cannot include Hop-2 merge inputs")
        else:
            if self.epsa_pass1.sufficiency_decision.sufficient:
                raise ValueError("sufficient Hop-1 evidence cannot trigger Hop-2")
            if query_text is None or self.hop2.query.text != query_text:
                raise ValueError("Hop-2 retrieval must execute EPSA's proposed query")
            if len(self.hop2.results) > self.top_k:
                raise ValueError("Hop-2 result count cannot exceed configured top_k")
            if self.epsa_pass2 is None or self.epsa_pass2.pass_number != 2:
                raise ValueError("Hop-2 retrieval requires EPSA pass 2")
            merged_ids = tuple(
                item.chunk.chunk_id for item in self.merged_retrieval.results
            )
            if self.epsa_pass2.input_chunk_ids != merged_ids:
                raise ValueError("EPSA pass 2 must consume the merged retrieval order")
            if self.epsa_pass2.question_analysis.raw_question != self.hop1.query.text:
                raise ValueError("EPSA pass 2 must reanalyze the original question")
            if self.epsa_pass2.next_hop_query is not None:
                raise ValueError("bounded EPSA pass 2 cannot propose another retrieval")

        expected_merged_ids = tuple(
            dict.fromkeys(
                (*hop1_ids, *(item.chunk.chunk_id for item in self.hop2.results))
                if self.hop2 is not None
                else hop1_ids
            )
        )
        actual_merged_ids = tuple(
            item.chunk.chunk_id for item in self.merged_retrieval.results
        )
        if actual_merged_ids != expected_merged_ids:
            raise ValueError("merged retrieval must exactly combine Hop-1 and Hop-2 order")

        final_pass = self.epsa_pass2 or self.epsa_pass1
        expected_state = _terminal_state(self.epsa_pass1, self.epsa_pass2)
        if self.terminal_state is not expected_state:
            raise ValueError("EPSA terminal state does not match its decisions")
        request = self.final_answer_request
        if request.question_id != self.question_id or request.question != self.hop1.query.text:
            raise ValueError("final answering must use the original question identity")
        if request.context.context_kind != "pruned_sentences":
            raise ValueError("EPSA final answer requires strict pruned sentence context")
        pruned = final_pass.pruned_context
        if (
            request.context.chunk_ids != pruned.selected_chunk_ids
            or request.context.text != pruned.selected_context_text
        ):
            raise ValueError("final-answer context must equal final EPSA pruned context")
        return self


def _terminal_state(
    first: EPSAPassResult, second: EPSAPassResult | None
) -> EPSATerminalState:
    if first.sufficiency_decision.sufficient:
        return EPSATerminalState.SUFFICIENT_HOP1
    if second is None:
        return EPSATerminalState.INSUFFICIENT_NO_QUERY
    if second.sufficiency_decision.sufficient:
        return EPSATerminalState.SUFFICIENT_HOP2
    return EPSATerminalState.INSUFFICIENT_AFTER_HOP2
