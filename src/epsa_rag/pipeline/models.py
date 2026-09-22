"""Immutable contracts for deterministic multi-hop retrieval orchestration."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel, RankedParagraphChunk, RetrievalQuery
from epsa_rag.retrieval.models import RetrievalResult


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


class RenderedContext(ContractModel):
    """Deterministically rendered context with structural provenance."""

    format_version: Identifier
    context_kind: Literal["full_paragraphs", "pruned_sentences"]
    chunk_ids: tuple[Identifier, ...]
    text: str
    character_count: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def require_matching_character_count(self) -> RenderedContext:
        if self.character_count != len(self.text):
            raise ValueError("context character count must match rendered text")
        if len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError("rendered context chunk IDs must be unique")
        return self


class AnswerGenerationRequest(ContractModel):
    """Provider-neutral final-answer request supplied identically by every system."""

    question_id: Identifier
    question: str
    context: RenderedContext

    @field_validator("question")
    @classmethod
    def require_nonblank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("final-answer question must not be blank")
        return value


class FinalAnswerPayload(ContractModel):
    """Structured answer-only schema returned by the final-answer model."""

    answer: str

    @field_validator("answer")
    @classmethod
    def require_nonblank_answer(cls, value: str) -> str:
        answer = value.strip()
        if not answer:
            raise ValueError("generated answer must not be blank")
        return answer


class LLMTokenUsage(ContractModel):
    """API-reported token usage for one model request."""

    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    total_tokens: Annotated[int, Field(ge=0)]
    cached_input_tokens: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def require_consistent_total(self) -> LLMTokenUsage:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total tokens must equal input plus output tokens")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached input tokens cannot exceed input tokens")
        return self


class FinalAnswer(ContractModel):
    """Inspectable final-answer output with exact model and usage provenance."""

    answer: str
    response_id: Identifier
    model: Identifier
    generator_version: Identifier
    prompt_version: Identifier
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    usage: LLMTokenUsage
    latency_ms: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("answer")
    @classmethod
    def require_nonblank_final_answer(cls, value: str) -> str:
        answer = value.strip()
        if not answer:
            raise ValueError("final answer must not be blank")
        return answer


class FixedBaselineTrace(ContractModel):
    """Complete output of one fixed one-hop baseline execution."""

    pipeline_version: Literal["fixed-one-hop-rag-v1"] = "fixed-one-hop-rag-v1"
    question_id: Identifier
    top_k: Annotated[int, Field(ge=1)]
    hop1: RetrievalResult
    merged_retrieval: HopMergeResult
    final_answer_request: AnswerGenerationRequest
    final_answer: FinalAnswer
    latency_ms: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_consistent_question_and_context(self) -> FixedBaselineTrace:
        if self.hop1.query.question_id != self.question_id:
            raise ValueError("baseline trace question ID must match Hop-1")
        if self.final_answer_request.question_id != self.question_id:
            raise ValueError("baseline trace question ID must match final-answer request")
        if len(self.hop1.results) > self.top_k:
            raise ValueError("Hop-1 result count cannot exceed configured top_k")
        merged_ids = tuple(result.chunk.chunk_id for result in self.merged_retrieval.results)
        if self.final_answer_request.context.chunk_ids != merged_ids:
            raise ValueError("final-answer context must match the merged retrieval order")
        return self


class AdaptiveReasonCode(StrEnum):
    """Stable controller outcomes used for diagnostics and stratified evaluation."""

    SUFFICIENT = "sufficient"
    MISSING_BRIDGE_EVIDENCE = "missing_bridge_evidence"
    MISSING_COMPARISON_EVIDENCE = "missing_comparison_evidence"
    MISSING_ANSWER_EVIDENCE = "missing_answer_evidence"
    OTHER_MISSING_EVIDENCE = "other_missing_evidence"
    INSUFFICIENT_NO_GROUNDED_QUERY = "insufficient_no_grounded_query"


class AdaptiveControlRequest(ContractModel):
    """Original question and complete Hop-1 context supplied to the LLM controller."""

    question_id: Identifier
    question: str
    hop1_context: RenderedContext

    @model_validator(mode="after")
    def require_valid_request(self) -> AdaptiveControlRequest:
        if not self.question.strip():
            raise ValueError("adaptive-controller question must not be blank")
        if self.hop1_context.context_kind != "full_paragraphs":
            raise ValueError("adaptive controller requires full Hop-1 paragraphs")
        return self


class AdaptiveControlOutput(ContractModel):
    """Five-field adaptive decision approved for the LLM baseline."""

    sufficient: bool
    reason_code: AdaptiveReasonCode
    missing_evidence: str | None
    evidence_document_numbers: tuple[Annotated[int, Field(ge=1)], ...]
    next_hop_query: str | None

    @field_validator("missing_evidence", "next_hop_query")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def require_consistent_decision(self) -> AdaptiveControlOutput:
        if tuple(sorted(set(self.evidence_document_numbers))) != self.evidence_document_numbers:
            raise ValueError("evidence document numbers must be unique and sorted")
        if self.sufficient:
            if (
                self.reason_code is not AdaptiveReasonCode.SUFFICIENT
                or self.missing_evidence is not None
                or self.next_hop_query is not None
            ):
                raise ValueError("sufficient decisions cannot request missing evidence or Hop-2")
            return self
        if self.reason_code is AdaptiveReasonCode.SUFFICIENT or self.missing_evidence is None:
            raise ValueError("insufficient decisions require missing evidence and a reason")
        if (
            self.next_hop_query is None
            and self.reason_code is not AdaptiveReasonCode.INSUFFICIENT_NO_GROUNDED_QUERY
        ):
            raise ValueError("an unavailable query requires the no-grounded-query reason")
        if (
            self.next_hop_query is not None
            and self.reason_code is AdaptiveReasonCode.INSUFFICIENT_NO_GROUNDED_QUERY
        ):
            raise ValueError("the no-grounded-query reason cannot contain a query")
        return self


class AdaptiveControlDecision(ContractModel):
    """Controller output with exact model, prompt, usage, and latency provenance."""

    output: AdaptiveControlOutput
    response_id: Identifier
    model: Identifier
    controller_version: Identifier
    prompt_version: Identifier
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    usage: LLMTokenUsage
    latency_ms: float = Field(ge=0, allow_inf_nan=False)


class AdaptiveBaselineTrace(ContractModel):
    """Complete execution trace for one LLM-controlled adaptive baseline question."""

    pipeline_version: Literal["adaptive-llm-two-hop-rag-v1"] = (
        "adaptive-llm-two-hop-rag-v1"
    )
    question_id: Identifier
    top_k: Annotated[int, Field(ge=1)]
    hop1: RetrievalResult
    controller_request: AdaptiveControlRequest
    controller_decision: AdaptiveControlDecision
    hop2: RetrievalResult | None
    merged_retrieval: HopMergeResult
    final_answer_request: AnswerGenerationRequest
    final_answer: FinalAnswer
    latency_ms: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_consistent_adaptive_flow(self) -> AdaptiveBaselineTrace:
        if self.hop1.query.question_id != self.question_id:
            raise ValueError("adaptive trace question ID must match Hop-1")
        if self.controller_request.question_id != self.question_id:
            raise ValueError("adaptive trace question ID must match controller request")
        if self.final_answer_request.question_id != self.question_id:
            raise ValueError("adaptive trace question ID must match final-answer request")
        if len(self.hop1.results) > self.top_k:
            raise ValueError("Hop-1 result count cannot exceed configured top_k")
        query = self.controller_decision.output.next_hop_query
        if query is None and self.hop2 is not None:
            raise ValueError("Hop-2 retrieval requires a controller query")
        if query is not None:
            if self.hop2 is None or self.hop2.query.text != query:
                raise ValueError("Hop-2 retrieval must execute the controller query")
            if len(self.hop2.results) > self.top_k:
                raise ValueError("Hop-2 result count cannot exceed configured top_k")
        max_document = len(self.controller_request.hop1_context.chunk_ids)
        if any(
            number > max_document
            for number in self.controller_decision.output.evidence_document_numbers
        ):
            raise ValueError("controller evidence document number exceeds Hop-1 context")
        merged_ids = tuple(result.chunk.chunk_id for result in self.merged_retrieval.results)
        if self.final_answer_request.context.chunk_ids != merged_ids:
            raise ValueError("final-answer context must match adaptive merged retrieval")
        return self
