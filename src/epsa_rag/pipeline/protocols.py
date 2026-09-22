"""Replaceable boundaries used by system-level orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from epsa_rag.core.models import RankedParagraphChunk, RetrievalQuery
from epsa_rag.epsa.chunk_analysis.models import CandidateChunkEvidence
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.instrumentation import TraceContext
from epsa_rag.pipeline.epsa_models import EPSAPassResult
from epsa_rag.pipeline.models import (
    AdaptiveControlDecision,
    AdaptiveControlRequest,
    AnswerGenerationRequest,
    FinalAnswer,
)
from epsa_rag.retrieval.models import RetrievalResult


@runtime_checkable
class HybridRetrieverProtocol(Protocol):
    """Canonical retriever boundary shared by every evaluated system."""

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Return backend-independent ranked paragraph chunks."""


@runtime_checkable
class FinalAnswerGeneratorProtocol(Protocol):
    """Provider-neutral final-answer boundary shared by every evaluated system."""

    def generate(self, request: AnswerGenerationRequest) -> FinalAnswer:
        """Generate one answer from the original question and supplied context."""


@runtime_checkable
class AdaptiveRetrievalControllerProtocol(Protocol):
    """LLM controller boundary used only by the adaptive baseline."""

    def decide(self, request: AdaptiveControlRequest) -> AdaptiveControlDecision:
        """Decide sufficiency and optionally propose one grounded retrieval query."""


@runtime_checkable
class ContextualChunkAnalyzerProtocol(Protocol):
    """Component 02 boundary that preserves retrieved-set connectivity analysis."""

    def analyze_batch(
        self,
        chunks: tuple[object, ...],
        question_analysis: QuestionAnalysis | None,
        *,
        trace_context: TraceContext | None = None,
    ) -> tuple[CandidateChunkEvidence, ...]: ...


@runtime_checkable
class EPSAControllerProtocol(Protocol):
    """Execute one inspectable EPSA component pass over a ranked candidate set."""

    def run_pass(
        self,
        *,
        question: str,
        ranked_chunks: Sequence[RankedParagraphChunk],
        pass_number: Literal[1, 2],
        generate_next_hop: bool,
        trace_context: TraceContext | None = None,
    ) -> EPSAPassResult: ...
