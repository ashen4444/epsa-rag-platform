"""Replaceable boundaries used by system-level orchestration."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from epsa_rag.core.models import RetrievalQuery
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
