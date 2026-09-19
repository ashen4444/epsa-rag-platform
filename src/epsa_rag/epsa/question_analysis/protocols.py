"""Replaceable interfaces for future Component 01 implementations/providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.instrumentation.context import TraceContext


@runtime_checkable
class QuestionAnalyzerProtocol(Protocol):
    """Stable downstream boundary for all Question Analyzer versions."""

    def analyze(
        self, question: str, *, trace_context: TraceContext | None = None
    ) -> QuestionAnalysis:
        """Analyze one nonblank natural-language question."""
