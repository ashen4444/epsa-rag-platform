"""Replaceable Component 02 boundary for later versions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from epsa_rag.epsa.chunk_analysis.models import CandidateChunkEvidence
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.instrumentation.context import TraceContext


@runtime_checkable
class ChunkAnalyzerProtocol(Protocol):
    def analyze(
        self,
        chunk: object,
        question_analysis: QuestionAnalysis | None = None,
        retrieval_rank: int | None = None,
        retrieval_score: float | None = None,
        *,
        trace_context: TraceContext | None = None,
    ) -> CandidateChunkEvidence: ...
