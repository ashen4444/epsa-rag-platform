"""Replaceable interface at the Component 05 graph-building boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from epsa_rag.epsa.evidence_graph.models import EvidenceGraph
from epsa_rag.epsa.evidence_scoring.models import ScoredEvidenceUnit
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.instrumentation import TraceContext


@runtime_checkable
class EvidenceGraphBuilderProtocol(Protocol):
    def build(
        self,
        question_analysis: QuestionAnalysis,
        scored_evidence_units: Sequence[ScoredEvidenceUnit],
        *,
        trace_context: TraceContext | None = None,
    ) -> EvidenceGraph: ...
