"""Replaceable interface at the Component 06 path-search boundary."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from epsa_rag.epsa.evidence_graph.models import EvidenceGraph
from epsa_rag.epsa.evidence_path_search.models import EvidencePath
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.instrumentation import TraceContext


@runtime_checkable
class EvidencePathSearcherProtocol(Protocol):
    def search_paths(
        self,
        evidence_graph: EvidenceGraph,
        question_analysis: QuestionAnalysis,
        max_paths: int = 10,
        *,
        trace_context: TraceContext | None = None,
    ) -> list[EvidencePath]: ...
