"""Replaceable interface at the Component 07 decision boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from epsa_rag.epsa.evidence_graph.models import EvidenceGraph
from epsa_rag.epsa.evidence_path_search.models import EvidencePath
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.epsa.sufficiency_decision.models import SufficiencyDecision
from epsa_rag.instrumentation import TraceContext


@runtime_checkable
class SufficiencyDecisionEngineProtocol(Protocol):
    def decide(
        self,
        question_analysis: QuestionAnalysis,
        evidence_graph: EvidenceGraph,
        evidence_paths: Sequence[EvidencePath],
        *,
        trace_context: TraceContext | None = None,
    ) -> SufficiencyDecision: ...
