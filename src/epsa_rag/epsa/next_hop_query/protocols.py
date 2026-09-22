"""Replaceable interface at the Component 09 next-hop-query boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from epsa_rag.epsa.evidence_graph.models import EvidenceGraph
from epsa_rag.epsa.evidence_path_search.models import EvidencePath
from epsa_rag.epsa.next_hop_query.models import NextHopQuery
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.epsa.sufficiency_decision.models import SufficiencyDecision
from epsa_rag.instrumentation import TraceContext


@runtime_checkable
class NextHopQueryGeneratorProtocol(Protocol):
    """Stable Component 09 interface; retrieval execution remains external."""

    def generate(
        self,
        question_analysis: QuestionAnalysis,
        sufficiency_decision: SufficiencyDecision,
        evidence_graph: EvidenceGraph,
        evidence_paths: Sequence[EvidencePath],
        *,
        trace_context: TraceContext | None = None,
    ) -> NextHopQuery:
        """Return a deterministic retrieval proposal or explicit no-query result."""
