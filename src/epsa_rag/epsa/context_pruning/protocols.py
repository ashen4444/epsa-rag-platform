"""Replaceable interface at the Component 08 context-pruning boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from epsa_rag.epsa.context_pruning.models import PrunedContext
from epsa_rag.epsa.evidence_scoring.models import ScoredEvidenceUnit
from epsa_rag.epsa.sufficiency_decision.models import SufficiencyDecision
from epsa_rag.instrumentation import TraceContext


@runtime_checkable
class ContextPrunerProtocol(Protocol):
    """Produce diagnostic context from Component 07's selected evidence only."""

    def prune(
        self,
        sufficiency_decision: SufficiencyDecision,
        scored_evidence_units: Sequence[ScoredEvidenceUnit],
        *,
        trace_context: TraceContext | None = None,
    ) -> PrunedContext: ...
