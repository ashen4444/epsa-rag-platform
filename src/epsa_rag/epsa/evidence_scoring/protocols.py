"""Replaceable interfaces at the Component 04 boundary."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from epsa_rag.epsa.evidence_scoring.models import EvidenceFeatureVector, ScoredEvidenceUnit
from epsa_rag.epsa.evidence_units.models import EvidenceUnit
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.instrumentation import TraceContext


class EvidenceFeatureExtractorProtocol(Protocol):
    def extract(
        self, evidence_unit: EvidenceUnit, question_analysis: QuestionAnalysis
    ) -> EvidenceFeatureVector: ...


@runtime_checkable
class EvidenceScorerProtocol(Protocol):
    def score(
        self,
        evidence_unit: EvidenceUnit,
        question_analysis: QuestionAnalysis,
        *,
        trace_context: TraceContext | None = None,
    ) -> ScoredEvidenceUnit: ...
    def score_many(
        self,
        evidence_units: Iterable[EvidenceUnit],
        question_analysis: QuestionAnalysis,
        *,
        trace_context: TraceContext | None = None,
    ) -> tuple[ScoredEvidenceUnit, ...]: ...
