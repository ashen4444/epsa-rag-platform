"""Replaceable Component 03 boundary."""

from __future__ import annotations

from typing import Protocol

from epsa_rag.epsa.chunk_analysis.models import CandidateChunkEvidence
from epsa_rag.epsa.evidence_units.models import EvidenceUnit
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.instrumentation import TraceContext


class EvidenceUnitExtractorProtocol(Protocol):
    def extract_from_chunk(
        self,
        candidate_evidence: CandidateChunkEvidence,
        chunk: object,
        question_analysis: QuestionAnalysis,
        *,
        trace_context: TraceContext | None = None,
    ) -> tuple[EvidenceUnit, ...]: ...
