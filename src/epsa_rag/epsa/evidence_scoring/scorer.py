"""Deterministic, observable Component 04 score aggregation."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

from pydantic import ValidationError

from epsa_rag.core.exceptions import EvidenceScoringError
from epsa_rag.epsa.evidence_scoring.config import EvidenceScorerV1Config
from epsa_rag.epsa.evidence_scoring.features import ResearchV1EvidenceFeatureExtractor
from epsa_rag.epsa.evidence_scoring.models import EvidenceScoreMetadata, ScoredEvidenceUnit
from epsa_rag.epsa.evidence_units.models import EvidenceUnit
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.instrumentation import InstrumentationEvent, InstrumentationSink, TraceContext


class RuleBasedEvidenceScorerV1:
    """Exact, non-probabilistic ``research_v1`` evidence-quality scorer."""

    def __init__(
        self,
        *,
        config: EvidenceScorerV1Config | None = None,
        instrumentation_sink: InstrumentationSink | None = None,
    ) -> None:
        self._config = config or EvidenceScorerV1Config()
        self._sink = instrumentation_sink
        self._features = ResearchV1EvidenceFeatureExtractor(self._config)

    @property
    def config(self) -> EvidenceScorerV1Config:
        return self._config

    def score(
        self,
        evidence_unit: EvidenceUnit,
        question_analysis: QuestionAnalysis,
        *,
        trace_context: TraceContext | None = None,
    ) -> ScoredEvidenceUnit:
        started = perf_counter()
        try:
            if not isinstance(evidence_unit, EvidenceUnit):
                raise EvidenceScoringError("evidence unit must use Component 03 contract")
            if not isinstance(question_analysis, QuestionAnalysis):
                raise EvidenceScoringError("question analysis must use Component 01 contract")
            features = self._features.extract(evidence_unit, question_analysis)
            raw = (
                self._config.entity_match_weight * features.entity_match_score
                + self._config.relation_match_weight * features.relation_match_score
                + self._config.answer_type_match_weight * features.answer_type_match_score
                + self._config.token_overlap_weight * features.token_overlap_score
                + self._config.title_match_weight * features.title_match_score
                + self._config.retrieval_component_weight * features.retrieval_score_component
                + self._config.bridge_entity_weight * features.bridge_entity_score
                - features.noise_penalty
            )
            result = ScoredEvidenceUnit(
                evidence_unit=evidence_unit,
                final_score=round(max(0.0, min(1.0, raw)), 6),
                score_breakdown=features,
                metadata=EvidenceScoreMetadata(
                    configuration_fingerprint=self._config.fingerprint(), raw_weighted_score=raw
                ),
            )
        except (EvidenceScoringError, ValidationError, ValueError) as error:
            wrapped = (
                error
                if isinstance(error, EvidenceScoringError)
                else EvidenceScoringError(str(error))
            )
            self._emit_failed(evidence_unit, wrapped, trace_context, started)
            if wrapped is error:
                raise
            raise wrapped from error
        except Exception as error:
            wrapped = EvidenceScoringError("evidence scoring failed")
            self._emit_failed(evidence_unit, wrapped, trace_context, started)
            raise wrapped from error
        self._emit_completed(result, trace_context, started)
        return result

    def score_many(
        self,
        evidence_units: Iterable[EvidenceUnit],
        question_analysis: QuestionAnalysis,
        *,
        trace_context: TraceContext | None = None,
    ) -> tuple[ScoredEvidenceUnit, ...]:
        return tuple(
            self.score(unit, question_analysis, trace_context=trace_context)
            for unit in evidence_units
        )

    def rank(
        self,
        evidence_units: Iterable[EvidenceUnit],
        question_analysis: QuestionAnalysis,
        *,
        trace_context: TraceContext | None = None,
    ) -> tuple[ScoredEvidenceUnit, ...]:
        return tuple(
            sorted(
                self.score_many(evidence_units, question_analysis, trace_context=trace_context),
                key=lambda item: item.final_score,
                reverse=True,
            )
        )

    def _emit_completed(
        self, scored: ScoredEvidenceUnit, context: TraceContext | None, started: float
    ) -> None:
        if self._sink is not None and context is not None:
            self._sink.emit(
                InstrumentationEvent(
                    context=context,
                    event_type="epsa.evidence_scoring.completed",
                    source="epsa.evidence_scorer",
                    source_version=self._config.mode,
                    payload={
                        "scored_evidence": scored.model_dump(mode="json"),
                        "latency_ms": (perf_counter() - started) * 1000,
                    },
                )
            )

    def _emit_failed(
        self,
        unit: object,
        error: EvidenceScoringError,
        context: TraceContext | None,
        started: float,
    ) -> None:
        if self._sink is not None and context is not None:
            self._sink.emit(
                InstrumentationEvent(
                    context=context,
                    event_type="epsa.evidence_scoring.failed",
                    source="epsa.evidence_scorer",
                    source_version=self._config.mode,
                    payload={
                        "evidence_unit_id": str(getattr(unit, "evidence_unit_id", "")),
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "latency_ms": (perf_counter() - started) * 1000,
                    },
                )
            )
