"""The deterministic, non-LLM implementation of EPSA Component 01."""

from __future__ import annotations

from time import perf_counter

from epsa_rag.core.exceptions import QuestionAnalysisError
from epsa_rag.epsa.question_analysis.answer_type import infer_answer_type
from epsa_rag.epsa.question_analysis.comparison_extractor import analyze_comparison
from epsa_rag.epsa.question_analysis.config import QuestionAnalyzerConfig
from epsa_rag.epsa.question_analysis.entity_extractor import extract_seed_entities
from epsa_rag.epsa.question_analysis.models import (
    AnalysisMetadata,
    AnswerType,
    AnswerTypeCandidate,
    QuestionAnalysis,
    QuestionType,
)
from epsa_rag.epsa.question_analysis.normalizer import normalize_matching_text, normalize_question
from epsa_rag.epsa.question_analysis.question_type import infer_question_type
from epsa_rag.epsa.question_analysis.relation_extractor import extract_relation_hints
from epsa_rag.instrumentation import InstrumentationEvent, InstrumentationSink, TraceContext


class RuleBasedQuestionAnalyzer:
    """Reproducible regex/capitalization analyzer retained for thesis compatibility."""

    def __init__(
        self,
        *,
        config: QuestionAnalyzerConfig | None = None,
        instrumentation_sink: InstrumentationSink | None = None,
    ) -> None:
        self._config = config or QuestionAnalyzerConfig()
        self._instrumentation_sink = instrumentation_sink

    @property
    def config(self) -> QuestionAnalyzerConfig:
        """Expose the immutable exact configuration used by this analyzer."""

        return self._config

    def analyze(
        self, question: str, *, trace_context: TraceContext | None = None
    ) -> QuestionAnalysis:
        """Return deterministic structured requirements or raise an explicit component error."""

        started = perf_counter()
        try:
            if not isinstance(question, str):
                raise QuestionAnalysisError("question must be a string")
            if not question.strip():
                raise QuestionAnalysisError("question must be a non-empty string")
            matching_question = normalize_matching_text(question)
            normalized_question = normalize_question(question)
            expected_answer_type = infer_answer_type(normalized_question)
            seed_entities = extract_seed_entities(matching_question, self._config)
            relation_hints = extract_relation_hints(
                normalized_question, self._config, surface_question=matching_question
            )
            comparison = analyze_comparison(matching_question, seed_entities, self._config)
            question_type = infer_question_type(
                normalized_question,
                expected_answer_type=expected_answer_type,
                relation_hint_count=len(relation_hints),
                comparison_intent=comparison is not None,
            )
            if (
                question_type is QuestionType.COMPARISON
                and expected_answer_type is AnswerType.BOOLEAN
            ):
                # An auxiliary-led scalar comparison returns the winning target, never yes/no.
                expected_answer_type = AnswerType.ENTITY
            comparison_targets = comparison if comparison is not None else ()
            answer_score = (
                self._config.unknown_answer_type_score
                if expected_answer_type.value == "UNKNOWN"
                else self._config.known_answer_type_score
            )
            analysis = QuestionAnalysis(
                raw_question=question,
                normalized_question=normalized_question,
                question_type=question_type,
                expected_answer_type=expected_answer_type,
                seed_entities=seed_entities,
                required_relation_hints=relation_hints,
                comparison_targets=comparison_targets,
                answer_type_candidates=(
                    AnswerTypeCandidate(
                        answer_type=expected_answer_type,
                        text=expected_answer_type.value,
                        confidence=answer_score,
                    ),
                ),
                metadata=AnalysisMetadata(configuration_fingerprint=self._config.fingerprint()),
            )
        except QuestionAnalysisError as error:
            self._emit_failure(question, error, trace_context, started)
            raise
        except Exception as error:
            wrapped = QuestionAnalysisError("rule-based question analysis failed")
            self._emit_failure(question, wrapped, trace_context, started)
            raise wrapped from error
        self._emit_completed(analysis, trace_context, started)
        return analysis

    def _emit_completed(
        self,
        analysis: QuestionAnalysis,
        trace_context: TraceContext | None,
        started: float,
    ) -> None:
        if self._instrumentation_sink is None or trace_context is None:
            return
        self._instrumentation_sink.emit(
            InstrumentationEvent(
                context=trace_context,
                event_type="epsa.question_analysis.completed",
                source="epsa.question_analyzer",
                source_version=self._config.version,
                payload={
                    "analysis": analysis.model_dump(mode="json"),
                    "latency_ms": (perf_counter() - started) * 1000,
                },
            )
        )

    def _emit_failure(
        self,
        question: object,
        error: QuestionAnalysisError,
        trace_context: TraceContext | None,
        started: float,
    ) -> None:
        if self._instrumentation_sink is None or trace_context is None:
            return
        self._instrumentation_sink.emit(
            InstrumentationEvent(
                context=trace_context,
                event_type="epsa.question_analysis.failed",
                source="epsa.question_analyzer",
                source_version=self._config.version,
                payload={
                    "raw_question": question if isinstance(question, str) else repr(question),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "latency_ms": (perf_counter() - started) * 1000,
                },
            )
        )
