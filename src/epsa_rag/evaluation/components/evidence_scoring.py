"""Development-only inference diagnostics for EPSA Components 01 through 04."""

from __future__ import annotations

from collections import Counter
from time import perf_counter
from typing import Literal

from pydantic import Field

from epsa_rag.core.config import ConfigModel
from epsa_rag.core.exceptions import (
    ChunkAnalysisError,
    EvidenceScoringError,
    EvidenceUnitExtractionError,
    QuestionAnalysisError,
)
from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel
from epsa_rag.epsa.chunk_analysis import RuleBasedV2CandidateChunkEvidenceAnalyzer
from epsa_rag.epsa.chunk_analysis.models import CandidateChunkEvidence
from epsa_rag.epsa.evidence_scoring import (
    EvidenceFeatureVector,
    RuleBasedEvidenceScorerV1,
    ScoredEvidenceUnit,
)
from epsa_rag.epsa.evidence_units import EvidenceUnit, RuleBasedV2EvidenceUnitExtractor
from epsa_rag.epsa.question_analysis import QuestionAnalysis, RuleBasedQuestionAnalyzer
from epsa_rag.evaluation.components.chunk_analyzer import InferenceRetrieval
from epsa_rag.instrumentation import InMemoryInstrumentationSink, InstrumentationEvent, TraceContext


class EvidenceScoringEvaluationConfig(ConfigModel):
    evaluator_version: Literal["evidence-scoring-evaluation-v1"] = "evidence-scoring-evaluation-v1"
    retrieval_depth: int = Field(default=10, ge=1)


class EvidenceScoringTrace(ContractModel):
    question_id: Identifier
    status: Literal["completed", "failed"]
    question_analysis: QuestionAnalysis | None
    chunk_evidence: tuple[CandidateChunkEvidence, ...]
    evidence_units: tuple[EvidenceUnit, ...]
    scored_evidence_units: tuple[ScoredEvidenceUnit, ...]
    error_type: str | None = None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)


class EvidenceScoringDiagnostics(ContractModel):
    evidence_units: int = Field(ge=0)
    scored_evidence_units: int = Field(ge=0)
    zero_scores: int = Field(ge=0)
    clipped_high_scores: int = Field(ge=0)
    feature_means: dict[str, float]
    final_score_histogram: dict[str, int]


class EvidenceScoringRunSummary(ContractModel):
    schema_version: Literal["evidence-scoring-run-v1"] = "evidence-scoring-run-v1"
    run_id: Identifier
    configuration: EvidenceScoringEvaluationConfig
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_ids: tuple[Identifier, ...]
    completed_questions: int = Field(ge=0)
    failed_questions: int = Field(ge=0)
    diagnostics: EvidenceScoringDiagnostics


def evaluate_evidence_scoring(
    inputs: tuple[InferenceRetrieval, ...],
    *,
    run_id: str,
    config: EvidenceScoringEvaluationConfig | None = None,
) -> tuple[
    EvidenceScoringRunSummary, tuple[EvidenceScoringTrace, ...], tuple[InstrumentationEvent, ...]
]:
    """Run only inference-visible Components 01 → 04; no gold labels are accepted."""
    if not inputs:
        raise ValueError("Component 04 evaluation requires inference inputs")
    selected = config or EvidenceScoringEvaluationConfig()
    sink = InMemoryInstrumentationSink()
    questions = RuleBasedQuestionAnalyzer(instrumentation_sink=sink)
    chunks = RuleBasedV2CandidateChunkEvidenceAnalyzer(instrumentation_sink=sink)
    extractor = RuleBasedV2EvidenceUnitExtractor(instrumentation_sink=sink)
    scorer = RuleBasedEvidenceScorerV1(instrumentation_sink=sink)
    traces: list[EvidenceScoringTrace] = []
    for item in inputs:
        started = perf_counter()
        context = TraceContext.start(run_id=run_id, question_id=item.question.question_id)
        analysis: QuestionAnalysis | None = None
        candidates: list[CandidateChunkEvidence] = []
        units: list[EvidenceUnit] = []
        try:
            analysis = questions.analyze(item.question.text, trace_context=context)
            ranked = item.chunks[: selected.retrieval_depth]
            candidates.extend(chunks.analyze_batch(ranked, analysis, trace_context=context))
            for candidate, chunk in zip(candidates, ranked, strict=True):
                units.extend(
                    extractor.extract_from_chunk(candidate, chunk, analysis, trace_context=context)
                )
            scored = scorer.score_many(units, analysis, trace_context=context)
        except (
            QuestionAnalysisError,
            ChunkAnalysisError,
            EvidenceUnitExtractionError,
            EvidenceScoringError,
            ValueError,
        ) as error:
            traces.append(
                EvidenceScoringTrace(
                    question_id=item.question.question_id,
                    status="failed",
                    question_analysis=analysis,
                    chunk_evidence=tuple(candidates),
                    evidence_units=tuple(units),
                    scored_evidence_units=(),
                    error_type=type(error).__name__,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
        else:
            traces.append(
                EvidenceScoringTrace(
                    question_id=item.question.question_id,
                    status="completed",
                    question_analysis=analysis,
                    chunk_evidence=tuple(candidates),
                    evidence_units=tuple(units),
                    scored_evidence_units=scored,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
    scored_units = [item for trace in traces for item in trace.scored_evidence_units]
    keys = tuple(EvidenceFeatureVector.model_fields)
    sums = Counter({key: 0.0 for key in keys})
    for scored_unit in scored_units:
        sums.update(scored_unit.score_breakdown.model_dump())
    histogram = Counter(
        "zero"
        if scored_unit.final_score == 0
        else "one"
        if scored_unit.final_score == 1
        else "interior"
        for scored_unit in scored_units
    )
    summary = EvidenceScoringRunSummary(
        run_id=run_id,
        configuration=selected,
        configuration_fingerprint=selected.fingerprint(),
        question_ids=tuple(item.question.question_id for item in inputs),
        completed_questions=sum(trace.status == "completed" for trace in traces),
        failed_questions=sum(trace.status == "failed" for trace in traces),
        diagnostics=EvidenceScoringDiagnostics(
            evidence_units=sum(len(trace.evidence_units) for trace in traces),
            scored_evidence_units=len(scored_units),
            zero_scores=histogram["zero"],
            clipped_high_scores=histogram["one"],
            feature_means={
                key: round(sums[key] / len(scored_units), 6) if scored_units else 0.0
                for key in keys
            },
            final_score_histogram=dict(sorted(histogram.items())),
        ),
    )
    return summary, tuple(traces), sink.events
