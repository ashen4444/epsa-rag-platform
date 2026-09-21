"""Inference-only structural diagnostics for EPSA Components 01 through 07."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import Field

from epsa_rag.core.config import ConfigModel
from epsa_rag.core.exceptions import (
    ChunkAnalysisError,
    EvidenceGraphBuildError,
    EvidencePathSearchError,
    EvidenceScoringError,
    EvidenceUnitExtractionError,
    QuestionAnalysisError,
    SufficiencyDecisionError,
)
from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel
from epsa_rag.data.io import sha256_file, write_json_exclusive, write_jsonl_exclusive
from epsa_rag.data.manifests import ArtifactFile
from epsa_rag.epsa.chunk_analysis import RuleBasedV2CandidateChunkEvidenceAnalyzer
from epsa_rag.epsa.chunk_analysis.models import CandidateChunkEvidence
from epsa_rag.epsa.evidence_graph import EvidenceGraph, EvidenceGraphBuilderV1
from epsa_rag.epsa.evidence_path_search import EvidencePath, EvidencePathSearcherV1
from epsa_rag.epsa.evidence_scoring import RuleBasedEvidenceScorerV1, ScoredEvidenceUnit
from epsa_rag.epsa.evidence_units import EvidenceUnit, RuleBasedV2EvidenceUnitExtractor
from epsa_rag.epsa.question_analysis import QuestionAnalysis, RuleBasedQuestionAnalyzer
from epsa_rag.epsa.sufficiency_decision import (
    RuleBasedSufficiencyEngineV1,
    SufficiencyDecision,
)
from epsa_rag.evaluation.components.chunk_analyzer import InferenceRetrieval
from epsa_rag.instrumentation import (
    InMemoryInstrumentationSink,
    InstrumentationEvent,
    NoOpInstrumentationSink,
    TraceContext,
)


class SufficiencyDecisionEvaluationConfig(ConfigModel):
    evaluator_version: Literal["sufficiency-decision-evaluation-v1"] = (
        "sufficiency-decision-evaluation-v1"
    )
    retrieval_depth: int = Field(default=10, ge=1)
    max_paths: int = Field(default=10, ge=0)
    retain_instrumentation_events: bool = True


class SufficiencyDecisionTrace(ContractModel):
    question_id: Identifier
    status: Literal["completed", "failed"]
    question_analysis: QuestionAnalysis | None
    chunk_evidence: tuple[CandidateChunkEvidence, ...]
    evidence_units: tuple[EvidenceUnit, ...]
    scored_evidence_units: tuple[ScoredEvidenceUnit, ...]
    evidence_graph: EvidenceGraph | None
    candidate_paths: tuple[EvidencePath, ...]
    decision: SufficiencyDecision | None
    error_type: str | None = None
    latency_ms: float = Field(ge=0)


class SufficiencyDecisionDiagnostics(ContractModel):
    decisions: int = Field(ge=0)
    sufficient_decisions: int = Field(ge=0)
    insufficient_decisions: int = Field(ge=0)
    decisions_missing_path_provenance: int = Field(ge=0)
    decisions_with_graph_identity_mismatch: int = Field(ge=0)
    reason_code_counts: dict[str, int]


class SufficiencyDecisionRunSummary(ContractModel):
    schema_version: Literal["sufficiency-decision-run-v1"] = "sufficiency-decision-run-v1"
    run_id: Identifier
    configuration: SufficiencyDecisionEvaluationConfig
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_ids: tuple[Identifier, ...]
    completed_questions: int = Field(ge=0)
    failed_questions: int = Field(ge=0)
    diagnostics: SufficiencyDecisionDiagnostics


class SufficiencyDecisionExportManifest(ContractModel):
    """Checksummed, immutable inference-only Component 07 export inventory."""

    schema_version: Literal["sufficiency-decision-export-v1"] = "sufficiency-decision-export-v1"
    run_id: Identifier
    files: tuple[ArtifactFile, ...]


def evaluate_sufficiency_decisions(
    inputs: tuple[InferenceRetrieval, ...],
    *,
    run_id: str,
    config: SufficiencyDecisionEvaluationConfig | None = None,
) -> tuple[
    SufficiencyDecisionRunSummary,
    tuple[SufficiencyDecisionTrace, ...],
    tuple[InstrumentationEvent, ...],
]:
    """Run Components 01-07 over retrieval outputs, without any gold fields."""

    if not inputs:
        raise ValueError("Component 07 evaluation requires inference inputs")
    selected = config or SufficiencyDecisionEvaluationConfig()
    sink = (
        InMemoryInstrumentationSink()
        if selected.retain_instrumentation_events
        else NoOpInstrumentationSink()
    )
    analyzer = RuleBasedQuestionAnalyzer(instrumentation_sink=sink)
    chunks = RuleBasedV2CandidateChunkEvidenceAnalyzer(instrumentation_sink=sink)
    extractor = RuleBasedV2EvidenceUnitExtractor(instrumentation_sink=sink)
    scorer = RuleBasedEvidenceScorerV1(instrumentation_sink=sink)
    builder = EvidenceGraphBuilderV1(instrumentation_sink=sink)
    searcher = EvidencePathSearcherV1(instrumentation_sink=sink)
    engine = RuleBasedSufficiencyEngineV1(instrumentation_sink=sink)
    traces: list[SufficiencyDecisionTrace] = []
    for item in inputs:
        started = perf_counter()
        analysis: QuestionAnalysis | None = None
        candidates: list[CandidateChunkEvidence] = []
        units: list[EvidenceUnit] = []
        scored: tuple[ScoredEvidenceUnit, ...] = ()
        graph: EvidenceGraph | None = None
        paths: list[EvidencePath] = []
        try:
            context = TraceContext.start(run_id=run_id, question_id=item.question.question_id)
            analysis = analyzer.analyze(item.question.text, trace_context=context)
            ranked = item.chunks[: selected.retrieval_depth]
            candidates.extend(chunks.analyze_batch(ranked, analysis, trace_context=context))
            for candidate, chunk in zip(candidates, ranked, strict=True):
                units.extend(
                    extractor.extract_from_chunk(candidate, chunk, analysis, trace_context=context)
                )
            scored = scorer.score_many(units, analysis, trace_context=context)
            graph = builder.build(analysis, scored, trace_context=context)
            paths = searcher.search_paths(
                graph, analysis, selected.max_paths, trace_context=context
            )
            decision = engine.decide(analysis, graph, paths, trace_context=context)
        except (
            QuestionAnalysisError,
            ChunkAnalysisError,
            EvidenceUnitExtractionError,
            EvidenceScoringError,
            EvidenceGraphBuildError,
            EvidencePathSearchError,
            SufficiencyDecisionError,
            ValueError,
        ) as error:
            traces.append(
                SufficiencyDecisionTrace(
                    question_id=item.question.question_id,
                    status="failed",
                    question_analysis=analysis,
                    chunk_evidence=tuple(candidates),
                    evidence_units=tuple(units),
                    scored_evidence_units=scored,
                    evidence_graph=graph,
                    candidate_paths=tuple(paths),
                    decision=None,
                    error_type=type(error).__name__,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
        else:
            traces.append(
                SufficiencyDecisionTrace(
                    question_id=item.question.question_id,
                    status="completed",
                    question_analysis=analysis,
                    chunk_evidence=tuple(candidates),
                    evidence_units=tuple(units),
                    scored_evidence_units=scored,
                    evidence_graph=graph,
                    candidate_paths=tuple(paths),
                    decision=decision,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
    completed = tuple(trace for trace in traces if trace.status == "completed")
    decisions = tuple(trace.decision for trace in completed if trace.decision is not None)
    reasons = Counter(decision.decision_reason.value for decision in decisions)
    provenance_missing = sum(
        not decision.candidate_paths_considered
        and decision.decision_reason.value != "no_candidate_paths"
        for decision in decisions
    )
    identity_mismatches = sum(
        trace.evidence_graph is not None
        and decision.metadata.source_graph != trace.evidence_graph.metadata
        for trace in completed
        for decision in (trace.decision,)
        if decision is not None
    )
    return (
        SufficiencyDecisionRunSummary(
            run_id=run_id,
            configuration=selected,
            configuration_fingerprint=selected.fingerprint(),
            question_ids=tuple(item.question.question_id for item in inputs),
            completed_questions=len(completed),
            failed_questions=len(traces) - len(completed),
            diagnostics=SufficiencyDecisionDiagnostics(
                decisions=len(decisions),
                sufficient_decisions=sum(decision.sufficient for decision in decisions),
                insufficient_decisions=sum(not decision.sufficient for decision in decisions),
                decisions_missing_path_provenance=provenance_missing,
                decisions_with_graph_identity_mismatch=identity_mismatches,
                reason_code_counts=dict(sorted(reasons.items())),
            ),
        ),
        tuple(traces),
        sink.events if isinstance(sink, InMemoryInstrumentationSink) else (),
    )


def write_sufficiency_decision_evaluation(
    directory: Path,
    summary: SufficiencyDecisionRunSummary,
    traces: tuple[SufficiencyDecisionTrace, ...],
    events: tuple[InstrumentationEvent, ...],
) -> SufficiencyDecisionExportManifest:
    """Write a new, checksummed structured export without overwriting a run."""

    directory.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(directory / "run.json", summary)
    traces_file = write_jsonl_exclusive(
        directory / "traces.jsonl", traces, relative_path="traces.jsonl"
    )
    events_file = write_jsonl_exclusive(
        directory / "events.jsonl", events, relative_path="events.jsonl"
    )
    manifest = SufficiencyDecisionExportManifest(
        run_id=summary.run_id,
        files=(
            ArtifactFile(
                relative_path="run.json",
                sha256=sha256_file(directory / "run.json"),
                byte_count=(directory / "run.json").stat().st_size,
                record_count=1,
            ),
            traces_file,
            events_file,
        ),
    )
    write_json_exclusive(directory / "manifest.json", manifest)
    return manifest
