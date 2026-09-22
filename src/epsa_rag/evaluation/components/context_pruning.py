"""Inference-only structural diagnostics for EPSA Components 01 through 08."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import Field

from epsa_rag.core.config import ConfigModel
from epsa_rag.core.exceptions import (
    ChunkAnalysisError,
    ContextPruningError,
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
from epsa_rag.epsa.context_pruning import PrunedContext, ResearchContextPrunerV1
from epsa_rag.epsa.evidence_graph import EvidenceGraph, EvidenceGraphBuilderV1
from epsa_rag.epsa.evidence_path_search import EvidencePath, EvidencePathSearcherV1
from epsa_rag.epsa.evidence_scoring import RuleBasedEvidenceScorerV1, ScoredEvidenceUnit
from epsa_rag.epsa.evidence_units import EvidenceUnit, RuleBasedV2EvidenceUnitExtractor
from epsa_rag.epsa.question_analysis import QuestionAnalysis, RuleBasedQuestionAnalyzer
from epsa_rag.epsa.sufficiency_decision import RuleBasedSufficiencyEngineV1, SufficiencyDecision
from epsa_rag.evaluation.components.chunk_analyzer import InferenceRetrieval
from epsa_rag.instrumentation import (
    InMemoryInstrumentationSink,
    InstrumentationEvent,
    NoOpInstrumentationSink,
    TraceContext,
)


class ContextPruningEvaluationConfig(ConfigModel):
    evaluator_version: Literal["context-pruning-evaluation-v1"] = "context-pruning-evaluation-v1"
    retrieval_depth: int = Field(default=10, ge=1)
    max_paths: int = Field(default=10, ge=0)
    retain_instrumentation_events: bool = True


class ContextPruningTrace(ContractModel):
    question_id: Identifier
    status: Literal["completed", "failed"]
    question_analysis: QuestionAnalysis | None
    chunk_evidence: tuple[CandidateChunkEvidence, ...]
    evidence_units: tuple[EvidenceUnit, ...]
    scored_evidence_units: tuple[ScoredEvidenceUnit, ...]
    evidence_graph: EvidenceGraph | None
    candidate_paths: tuple[EvidencePath, ...]
    decision: SufficiencyDecision | None
    pruned_context: PrunedContext | None
    error_type: str | None = None
    latency_ms: float = Field(ge=0)


class ContextPruningDiagnostics(ContractModel):
    contexts: int = Field(ge=0)
    sufficient_decisions: int = Field(ge=0)
    insufficient_decisions: int = Field(ge=0)
    empty_contexts: int = Field(ge=0)
    missing_requested_evidence_ids: int = Field(ge=0)
    selected_evidence_units: int = Field(ge=0)
    estimated_context_tokens: int = Field(ge=0)
    strategy_counts: dict[str, int]


class ContextPruningRunSummary(ContractModel):
    schema_version: Literal["context-pruning-run-v1"] = "context-pruning-run-v1"
    run_id: Identifier
    configuration: ContextPruningEvaluationConfig
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_ids: tuple[Identifier, ...]
    completed_questions: int = Field(ge=0)
    failed_questions: int = Field(ge=0)
    diagnostics: ContextPruningDiagnostics


class ContextPruningExportManifest(ContractModel):
    schema_version: Literal["context-pruning-export-v1"] = "context-pruning-export-v1"
    run_id: Identifier
    files: tuple[ArtifactFile, ...]


def evaluate_context_pruning(
    inputs: tuple[InferenceRetrieval, ...],
    *,
    run_id: str,
    config: ContextPruningEvaluationConfig | None = None,
) -> tuple[
    ContextPruningRunSummary,
    tuple[ContextPruningTrace, ...],
    tuple[InstrumentationEvent, ...],
]:
    """Run Components 01-08 on inference-only retrieval inputs."""

    if not inputs:
        raise ValueError("Component 08 evaluation requires inference inputs")
    selected = config or ContextPruningEvaluationConfig()
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
    pruner = ResearchContextPrunerV1(instrumentation_sink=sink)
    traces: list[ContextPruningTrace] = []
    for item in inputs:
        started = perf_counter()
        analysis: QuestionAnalysis | None = None
        candidates: list[CandidateChunkEvidence] = []
        units: list[EvidenceUnit] = []
        scored: tuple[ScoredEvidenceUnit, ...] = ()
        graph: EvidenceGraph | None = None
        paths: list[EvidencePath] = []
        decision: SufficiencyDecision | None = None
        try:
            trace_context = TraceContext.start(run_id=run_id, question_id=item.question.question_id)
            analysis = analyzer.analyze(item.question.text, trace_context=trace_context)
            ranked = item.chunks[: selected.retrieval_depth]
            candidates.extend(chunks.analyze_batch(ranked, analysis, trace_context=trace_context))
            for candidate, chunk in zip(candidates, ranked, strict=True):
                units.extend(
                    extractor.extract_from_chunk(
                        candidate, chunk, analysis, trace_context=trace_context
                    )
                )
            scored = scorer.score_many(units, analysis, trace_context=trace_context)
            graph = builder.build(analysis, scored, trace_context=trace_context)
            paths = searcher.search_paths(
                graph, analysis, selected.max_paths, trace_context=trace_context
            )
            decision = engine.decide(analysis, graph, paths, trace_context=trace_context)
            context = pruner.prune(decision, scored, trace_context=trace_context)
        except (
            QuestionAnalysisError,
            ChunkAnalysisError,
            EvidenceUnitExtractionError,
            EvidenceScoringError,
            EvidenceGraphBuildError,
            EvidencePathSearchError,
            SufficiencyDecisionError,
            ContextPruningError,
            ValueError,
        ) as error:
            traces.append(
                ContextPruningTrace(
                    question_id=item.question.question_id,
                    status="failed",
                    question_analysis=analysis,
                    chunk_evidence=tuple(candidates),
                    evidence_units=tuple(units),
                    scored_evidence_units=scored,
                    evidence_graph=graph,
                    candidate_paths=tuple(paths),
                    decision=decision,
                    pruned_context=None,
                    error_type=type(error).__name__,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
        else:
            traces.append(
                ContextPruningTrace(
                    question_id=item.question.question_id,
                    status="completed",
                    question_analysis=analysis,
                    chunk_evidence=tuple(candidates),
                    evidence_units=tuple(units),
                    scored_evidence_units=scored,
                    evidence_graph=graph,
                    candidate_paths=tuple(paths),
                    decision=decision,
                    pruned_context=context,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
    completed = tuple(trace for trace in traces if trace.status == "completed")
    contexts = tuple(
        trace.pruned_context for trace in completed if trace.pruned_context is not None
    )
    decisions = tuple(trace.decision for trace in completed if trace.decision is not None)
    strategies = Counter(context.pruning_strategy.value for context in contexts)
    return (
        ContextPruningRunSummary(
            run_id=run_id,
            configuration=selected,
            configuration_fingerprint=selected.fingerprint(),
            question_ids=tuple(item.question.question_id for item in inputs),
            completed_questions=len(completed),
            failed_questions=len(traces) - len(completed),
            diagnostics=ContextPruningDiagnostics(
                contexts=len(contexts),
                sufficient_decisions=sum(decision.sufficient for decision in decisions),
                insufficient_decisions=sum(not decision.sufficient for decision in decisions),
                empty_contexts=sum(not context.selected_evidence_unit_ids for context in contexts),
                missing_requested_evidence_ids=sum(
                    len(context.diagnostics.missing_requested_evidence_unit_ids)
                    for context in contexts
                ),
                selected_evidence_units=sum(
                    len(context.selected_evidence_unit_ids) for context in contexts
                ),
                estimated_context_tokens=sum(
                    context.estimated_context_tokens for context in contexts
                ),
                strategy_counts=dict(sorted(strategies.items())),
            ),
        ),
        tuple(traces),
        sink.events if isinstance(sink, InMemoryInstrumentationSink) else (),
    )


def write_context_pruning_evaluation(
    directory: Path,
    summary: ContextPruningRunSummary,
    traces: tuple[ContextPruningTrace, ...],
    events: tuple[InstrumentationEvent, ...],
) -> ContextPruningExportManifest:
    """Write a new checksummed Component 08 export without overwriting a run."""

    directory.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(directory / "run.json", summary)
    traces_file = write_jsonl_exclusive(
        directory / "traces.jsonl", traces, relative_path="traces.jsonl"
    )
    events_file = write_jsonl_exclusive(
        directory / "events.jsonl", events, relative_path="events.jsonl"
    )
    manifest = ContextPruningExportManifest(
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
