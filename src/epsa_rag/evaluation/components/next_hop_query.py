"""Inference-only structural diagnostics for EPSA Components 01 through 09."""

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
    NextHopQueryGenerationError,
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
from epsa_rag.epsa.next_hop_query import (
    NextHopQuery,
    RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1,
)
from epsa_rag.epsa.question_analysis import QuestionAnalysis, RuleBasedQuestionAnalyzer
from epsa_rag.epsa.sufficiency_decision import RuleBasedSufficiencyEngineV1, SufficiencyDecision
from epsa_rag.evaluation.components.chunk_analyzer import InferenceRetrieval
from epsa_rag.instrumentation import (
    InMemoryInstrumentationSink,
    InstrumentationEvent,
    NoOpInstrumentationSink,
    TraceContext,
)


class NextHopQueryEvaluationConfig(ConfigModel):
    """Frozen execution settings for inference-only Component 09 diagnostics."""

    evaluator_version: Literal["next-hop-query-evaluation-v1"] = "next-hop-query-evaluation-v1"
    retrieval_depth: int = Field(default=10, ge=1)
    max_paths: int = Field(default=10, ge=0)
    retain_instrumentation_events: bool = True


class NextHopQueryTrace(ContractModel):
    """One Component 01-09 inference trace without retrieval execution or gold labels."""

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
    next_hop_query: NextHopQuery | None
    error_type: str | None = None
    latency_ms: float = Field(ge=0)


class NextHopQueryDiagnostics(ContractModel):
    """Operational Component 09 counts; no answer-quality or gold-label metrics."""

    queries: int = Field(ge=0)
    no_queries: int = Field(ge=0)
    sufficient_decisions: int = Field(ge=0)
    insufficient_decisions: int = Field(ge=0)
    queries_missing_selected_path_provenance: int = Field(ge=0)
    queries_with_graph_identity_mismatch: int = Field(ge=0)
    query_type_counts: dict[str, int]
    source_counts: dict[str, int]
    reason_code_counts: dict[str, int]
    confidence_sum: float = Field(ge=0)


class NextHopQueryRunSummary(ContractModel):
    """Immutable run-level Component 09 provenance and operational diagnostics."""

    schema_version: Literal["next-hop-query-run-v1"] = "next-hop-query-run-v1"
    run_id: Identifier
    configuration: NextHopQueryEvaluationConfig
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_ids: tuple[Identifier, ...]
    completed_questions: int = Field(ge=0)
    failed_questions: int = Field(ge=0)
    diagnostics: NextHopQueryDiagnostics


class NextHopQueryExportManifest(ContractModel):
    """Checksum manifest for one exclusive Component 09 structured export."""

    schema_version: Literal["next-hop-query-export-v1"] = "next-hop-query-export-v1"
    run_id: Identifier
    files: tuple[ArtifactFile, ...]


def evaluate_next_hop_queries(
    inputs: tuple[InferenceRetrieval, ...],
    *,
    run_id: str,
    config: NextHopQueryEvaluationConfig | None = None,
) -> tuple[
    NextHopQueryRunSummary,
    tuple[NextHopQueryTrace, ...],
    tuple[InstrumentationEvent, ...],
]:
    """Run Components 01-09 on inference-only Hop-1 inputs without retrieval execution."""

    if not inputs:
        raise ValueError("Component 09 evaluation requires inference inputs")
    selected = config or NextHopQueryEvaluationConfig()
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
    generator = RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1(instrumentation_sink=sink)
    traces: list[NextHopQueryTrace] = []
    for item in inputs:
        started = perf_counter()
        analysis: QuestionAnalysis | None = None
        candidates: list[CandidateChunkEvidence] = []
        units: list[EvidenceUnit] = []
        scored: tuple[ScoredEvidenceUnit, ...] = ()
        graph: EvidenceGraph | None = None
        paths: list[EvidencePath] = []
        decision: SufficiencyDecision | None = None
        context: PrunedContext | None = None
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
            query = generator.generate(
                analysis, decision, graph, paths, trace_context=trace_context
            )
        except (
            QuestionAnalysisError,
            ChunkAnalysisError,
            EvidenceUnitExtractionError,
            EvidenceScoringError,
            EvidenceGraphBuildError,
            EvidencePathSearchError,
            SufficiencyDecisionError,
            ContextPruningError,
            NextHopQueryGenerationError,
            ValueError,
        ) as error:
            traces.append(
                NextHopQueryTrace(
                    question_id=item.question.question_id,
                    status="failed",
                    question_analysis=analysis,
                    chunk_evidence=tuple(candidates),
                    evidence_units=tuple(units),
                    scored_evidence_units=scored,
                    evidence_graph=graph,
                    candidate_paths=tuple(paths),
                    decision=decision,
                    pruned_context=context,
                    next_hop_query=None,
                    error_type=type(error).__name__,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
        else:
            traces.append(
                NextHopQueryTrace(
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
                    next_hop_query=query,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
    completed = tuple(trace for trace in traces if trace.status == "completed")
    queries = tuple(trace.next_hop_query for trace in completed if trace.next_hop_query is not None)
    decisions = tuple(trace.decision for trace in completed if trace.decision is not None)
    query_types = Counter(query.query_type.value for query in queries)
    sources = Counter(query.source.value for query in queries)
    reasons = Counter(query.metadata.reason_code.value for query in queries)
    provenance_missing = sum(
        query.query is not None and query.metadata.selected_path_id is None for query in queries
    )
    graph_mismatches = sum(
        trace.evidence_graph is not None
        and query.metadata.source_graph != trace.evidence_graph.metadata
        for trace in completed
        for query in (trace.next_hop_query,)
        if query is not None
    )
    return (
        NextHopQueryRunSummary(
            run_id=run_id,
            configuration=selected,
            configuration_fingerprint=selected.fingerprint(),
            question_ids=tuple(item.question.question_id for item in inputs),
            completed_questions=len(completed),
            failed_questions=len(traces) - len(completed),
            diagnostics=NextHopQueryDiagnostics(
                queries=sum(query.query is not None for query in queries),
                no_queries=sum(query.query is None for query in queries),
                sufficient_decisions=sum(decision.sufficient for decision in decisions),
                insufficient_decisions=sum(not decision.sufficient for decision in decisions),
                queries_missing_selected_path_provenance=provenance_missing,
                queries_with_graph_identity_mismatch=graph_mismatches,
                query_type_counts=dict(sorted(query_types.items())),
                source_counts=dict(sorted(sources.items())),
                reason_code_counts=dict(sorted(reasons.items())),
                confidence_sum=round(sum(query.confidence for query in queries), 6),
            ),
        ),
        tuple(traces),
        sink.events if isinstance(sink, InMemoryInstrumentationSink) else (),
    )


def write_next_hop_query_evaluation(
    directory: Path,
    summary: NextHopQueryRunSummary,
    traces: tuple[NextHopQueryTrace, ...],
    events: tuple[InstrumentationEvent, ...],
) -> NextHopQueryExportManifest:
    """Write one exclusive checksummed Component 09 export without overwriting a run."""

    directory.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(directory / "run.json", summary)
    traces_file = write_jsonl_exclusive(
        directory / "traces.jsonl", traces, relative_path="traces.jsonl"
    )
    events_file = write_jsonl_exclusive(
        directory / "events.jsonl", events, relative_path="events.jsonl"
    )
    manifest = NextHopQueryExportManifest(
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
