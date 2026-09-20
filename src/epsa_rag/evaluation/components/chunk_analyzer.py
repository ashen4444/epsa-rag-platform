"""Inference-only Component 02 development diagnostics over frozen retrieval traces."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import Field

from epsa_rag.core.config import ConfigModel
from epsa_rag.core.exceptions import (
    ChunkAnalysisError,
    FrozenArtifactError,
    QuestionAnalysisError,
    SourceValidationError,
)
from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel, RankedParagraphChunk
from epsa_rag.data.io import read_jsonl, sha256_file, write_json_exclusive, write_jsonl_exclusive
from epsa_rag.data.manifests import ArtifactFile, DatasetManifest
from epsa_rag.data.models import QuestionInput
from epsa_rag.epsa.chunk_analysis import (
    CandidateChunkEvidence,
    ChunkAnalyzerConfig,
    RuleBasedCandidateChunkEvidenceAnalyzer,
    RuleBasedV2CandidateChunkEvidenceAnalyzer,
    RuleBasedV2ChunkAnalyzerConfig,
)
from epsa_rag.epsa.chunk_analysis.models import ChunkAnalysisMetadataV2
from epsa_rag.epsa.question_analysis import QuestionAnalysis, RuleBasedQuestionAnalyzer
from epsa_rag.epsa.question_analysis.models import AnswerType, QuestionType
from epsa_rag.evaluation.components.question_analyzer import load_inference_questions
from epsa_rag.evaluation.retrieval.exports import load_export
from epsa_rag.instrumentation import InMemoryInstrumentationSink, InstrumentationEvent, TraceContext


class ChunkAnalyzerEvaluationConfig(ConfigModel):
    evaluator_version: Literal["chunk-analyzer-evaluation-v1"] = "chunk-analyzer-evaluation-v1"
    analyzer: ChunkAnalyzerConfig | RuleBasedV2ChunkAnalyzerConfig = Field(
        default_factory=ChunkAnalyzerConfig
    )
    retrieval_depth: int = Field(default=10, ge=1)


class InferenceRetrieval(ContractModel):
    """Only fields passed from a retrieval export into Component 02 evaluation."""

    question: QuestionInput
    chunks: tuple[RankedParagraphChunk, ...]
    retriever_version: Identifier


class ChunkAnalysisTrace(ContractModel):
    question: QuestionInput
    status: Literal["completed", "failed"]
    question_analysis: QuestionAnalysis | None
    evidence: tuple[CandidateChunkEvidence, ...]
    error_type: str | None = None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)


class ChunkAnalyzerDiagnostics(ContractModel):
    """Descriptive counts only; no Component 02 gold labels are present."""

    analyzed_chunks: int = Field(ge=0)
    chunks_with_question_entity_overlap: int = Field(ge=0)
    chunks_with_title_match: int = Field(ge=0)
    chunks_with_bridge_candidates: int = Field(ge=0)
    chunks_without_relation_hints: int = Field(ge=0)
    answer_candidate_type_counts: dict[str, int]
    bridge_decision_reason_counts: dict[str, int]
    relation_grounding_type_counts: dict[str, int] = Field(default_factory=dict)
    bridge_questions: int = Field(default=0, ge=0)
    bridge_questions_with_candidates: int = Field(default=0, ge=0)
    span_integrity_failures: int | None = Field(default=None, ge=0)
    analyzer_fingerprint_mismatches: int = Field(default=0, ge=0)


def _v2_span_failures(evidence: CandidateChunkEvidence) -> int:
    """Count broken source slices in one v2 evidence record."""

    body = evidence.paragraph_text
    failures = 0
    for item in (*evidence.entities, *evidence.answer_type_candidates):
        scope = getattr(item, "span_scope", None)
        if scope == "doc_title":
            valid = (
                item.start_char is None and item.end_char is None
                and item.text == evidence.doc_title.strip()
            )
        elif scope == "paragraph_text":
            valid = (
                item.start_char is not None and item.end_char is not None
                and body[item.start_char:item.end_char] == item.text
            )
        else:
            valid = False
        failures += not valid
    for hint in evidence.relation_hints:
        failures += not (
            getattr(hint, "span_scope", None) == "paragraph_text"
            and body[hint.start_char:hint.end_char] == hint.matched_text
        )
    return failures


class ChunkAnalyzerRunSummary(ContractModel):
    schema_version: Literal["chunk-analyzer-run-v1"] = "chunk-analyzer-run-v1"
    run_id: Identifier
    research_role: Literal["development"] = "development"
    dataset_version: Identifier
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_version: Identifier
    retrieval_run_id: Identifier
    retriever_version: Identifier
    git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    git_dirty: bool
    source_sha256: dict[str, str]
    runtime: dict[str, str]
    question_ids: tuple[Identifier, ...]
    configuration: ChunkAnalyzerEvaluationConfig
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_questions: int = Field(ge=0)
    failed_questions: int = Field(ge=0)
    diagnostics: ChunkAnalyzerDiagnostics


class ChunkAnalyzerExportManifest(ContractModel):
    schema_version: Literal["chunk-analyzer-export-v1"] = "chunk-analyzer-export-v1"
    run_id: Identifier
    files: tuple[ArtifactFile, ...]


def load_development_inputs(
    dataset_directory: Path, retrieval_export_directory: Path, *, depth: int
) -> tuple[DatasetManifest, str, str, str, tuple[InferenceRetrieval, ...]]:
    """Validate both frozen artifacts, then discard evaluation-only labels."""

    manifest, questions = load_inference_questions(dataset_directory)
    if manifest.version != "hotpotqa_1000_v1":
        raise SourceValidationError("Component 02 diagnostic requires the fixed development set")
    retrieval_run, traces = load_export(retrieval_export_directory)
    metadata = retrieval_run.metadata
    if (
        retrieval_run.status != "completed"
        or not retrieval_run.full_benchmark
        or metadata.dataset_version != manifest.version
        or metadata.dataset_manifest_sha256 != sha256_file(dataset_directory / "manifest.json")
        or metadata.question_ids != tuple(question.question_id for question in questions)
        or len(traces) != len(questions)
    ):
        raise SourceValidationError("retrieval export does not match the complete development set")
    inputs: list[InferenceRetrieval] = []
    for question, trace in zip(questions, traces, strict=True):
        if trace.status != "completed" or trace.retrieval is None:
            raise SourceValidationError("retrieval export has a failed development question")
        if trace.question != question:
            raise SourceValidationError("retrieval question differs from frozen inference input")
        inputs.append(
            InferenceRetrieval(
                question=question,
                chunks=trace.retrieval.results[:depth],
                retriever_version=trace.retrieval.retriever_version,
            )
        )
    versions = {item.retriever_version for item in inputs}
    if len(versions) != 1:
        raise SourceValidationError("retrieval export contains mixed retriever versions")
    return manifest, metadata.corpus_version, metadata.run_id, versions.pop(), tuple(inputs)


def evaluate_chunks(
    inputs: tuple[InferenceRetrieval, ...],
    *,
    run_id: str,
    dataset_version: str,
    dataset_manifest_sha256: str,
    corpus_version: str,
    retrieval_run_id: str,
    git_commit_sha: str,
    git_dirty: bool,
    source_sha256: dict[str, str],
    runtime: dict[str, str],
    config: ChunkAnalyzerEvaluationConfig | None = None,
) -> tuple[
    ChunkAnalyzerRunSummary, tuple[ChunkAnalysisTrace, ...], tuple[InstrumentationEvent, ...]
]:
    """Analyze fixed retrieved paragraphs and aggregate non-accuracy diagnostics."""

    if not inputs:
        raise ValueError("chunk analyzer evaluation requires at least one question")
    evaluation_config = config or ChunkAnalyzerEvaluationConfig()
    sink = InMemoryInstrumentationSink()
    if isinstance(evaluation_config.analyzer, RuleBasedV2ChunkAnalyzerConfig):
        analyzer: (
            RuleBasedCandidateChunkEvidenceAnalyzer | RuleBasedV2CandidateChunkEvidenceAnalyzer
        ) = (
            RuleBasedV2CandidateChunkEvidenceAnalyzer(
                config=evaluation_config.analyzer, instrumentation_sink=sink
            )
        )
    else:
        analyzer = RuleBasedCandidateChunkEvidenceAnalyzer(
            config=evaluation_config.analyzer, instrumentation_sink=sink
        )
    question_analyzer = RuleBasedQuestionAnalyzer(instrumentation_sink=sink)
    traces: list[ChunkAnalysisTrace] = []
    for item in inputs:
        started = perf_counter()
        context = TraceContext.start(run_id=run_id, question_id=item.question.question_id)
        evidence: list[CandidateChunkEvidence] = []
        analysis: QuestionAnalysis | None = None
        try:
            analysis = question_analyzer.analyze(item.question.text, trace_context=context)
            ranked_chunks = item.chunks[: evaluation_config.retrieval_depth]
            if isinstance(analyzer, RuleBasedV2CandidateChunkEvidenceAnalyzer):
                evidence.extend(
                    analyzer.analyze_batch(ranked_chunks, analysis, trace_context=context)
                )
            else:
                for ranked in ranked_chunks:
                    evidence.append(analyzer.analyze(ranked, analysis, trace_context=context))
        except (ChunkAnalysisError, QuestionAnalysisError, ValueError) as error:
            traces.append(
                ChunkAnalysisTrace(
                    question=item.question,
                    status="failed",
                    question_analysis=analysis,
                    evidence=tuple(evidence),
                    error_type=type(error).__name__,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
        else:
            traces.append(
                ChunkAnalysisTrace(
                    question=item.question,
                    status="completed",
                    question_analysis=analysis,
                    evidence=tuple(evidence),
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
    all_evidence = [evidence for trace in traces for evidence in trace.evidence]
    answer_counts = Counter(
        candidate.answer_type.value
        for evidence in all_evidence
        for candidate in evidence.answer_type_candidates
    )
    reason_counts = Counter(
        decision.reason
        for evidence in all_evidence
        for decision in evidence.metadata.bridge_decisions
    )
    v2_evidence = [
        evidence for evidence in all_evidence
        if isinstance(evidence.metadata, ChunkAnalysisMetadataV2)
    ]
    grounding_counts = Counter(
        getattr(hint, "grounding_type", "lexical_hint")
        for evidence in v2_evidence for hint in evidence.relation_hints
    )
    bridge_traces = [
        trace for trace in traces if trace.question_analysis is not None
        and trace.question_analysis.question_type is QuestionType.BRIDGE
    ]
    diagnostics = ChunkAnalyzerDiagnostics(
        analyzed_chunks=len(all_evidence),
        chunks_with_question_entity_overlap=sum(
            bool(e.question_entity_overlap) for e in all_evidence
        ),
        chunks_with_title_match=sum(e.is_title_match for e in all_evidence),
        chunks_with_bridge_candidates=sum(bool(e.potential_bridge_entities) for e in all_evidence),
        chunks_without_relation_hints=sum(not e.relation_hints for e in all_evidence),
        answer_candidate_type_counts={
            kind.value: answer_counts[kind.value] for kind in (
                AnswerType.PERSON, AnswerType.LOCATION, AnswerType.ORGANIZATION,
                AnswerType.TITLE_OR_WORK, AnswerType.DATE, AnswerType.NUMBER,
                AnswerType.ENTITY,
            )
        },
        bridge_decision_reason_counts=dict(sorted(reason_counts.items())),
        relation_grounding_type_counts=dict(sorted(grounding_counts.items())),
        bridge_questions=len(bridge_traces),
        bridge_questions_with_candidates=sum(
            any(e.potential_bridge_entities for e in trace.evidence)
            for trace in bridge_traces
        ),
        span_integrity_failures=(
            sum(_v2_span_failures(evidence) for evidence in v2_evidence)
            if isinstance(evaluation_config.analyzer, RuleBasedV2ChunkAnalyzerConfig) else None
        ),
        analyzer_fingerprint_mismatches=sum(
            evidence.metadata.configuration_fingerprint != analyzer.config.fingerprint()
            for evidence in all_evidence
        ),
    )
    summary = ChunkAnalyzerRunSummary(
        run_id=run_id,
        dataset_version=dataset_version,
        dataset_manifest_sha256=dataset_manifest_sha256,
        corpus_version=corpus_version,
        retrieval_run_id=retrieval_run_id,
        retriever_version=inputs[0].retriever_version,
        git_commit_sha=git_commit_sha,
        git_dirty=git_dirty,
        source_sha256=source_sha256,
        runtime=runtime,
        question_ids=tuple(item.question.question_id for item in inputs),
        configuration=evaluation_config,
        configuration_fingerprint=evaluation_config.fingerprint(),
        completed_questions=sum(trace.status == "completed" for trace in traces),
        failed_questions=sum(trace.status == "failed" for trace in traces),
        diagnostics=diagnostics,
    )
    return summary, tuple(traces), sink.events


def write_export(
    export_root: Path,
    *,
    summary: ChunkAnalyzerRunSummary,
    traces: tuple[ChunkAnalysisTrace, ...],
    events: tuple[InstrumentationEvent, ...],
) -> Path:
    """Persist a named, checksummed, exclusive local diagnostic export."""

    directory = export_root / summary.run_id
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FrozenArtifactError(f"refusing to overwrite frozen artifact: {directory}") from error
    write_json_exclusive(directory / "run.json", summary)
    trace_file = write_jsonl_exclusive(
        directory / "traces.jsonl", traces, relative_path="traces.jsonl"
    )
    event_file = write_jsonl_exclusive(
        directory / "events.jsonl", events, relative_path="events.jsonl"
    )
    run_file = ArtifactFile(
        relative_path="run.json",
        sha256=sha256_file(directory / "run.json"),
        byte_count=(directory / "run.json").stat().st_size,
        record_count=1,
    )
    write_json_exclusive(
        directory / "manifest.json",
        ChunkAnalyzerExportManifest(
            run_id=summary.run_id, files=(run_file, trace_file, event_file)
        ),
    )
    return directory


def inspect_trace(directory: Path, question_id: str) -> ChunkAnalysisTrace:
    """Return one stored question trace for structured manual inspection."""

    for record in read_jsonl(directory / "traces.jsonl"):
        question = record.get("question")
        if isinstance(question, dict) and question.get("question_id") == question_id:
            return ChunkAnalysisTrace.model_validate(record)
    raise ValueError("question ID is not present in this export")
