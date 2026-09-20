"""Development-only Component 01 → 02 → 03 inference diagnostics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import Field

from epsa_rag.core.config import ConfigModel
from epsa_rag.core.exceptions import (
    ChunkAnalysisError,
    EvidenceUnitExtractionError,
    FrozenArtifactError,
    QuestionAnalysisError,
    SourceValidationError,
)
from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel
from epsa_rag.data.io import read_jsonl, sha256_file, write_json_exclusive, write_jsonl_exclusive
from epsa_rag.data.manifests import ArtifactFile
from epsa_rag.data.models import BenchmarkExample, QuestionInput
from epsa_rag.epsa.chunk_analysis import (
    CandidateChunkEvidence,
    RuleBasedCandidateChunkEvidenceAnalyzer,
    RuleBasedV2CandidateChunkEvidenceAnalyzer,
)
from epsa_rag.epsa.evidence_units import (
    EvidenceUnit,
    RuleBasedEvidenceUnitExtractor,
    RuleBasedV2EvidenceUnitExtractor,
)
from epsa_rag.epsa.question_analysis import QuestionAnalysis, RuleBasedQuestionAnalyzer
from epsa_rag.evaluation.components.chunk_analyzer import (
    InferenceRetrieval,
    load_development_inputs,
)
from epsa_rag.evaluation.components.question_analyzer import load_inference_questions
from epsa_rag.instrumentation import InMemoryInstrumentationSink, InstrumentationEvent, TraceContext

ChunkMode = Literal["rule_based_v1", "rule_based_v2"]
UnitMode = Literal["rule_based_v1", "rule_based_v2"]


class EvidenceUnitEvaluationConfig(ConfigModel):
    evaluator_version: Literal["evidence-unit-evaluation-v1"] = "evidence-unit-evaluation-v1"
    chunk_mode: ChunkMode = "rule_based_v2"
    unit_mode: UnitMode = "rule_based_v2"
    retrieval_depth: int = Field(default=10, ge=1)


class EvidenceUnitTrace(ContractModel):
    question: QuestionInput
    status: Literal["completed", "failed"]
    question_analysis: QuestionAnalysis | None
    chunk_evidence: tuple[CandidateChunkEvidence, ...]
    evidence_units: tuple[EvidenceUnit, ...]
    error_type: str | None = None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)


class EvidenceUnitDiagnostics(ContractModel):
    analyzed_chunks: int = Field(ge=0)
    evidence_units: int = Field(ge=0)
    segmented_units: int = Field(ge=0)
    native_metadata_units: int = Field(ge=0)
    changed_resolutions: int = Field(ge=0)
    ambiguous_resolutions: int = Field(ge=0)
    units_with_question_entity_overlap: int = Field(ge=0)
    units_with_relation_hints: int = Field(ge=0)
    units_with_structured_candidates: int = Field(ge=0)
    source_span_failures: int = Field(ge=0)
    answer_type_counts: dict[str, int]


class EvidenceUnitRunSummary(ContractModel):
    schema_version: Literal["evidence-unit-run-v1"] = "evidence-unit-run-v1"
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
    configuration: EvidenceUnitEvaluationConfig
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_questions: int = Field(ge=0)
    failed_questions: int = Field(ge=0)
    diagnostics: EvidenceUnitDiagnostics


class EvidenceUnitExportManifest(ContractModel):
    schema_version: Literal["evidence-unit-export-v1"] = "evidence-unit-export-v1"
    run_id: Identifier
    files: tuple[ArtifactFile, ...]


def evaluate_evidence_units(
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
    config: EvidenceUnitEvaluationConfig | None = None,
) -> tuple[EvidenceUnitRunSummary, tuple[EvidenceUnitTrace, ...], tuple[InstrumentationEvent, ...]]:
    if not inputs:
        raise ValueError("Component 03 evaluation requires inference inputs")
    selected = config or EvidenceUnitEvaluationConfig()
    sink = InMemoryInstrumentationSink()
    questions = RuleBasedQuestionAnalyzer(instrumentation_sink=sink)
    chunks = (
        RuleBasedV2CandidateChunkEvidenceAnalyzer(instrumentation_sink=sink)
        if selected.chunk_mode == "rule_based_v2"
        else RuleBasedCandidateChunkEvidenceAnalyzer(instrumentation_sink=sink)
    )
    extractor = (
        RuleBasedV2EvidenceUnitExtractor(instrumentation_sink=sink)
        if selected.unit_mode == "rule_based_v2"
        else RuleBasedEvidenceUnitExtractor(instrumentation_sink=sink)
    )
    traces: list[EvidenceUnitTrace] = []
    for item in inputs:
        started = perf_counter()
        context = TraceContext.start(run_id=run_id, question_id=item.question.question_id)
        analysis: QuestionAnalysis | None = None
        chunk_evidence: list[CandidateChunkEvidence] = []
        units: list[EvidenceUnit] = []
        try:
            analysis = questions.analyze(item.question.text, trace_context=context)
            ranked = item.chunks[: selected.retrieval_depth]
            if isinstance(chunks, RuleBasedV2CandidateChunkEvidenceAnalyzer):
                chunk_evidence.extend(chunks.analyze_batch(ranked, analysis, trace_context=context))
            else:
                chunk_evidence.extend(
                    chunks.analyze(chunk, analysis, trace_context=context) for chunk in ranked
                )
            for candidate, chunk in zip(chunk_evidence, ranked, strict=True):
                units.extend(
                    extractor.extract_from_chunk(candidate, chunk, analysis, trace_context=context)
                )
        except (
            QuestionAnalysisError,
            ChunkAnalysisError,
            EvidenceUnitExtractionError,
            ValueError,
        ) as error:
            traces.append(
                EvidenceUnitTrace(
                    question=item.question,
                    status="failed",
                    question_analysis=analysis,
                    chunk_evidence=tuple(chunk_evidence),
                    evidence_units=tuple(units),
                    error_type=type(error).__name__,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
        else:
            traces.append(
                EvidenceUnitTrace(
                    question=item.question,
                    status="completed",
                    question_analysis=analysis,
                    chunk_evidence=tuple(chunk_evidence),
                    evidence_units=tuple(units),
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
    all_units = [unit for trace in traces for unit in trace.evidence_units]
    counts = Counter(kind.value for unit in all_units for kind in unit.answer_type_candidates)
    source_failures = 0
    for trace in traces:
        body_by_chunk = {item.chunk_id: item.paragraph_text for item in trace.chunk_evidence}
        for unit in trace.evidence_units:
            if unit.start_char is not None and unit.end_char is not None:
                source_failures += (
                    body_by_chunk[unit.chunk_id][unit.start_char : unit.end_char]
                    != unit.sentence_text
                )
            for feature in unit.entity_features:
                if feature.start_char is not None and feature.end_char is not None:
                    source_failures += (
                        unit.sentence_text[feature.start_char : feature.end_char] != feature.text
                    )
            for answer_candidate in unit.structured_answer_candidates:
                if (
                    answer_candidate.start_char is not None
                    and answer_candidate.end_char is not None
                ):
                    source_failures += (
                        unit.sentence_text[answer_candidate.start_char : answer_candidate.end_char]
                        != answer_candidate.text
                    )
    summary = EvidenceUnitRunSummary(
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
        configuration=selected,
        configuration_fingerprint=selected.fingerprint(),
        completed_questions=sum(trace.status == "completed" for trace in traces),
        failed_questions=sum(trace.status == "failed" for trace in traces),
        diagnostics=EvidenceUnitDiagnostics(
            analyzed_chunks=sum(len(trace.chunk_evidence) for trace in traces),
            evidence_units=len(all_units),
            segmented_units=sum(unit.metadata.sentence_source == "segmented" for unit in all_units),
            native_metadata_units=sum(
                unit.metadata.sentence_source == "metadata" for unit in all_units
            ),
            changed_resolutions=sum(unit.metadata.resolution.changed for unit in all_units),
            ambiguous_resolutions=sum(
                unit.metadata.resolution.method == "ambiguous_context" for unit in all_units
            ),
            units_with_question_entity_overlap=sum(
                bool(unit.question_entity_overlap) for unit in all_units
            ),
            units_with_relation_hints=sum(bool(unit.relation_hints) for unit in all_units),
            units_with_structured_candidates=sum(
                bool(unit.structured_answer_candidates) for unit in all_units
            ),
            source_span_failures=source_failures,
            answer_type_counts=dict(sorted(counts.items())),
        ),
    )
    return summary, tuple(traces), sink.events


def write_export(
    export_root: Path,
    *,
    summary: EvidenceUnitRunSummary,
    traces: tuple[EvidenceUnitTrace, ...],
    events: tuple[InstrumentationEvent, ...],
) -> Path:
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
        EvidenceUnitExportManifest(run_id=summary.run_id, files=(run_file, trace_file, event_file)),
    )
    return directory


def inspect_trace(directory: Path, question_id: str) -> EvidenceUnitTrace:
    _, traces = load_export(directory)
    for trace in traces:
        if trace.question.question_id == question_id:
            return trace
    raise ValueError("question ID is not present in this export")


def load_export(directory: Path) -> tuple[EvidenceUnitRunSummary, tuple[EvidenceUnitTrace, ...]]:
    """Verify named export checksums and question order before inspection."""

    manifest = EvidenceUnitExportManifest.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    if tuple(item.relative_path for item in manifest.files) != (
        "run.json",
        "traces.jsonl",
        "events.jsonl",
    ):
        raise SourceValidationError("unexpected Component 03 export files")
    for item in manifest.files:
        path = directory / item.relative_path
        if path.stat().st_size != item.byte_count or sha256_file(path) != item.sha256:
            raise SourceValidationError(
                f"Component 03 export integrity failure: {item.relative_path}"
            )
    summary = EvidenceUnitRunSummary.model_validate_json(
        (directory / "run.json").read_text(encoding="utf-8")
    )
    traces = tuple(
        EvidenceUnitTrace.model_validate(record)
        for record in read_jsonl(directory / "traces.jsonl")
    )
    if (
        manifest.run_id != summary.run_id
        or len(traces) != manifest.files[1].record_count
        or tuple(trace.question.question_id for trace in traces) != summary.question_ids
        or sum(trace.status == "completed" for trace in traces) != summary.completed_questions
        or sum(trace.status == "failed" for trace in traces) != summary.failed_questions
    ):
        raise SourceValidationError("Component 03 export identity or trace count mismatch")
    return summary, traces


class GoldCoverageDiagnostic(ContractModel):
    """Evaluation-only retrieval coverage; never an extractor feature."""

    question_id: Identifier
    gold_sentence_count: int = Field(ge=0)
    retrieved_gold_sentence_count: int = Field(ge=0)
    missing_gold_evidence_unit_ids: tuple[Identifier, ...]


def gold_coverage_diagnostics(
    traces: tuple[EvidenceUnitTrace, ...], dataset_directory: Path
) -> tuple[GoldCoverageDiagnostic, ...]:
    """Join gold labels only after inference traces have been finalized."""

    manifest, questions = load_inference_questions(dataset_directory)
    if manifest.version != "hotpotqa_1000_v1":
        raise SourceValidationError("gold diagnostics require the development benchmark")
    if tuple(trace.question for trace in traces) != questions:
        raise SourceValidationError("gold diagnostics question order differs from benchmark")
    labels = {
        item.inference.question_id: item.evaluation.supporting_facts
        for item in (
            BenchmarkExample.model_validate(record)
            for record in read_jsonl(dataset_directory / "dataset.jsonl")
        )
    }
    results: list[GoldCoverageDiagnostic] = []
    for trace in traces:
        gold = {item.evidence_unit_id for item in labels[trace.question.question_id]}
        extracted = {unit.evidence_unit_id for unit in trace.evidence_units}
        missing = tuple(sorted(gold - extracted))
        results.append(
            GoldCoverageDiagnostic(
                question_id=trace.question.question_id,
                gold_sentence_count=len(gold),
                retrieved_gold_sentence_count=len(gold & extracted),
                missing_gold_evidence_unit_ids=missing,
            )
        )
    return tuple(results)


__all__ = [
    "EvidenceUnitEvaluationConfig",
    "EvidenceUnitRunSummary",
    "EvidenceUnitTrace",
    "evaluate_evidence_units",
    "gold_coverage_diagnostics",
    "inspect_trace",
    "load_development_inputs",
    "load_export",
    "write_export",
]
