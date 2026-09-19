"""Development-benchmark diagnostics for EPSA Component 01.

This module intentionally consumes only inference question IDs/text. HotPotQA answer and supporting
fact labels do not provide gold labels for Question Analyzer outputs, so it reports coverage and
diagnostic counts rather than accuracy.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import Field

from epsa_rag.core.config import ConfigModel
from epsa_rag.core.exceptions import (
    FrozenArtifactError,
    QuestionAnalysisError,
    SourceValidationError,
)
from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel
from epsa_rag.data.io import read_jsonl, sha256_file, write_json_exclusive, write_jsonl_exclusive
from epsa_rag.data.manifests import ArtifactFile, DatasetManifest
from epsa_rag.data.models import QuestionInput
from epsa_rag.data.validation import validate_dataset_artifact
from epsa_rag.epsa.question_analysis import (
    QuestionAnalysis,
    QuestionAnalyzerConfig,
    RuleBasedQuestionAnalyzer,
)
from epsa_rag.instrumentation import InMemoryInstrumentationSink, InstrumentationEvent, TraceContext


class QuestionAnalyzerEvaluationConfig(ConfigModel):
    """Explicit fixed choices for a Component 01 diagnostic benchmark run."""

    evaluator_version: Literal["question-analyzer-evaluation-v1"] = (
        "question-analyzer-evaluation-v1"
    )
    analyzer: QuestionAnalyzerConfig = Field(default_factory=QuestionAnalyzerConfig)


class QuestionAnalysisTrace(ContractModel):
    """Inference-only per-question analyzer output and diagnostic timing."""

    question: QuestionInput
    status: Literal["completed", "failed"]
    analysis: QuestionAnalysis | None
    error_type: str | None = None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)


class QuestionAnalyzerDiagnosticSummary(ContractModel):
    """Descriptive counts, deliberately not component accuracy metrics."""

    question_type_counts: dict[str, int]
    expected_answer_type_counts: dict[str, int]
    questions_without_seed_entities: int = Field(ge=0)
    questions_without_relation_hints: int = Field(ge=0)
    comparison_questions_without_two_targets: int = Field(ge=0)


class QuestionAnalyzerRunSummary(ContractModel):
    """Versioned run provenance and non-accuracy diagnostics."""

    schema_version: Literal["question-analyzer-run-v1"] = "question-analyzer-run-v1"
    run_id: Identifier
    dataset_version: Identifier
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_ids: tuple[Identifier, ...]
    configuration: QuestionAnalyzerEvaluationConfig
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_questions: int = Field(ge=0)
    failed_questions: int = Field(ge=0)
    diagnostics: QuestionAnalyzerDiagnosticSummary


class QuestionAnalyzerExportManifest(ContractModel):
    """Checksums for an immutable local diagnostic export, not authoritative storage."""

    schema_version: Literal["question-analyzer-export-v1"] = "question-analyzer-export-v1"
    run_id: Identifier
    files: tuple[ArtifactFile, ...]


def load_inference_questions(
    dataset_directory: Path,
) -> tuple[DatasetManifest, tuple[QuestionInput, ...]]:
    """Validate a frozen dataset then read only its inference-visible question objects."""

    manifest_path = dataset_directory / "manifest.json"
    dataset_path = dataset_directory / "dataset.jsonl"
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if len(manifest.files) != 1 or manifest.files[0].relative_path != "dataset.jsonl":
        raise SourceValidationError("expected exactly one dataset.jsonl artifact")
    validate_dataset_artifact(dataset_path, manifest)
    questions: list[QuestionInput] = []
    for record in read_jsonl(dataset_path):
        inference = record.get("inference")
        if not isinstance(inference, dict):
            raise SourceValidationError("benchmark record lacks an inference object")
        questions.append(QuestionInput.model_validate(inference))
    question_ids = tuple(question.question_id for question in questions)
    if (
        len(questions) != manifest.selected_question_count
        or question_ids != manifest.question_ids
        or len(set(question_ids)) != len(question_ids)
    ):
        raise SourceValidationError("dataset inference questions do not match the frozen manifest")
    return manifest, tuple(questions)


def evaluate_questions(
    questions: tuple[QuestionInput, ...],
    *,
    run_id: str,
    dataset_version: str,
    dataset_manifest_sha256: str,
    config: QuestionAnalyzerEvaluationConfig | None = None,
) -> tuple[
    QuestionAnalyzerRunSummary, tuple[QuestionAnalysisTrace, ...], tuple[InstrumentationEvent, ...]
]:
    """Analyze all questions with per-question events and diagnostic-only aggregation."""

    if not questions:
        raise ValueError("question analyzer evaluation requires at least one question")
    evaluation_config = config or QuestionAnalyzerEvaluationConfig()
    sink = InMemoryInstrumentationSink()
    analyzer = RuleBasedQuestionAnalyzer(
        config=evaluation_config.analyzer,
        instrumentation_sink=sink,
    )
    traces: list[QuestionAnalysisTrace] = []
    for question in questions:
        started = perf_counter()
        try:
            analysis = analyzer.analyze(
                question.text,
                trace_context=TraceContext.start(run_id=run_id, question_id=question.question_id),
            )
        except QuestionAnalysisError as error:
            traces.append(
                QuestionAnalysisTrace(
                    question=question,
                    status="failed",
                    analysis=None,
                    error_type=type(error).__name__,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
        else:
            traces.append(
                QuestionAnalysisTrace(
                    question=question,
                    status="completed",
                    analysis=analysis,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
    completed = [trace.analysis for trace in traces if trace.analysis is not None]
    question_types = Counter(analysis.question_type.value for analysis in completed)
    answer_types = Counter(analysis.expected_answer_type.value for analysis in completed)
    diagnostics = QuestionAnalyzerDiagnosticSummary(
        question_type_counts=dict(sorted(question_types.items())),
        expected_answer_type_counts=dict(sorted(answer_types.items())),
        questions_without_seed_entities=sum(not analysis.seed_entities for analysis in completed),
        questions_without_relation_hints=sum(
            not analysis.required_relation_hints for analysis in completed
        ),
        comparison_questions_without_two_targets=sum(
            analysis.question_type.value == "comparison" and len(analysis.comparison_targets) < 2
            for analysis in completed
        ),
    )
    summary = QuestionAnalyzerRunSummary(
        run_id=run_id,
        dataset_version=dataset_version,
        dataset_manifest_sha256=dataset_manifest_sha256,
        question_ids=tuple(question.question_id for question in questions),
        configuration=evaluation_config,
        configuration_fingerprint=evaluation_config.fingerprint(),
        completed_questions=len(completed),
        failed_questions=len(traces) - len(completed),
        diagnostics=diagnostics,
    )
    return summary, tuple(traces), sink.events


def write_export(
    export_root: Path,
    *,
    summary: QuestionAnalyzerRunSummary,
    traces: tuple[QuestionAnalysisTrace, ...],
    events: tuple[InstrumentationEvent, ...],
) -> Path:
    """Write one named immutable local export for researcher inspection."""

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
        QuestionAnalyzerExportManifest(
            run_id=summary.run_id, files=(run_file, trace_file, event_file)
        ),
    )
    return directory


def render_development_report(
    traces: tuple[QuestionAnalysisTrace, ...], *, run_id: str, dataset_version: str
) -> str:
    """Render deterministic per-question traces for manual development-set review."""

    lines = [
        "# Component 01 Question Analyzer — Development Set Output",
        "",
        f"Run ID: {run_id}",
        f"Dataset: {dataset_version}",
        "This deterministic display contains inference-only analyses and is not an accuracy "
        "report.",
    ]
    for index, trace in enumerate(traces, start=1):
        lines.extend(
            ("", f"## {index}. Question ID: {trace.question.question_id}", "", "Question:")
        )
        lines.extend((trace.question.text, ""))
        if trace.analysis is None:
            lines.extend(("Status: failed", f"Error: {trace.error_type or 'Unknown error'}"))
            continue
        analysis = trace.analysis
        seeds = "; ".join(entity.text for entity in analysis.seed_entities) or "none"
        hints = "; ".join(
            f"{hint.relation} ({hint.matched_text})" for hint in analysis.required_relation_hints
        ) or "none"
        targets = "; ".join(target.text for target in analysis.comparison_targets) or "none"
        lines.extend(
            (
                f"Question type: {analysis.question_type.value}",
                f"Expected answer type: {analysis.expected_answer_type.value}",
                f"Seed entities: {seeds}",
                f"Relation hints: {hints}",
                f"Comparison targets: {targets}",
            )
        )
    return "\n".join(lines) + "\n"
