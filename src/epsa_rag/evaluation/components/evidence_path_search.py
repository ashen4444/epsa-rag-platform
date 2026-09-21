"""Inference-only structural diagnostics for EPSA Components 01 through 06."""

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
from epsa_rag.evaluation.components.chunk_analyzer import InferenceRetrieval
from epsa_rag.instrumentation import (
    InMemoryInstrumentationSink,
    InstrumentationEvent,
    NoOpInstrumentationSink,
    TraceContext,
)


class EvidencePathSearchEvaluationConfig(ConfigModel):
    evaluator_version: Literal["evidence-path-search-evaluation-v1"] = (
        "evidence-path-search-evaluation-v1"
    )
    retrieval_depth: int = Field(default=10, ge=1)
    max_paths: int = Field(default=10, ge=0)
    retain_instrumentation_events: bool = True


class EvidencePathSearchTrace(ContractModel):
    question_id: Identifier
    status: Literal["completed", "failed"]
    question_analysis: QuestionAnalysis | None
    chunk_evidence: tuple[CandidateChunkEvidence, ...]
    evidence_units: tuple[EvidenceUnit, ...]
    scored_evidence_units: tuple[ScoredEvidenceUnit, ...]
    evidence_graph: EvidenceGraph | None
    candidate_paths: tuple[EvidencePath, ...]
    error_type: str | None = None
    latency_ms: float = Field(ge=0)


class EvidencePathSearchDiagnostics(ContractModel):
    graphs: int = Field(ge=0)
    candidate_paths: int = Field(ge=0)
    questions_with_no_candidate_paths: int = Field(ge=0)
    paths_missing_scored_evidence_provenance: int = Field(ge=0)
    paths_with_graph_identity_mismatch: int = Field(ge=0)
    question_type_counts: dict[str, int]
    path_kind_counts: dict[str, int]


class EvidencePathSearchRunSummary(ContractModel):
    schema_version: Literal["evidence-path-search-run-v1"] = "evidence-path-search-run-v1"
    run_id: Identifier
    configuration: EvidencePathSearchEvaluationConfig
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_ids: tuple[Identifier, ...]
    completed_questions: int = Field(ge=0)
    failed_questions: int = Field(ge=0)
    diagnostics: EvidencePathSearchDiagnostics


class EvidencePathSearchExportManifest(ContractModel):
    """Checksummed, immutable diagnostic-output inventory for Component 06."""

    schema_version: Literal["evidence-path-search-export-v1"] = "evidence-path-search-export-v1"
    run_id: Identifier
    files: tuple[ArtifactFile, ...]


def evaluate_evidence_paths(
    inputs: tuple[InferenceRetrieval, ...],
    *,
    run_id: str,
    config: EvidencePathSearchEvaluationConfig | None = None,
) -> tuple[
    EvidencePathSearchRunSummary,
    tuple[EvidencePathSearchTrace, ...],
    tuple[InstrumentationEvent, ...],
]:
    """Run Components 01-06 over retrieved inference inputs, without gold labels."""

    if not inputs:
        raise ValueError("Component 06 evaluation requires inference inputs")
    selected = config or EvidencePathSearchEvaluationConfig()
    sink = (
        InMemoryInstrumentationSink()
        if selected.retain_instrumentation_events
        else NoOpInstrumentationSink()
    )
    questions = RuleBasedQuestionAnalyzer(instrumentation_sink=sink)
    chunks = RuleBasedV2CandidateChunkEvidenceAnalyzer(instrumentation_sink=sink)
    extractor = RuleBasedV2EvidenceUnitExtractor(instrumentation_sink=sink)
    scorer = RuleBasedEvidenceScorerV1(instrumentation_sink=sink)
    builder = EvidenceGraphBuilderV1(instrumentation_sink=sink)
    searcher = EvidencePathSearcherV1(instrumentation_sink=sink)
    traces: list[EvidencePathSearchTrace] = []
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
            analysis = questions.analyze(item.question.text, trace_context=context)
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
        except (
            QuestionAnalysisError,
            ChunkAnalysisError,
            EvidenceUnitExtractionError,
            EvidenceScoringError,
            EvidenceGraphBuildError,
            EvidencePathSearchError,
            ValueError,
        ) as error:
            traces.append(
                EvidencePathSearchTrace(
                    question_id=item.question.question_id,
                    status="failed",
                    question_analysis=analysis,
                    chunk_evidence=tuple(candidates),
                    evidence_units=tuple(units),
                    scored_evidence_units=scored,
                    evidence_graph=graph,
                    candidate_paths=tuple(paths),
                    error_type=type(error).__name__,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
        else:
            traces.append(
                EvidencePathSearchTrace(
                    question_id=item.question.question_id,
                    status="completed",
                    question_analysis=analysis,
                    chunk_evidence=tuple(candidates),
                    evidence_units=tuple(units),
                    scored_evidence_units=scored,
                    evidence_graph=graph,
                    candidate_paths=tuple(paths),
                    latency_ms=(perf_counter() - started) * 1000,
                )
            )
    completed = tuple(trace for trace in traces if trace.status == "completed")
    all_paths = tuple(path for trace in completed for path in trace.candidate_paths)
    kinds = Counter(path.metadata.path_kind for path in all_paths)
    question_types = Counter(path.question_type.value for path in all_paths)
    provenance_failures = sum(not path.scored_evidence_units for path in all_paths)
    identity_mismatches = sum(
        path.metadata.source_graph != trace.evidence_graph.metadata
        for trace in completed
        if trace.evidence_graph is not None
        for path in trace.candidate_paths
    )
    return (
        EvidencePathSearchRunSummary(
            run_id=run_id,
            configuration=selected,
            configuration_fingerprint=selected.fingerprint(),
            question_ids=tuple(item.question.question_id for item in inputs),
            completed_questions=len(completed),
            failed_questions=len(traces) - len(completed),
            diagnostics=EvidencePathSearchDiagnostics(
                graphs=sum(trace.evidence_graph is not None for trace in completed),
                candidate_paths=len(all_paths),
                questions_with_no_candidate_paths=sum(
                    not trace.candidate_paths for trace in completed
                ),
                paths_missing_scored_evidence_provenance=provenance_failures,
                paths_with_graph_identity_mismatch=identity_mismatches,
                question_type_counts=dict(sorted(question_types.items())),
                path_kind_counts=dict(sorted(kinds.items())),
            ),
        ),
        tuple(traces),
        sink.events if isinstance(sink, InMemoryInstrumentationSink) else (),
    )


def write_evidence_path_evaluation(
    directory: Path,
    summary: EvidencePathSearchRunSummary,
    traces: tuple[EvidencePathSearchTrace, ...],
    events: tuple[InstrumentationEvent, ...],
) -> EvidencePathSearchExportManifest:
    """Persist a new immutable Component 06 diagnostic export without overwriting a run."""

    directory.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(directory / "run.json", summary)
    traces_file = write_jsonl_exclusive(
        directory / "traces.jsonl", traces, relative_path="traces.jsonl"
    )
    events_file = write_jsonl_exclusive(
        directory / "events.jsonl", events, relative_path="events.jsonl"
    )
    manifest = EvidencePathSearchExportManifest(
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


def write_evidence_path_markdown_report(
    output_path: Path,
    summary: EvidencePathSearchRunSummary,
    traces: tuple[EvidencePathSearchTrace, ...],
) -> None:
    """Write a non-overwriting, inference-only Markdown inspection report."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {output_path}")
    lines = [
        "# EPSA Component 06 - 1,000-Question Development Results",
        "",
        "Inference-visible Components 01-06 diagnostic output. Candidate paths are not answers "
        "or sufficiency decisions. No HotPotQA gold answers or supporting-fact labels are "
        "included.",
        "",
        f"Run ID: `{summary.run_id}`",
        f"Completed: {summary.completed_questions}; failed: {summary.failed_questions}",
        f"Diagnostics: `{summary.diagnostics.model_dump_json()}`",
        "",
    ]
    for trace in traces:
        question = (
            trace.question_analysis.raw_question
            if trace.question_analysis is not None
            else "[unavailable]"
        )
        lines.extend(
            (
                f"## {trace.question_id}",
                "",
                f"Question: {question}",
                f"Status: {trace.status}",
                f"Candidate paths: {len(trace.candidate_paths)}",
                f"Error: {trace.error_type or 'none'}",
                "",
                "| Rank | Path ID | Type | Answer candidate | Score | Evidence units |",
                "| ---: | --- | --- | --- | ---: | --- |",
            )
        )
        for rank, path in enumerate(trace.candidate_paths, start=1):
            answer = (path.answer_candidate or "").replace("|", "\\|").replace("\n", " ")
            evidence_units = ", ".join(path.evidence_unit_ids).replace("|", "\\|")
            lines.append(
                f"| {rank} | `{path.path_id}` | {path.metadata.path_kind} | {answer} | "
                f"{path.score:.6f} | {evidence_units} |"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_streaming_evidence_path_markdown_report(
    output_path: Path,
    inputs: tuple[InferenceRetrieval, ...],
    *,
    run_id: str,
    batch_size: int = 10,
) -> dict[str, int]:
    """Evaluate fixed inference inputs in bounded batches and stream a non-overwriting report."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {output_path}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    completed = 0
    failed = 0
    paths = 0
    no_paths = 0
    configuration = EvidencePathSearchEvaluationConfig(retain_instrumentation_events=False)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("# EPSA Component 06 - 1,000-Question Development Results\n\n")
        stream.write(
            "Inference-visible Components 01-06 diagnostic output. Candidate paths are not "
            "answers or sufficiency decisions. No HotPotQA gold answers or supporting-fact labels "
            "are included.\n\n"
        )
        stream.write(f"Run ID prefix: `{run_id}`\n\n")
        for offset in range(0, len(inputs), batch_size):
            summary, traces, _ = evaluate_evidence_paths(
                inputs[offset : offset + batch_size],
                run_id=f"{run_id}-{offset // batch_size + 1}",
                config=configuration,
            )
            completed += summary.completed_questions
            failed += summary.failed_questions
            paths += summary.diagnostics.candidate_paths
            no_paths += summary.diagnostics.questions_with_no_candidate_paths
            for trace in traces:
                question = (
                    trace.question_analysis.raw_question
                    if trace.question_analysis is not None
                    else "[unavailable]"
                )
                stream.write(f"## {trace.question_id}\n\n")
                stream.write(f"Question: {question}\n\n")
                stream.write(f"Status: {trace.status}\n\n")
                stream.write(f"Candidate paths: {len(trace.candidate_paths)}\n\n")
                stream.write(f"Error: {trace.error_type or 'none'}\n\n")
                stream.write(
                    "| Rank | Path ID | Type | Answer candidate | Score | Evidence units |\n"
                )
                stream.write("| ---: | --- | --- | --- | ---: | --- |\n")
                for rank, path in enumerate(trace.candidate_paths, start=1):
                    answer = (path.answer_candidate or "").replace("|", "\\|").replace("\n", " ")
                    evidence_units = ", ".join(path.evidence_unit_ids).replace("|", "\\|")
                    stream.write(
                        f"| {rank} | `{path.path_id}` | {path.metadata.path_kind} | {answer} | "
                        f"{path.score:.6f} | {evidence_units} |\n"
                    )
                stream.write("\n")
            stream.flush()
            print(f"completed batch {offset // batch_size + 1}", flush=True)
        stream.write("## Aggregate structural diagnostics\n\n")
        stream.write(f"Completed questions: {completed}\n\n")
        stream.write(f"Failed questions: {failed}\n\n")
        stream.write(f"Candidate paths: {paths}\n\n")
        stream.write(f"Questions with no candidate paths: {no_paths}\n")
    return {
        "completed_questions": completed,
        "failed_questions": failed,
        "candidate_paths": paths,
        "questions_with_no_candidate_paths": no_paths,
    }


def append_streaming_evidence_path_markdown_report(
    output_path: Path,
    inputs: tuple[InferenceRetrieval, ...],
    *,
    run_id: str,
    batch_size: int = 10,
) -> dict[str, int]:
    """Append bounded inference-only question sections to an existing report."""

    if not output_path.exists():
        raise FileNotFoundError(f"report does not exist: {output_path}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    completed = 0
    failed = 0
    paths = 0
    configuration = EvidencePathSearchEvaluationConfig(retain_instrumentation_events=False)
    with output_path.open("a", encoding="utf-8", newline="\n") as stream:
        for offset in range(0, len(inputs), batch_size):
            summary, traces, _ = evaluate_evidence_paths(
                inputs[offset : offset + batch_size],
                run_id=f"{run_id}-{offset // batch_size + 1}",
                config=configuration,
            )
            completed += summary.completed_questions
            failed += summary.failed_questions
            paths += summary.diagnostics.candidate_paths
            for trace in traces:
                question = (
                    trace.question_analysis.raw_question
                    if trace.question_analysis is not None
                    else "[unavailable]"
                )
                stream.write(f"## {trace.question_id}\n\nQuestion: {question}\n\n")
                stream.write(f"Status: {trace.status}\n\n")
                stream.write(f"Candidate paths: {len(trace.candidate_paths)}\n\n")
                stream.write(f"Error: {trace.error_type or 'none'}\n\n")
                stream.write(
                    "| Rank | Path ID | Type | Answer candidate | Score | Evidence units |\n"
                )
                stream.write("| ---: | --- | --- | --- | ---: | --- |\n")
                for rank, path in enumerate(trace.candidate_paths, start=1):
                    answer = (path.answer_candidate or "").replace("|", "\\|").replace("\n", " ")
                    evidence_units = ", ".join(path.evidence_unit_ids).replace("|", "\\|")
                    stream.write(
                        f"| {rank} | `{path.path_id}` | {path.metadata.path_kind} | {answer} | "
                        f"{path.score:.6f} | {evidence_units} |\n"
                    )
                stream.write("\n")
            stream.flush()
    return {"completed_questions": completed, "failed_questions": failed, "candidate_paths": paths}
