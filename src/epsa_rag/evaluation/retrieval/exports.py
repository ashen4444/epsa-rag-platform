"""Immutable Phase 4 diagnostic exports, awaiting the Phase 5 authoritative store."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import TextIO

from epsa_rag.core.exceptions import FrozenArtifactError, SourceValidationError
from epsa_rag.core.models import ContractModel
from epsa_rag.data.io import canonical_json, read_jsonl, sha256_file, write_json_exclusive
from epsa_rag.data.manifests import ArtifactFile
from epsa_rag.evaluation.retrieval.models import QuestionTrace, RunMetadata, RunSummary
from epsa_rag.instrumentation.events import InstrumentationEvent
from epsa_rag.instrumentation.sinks import InstrumentationSink


class ExportManifest(ContractModel):
    schema_version: str = "1.0"
    purpose: str = "diagnostic-export-pending-postgresql-import"
    run_id: str
    files: tuple[ArtifactFile, ...]


class DiagnosticExportSink:
    """Reserve a stable run ID and flush events incrementally, preserving interrupted runs.

    Only a finalized export has run.json and manifest.json. Existing directories are never reused.
    An optional downstream sink lets Phase 5 consume the same events without changing evaluation.
    """

    def __init__(
        self, root: Path, metadata: RunMetadata, downstream: InstrumentationSink | None = None
    ) -> None:
        self.directory = root / metadata.run_id
        # Windows device names are special even without a file extension.
        if metadata.run_id.upper() in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)),
        }:
            raise ValueError("run_id is a reserved device name")
        root.mkdir(parents=True, exist_ok=True)
        try:
            self.directory.mkdir()
        except FileExistsError as error:
            raise FrozenArtifactError(f"run ID already exists: {metadata.run_id}") from error
        self._stream: TextIO = (self.directory / "events.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        )
        self._downstream = downstream
        self._count = 0
        self._run_id = metadata.run_id

    def __enter__(self) -> DiagnosticExportSink:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stream.close()

    def emit(self, event: InstrumentationEvent) -> None:
        if event.context.run_id != self._run_id:
            raise ValueError("event belongs to a different run")
        self._stream.write(canonical_json(event.model_dump(mode="json")) + "\n")
        self._stream.flush()
        self._count += 1
        if self._downstream is not None:
            self._downstream.emit(event)

    def finalize(self, summary: RunSummary) -> None:
        if summary.metadata.run_id != self._run_id:
            raise ValueError("summary belongs to a different run")
        self._stream.close()
        write_json_exclusive(self.directory / "run.json", summary)
        files = tuple(
            ArtifactFile(
                relative_path=name,
                sha256=sha256_file(self.directory / name),
                byte_count=(self.directory / name).stat().st_size,
                record_count=count,
            )
            for name, count in (("events.jsonl", self._count), ("run.json", 1))
        )
        write_json_exclusive(
            self.directory / "manifest.json", ExportManifest(run_id=self._run_id, files=files)
        )


def load_export(directory: Path) -> tuple[RunSummary, tuple[QuestionTrace, ...]]:
    """Verify content hashes and event/run consistency before inspecting or comparing results."""

    manifest = ExportManifest.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    if tuple(item.relative_path for item in manifest.files) != ("events.jsonl", "run.json"):
        raise SourceValidationError("unexpected export file list")
    for item in manifest.files:
        path = directory / item.relative_path
        if path.stat().st_size != item.byte_count or sha256_file(path) != item.sha256:
            raise SourceValidationError(f"export integrity failure: {item.relative_path}")
    summary = RunSummary.model_validate_json((directory / "run.json").read_text(encoding="utf-8"))
    events = tuple(
        InstrumentationEvent.model_validate(item) for item in read_jsonl(directory / "events.jsonl")
    )
    if len(events) != manifest.files[0].record_count or not events:
        raise SourceValidationError("export event count mismatch")
    if manifest.run_id != summary.metadata.run_id or any(
        event.context.run_id != manifest.run_id for event in events
    ):
        raise SourceValidationError("export run identity mismatch")
    if (
        events[0].event_type != "evaluation.run.started"
        or events[0].payload != summary.metadata.model_dump(mode="json")
        or events[-1].event_type != "evaluation.run.finished"
        or events[-1].payload != summary.model_dump(mode="json")
    ):
        raise SourceValidationError("export lifecycle does not match summary")
    traces = tuple(
        QuestionTrace.model_validate(event.payload)
        for event in events
        if event.event_type == "evaluation.question.completed"
    )
    if (
        len(traces) != summary.completed_questions + summary.failed_questions
        or tuple(trace.question.question_id for trace in traces)
        != summary.metadata.question_ids[: len(traces)]
    ):
        raise SourceValidationError("export traces do not match run question order/count")
    return summary, traces


def compare_exports(left: Path, right: Path, *, metric: str = "recall@10") -> dict[str, object]:
    """Paired diagnostic comparison with explicit improved/regressed/unchanged question IDs."""

    before, before_traces = load_export(left)
    after, after_traces = load_export(right)
    if before.status != "completed" or after.status != "completed":
        raise ValueError("only completed runs can be compared")
    fields = (
        "dataset_file_sha256",
        "dataset_manifest_sha256",
        "corpus_file_sha256",
        "corpus_manifest_sha256",
        "question_ids",
    )
    if any(getattr(before.metadata, name) != getattr(after.metadata, name) for name in fields):
        raise ValueError("paired comparison requires identical benchmark and corpus provenance")
    if before.metadata.configuration.relevance != after.metadata.configuration.relevance:
        raise ValueError("paired comparison requires identical relevance definitions")
    if metric not in before.metrics or metric not in after.metrics:
        raise ValueError(f"metric unavailable: {metric}")
    groups: dict[str, list[str]] = {"improved": [], "regressed": [], "unchanged": []}
    lower_is_better = metric.startswith(("missing_", "any_gold_missing"))
    for old, new in zip(before_traces, after_traces, strict=True):
        if metric not in old.metrics or metric not in new.metrics:
            continue
        delta = new.metrics[metric] - old.metrics[metric]
        direction = -delta if lower_is_better else delta
        group = "improved" if direction > 0 else "regressed" if direction < 0 else "unchanged"
        groups[group].append(old.question.question_id)
    return {
        "before_run_id": before.metadata.run_id,
        "after_run_id": after.metadata.run_id,
        "metric": metric,
        "delta": after.metrics[metric] - before.metrics[metric],
        "question_ids": groups,
        "note": "Diagnostic comparison; no statistical significance or acceptance claim.",
    }
