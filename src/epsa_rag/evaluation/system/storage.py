"""Resumable, checksummed local storage for paired system evaluations."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from epsa_rag.core.exceptions import FrozenArtifactError, SourceValidationError
from epsa_rag.core.models import ContractModel
from epsa_rag.data.io import canonical_json, sha256_file, write_json_exclusive
from epsa_rag.data.manifests import ArtifactFile
from epsa_rag.evaluation.system.models import (
    SystemCondition,
    SystemQuestionTrace,
    SystemRunMetadata,
    SystemRunSummary,
)


class SystemExportManifest(ContractModel):
    """Final immutable file inventory for one completed resumable run."""

    schema_version: str = "1.0"
    run_id: str
    files: tuple[ArtifactFile, ...]


class ResumableSystemRunStore:
    """Persist each question-condition result before moving to the next API call."""

    def __init__(self, root: Path, metadata: SystemRunMetadata) -> None:
        self.directory = root / metadata.run_id
        self.metadata_path = self.directory / "metadata.json"
        self.summary_path = self.directory / "summary.json"
        if self.summary_path.exists():
            raise FrozenArtifactError(f"completed run ID already exists: {metadata.run_id}")
        if self.metadata_path.exists():
            existing = SystemRunMetadata.model_validate_json(
                self.metadata_path.read_text(encoding="utf-8")
            )
            if existing != metadata:
                raise SourceValidationError("resume metadata differs from the reserved run")
        else:
            self.directory.mkdir(parents=True, exist_ok=False)
            write_json_exclusive(self.metadata_path, metadata)
        self.metadata = metadata

    def trace_path(self, condition: SystemCondition, question_id: str) -> Path:
        return self.directory / "questions" / condition.key / f"{question_id}.json"

    def load_trace(
        self, condition: SystemCondition, question_id: str
    ) -> SystemQuestionTrace | None:
        path = self.trace_path(condition, question_id)
        if not path.exists():
            return None
        trace = SystemQuestionTrace.model_validate_json(path.read_text(encoding="utf-8"))
        if trace.condition != condition or trace.question.question_id != question_id:
            raise SourceValidationError("stored system trace identity mismatch")
        return trace

    def write_trace(self, trace: SystemQuestionTrace) -> None:
        path = self.trace_path(trace.condition, trace.question.question_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = SystemQuestionTrace.model_validate_json(path.read_text(encoding="utf-8"))
            if existing != trace:
                raise FrozenArtifactError("refusing to replace a completed system trace")
            return
        temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        try:
            temporary.write_text(
                canonical_json(trace.model_dump(mode="json")) + "\n", encoding="utf-8"
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def finalize(self, summary: SystemRunSummary) -> None:
        if summary.metadata != self.metadata:
            raise ValueError("summary metadata differs from reserved run")
        write_json_exclusive(self.summary_path, summary)
        paths = [self.metadata_path, self.summary_path]
        paths.extend(sorted((self.directory / "questions").rglob("*.json")))
        files = tuple(
            ArtifactFile(
                relative_path=path.relative_to(self.directory).as_posix(),
                sha256=sha256_file(path),
                byte_count=path.stat().st_size,
                record_count=1,
            )
            for path in paths
        )
        manifest = SystemExportManifest(run_id=self.metadata.run_id, files=files)
        write_json_exclusive(self.directory / "manifest.json", manifest)


def load_system_export(
    directory: Path,
) -> tuple[SystemRunSummary, tuple[SystemQuestionTrace, ...]]:
    manifest = SystemExportManifest.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    for artifact in manifest.files:
        path = directory / artifact.relative_path
        if path.stat().st_size != artifact.byte_count or sha256_file(path) != artifact.sha256:
            raise SourceValidationError(
                f"system export integrity failure: {artifact.relative_path}"
            )
    summary = SystemRunSummary.model_validate_json(
        (directory / "summary.json").read_text(encoding="utf-8")
    )
    traces = tuple(
        SystemQuestionTrace.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((directory / "questions").rglob("*.json"))
    )
    if summary.metadata.run_id != manifest.run_id:
        raise SourceValidationError("system export run identity mismatch")
    expected = len(summary.metadata.question_ids) * len(
        summary.metadata.configuration.conditions()
    )
    if len(traces) != expected:
        raise SourceValidationError("system export trace count mismatch")
    return summary, traces
