"""Exclusive serialization helpers for immutable retrieval indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from epsa_rag.core.exceptions import FrozenArtifactError, IndexIntegrityError
from epsa_rag.data.io import canonical_json, sha256_file
from epsa_rag.data.manifests import ArtifactFile


def write_json_artifact_exclusive(
    path: Path,
    value: Any,
    *,
    relative_path: str,
    record_count: int,
) -> ArtifactFile:
    """Write deterministic compact JSON and return its integrity metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(value))
            stream.write("\n")
    except FileExistsError as error:
        raise FrozenArtifactError(f"refusing to overwrite retrieval artifact: {path}") from error
    return describe_artifact(path, relative_path=relative_path, record_count=record_count)


def write_manifest_exclusive(path: Path, manifest: BaseModel) -> None:
    """Write a human-readable retrieval manifest without overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                manifest.model_dump(mode="json"),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
    except FileExistsError as error:
        raise FrozenArtifactError(f"refusing to overwrite retrieval manifest: {path}") from error


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object with an index-specific error."""

    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise IndexIntegrityError(f"unable to read retrieval artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise IndexIntegrityError(f"retrieval artifact is not a JSON object: {path}")
    return value


def describe_artifact(path: Path, *, relative_path: str, record_count: int) -> ArtifactFile:
    """Create checksummed metadata for a file already written to staging."""

    return ArtifactFile(
        relative_path=relative_path,
        sha256=sha256_file(path),
        byte_count=path.stat().st_size,
        record_count=record_count,
    )


def validate_artifact(path: Path, artifact: ArtifactFile) -> None:
    """Verify a persisted index file against manifest integrity metadata."""

    if not path.is_file():
        raise IndexIntegrityError(f"retrieval artifact does not exist: {path}")
    if path.stat().st_size != artifact.byte_count or sha256_file(path) != artifact.sha256:
        raise IndexIntegrityError(f"retrieval artifact integrity check failed: {path}")
