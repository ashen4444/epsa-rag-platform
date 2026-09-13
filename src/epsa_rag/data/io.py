"""Canonical serialization and integrity helpers for Phase 2 artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from epsa_rag.core.exceptions import FrozenArtifactError
from epsa_rag.data.manifests import ArtifactFile


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    """Calculate a streaming SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically while retaining Unicode text."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def write_jsonl_exclusive(
    path: Path,
    records: Iterable[BaseModel],
    *,
    relative_path: str,
) -> ArtifactFile:
    """Write canonical JSON Lines without permitting version overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(canonical_json(record.model_dump(mode="json")))
                stream.write("\n")
                count += 1
    except FileExistsError as error:
        raise FrozenArtifactError(f"refusing to overwrite frozen artifact: {path}") from error
    return ArtifactFile(
        relative_path=relative_path,
        sha256=sha256_file(path),
        byte_count=path.stat().st_size,
        record_count=count,
    )


def write_json_exclusive(path: Path, model: BaseModel) -> None:
    """Write a human-readable manifest without overwriting an existing version."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                model.model_dump(mode="json"),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
    except FileExistsError as error:
        raise FrozenArtifactError(f"refusing to overwrite frozen manifest: {path}") from error


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield decoded objects from a UTF-8 JSON Lines file."""

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} of {path} is not a JSON object")
            yield value
