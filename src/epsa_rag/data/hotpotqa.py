"""Official HotPotQA distractor-development source acquisition and loading."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.request import Request, urlopen
from uuid import uuid4

from pydantic import ValidationError

from epsa_rag.core.exceptions import SourceValidationError
from epsa_rag.core.ids import stable_digest, validate_identifier
from epsa_rag.data.config import PreparationConfig
from epsa_rag.data.io import sha256_file
from epsa_rag.data.models import HotPotQASourceExample

OpenUrl = Callable[[Request], BinaryIO]


@dataclass(frozen=True)
class SelectedSource:
    """Selected valid examples plus whole-source audit information."""

    examples: tuple[HotPotQASourceExample, ...]
    source_record_count: int
    invalid_unselected_question_ids: tuple[str, ...]


def download_source(
    *,
    uri: str,
    destination: Path,
    expected_sha256: str,
    open_url: OpenUrl | None = None,
) -> Path:
    """Download the official source atomically and verify its expected digest."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _require_digest(destination, expected_sha256)
        return destination

    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    opener = open_url or _open_request
    request = Request(uri, headers={"User-Agent": "epsa-rag-platform/0.1"})
    try:
        with opener(request) as response, temporary.open("xb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        _require_digest(temporary, expected_sha256)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _open_request(request: Request) -> BinaryIO:
    """Open a source request with a bounded network timeout."""

    return cast(BinaryIO, urlopen(request, timeout=120))


def _require_digest(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise SourceValidationError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, found {actual}"
        )


def load_source(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[HotPotQASourceExample, ...]:
    """Load and validate all examples from an official HotPotQA JSON file."""

    if expected_sha256 is not None:
        _require_digest(path, expected_sha256)
    raw = _load_raw_source(path)
    question_ids = _validate_source_question_ids(raw)
    examples: list[HotPotQASourceExample] = []
    for index, (value, question_id) in enumerate(zip(raw, question_ids, strict=True)):
        try:
            example = HotPotQASourceExample.model_validate(value)
        except ValidationError as error:
            raise SourceValidationError(
                f"invalid HotPotQA example at index {index}: {error}"
            ) from error
        if example.question_id != question_id:
            raise SourceValidationError(f"question id changed during validation: {question_id}")
        examples.append(example)
    return tuple(examples)


def load_selected_source(
    path: Path,
    *,
    config: PreparationConfig,
) -> SelectedSource:
    """Audit the whole source and fully validate the deterministically selected records."""

    _require_digest(path, config.expected_source_sha256)
    raw = _load_raw_source(path)
    question_ids = _validate_source_question_ids(raw)
    if config.question_count > len(raw):
        raise SourceValidationError(
            f"requested {config.question_count} questions from a source containing {len(raw)}"
        )
    ranked_ids = sorted(
        question_ids,
        key=lambda question_id: (
            stable_digest(str(config.selection_seed), question_id),
            question_id,
        ),
    )
    selected_ids = tuple(ranked_ids[: config.question_count])
    selected_id_set = set(selected_ids)
    selected_by_id: dict[str, HotPotQASourceExample] = {}
    invalid_unselected: list[str] = []

    for index, (value, question_id) in enumerate(zip(raw, question_ids, strict=True)):
        try:
            example = HotPotQASourceExample.model_validate(value)
        except ValidationError as error:
            if question_id in selected_id_set:
                raise SourceValidationError(
                    f"selected HotPotQA example {question_id!r} at index {index} "
                    f"is invalid: {error}"
                ) from error
            invalid_unselected.append(question_id)
            continue
        if question_id in selected_id_set:
            selected_by_id[question_id] = example

    return SelectedSource(
        examples=tuple(selected_by_id[question_id] for question_id in selected_ids),
        source_record_count=len(raw),
        invalid_unselected_question_ids=tuple(sorted(invalid_unselected)),
    )


def _load_raw_source(path: Path) -> list[Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            raw: Any = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise SourceValidationError(f"unable to load HotPotQA source {path}: {error}") from error
    if not isinstance(raw, list):
        raise SourceValidationError("HotPotQA source root must be a JSON array")
    if not raw:
        raise SourceValidationError("HotPotQA source must contain at least one example")
    return raw


def _validate_source_question_ids(raw: list[Any]) -> tuple[str, ...]:
    question_ids: list[str] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(raw):
        if not isinstance(value, dict) or not isinstance(value.get("_id"), str):
            raise SourceValidationError(f"HotPotQA example at index {index} has no string _id")
        try:
            question_id = validate_identifier(value["_id"])
        except ValueError as error:
            raise SourceValidationError(
                f"HotPotQA example at index {index} has an invalid _id"
            ) from error
        if question_id in seen_ids:
            raise SourceValidationError(f"duplicate question id: {question_id}")
        seen_ids.add(question_id)
        question_ids.append(question_id)
    return tuple(question_ids)
