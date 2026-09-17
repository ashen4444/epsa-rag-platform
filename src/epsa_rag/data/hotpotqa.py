"""Official HotPotQA distractor-development source acquisition and loading."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.request import Request, urlopen
from uuid import uuid4

from pydantic import ValidationError

from epsa_rag.core.exceptions import SourceValidationError
from epsa_rag.core.ids import stable_digest, validate_identifier
from epsa_rag.data.config import PreparationConfiguration
from epsa_rag.data.io import sha256_file
from epsa_rag.data.models import HotPotQASourceExample

OpenUrl = Callable[[Request], BinaryIO]


@dataclass(frozen=True)
class SelectedSource:
    """Selected valid examples plus whole-source audit information."""

    examples: tuple[HotPotQASourceExample, ...]
    source_record_count: int
    eligible_record_count: int
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
    config: PreparationConfiguration,
) -> SelectedSource:
    """Audit, filter, and deterministically select records without loading the source at once."""

    _require_digest(path, config.expected_source_sha256)
    difficulty_filter = getattr(config, "difficulty_filter", None)
    requires_valid_eligibility = config.selection_method == "sha256-rank-valid-v1"
    question_ids: list[str] = []
    eligible_ids: list[str] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(_iter_raw_source(path)):
        question_id = _validate_source_question_id(value, index=index, seen_ids=seen_ids)
        question_ids.append(question_id)
        satisfies_difficulty = difficulty_filter is None or (
            isinstance(value, dict) and value.get("level") == difficulty_filter
        )
        if not satisfies_difficulty:
            continue
        if requires_valid_eligibility:
            try:
                example = HotPotQASourceExample.model_validate(value)
            except ValidationError:
                continue
            if example.question_id != question_id:
                raise SourceValidationError(
                    f"question id changed during validation: {question_id}"
                )
        eligible_ids.append(question_id)

    if not question_ids:
        raise SourceValidationError("HotPotQA source must contain at least one example")
    if config.question_count > len(eligible_ids):
        raise SourceValidationError(
            f"requested {config.question_count} questions from {len(eligible_ids)} eligible "
            f"records in a source containing {len(question_ids)}"
        )
    ranked_ids = sorted(
        eligible_ids,
        key=lambda question_id: (
            stable_digest(str(config.selection_seed), question_id),
            question_id,
        ),
    )
    selected_ids = tuple(ranked_ids[: config.question_count])
    selected_id_set = set(selected_ids)
    selected_by_id: dict[str, HotPotQASourceExample] = {}
    invalid_unselected: list[str] = []

    second_pass_count = 0
    for index, value in enumerate(_iter_raw_source(path)):
        if index >= len(question_ids):
            raise SourceValidationError("HotPotQA source changed while it was being read")
        question_id = question_ids[index]
        second_pass_count += 1
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
        if example.question_id != question_id:
            raise SourceValidationError("HotPotQA source changed while it was being read")
        if question_id in selected_id_set:
            if difficulty_filter is not None and example.level != difficulty_filter:
                raise SourceValidationError(
                    f"selected HotPotQA example {question_id!r} does not satisfy "
                    f"difficulty filter {difficulty_filter!r}"
                )
            selected_by_id[question_id] = example

    if second_pass_count != len(question_ids):
        raise SourceValidationError("HotPotQA source changed while it was being read")

    return SelectedSource(
        examples=tuple(selected_by_id[question_id] for question_id in selected_ids),
        source_record_count=len(question_ids),
        eligible_record_count=len(eligible_ids),
        invalid_unselected_question_ids=tuple(sorted(invalid_unselected)),
    )


def _iter_raw_source(path: Path) -> Iterator[Any]:
    """Stream objects from a top-level JSON array with bounded working memory."""

    decoder = json.JSONDecoder()
    try:
        with path.open(encoding="utf-8") as stream:
            buffer = ""
            position = 0
            end_of_file = False

            def read_more() -> None:
                nonlocal buffer, position, end_of_file
                buffer = buffer[position:]
                position = 0
                block = stream.read(1024 * 1024)
                if block:
                    buffer += block
                else:
                    end_of_file = True

            def skip_whitespace() -> None:
                nonlocal position
                while True:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position < len(buffer) or end_of_file:
                        return
                    read_more()

            read_more()
            skip_whitespace()
            if position >= len(buffer) or buffer[position] != "[":
                raise SourceValidationError("HotPotQA source root must be a JSON array")
            position += 1
            first = True

            while True:
                skip_whitespace()
                if position >= len(buffer):
                    raise SourceValidationError(
                        f"unable to load HotPotQA source {path}: truncated JSON"
                    )
                if buffer[position] == "]":
                    position += 1
                    break
                if not first:
                    if buffer[position] != ",":
                        raise SourceValidationError(
                            f"unable to load HotPotQA source {path}: expected a comma"
                        )
                    position += 1
                    skip_whitespace()

                while True:
                    try:
                        value, end = decoder.raw_decode(buffer, position)
                    except json.JSONDecodeError as error:
                        if end_of_file:
                            raise SourceValidationError(
                                f"unable to load HotPotQA source {path}: {error}"
                            ) from error
                        read_more()
                        continue
                    position = end
                    first = False
                    yield value
                    break

            skip_whitespace()
            while not end_of_file:
                read_more()
                skip_whitespace()
            if position < len(buffer):
                raise SourceValidationError(
                    f"unable to load HotPotQA source {path}: content follows the JSON array"
                )
    except (OSError, UnicodeError) as error:
        raise SourceValidationError(f"unable to load HotPotQA source {path}: {error}") from error


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
        question_ids.append(
            _validate_source_question_id(value, index=index, seen_ids=seen_ids)
        )
    return tuple(question_ids)


def _validate_source_question_id(
    value: Any,
    *,
    index: int,
    seen_ids: set[str],
) -> str:
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
    return question_id
