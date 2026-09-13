"""Validation and deterministic helpers for project identifiers."""

from __future__ import annotations

import hashlib
import re
from typing import Annotated

from pydantic import AfterValidator

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_MAX_IDENTIFIER_LENGTH = 255


def validate_identifier(value: str) -> str:
    """Validate a portable, non-empty project identifier."""

    normalized = value.strip()
    if not normalized:
        raise ValueError("identifier must not be empty")
    if len(normalized) > _MAX_IDENTIFIER_LENGTH:
        raise ValueError(f"identifier must be at most {_MAX_IDENTIFIER_LENGTH} characters")
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "identifier must start with an ASCII letter or digit and contain only "
            "letters, digits, '.', '_', ':', or '-'"
        )
    return normalized


Identifier = Annotated[str, AfterValidator(validate_identifier)]


def stable_digest(*parts: str) -> str:
    """Hash ordered text parts without delimiter-collision ambiguity."""

    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def make_content_id(namespace: str, *parts: str) -> str:
    """Build a deterministic identifier while leaving content selection to the caller."""

    validated_namespace = validate_identifier(namespace)
    if not parts:
        raise ValueError("at least one content part is required")
    return f"{validated_namespace}:{stable_digest(*parts)}"


def make_evidence_unit_id(chunk_id: str, sentence_index: int) -> str:
    """Build the documented stable identifier for a sentence within a chunk."""

    validated_chunk_id = validate_identifier(chunk_id)
    if sentence_index < 0:
        raise ValueError("sentence_index must be non-negative")
    return f"{validated_chunk_id}::s{sentence_index}"

