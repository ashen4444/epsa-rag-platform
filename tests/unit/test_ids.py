from __future__ import annotations

import pytest

from epsa_rag.core.ids import (
    make_content_id,
    make_evidence_unit_id,
    stable_digest,
    validate_identifier,
)


def test_identifier_validation_normalizes_outer_whitespace() -> None:
    assert validate_identifier("  run-01  ") == "run-01"


@pytest.mark.parametrize("value", ["", "has space", "bad/value", "-leading"])
def test_identifier_validation_rejects_non_portable_values(value: str) -> None:
    with pytest.raises(ValueError, match="identifier"):
        validate_identifier(value)


def test_stable_digest_preserves_part_boundaries() -> None:
    assert stable_digest("ab", "c") != stable_digest("a", "bc")
    assert stable_digest("ab", "c") == stable_digest("ab", "c")


def test_content_id_is_deterministic_and_namespaced() -> None:
    identifier = make_content_id("chunk", "Title", "Paragraph")

    assert identifier.startswith("chunk:")
    assert identifier == make_content_id("chunk", "Title", "Paragraph")


def test_content_id_requires_content() -> None:
    with pytest.raises(ValueError, match="at least one"):
        make_content_id("chunk")


def test_evidence_unit_id_uses_native_sentence_index() -> None:
    assert make_evidence_unit_id("chunk:abc", 2) == "chunk:abc::s2"

    with pytest.raises(ValueError, match="non-negative"):
        make_evidence_unit_id("chunk:abc", -1)
