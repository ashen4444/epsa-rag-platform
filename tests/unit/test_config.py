from __future__ import annotations

import pytest
from pydantic import ValidationError

from epsa_rag.core.config import ConfigModel


class ExampleConfig(ConfigModel):
    limit: int
    strategy: str


def test_config_serialization_and_fingerprint_are_deterministic() -> None:
    first = ExampleConfig(limit=10, strategy="rrf")
    second = ExampleConfig(strategy="rrf", limit=10)

    assert first.canonical_json() == '{"limit":10,"strategy":"rrf"}'
    assert first.fingerprint() == second.fingerprint()


def test_config_is_frozen_and_rejects_extra_fields() -> None:
    config = ExampleConfig(limit=10, strategy="rrf")

    with pytest.raises(ValidationError):
        config.limit = 20

    with pytest.raises(ValidationError):
        ExampleConfig(limit=10, strategy="rrf", undocumented=True)

