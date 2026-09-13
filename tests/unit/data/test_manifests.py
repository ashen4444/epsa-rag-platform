from datetime import datetime

import pytest
from pydantic import ValidationError

from epsa_rag.data.manifests import GenerationManifest


def test_generation_manifest_requires_timezone_aware_creation_time() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        GenerationManifest(generated_at=datetime(2026, 9, 13), configuration_fingerprint="a" * 64)

