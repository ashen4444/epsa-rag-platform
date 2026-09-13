"""Configuration primitives shared by research components."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    """Strict, immutable base for explicit research configuration.

    Environment loading is intentionally outside this contract. Research-critical values must be
    represented explicitly and persisted by the experiment layer introduced in a later phase.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    def canonical_json(self) -> str:
        """Return deterministic JSON suitable for persistence and hashing."""

        data: dict[str, Any] = self.model_dump(mode="json")
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def fingerprint(self) -> str:
        """Return a SHA-256 fingerprint of the complete serialized configuration."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

