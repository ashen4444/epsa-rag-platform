"""Explicit, minimal application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """Runtime settings; a missing URL leaves only liveness available."""

    database_url: str | None
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_environment(cls) -> ApiSettings:
        database_url = os.environ.get("EPSA_DATABASE_URL")
        return cls(database_url=database_url.strip() if database_url else None)
