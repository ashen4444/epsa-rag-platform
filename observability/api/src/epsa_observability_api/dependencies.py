"""FastAPI dependencies, with a replaceable read repository for tests."""

from __future__ import annotations

from typing import cast

from fastapi import Request

from epsa_observability_api.errors import DatabaseUnavailableError
from epsa_observability_api.service import ReadRepository


def get_read_repository(request: Request) -> ReadRepository:
    """Return the configured query adapter without connecting during application startup."""

    repository = getattr(request.app.state, "read_repository", None)
    if repository is None:
        raise DatabaseUnavailableError("Database is unavailable or its schema is incompatible.")
    return cast(ReadRepository, repository)
