"""Liveness and Phase 5A database-readiness routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from epsa_observability.schema import REVISION
from epsa_observability_api import __version__
from epsa_observability_api.dependencies import get_read_repository
from epsa_observability_api.models import DatabaseHealthResponse, HealthResponse
from epsa_observability_api.service import ReadRepository

router = APIRouter(tags=["health"])
ReadRepositoryDependency = Annotated[ReadRepository, Depends(get_read_repository)]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report process liveness without making PostgreSQL a startup dependency."""

    return HealthResponse(version=__version__)


@router.get("/health/database", response_model=DatabaseHealthResponse)
def database_health(
    repository: ReadRepositoryDependency,
) -> DatabaseHealthResponse:
    """Confirm the configured database has exactly the Phase 5A schema revision."""

    repository.check_compatibility()
    return DatabaseHealthResponse(version=__version__, expected_revision=REVISION)
