"""Application factory and executable entry point for the read-only API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Engine

from epsa_observability.connection import create_engine
from epsa_observability.errors import StorageError
from epsa_observability_api import __version__
from epsa_observability_api.errors import (
    DatabaseUnavailableError,
    IncompatibleRunsError,
    NotFoundError,
    StoredRecordError,
)
from epsa_observability_api.models import ApiError
from epsa_observability_api.read_repository import ObservabilityReadRepository
from epsa_observability_api.routes.health import router as health_router
from epsa_observability_api.routes.runs import router as runs_router
from epsa_observability_api.service import ReadRepository
from epsa_observability_api.settings import ApiSettings


def _error_response(
    status_code: int, code: str, message: str, details: dict[str, Any] | None = None
) -> JSONResponse:
    payload = ApiError(code=code, message=message, details=details)
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def _configure_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def not_found(_: Request, error: NotFoundError) -> JSONResponse:
        return _error_response(404, "not_found", str(error))

    @app.exception_handler(IncompatibleRunsError)
    async def incompatible(_: Request, error: IncompatibleRunsError) -> JSONResponse:
        return _error_response(
            409,
            "incompatible_runs",
            str(error),
            {"criteria": list(error.criteria)},
        )

    @app.exception_handler(DatabaseUnavailableError)
    async def unavailable(_: Request, error: DatabaseUnavailableError) -> JSONResponse:
        return _error_response(503, "database_unavailable", str(error))

    @app.exception_handler(StoredRecordError)
    async def invalid_record(_: Request, error: StoredRecordError) -> JSONResponse:
        return _error_response(500, "stored_record_invalid", "Stored experiment data is invalid.")

    @app.exception_handler(RequestValidationError)
    async def validation(_: Request, error: RequestValidationError) -> JSONResponse:
        fields = [
            ".".join(str(part) for part in detail["loc"])
            for detail in error.errors()
            if "loc" in detail
        ]
        return _error_response(
            422,
            "validation_error",
            "Request parameters are invalid.",
            {"fields": fields},
        )

    @app.exception_handler(ValueError)
    async def invalid_cursor(_: Request, error: ValueError) -> JSONResponse:
        return _error_response(422, "validation_error", "Request parameters are invalid.")


def create_app(
    settings: ApiSettings | None = None,
    repository: ReadRepository | None = None,
) -> FastAPI:
    """Create an app without connecting to PostgreSQL until a database route is requested."""

    configured_settings = settings or ApiSettings.from_environment()
    engine: Engine | None = None
    if repository is None and configured_settings.database_url is not None:
        try:
            engine = create_engine(configured_settings.database_url)
            repository = ObservabilityReadRepository(engine)
        except StorageError:
            repository = None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if engine is not None:
            engine.dispose()

    app = FastAPI(
        title="EPSA-RAG Observability API",
        version=__version__,
        description="Read-only inspection of Phase 5A PostgreSQL experiment storage.",
        lifespan=lifespan,
    )
    app.state.read_repository = repository
    _configure_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(runs_router)
    return app


app = create_app()


def main() -> None:
    """Run a local development server; deployment infrastructure remains out of scope."""

    settings = ApiSettings.from_environment()
    uvicorn.run("epsa_observability_api.app:app", host=settings.host, port=settings.port)
