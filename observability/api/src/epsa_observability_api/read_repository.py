"""Read-only SQLAlchemy queries over the frozen Phase 5A schema."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from epsa_observability.connection import check_schema
from epsa_observability.errors import StorageError
from epsa_observability.schema import run_questions, runs, trace_events
from epsa_observability_api.errors import DatabaseUnavailableError


class ObservabilityReadRepository:
    """A deliberately narrow query adapter with no mutation methods."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def check_compatibility(self) -> None:
        try:
            check_schema(self._engine)
        except StorageError:
            raise DatabaseUnavailableError(
                "Database is unavailable or its schema is incompatible."
            ) from None

    @contextmanager
    def _connection(self, *, repeatable_read: bool = False) -> Iterator[Connection]:
        try:
            with self._engine.connect() as raw_connection:
                connection = (
                    raw_connection.execution_options(isolation_level="REPEATABLE READ")
                    if repeatable_read
                    else raw_connection
                )
                with connection.begin():
                    yield connection
        except SQLAlchemyError:
            raise DatabaseUnavailableError(
                "Database is unavailable or its schema is incompatible."
            ) from None

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = (
                connection.execute(sa.select(runs).where(runs.c.run_id == run_id))
                .mappings()
                .one_or_none()
            )
            return dict(row) if row is not None else None

    def list_runs(
        self,
        *,
        limit: int,
        benchmark_role: str | None,
        status: str | None,
        experiment_type: str | None,
        dataset_version: str | None,
        corpus_version: str | None,
        retriever_version: str | None,
        after_registered_at: datetime | None,
        after_run_id: str | None,
    ) -> list[dict[str, Any]]:
        columns = (
            runs.c.run_id,
            runs.c.experiment_type,
            runs.c.benchmark_role,
            runs.c.dataset_version,
            runs.c.corpus_version,
            runs.c.retriever_version,
            runs.c.status,
            runs.c.registered_at,
            runs.c.planned_questions,
            runs.c.completed_questions,
            runs.c.failed_questions,
        )
        query = sa.select(*columns)
        if benchmark_role is not None:
            query = query.where(runs.c.benchmark_role == benchmark_role)
        if status is not None:
            query = query.where(runs.c.status == status)
        if experiment_type is not None:
            query = query.where(runs.c.experiment_type == experiment_type)
        if dataset_version is not None:
            query = query.where(runs.c.dataset_version == dataset_version)
        if corpus_version is not None:
            query = query.where(runs.c.corpus_version == corpus_version)
        if retriever_version is not None:
            query = query.where(runs.c.retriever_version == retriever_version)
        if after_registered_at is not None and after_run_id is not None:
            query = query.where(
                sa.or_(
                    runs.c.registered_at < after_registered_at,
                    sa.and_(
                        runs.c.registered_at == after_registered_at,
                        runs.c.run_id > after_run_id,
                    ),
                )
            )
        query = query.order_by(runs.c.registered_at.desc(), runs.c.run_id).limit(limit + 1)
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(query).mappings()]

    def list_questions(
        self,
        run_id: str,
        *,
        limit: int,
        status: str | None,
        has_failure: bool | None,
        question_id: str | None,
        after_position: int | None,
    ) -> list[dict[str, Any]]:
        columns = (
            run_questions.c.question_id,
            run_questions.c.position,
            run_questions.c.status,
            run_questions.c.error_type,
            run_questions.c.metrics,
            run_questions.c.latency_ms,
            run_questions.c.retrieval_core_latency_ms,
        )
        query = sa.select(*columns).where(run_questions.c.run_id == run_id)
        if status is not None:
            query = query.where(run_questions.c.status == status)
        if has_failure is not None:
            query = query.where(
                run_questions.c.error_type.is_not(None)
                if has_failure
                else run_questions.c.error_type.is_(None)
            )
        if question_id is not None:
            query = query.where(run_questions.c.question_id == question_id)
        if after_position is not None:
            query = query.where(run_questions.c.position > after_position)
        query = query.order_by(run_questions.c.position).limit(limit + 1)
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(query).mappings()]

    def get_question_trace(self, run_id: str, question_id: str) -> dict[str, Any] | None:
        question_columns = (
            run_questions.c.question_id,
            run_questions.c.position,
            run_questions.c.status,
            run_questions.c.error_type,
            run_questions.c.metrics,
            run_questions.c.latency_ms,
            run_questions.c.retrieval_core_latency_ms,
        )
        event_columns = (
            trace_events.c.event_id,
            trace_events.c.trace_id,
            trace_events.c.event_type,
            trace_events.c.source,
            trace_events.c.source_version,
            trace_events.c.occurred_at,
            trace_events.c.ingested_at,
            trace_events.c.sequence,
            trace_events.c.content_sha256,
            trace_events.c.envelope,
        )
        with self._connection(repeatable_read=True) as connection:
            question = (
                connection.execute(
                    sa.select(*question_columns).where(
                        run_questions.c.run_id == run_id,
                        run_questions.c.question_id == question_id,
                    )
                )
                .mappings()
                .one_or_none()
            )
            if question is None:
                return None
            events = connection.execute(
                sa.select(*event_columns)
                .where(
                    trace_events.c.run_id == run_id,
                    trace_events.c.question_id == question_id,
                )
                .order_by(trace_events.c.sequence)
            ).mappings()
            return {"question": dict(question), "events": [dict(event) for event in events]}
