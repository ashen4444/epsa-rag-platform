"""Transactional run registration, append-only events, and small query projections."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from epsa_rag.instrumentation.events import InstrumentationEvent
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine

from epsa_observability import __version__
from epsa_observability.connection import check_schema, transaction
from epsa_observability.errors import ConflictError, StorageError
from epsa_observability.schema import run_questions, runs, trace_events
from epsa_observability.validation import (
    FINISH,
    QUESTION,
    START,
    BenchmarkRole,
    canonical,
    digest,
    snapshot,
    validate_metadata,
    validate_question,
    validate_summary,
)


class ExperimentRepository:
    """The caller owns the engine. No schema migration occurs during construction."""

    def __init__(self, engine: Engine) -> None:
        check_schema(engine)
        self.engine = engine
        self.storage_identity = {
            "adapter_version": __version__,
            "dependencies": {name: version(name) for name in ("sqlalchemy", "psycopg", "alembic")},
            "source_fingerprint": digest(
                {
                    path.relative_to(Path(__file__).parent).as_posix(): path.read_text(
                        encoding="utf-8"
                    )
                    for path in sorted(Path(__file__).parent.rglob("*.py"))
                }
            ),
        }

    def register_run(
        self,
        metadata: dict[str, Any],
        *,
        benchmark_role: BenchmarkRole,
        component_versions: dict[str, str] | None = None,
    ) -> bool:
        """Reserve a run; exact re-registration is a no-op, conflicting identity is rejected."""
        with transaction(self.engine) as connection:
            _, created = self._register(
                connection,
                json.loads(canonical(metadata)),
                benchmark_role,
                component_versions or {},
                {"mode": "live", "delivery": "synchronous"},
            )
            return created

    def _register(
        self,
        connection: Connection,
        payload: dict[str, Any],
        role: BenchmarkRole,
        components: dict[str, str],
        provenance: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        metadata = validate_metadata(payload, role)
        if any(not key.strip() or not value.strip() for key, value in components.items()):
            raise StorageError("Component versions must have nonempty names and versions.")
        canonical(components)
        config = metadata.configuration
        retriever_version = (
            config.retriever.retriever_version
            if config.mode == "hybrid"
            else getattr(config.retriever, config.mode).index_version
        )
        created = (
            connection.execute(
                insert(runs)
                .values(
                    run_id=metadata.run_id,
                    experiment_type=metadata.experiment_type,
                    benchmark_role=role,
                    dataset_version=metadata.dataset_version,
                    corpus_version=metadata.corpus_version,
                    git_commit_sha=metadata.git_commit_sha,
                    git_dirty=metadata.git_dirty,
                    retriever_version=retriever_version,
                    evaluator_version=config.evaluator_version,
                    retrieval_depth=config.retriever.fusion.result_k,
                    configuration_fingerprint=metadata.configuration_fingerprint,
                    metadata_payload=payload,
                    component_versions=components,
                    storage_provenance={**self.storage_identity, **provenance},
                    status="registered",
                    registered_at=datetime.now(UTC),
                    planned_questions=len(metadata.question_ids),
                    attempted_questions=0,
                    completed_questions=0,
                    failed_questions=0,
                    event_count=0,
                )
                .on_conflict_do_nothing(index_elements=[runs.c.run_id])
                .returning(runs.c.run_id)
            ).scalar_one_or_none()
            is not None
        )
        row = self._locked_run(connection, metadata.run_id, include_payload=True)
        if (
            row["metadata_payload"] != payload
            or row["benchmark_role"] != role
            or row["component_versions"] != components
        ):
            raise ConflictError("Run ID already exists with different provenance.")
        if created:
            connection.execute(
                run_questions.insert(),
                [
                    {
                        "run_id": metadata.run_id,
                        "question_id": qid,
                        "position": position,
                        "status": "pending",
                    }
                    for position, qid in enumerate(metadata.question_ids)
                ],
            )
        return row, created

    @staticmethod
    def _locked_run(
        connection: Connection,
        run_id: str,
        *,
        include_payload: bool = False,
    ) -> dict[str, Any]:
        # A 10,000-question run's manifest can be large. Fetch it only at registration/finalization.
        columns = [
            column
            for column in runs.c
            if column.name not in {"metadata_payload", "summary_payload", "storage_provenance"}
        ]
        if include_payload:
            columns.append(runs.c.metadata_payload)
        row = (
            connection.execute(sa.select(*columns).where(runs.c.run_id == run_id).with_for_update())
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise StorageError("Run must be registered before accepting events.")
        return dict(row)

    def ingest_event(
        self,
        event: InstrumentationEvent,
        *,
        benchmark_role: BenchmarkRole,
        component_versions: dict[str, str] | None = None,
    ) -> bool:
        """Commit one event and projections together. Return False for an exact replay."""
        with transaction(self.engine) as connection:
            return self._ingest(
                connection,
                event,
                benchmark_role,
                component_versions or {},
                {"mode": "live", "delivery": "synchronous"},
            )

    def _ingest(
        self,
        connection: Connection,
        event: InstrumentationEvent,
        role: BenchmarkRole,
        components: dict[str, str],
        provenance: dict[str, Any],
    ) -> bool:
        event, envelope = snapshot(event)
        run_id = event.context.run_id
        if event.event_type == START:
            if event.payload.get("run_id") != run_id:
                raise StorageError("Start event and metadata run identities disagree.")
            row, _ = self._register(connection, event.payload, role, components, provenance)
        else:
            row = self._locked_run(connection, run_id)
            if row["benchmark_role"] != role or row["component_versions"] != components:
                raise ConflictError("Event registration context differs from the stored run.")
        content_hash = digest(envelope)
        existing = connection.execute(
            sa.select(trace_events.c.content_sha256).where(
                trace_events.c.event_id == event.event_id
            )
        ).scalar_one_or_none()
        if existing is not None:
            if existing != content_hash:
                raise ConflictError("Event ID already exists with different content.")
            return False
        if row["status"] in {"completed", "failed"}:
            raise ConflictError("Finalized runs cannot accept new events.")
        kind, qid = event.event_type, event.context.question_id
        if kind in {START, QUESTION, FINISH}:
            if (
                event.source != "retrieval-evaluation"
                or event.source_version != row["evaluator_version"]
            ):
                raise StorageError("Evaluation event source/version differs from run provenance.")
            if (kind == QUESTION) != (qid is not None):
                raise StorageError("Evaluation event has an invalid question context.")
        question_position = None
        if qid is not None:
            question_position = connection.execute(
                sa.select(run_questions.c.position).where(
                    run_questions.c.run_id == run_id,
                    run_questions.c.question_id == qid,
                )
            ).scalar_one_or_none()
            if question_position is None:
                raise StorageError("Event question is not part of the registered run.")
        updates: dict[str, Any] = {"event_count": row["event_count"] + 1}
        if kind == START:
            if row["status"] != "registered":
                raise ConflictError("Run already has a start event.")
            updates.update(status="running", started_event_at=event.occurred_at)
        else:
            if row["status"] != "running":
                raise StorageError("Run must have a start event before results or diagnostics.")
            if kind == QUESTION:
                trace = validate_question(
                    event,
                    retriever_version=row["retriever_version"],
                    retrieval_depth=row["retrieval_depth"],
                )
                position = row["attempted_questions"]
                if row["failed_questions"] or position != question_position:
                    raise ConflictError(
                        "Question result is duplicate, out of order, or after failure."
                    )
                connection.execute(
                    run_questions.update()
                    .where(
                        run_questions.c.run_id == run_id,
                        run_questions.c.question_id == qid,
                    )
                    .values(
                        status=trace.status,
                        error_type=trace.error_type,
                        metrics=trace.metrics,
                        latency_ms=trace.latency_ms,
                        retrieval_core_latency_ms=trace.retrieval_core_latency_ms,
                    )
                )
                updates.update(
                    attempted_questions=position + 1,
                    completed_questions=row["completed_questions"] + (trace.status == "completed"),
                    failed_questions=row["failed_questions"] + (trace.status == "failed"),
                )
            elif kind == FINISH:
                row["metadata_payload"] = connection.execute(
                    sa.select(runs.c.metadata_payload).where(runs.c.run_id == run_id)
                ).scalar_one()
                recorded_metrics = (
                    connection.execute(
                        sa.select(run_questions.c.metrics)
                        .where(
                            run_questions.c.run_id == run_id,
                            run_questions.c.status != "pending",
                        )
                        .order_by(run_questions.c.position)
                    )
                    .scalars()
                    .all()
                )
                summary = validate_summary(event, row, list(recorded_metrics))
                updates.update(
                    status=summary.status,
                    started_at=summary.started_at,
                    ended_at=summary.ended_at,
                    summary_payload=event.payload,
                )
        inserted = connection.execute(
            insert(trace_events)
            .values(
                event_id=event.event_id,
                run_id=run_id,
                question_id=qid,
                trace_id=event.context.trace_id,
                event_type=kind,
                source=event.source,
                source_version=event.source_version,
                occurred_at=event.occurred_at,
                ingested_at=datetime.now(UTC),
                sequence=updates["event_count"],
                content_sha256=content_hash,
                envelope=envelope,
            )
            .on_conflict_do_nothing(index_elements=[trace_events.c.event_id])
            .returning(trace_events.c.event_id)
        ).scalar_one_or_none()
        if inserted is None:
            # A concurrent different run won the globally unique event ID. Roll back projections.
            raise ConflictError("Event ID was concurrently claimed by another run.")
        connection.execute(runs.update().where(runs.c.run_id == run_id).values(**updates))
        return True

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with transaction(self.engine) as connection:
            row = (
                connection.execute(sa.select(runs).where(runs.c.run_id == run_id))
                .mappings()
                .one_or_none()
            )
            return dict(row) if row is not None else None

    def list_runs(
        self,
        *,
        limit: int = 50,
        benchmark_role: BenchmarkRole | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise StorageError("List limit must be between 1 and 1000.")
        query = (
            sa.select(
                runs.c.run_id,
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
            .order_by(runs.c.registered_at.desc(), runs.c.run_id)
            .limit(limit)
        )
        if benchmark_role is not None:
            query = query.where(runs.c.benchmark_role == benchmark_role)
        if status is not None:
            query = query.where(runs.c.status == status)
        with transaction(self.engine) as connection:
            return [dict(row) for row in connection.execute(query).mappings()]

    def get_question_trace(self, run_id: str, question_id: str) -> dict[str, Any] | None:
        with transaction(self.engine) as connection:
            connection.execute(sa.text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))
            row = (
                connection.execute(
                    sa.select(run_questions).where(
                        run_questions.c.run_id == run_id,
                        run_questions.c.question_id == question_id,
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                return None
            events = (
                connection.execute(
                    sa.select(trace_events.c.envelope)
                    .where(
                        trace_events.c.run_id == run_id,
                        trace_events.c.question_id == question_id,
                    )
                    .order_by(trace_events.c.sequence)
                )
                .scalars()
                .all()
            )
            return {**dict(row), "events": list(events)}
