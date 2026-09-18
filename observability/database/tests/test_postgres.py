from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
import sqlalchemy as sa
from alembic import command
from conftest import make_events
from epsa_rag.instrumentation.events import InstrumentationEvent
from epsa_rag.instrumentation.sinks import InstrumentationSink
from test_validation import write_export

from epsa_observability import cli
from epsa_observability.connection import check_schema
from epsa_observability.errors import ConflictError, StorageError
from epsa_observability.importer import import_export
from epsa_observability.migrate import migration_config, upgrade
from epsa_observability.repository import ExperimentRepository
from epsa_observability.schema import run_questions, runs, trace_events
from epsa_observability.sink import PostgresInstrumentationSink
from epsa_observability.validation import digest

pytestmark = pytest.mark.postgres


def ingest(repository, events):
    sink = PostgresInstrumentationSink(repository, benchmark_role="diagnostic")
    for event in events:
        sink.emit(event)
    return sink


def test_migration_cycle_and_revision_guard(database_engine):
    with pytest.raises(StorageError, match="missing"):
        ExperimentRepository(database_engine)
    upgrade(database_engine)
    upgrade(database_engine)
    with database_engine.begin() as connection:
        assert set(sa.inspect(connection).get_table_names(schema="epsa_experiments")) == {
            "runs",
            "run_questions",
            "trace_events",
            "alembic_version",
        }
        command.downgrade(migration_config(connection), "base")
    upgrade(database_engine)
    check_schema(database_engine)
    with database_engine.begin() as connection:
        connection.execute(
            sa.text("UPDATE epsa_experiments.alembic_version SET version_num='future'")
        )
    with pytest.raises(StorageError, match="incompatible"):
        check_schema(database_engine)
    with pytest.raises(StorageError, match="Migration history"):
        upgrade(database_engine)


def test_complete_trace_round_trip_and_idempotency(repository, events):
    sink = ingest(repository, events)
    assert isinstance(sink, InstrumentationSink)
    row = repository.get_run("storage-test")
    assert row["status"] == "completed"
    assert row["metadata_payload"] == events[0].payload
    assert row["summary_payload"] == events[-1].payload
    assert row["component_versions"] == {}
    assert row["storage_provenance"]["mode"] == "live"
    assert row["event_count"] == 4
    trace = repository.get_question_trace("storage-test", "q-0")
    assert trace["events"] == [events[1].model_dump(mode="json")]
    assert trace["latency_ms"] == events[1].payload["latency_ms"]
    for event in events:
        assert repository.ingest_event(event, benchmark_role="diagnostic") is False
    assert repository.get_run("storage-test") == row
    assert repository.list_runs(benchmark_role="diagnostic", status="completed")[0]["run_id"] == (
        "storage-test"
    )
    assert repository.list_runs(status="failed") == []
    assert repository.get_run("absent") is None
    assert repository.get_question_trace("storage-test", "absent") is None
    with pytest.raises(StorageError, match="limit"):
        repository.list_runs(limit=0)


def test_registration_and_pending_results(repository, events):
    assert repository.register_run(events[0].payload, benchmark_role="diagnostic")
    assert not repository.register_run(events[0].payload, benchmark_role="diagnostic")
    assert repository.get_run("storage-test")["status"] == "registered"
    assert repository.get_question_trace("storage-test", "q-1")["events"] == []
    with pytest.raises(StorageError, match="start event"):
        repository.ingest_event(events[1], benchmark_role="diagnostic")
    ingest(repository, events[:2])
    row = repository.get_run("storage-test")
    assert row["status"] == "running"
    assert row["summary_payload"] is None
    assert row["started_at"] is None
    assert row["started_event_at"] == events[0].occurred_at
    assert row["completed_questions"] == 1
    assert repository.get_question_trace("storage-test", "q-1")["status"] == "pending"


@pytest.mark.parametrize(
    "options", [dict(fail_at="q-1"), dict(fail_at="q-0", warmups=1), dict(limit=1)]
)
def test_failed_warmup_and_subset_runs(repository, options):
    events, summary = make_events(**options)
    ingest(repository, events)
    row = repository.get_run("storage-test")
    assert row["summary_payload"] == summary.model_dump(mode="json")
    assert not row["summary_payload"]["full_benchmark"]
    assert row["failed_questions"] == summary.failed_questions
    assert "sensitive error text" not in json.dumps(row, default=str)


@pytest.mark.parametrize(
    "problem",
    [
        "event_content",
        "run_metadata",
        "new_finish",
        "new_question",
        "new_start",
        "component_versions",
        "cross_run",
    ],
)
def test_immutable_identity_conflicts(repository, events, problem):
    ingest(repository, events)
    before = repository.get_run("storage-test")
    event = events[1]
    if problem == "event_content":
        event = event.model_copy(update={"payload": {**event.payload, "latency_ms": 999}})
    elif problem == "run_metadata":
        payload = deepcopy(events[0].payload)
        payload["git_commit_sha"] = "f" * 40
        with pytest.raises(ConflictError):
            repository.register_run(payload, benchmark_role="diagnostic")
        return
    elif problem == "component_versions":
        with pytest.raises(ConflictError):
            repository.ingest_event(
                events[0], benchmark_role="diagnostic", component_versions={"future": "v1"}
            )
        return
    elif problem == "cross_run":
        other, _ = make_events("other-run")
        event = other[0].model_copy(update={"event_id": events[0].event_id})
    else:
        event = {"new_finish": events[-1], "new_question": events[1], "new_start": events[0]}[
            problem
        ]
        event = event.model_copy(update={"event_id": "event:replacement"})
    with pytest.raises(ConflictError):
        repository.ingest_event(event, benchmark_role="diagnostic")
    assert repository.get_run("storage-test") == before
    assert repository.get_run("other-run") is None


@pytest.mark.parametrize(
    "problem",
    [
        "out_of_order",
        "source",
        "question_context",
        "missing_run",
        "unknown_question",
        "metadata_id",
    ],
)
def test_invalid_delivery_rolls_back(repository, events, problem):
    if problem != "missing_run":
        ingest(repository, events[:1])
    raw = events[2 if problem == "out_of_order" else 1].model_dump(mode="json")
    if problem == "source":
        raw["source_version"] = "wrong"
    elif problem == "question_context":
        raw["context"]["question_id"] = None
    elif problem == "unknown_question":
        raw["context"]["question_id"] = "unknown"
    elif problem == "metadata_id":
        raw = events[0].model_dump(mode="json")
        raw["context"]["run_id"] = "different"
    with pytest.raises(StorageError):
        repository.ingest_event(
            InstrumentationEvent.model_validate(raw), benchmark_role="diagnostic"
        )
    if problem != "missing_run":
        row = repository.get_run("storage-test")
        assert row["event_count"] == 1
        assert row["attempted_questions"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed_questions", 0),
        ("failed_questions", 1),
        ("unattempted_questions", 1),
        ("planned_questions", 3),
        ("full_benchmark", False),
        ("warmup_completed", 1),
        ("error_type", "Wrong"),
        ("metrics", {}),
        ("metric_denominators", {}),
        ("ended_at", "2000-01-01T00:00:00Z"),
    ],
)
def test_invalid_summary_never_finalizes(repository, events, field, value):
    ingest(repository, events[:-1])
    payload = deepcopy(events[-1].payload)
    payload[field] = value
    with pytest.raises(StorageError):
        repository.ingest_event(
            events[-1].model_copy(update={"payload": payload}), benchmark_role="diagnostic"
        )
    assert repository.get_run("storage-test")["status"] == "running"
    assert repository.get_run("storage-test")["event_count"] == 3


def test_aggregate_values_checked_without_rewriting(repository, events):
    ingest(repository, events[:-1])
    payload = deepcopy(events[-1].payload)
    payload["metrics"]["recall@10"] = 0.125
    with pytest.raises(StorageError, match="Aggregate"):
        repository.ingest_event(
            events[-1].model_copy(update={"payload": payload}), benchmark_role="diagnostic"
        )


def test_generic_diagnostics_preserved_without_future_schemas(repository, events):
    ingest(repository, events[:1])
    event = InstrumentationEvent(
        context=events[1].context,
        event_type="diagnostic.note",
        source="fixture",
        payload={"native_text": "  exact\n"},
    )
    ingest(repository, [event, *events[1:]])
    assert repository.get_question_trace("storage-test", "q-0")["events"][0] == (
        event.model_dump(mode="json")
    )


def test_concurrent_registration_and_duplicate_delivery(repository, events):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: repository.register_run(events[0].payload, benchmark_role="diagnostic"),
                range(4),
            )
        )
        assert results.count(True) == 1
        results = list(
            pool.map(
                lambda _: repository.ingest_event(events[0], benchmark_role="diagnostic"), range(4)
            )
        )
        assert results.count(True) == 1
    assert repository.get_run("storage-test")["event_count"] == 1


def test_atomic_import_and_source_files_unchanged(repository, events, tmp_path):
    directory = write_export(tmp_path / "archive", events)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    result = import_export(repository, directory, benchmark_role="diagnostic")
    assert result["inserted_events"] == 4
    assert import_export(repository, directory, benchmark_role="diagnostic")["already_present"]
    row = repository.get_run("storage-test")
    assert row["storage_provenance"]["mode"] == "import"
    assert row["summary_payload"] == events[-1].payload
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize(
    "problem",
    [
        "checksum",
        "size",
        "count",
        "path",
        "missing",
        "run_checksum",
        "summary",
        "duplicate",
        "late",
        "last_record",
    ],
)
def test_bad_import_is_atomic(repository, events, tmp_path, problem):
    directory = write_export(tmp_path / "archive", events)
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "events.jsonl"
    if problem == "checksum":
        manifest["files"][0]["sha256"] = "0" * 64
    elif problem == "size":
        manifest["files"][0]["byte_count"] += 1
    elif problem == "count":
        manifest["files"][0]["record_count"] += 1
    elif problem == "path":
        manifest["files"][0]["relative_path"] = "../events.jsonl"
    elif problem == "missing":
        (directory / "run.json").unlink()
    elif problem == "run_checksum":
        (directory / "run.json").write_text("{}")
    elif problem == "summary":
        data = json.loads((directory / "run.json").read_text())
        data["status"] = "failed"
        encoded = json.dumps(data).encode()
        (directory / "run.json").write_bytes(encoded)
        manifest["files"][1].update(
            sha256=hashlib.sha256(encoded).hexdigest(), byte_count=len(encoded)
        )
    else:
        lines = path.read_bytes().splitlines(keepends=True)
        if problem == "duplicate":
            lines.insert(2, lines[1])
        elif problem == "late":
            lines.append(lines[1])
        else:
            lines[-1] = b"{broken-json}\n"
        encoded = b"".join(lines)
        path.write_bytes(encoded)
        manifest["files"][0].update(
            sha256=hashlib.sha256(encoded).hexdigest(),
            byte_count=len(encoded),
            record_count=len(lines),
        )
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(StorageError):
        import_export(repository, directory, benchmark_role="diagnostic")
    assert repository.list_runs() == []


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_legacy_payload_fields_not_fabricated(repository, events, tmp_path, version):
    raw = [event.model_dump(mode="json") for event in events]
    metadata = raw[0]["payload"]
    evaluator = f"retrieval-evaluation-{version}"
    metadata["configuration"]["evaluator_version"] = evaluator
    if version != "v3":
        metadata.pop("query_embedding_collection")
        for key in ("query_embedding_cache", "query_embedding_cache_version"):
            metadata["configuration"].pop(key)
        for event in raw[1:-1]:
            event["payload"].pop("query_embedding")
            event["payload"].pop("retrieval_core_latency_ms")
        for key in list(raw[-1]["payload"]):
            if key.startswith("retrieval_core_"):
                raw[-1]["payload"].pop(key)
    metadata["configuration_fingerprint"] = digest(metadata["configuration"])
    raw[-1]["payload"]["metadata"] = deepcopy(metadata)
    for event in raw:
        event["source_version"] = evaluator
    converted = [InstrumentationEvent.model_validate(event) for event in raw]
    directory = write_export(tmp_path / "legacy", converted)
    import_export(repository, directory, benchmark_role="diagnostic")
    row = repository.get_run("storage-test")
    assert row["metadata_payload"] == metadata
    assert row["summary_payload"] == raw[-1]["payload"]
    if version != "v3":
        assert "retrieval_core_seconds" not in row["summary_payload"]
        assert (
            repository.get_question_trace("storage-test", "q-0")["retrieval_core_latency_ms"]
            is None
        )


def test_database_constraints_and_populated_downgrade_guard(repository, events):
    ingest(repository, events[:1])
    with pytest.raises(sa.exc.IntegrityError), repository.engine.begin() as connection:
        connection.execute(run_questions.update().values(position=-1))
    with pytest.raises(sa.exc.DBAPIError), repository.engine.begin() as connection:
        command.downgrade(migration_config(connection), "base")
    assert repository.get_run("storage-test")["event_count"] == 1


def test_cli_inspection_and_safe_connection_error(repository, events, monkeypatch, capsys):
    import os

    monkeypatch.setenv("EPSA_DATABASE_URL", os.environ["EPSA_TEST_DATABASE_URL"])
    ingest(repository, events)
    assert cli.main(["check"]) == 0
    capsys.readouterr()
    assert cli.main(["inspect", "storage-test", "--question-id", "q-0"]) == 0
    assert json.loads(capsys.readouterr().out)["events"][0]["payload"] == events[1].payload
    assert cli.main(["inspect", "absent"]) == 1
    assert "not found" in capsys.readouterr().err
    monkeypatch.setenv("EPSA_DATABASE_URL", "postgresql://secret:secret@127.0.0.1:1/epsa_test")
    assert cli.main(["check"]) == 1
    assert "secret" not in capsys.readouterr().err


def test_foreign_keys_and_global_event_identity(repository, events):
    ingest(repository, events)
    with repository.engine.connect() as connection:
        assert connection.scalar(sa.select(sa.func.count()).select_from(trace_events)) == 4
        assert connection.scalar(sa.select(sa.func.count()).select_from(run_questions)) == 2
        assert connection.scalar(sa.select(sa.func.count()).select_from(runs)) == 1


def test_sink_database_failure_rolls_back_question_and_remains_incomplete(repository, events):
    ingest(repository, events[:1])

    def fail_trace_insert(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO epsa_experiments.trace_events"):
            raise sa.exc.OperationalError(statement, parameters, RuntimeError("sensitive-db-error"))

    sa.event.listen(repository.engine, "before_cursor_execute", fail_trace_insert)
    try:
        with pytest.raises(StorageError) as error:
            ingest(repository, events[1:2])
        assert "sensitive-db-error" not in str(error.value)
    finally:
        sa.event.remove(repository.engine, "before_cursor_execute", fail_trace_insert)
    row = repository.get_run("storage-test")
    assert row["status"] == "running"
    assert row["event_count"] == 1
    assert row["attempted_questions"] == 0
    assert repository.get_question_trace("storage-test", "q-0")["status"] == "pending"
    ingest(repository, events[1:])
    assert repository.get_run("storage-test")["status"] == "completed"


def test_run_identity_and_failure_lifecycle_validation(repository, events):
    with pytest.raises(StorageError, match="Component versions"):
        repository.register_run(
            events[0].payload, benchmark_role="diagnostic", component_versions={"component": " "}
        )
    ingest(repository, events[:1])
    with pytest.raises(ConflictError, match="start event"):
        repository.ingest_event(
            events[0].model_copy(update={"event_id": "event:second-start"}),
            benchmark_role="diagnostic",
        )
    with pytest.raises(ConflictError, match="registration context"):
        repository.ingest_event(events[1], benchmark_role="test")
    ingest(repository, events[1:-1])
    payload = deepcopy(events[-1].payload)
    payload["metadata"]["git_dirty"] = True
    with pytest.raises(StorageError, match="metadata"):
        repository.ingest_event(
            events[-1].model_copy(update={"payload": payload}), benchmark_role="diagnostic"
        )
    payload = deepcopy(events[-1].payload)
    payload.update(status="failed", error_type="Error")
    with pytest.raises(StorageError, match="Failed run"):
        repository.ingest_event(
            events[-1].model_copy(update={"payload": payload}), benchmark_role="diagnostic"
        )


def test_question_after_failure_and_missing_warmups_rejected(repository):
    events, _ = make_events(fail_at="q-0")
    ingest(repository, events[:2])
    successful, _ = make_events()
    with pytest.raises(ConflictError, match="after failure"):
        repository.ingest_event(successful[2], benchmark_role="diagnostic")
    ingest(repository, events[-1:])
    warmed, _ = make_events("warmup-run", warmups=1)
    ingest(repository, warmed[:-1])
    payload = deepcopy(warmed[-1].payload)
    payload["warmup_completed"] = 0
    with pytest.raises(StorageError, match="warmups"):
        repository.ingest_event(
            warmed[-1].model_copy(update={"payload": payload}), benchmark_role="diagnostic"
        )


@pytest.mark.parametrize("problem", ["missing_identity", "missing_start", "unknown_event"])
def test_export_event_contract_required(repository, events, tmp_path, problem):
    directory = write_export(tmp_path / "archive", events)
    raw = [event.model_dump(mode="json") for event in events]
    if problem == "missing_identity":
        raw[1].pop("event_id")
    elif problem == "missing_start":
        raw = raw[1:]
    else:
        raw[1]["event_type"] = "future.unsupported"
    encoded = "".join(json.dumps(event) + "\n" for event in raw).encode()
    (directory / "events.jsonl").write_bytes(encoded)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["files"][0].update(
        sha256=hashlib.sha256(encoded).hexdigest(), byte_count=len(encoded), record_count=len(raw)
    )
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(StorageError):
        import_export(repository, directory, benchmark_role="diagnostic")
    assert repository.list_runs() == []


def test_cli_migrate_import_list_and_run_inspection(
    database_engine, events, tmp_path, monkeypatch, capsys
):
    import os

    monkeypatch.setenv("EPSA_DATABASE_URL", os.environ["EPSA_TEST_DATABASE_URL"])
    assert cli.main(["migrate"]) == 0
    capsys.readouterr()
    directory = write_export(tmp_path / "archive", events)
    assert cli.main(["import", str(directory), "--benchmark-role", "diagnostic"]) == 0
    assert json.loads(capsys.readouterr().out)["inserted_events"] == 4
    assert cli.main(["list"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["run_id"] == "storage-test"
    assert cli.main(["inspect", "storage-test"]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["status"] == "completed"
    assert row["registered_at"]
