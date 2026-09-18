from __future__ import annotations

import json
from copy import deepcopy

import pytest
from epsa_rag.instrumentation.events import InstrumentationEvent

from epsa_observability import cli
from epsa_observability.connection import create_engine
from epsa_observability.errors import StorageError
from epsa_observability.importer import decode_json
from epsa_observability.validation import digest, snapshot, validate_metadata, validate_question


def test_native_whitespace_and_deep_snapshot(events):
    event, original = snapshot(events[1])
    events[1].payload["question"]["text"] = "changed caller state"
    assert event.payload["question"]["text"] == "  Question 0?\n"
    assert original["payload"]["retrieval"]["results"][0]["chunk"]["sentences"][0]["text"] == (
        " First native sentence. \n"
    )


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b"[]", b"{bad", b"\xff"])
def test_strict_export_json(raw):
    with pytest.raises(StorageError):
        decode_json(raw)


@pytest.mark.parametrize("change", ["role", "fingerprint", "duplicate", "empty", "hash", "limit"])
def test_registration_validation(events, change):
    payload = deepcopy(events[0].payload)
    role = "diagnostic"
    if change == "role":
        role = "test"
    elif change == "fingerprint":
        payload["configuration_fingerprint"] = "0" * 64
    elif change == "duplicate":
        payload["question_ids"] = ["q-0", "q-0"]
    elif change == "empty":
        payload["question_ids"] = []
    elif change == "hash":
        payload["dataset_file_sha256"] = "bad"
    else:
        payload["question_ids"] = ["q-0"]
    with pytest.raises(StorageError):
        validate_metadata(payload, role)


def test_known_benchmark_cannot_be_relabelled(events):
    payload = deepcopy(events[0].payload)
    payload.update(dataset_version="hotpotqa_1000_v1", corpus_version="hotpotqa_10000_v1")
    payload["full_dataset_question_count"] = 1000
    payload["configuration"]["question_limit"] = 2
    payload["configuration_fingerprint"] = digest(payload["configuration"])
    assert validate_metadata(payload, "development").run_id == "storage-test"
    with pytest.raises(StorageError, match="Frozen"):
        validate_metadata(payload, "diagnostic")
    payload["full_dataset_question_count"] = 2
    with pytest.raises(StorageError, match="dataset count"):
        validate_metadata(payload, "development")


def test_configuration_fingerprint_uses_original_legacy_fields(events):
    payload = deepcopy(events[0].payload)
    payload["configuration"].pop("query_embedding_cache")
    payload["configuration_fingerprint"] = digest(payload["configuration"])
    validate_metadata(payload, "diagnostic")
    assert "query_embedding_cache" not in payload["configuration"]


@pytest.mark.parametrize("change", ["identity", "status", "retrieval", "query", "ranks", "version"])
def test_question_validation(events, change):
    raw = events[1].model_dump(mode="json")
    if change == "identity":
        raw["context"]["question_id"] = "other"
    elif change == "status":
        raw["payload"]["error_type"] = "Error"
    elif change == "retrieval":
        raw["payload"]["retrieval"] = None
    elif change == "query":
        raw["payload"]["retrieval"]["query"]["text"] = "changed"
    elif change == "ranks":
        raw["payload"]["retrieval"]["results"][0]["rank"] = 2
    else:
        raw["payload"]["retrieval"]["retriever_version"] = "other"
    with pytest.raises(StorageError):
        validate_question(
            InstrumentationEvent.model_validate(raw),
            retriever_version="hybrid-retriever-v2",
            retrieval_depth=10,
        )


def test_nonfinite_and_bad_payload_errors_are_safe(events):
    raw = events[1].model_dump(mode="json")
    raw["payload"]["metrics"]["secret"] = float("nan")
    with pytest.raises(StorageError, match="finite"):
        snapshot(InstrumentationEvent.model_validate(raw))
    with pytest.raises(StorageError) as error:
        validate_metadata({"password": "do-not-print"}, "diagnostic")
    assert "do-not-print" not in str(error.value)


@pytest.mark.parametrize("url", ["sqlite://", "not-a-url", "mysql://user:secret@localhost/db"])
def test_connection_configuration_rejects_other_backends(url):
    with pytest.raises(StorageError) as error:
        create_engine(url)
    assert "secret" not in str(error.value)


def test_cli_requires_environment_url(monkeypatch, capsys):
    monkeypatch.delenv("EPSA_DATABASE_URL", raising=False)
    assert cli.main(["check"]) == 1
    assert "EPSA_DATABASE_URL" in capsys.readouterr().err


def test_cli_uses_utf8_for_trace_output(monkeypatch):
    class Stream:
        def __init__(self):
            self.encodings: list[str] = []

        def reconfigure(self, *, encoding):
            self.encodings.append(encoding)

        def write(self, value):
            return len(value)

        def flush(self):
            return None

    import epsa_observability.cli as storage_cli

    stdout, stderr = Stream(), Stream()
    monkeypatch.setattr(storage_cli.sys, "stdout", stdout)
    monkeypatch.setattr(storage_cli.sys, "stderr", stderr)
    monkeypatch.delenv("EPSA_DATABASE_URL", raising=False)
    assert storage_cli.main(["check"]) == 1
    assert stdout.encodings == stderr.encodings == ["utf-8"]


def test_no_database_dependency_in_research_imports():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "src" / "epsa_rag"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import epsa_observability" not in text
        assert "from epsa_observability" not in text


def write_export(directory, events):
    """Write the existing archive format without normalizing original payload fields."""
    from hashlib import sha256

    directory.mkdir()
    contents = {
        "events.jsonl": ("".join(event.model_dump_json() + "\n" for event in events)).encode(),
        "run.json": json.dumps(events[-1].payload).encode(),
    }
    files = []
    for name, data in contents.items():
        (directory / name).write_bytes(data)
        files.append(
            {
                "relative_path": name,
                "sha256": sha256(data).hexdigest(),
                "byte_count": len(data),
                "record_count": len(events) if name == "events.jsonl" else 1,
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "purpose": "diagnostic-export-pending-postgresql-import",
                "run_id": events[0].context.run_id,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return directory
