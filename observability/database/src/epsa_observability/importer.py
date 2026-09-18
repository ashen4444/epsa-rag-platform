"""Stream finalized Phase 4 exports into one atomic PostgreSQL transaction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from epsa_rag.evaluation.retrieval.exports import ExportManifest
from epsa_rag.instrumentation.events import InstrumentationEvent

from epsa_observability.connection import transaction
from epsa_observability.errors import StorageError
from epsa_observability.repository import ExperimentRepository
from epsa_observability.validation import FINISH, QUESTION, START, BenchmarkRole, parse


def _reject_constant(value: str) -> Any:
    raise StorageError("Export contains a non-finite JSON value.")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StorageError("Export contains duplicate JSON object keys.")
        result[key] = value
    return result


def decode_json(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        if not isinstance(value, dict):
            raise StorageError("Export record must be a JSON object.")
        return value
    except (ValueError, UnicodeError):
        raise StorageError("Export contains invalid JSON.") from None


def import_export(
    repository: ExperimentRepository, directory: Path, *, benchmark_role: BenchmarkRole
) -> dict[str, Any]:
    """No retrieval, embedding, rewriting, or writes to the source export directory."""
    try:
        return _import(repository, directory, benchmark_role)
    except OSError:
        raise StorageError(
            "Cannot read a complete export: manifest, run, and events are required."
        ) from None


def _import(
    repository: ExperimentRepository, directory: Path, role: BenchmarkRole
) -> dict[str, Any]:
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = parse(ExportManifest, decode_json(manifest_bytes))
    if (
        manifest.schema_version != "1.0"
        or manifest.purpose != "diagnostic-export-pending-postgresql-import"
        or tuple(item.relative_path for item in manifest.files) != ("events.jsonl", "run.json")
    ):
        raise StorageError("Unsupported export manifest or file list.")
    events_file, run_file = manifest.files
    run_bytes = (directory / "run.json").read_bytes()
    if (
        len(run_bytes) != run_file.byte_count
        or hashlib.sha256(run_bytes).hexdigest() != run_file.sha256
        or run_file.record_count != 1
    ):
        raise StorageError("Run export checksum, size, or record count is invalid.")
    summary = decode_json(run_bytes)
    provenance = {
        "mode": "import",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "export_files": [item.model_dump(mode="json") for item in manifest.files],
    }
    count, inserted, byte_count = 0, 0, 0
    event_hash = hashlib.sha256()
    event_ids: set[str] = set()
    finished = False
    # Full trace bodies stream one at a time; metadata, IDs and metric projections are smaller.
    # The final hash check is inside the same transaction:
    # even a changed/truncated source or an invalid last record leaves no partial imported run.
    with (
        transaction(repository.engine) as connection,
        (directory / "events.jsonl").open("rb") as stream,
    ):
        for line in stream:
            byte_count += len(line)
            event_hash.update(line)
            raw = decode_json(line)
            if not {"event_id", "occurred_at", "context", "event_type", "source"} <= raw.keys():
                raise StorageError("Imported events must contain their original identity and time.")
            event = parse(InstrumentationEvent, raw)
            if event.context.run_id != manifest.run_id or event.event_id in event_ids or finished:
                raise StorageError(
                    "Export has conflicting identity, duplicate events, or late events."
                )
            event_ids.add(event.event_id)
            if count == 0:
                if event.event_type != START or event.payload != summary.get("metadata"):
                    raise StorageError("Export must begin with metadata matching its summary.")
            elif event.event_type not in {QUESTION, FINISH}:
                raise StorageError("Unsupported event in retrieval export.")
            if event.event_type == FINISH:
                if event.payload != summary:
                    raise StorageError("Finish event differs from the archived run summary.")
                finished = True
            inserted += repository._ingest(connection, event, role, {}, provenance)
            count += 1
        if (
            not finished
            or count != events_file.record_count
            or byte_count != events_file.byte_count
            or event_hash.hexdigest() != events_file.sha256
        ):
            raise StorageError("Export is unfinished or its event checksum/count/size is invalid.")
    return {
        "run_id": manifest.run_id,
        "events": count,
        "inserted_events": inserted,
        "already_present": inserted == 0,
    }
