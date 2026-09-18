"""Opt-in 10,000-question storage check using synthetic data, never the held-out benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from time import perf_counter

import pytest
from conftest import make_events

from epsa_observability.importer import import_export


@pytest.mark.postgres
@pytest.mark.skipif(
    os.environ.get("EPSA_RUN_SCALE_TEST") != "1", reason="Opt-in synthetic scale test"
)
def test_streaming_ten_thousand_question_import(repository, tmp_path):
    count = 10_000
    source, _ = make_events("synthetic-scale-test")
    start, question, finish = (deepcopy(source[i].model_dump(mode="json")) for i in (0, 1, -1))
    ids = [f"synthetic-q-{i}" for i in range(count)]
    start["payload"].update(question_ids=ids, full_dataset_question_count=count)
    finish["payload"].update(
        metadata=start["payload"], planned_questions=count, completed_questions=count
    )
    summary = finish["payload"]
    summary["metric_denominators"] = {name: count for name in summary["metrics"]}
    latency = question["payload"]["latency_ms"]
    for name in (
        "latency_p50_ms",
        "latency_p95_ms",
        "successful_latency_p50_ms",
        "successful_latency_p95_ms",
        "retrieval_core_latency_p50_ms",
        "retrieval_core_latency_p95_ms",
    ):
        summary[name] = latency
    seconds = latency * count / 1000
    for name in ("retrieval_seconds", "retrieval_core_seconds", "evaluation_wall_seconds"):
        summary[name] = seconds
    for name in (
        "throughput_questions_per_second",
        "retrieval_throughput_questions_per_second",
        "retrieval_core_throughput_questions_per_second",
    ):
        summary[name] = count / seconds if seconds else None
    directory = tmp_path / "synthetic-export"
    directory.mkdir()
    event_digest = hashlib.sha256()
    size = 0
    with (directory / "events.jsonl").open("wb") as stream:

        def write(event):
            nonlocal size
            data = (json.dumps(event) + "\n").encode()
            event_digest.update(data)
            size += len(data)
            stream.write(data)

        write(start)
        for i, qid in enumerate(ids):
            question["event_id"] = f"event:synthetic-{i}"
            question["context"].update(question_id=qid, trace_id=f"trace:synthetic-{i}")
            question["payload"]["question"]["question_id"] = qid
            question["payload"]["retrieval"]["query"]["question_id"] = qid
            write(question)
        write(finish)
    run_bytes = json.dumps(summary).encode()
    (directory / "run.json").write_bytes(run_bytes)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "purpose": "diagnostic-export-pending-postgresql-import",
                "run_id": "synthetic-scale-test",
                "files": [
                    {
                        "relative_path": "events.jsonl",
                        "sha256": event_digest.hexdigest(),
                        "byte_count": size,
                        "record_count": count + 2,
                    },
                    {
                        "relative_path": "run.json",
                        "sha256": hashlib.sha256(run_bytes).hexdigest(),
                        "byte_count": len(run_bytes),
                        "record_count": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    started = perf_counter()
    result = import_export(repository, directory, benchmark_role="diagnostic")
    elapsed = perf_counter() - started
    assert result["inserted_events"] == count + 2
    assert repository.get_run("synthetic-scale-test")["completed_questions"] == count
    assert repository.get_question_trace("synthetic-scale-test", ids[-1])["status"] == "completed"
    print(f"Synthetic storage check: {count} questions, {size} bytes, {elapsed:.2f}s import")
