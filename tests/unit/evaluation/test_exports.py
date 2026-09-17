from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from epsa_rag.core.exceptions import SourceValidationError
from epsa_rag.data.io import sha256_file
from epsa_rag.evaluation.retrieval.evaluator import evaluate
from epsa_rag.evaluation.retrieval.exports import DiagnosticExportSink, compare_exports, load_export
from epsa_rag.instrumentation.context import TraceContext
from epsa_rag.instrumentation.events import InstrumentationEvent
from epsa_rag.instrumentation.sinks import InMemoryInstrumentationSink
from epsa_rag.retrieval.models import RetrievalResult


@pytest.fixture
def completed_export(evaluation_data, evaluation_metadata, tmp_path):
    corpus, benchmark, _ = evaluation_data
    retriever = MagicMock()
    retriever.retrieve.side_effect = lambda query, **k: RetrievalResult(
        query=query, retriever_version="hybrid-retriever-v2", results=()
    )
    downstream = InMemoryInstrumentationSink()
    with DiagnosticExportSink(tmp_path / "exports", evaluation_metadata, downstream) as sink:
        summary = evaluate(
            examples=benchmark.examples,
            corpus=corpus,
            retriever=retriever,
            metadata=evaluation_metadata,
            sink=sink,
        )
        sink.finalize(summary)
    assert len(downstream.events) == 5
    return sink.directory, summary


def test_unfinished_export_cannot_be_mistaken_for_complete(evaluation_metadata, tmp_path):
    with DiagnosticExportSink(tmp_path, evaluation_metadata) as sink:
        event = InstrumentationEvent(
            context=TraceContext.start(run_id=evaluation_metadata.run_id),
            event_type="evaluation.run.started",
            source="test",
        )
        sink.emit(event)
    assert (sink.directory / "events.jsonl").is_file()
    with pytest.raises(FileNotFoundError):
        load_export(sink.directory)


def test_reserved_windows_names_and_cross_run_events(evaluation_metadata, tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        DiagnosticExportSink(tmp_path, evaluation_metadata.model_copy(update={"run_id": "con"}))
    with DiagnosticExportSink(tmp_path, evaluation_metadata) as sink:
        event = InstrumentationEvent(
            context=TraceContext.start(run_id="different"), event_type="test", source="test"
        )
        with pytest.raises(ValueError, match="different run"):
            sink.emit(event)


def test_cross_run_summary_rejected(completed_export, evaluation_metadata, tmp_path):
    _, summary = completed_export
    with DiagnosticExportSink(tmp_path / "another", evaluation_metadata) as sink:
        changed = summary.model_copy(
            update={"metadata": evaluation_metadata.model_copy(update={"run_id": "different"})}
        )
        with pytest.raises(ValueError, match="different run"):
            sink.finalize(changed)


@pytest.mark.parametrize("problem", ["path", "count", "run", "lifecycle", "traces"])
def test_export_consistency_beyond_checksums(completed_export, problem):
    directory, _ = completed_export
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if problem == "path":
        manifest["files"][0]["relative_path"] = "../escape"
    elif problem == "count":
        manifest["files"][0]["record_count"] = 100
    elif problem == "run":
        manifest["run_id"] = "different"
    else:
        path = directory / "events.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if problem == "lifecycle":
            events[0]["event_type"] = "wrong"
        else:
            events[1]["payload"]["question"]["question_id"] = "wrong"
        path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        manifest["files"][0]["sha256"] = sha256_file(path)
        manifest["files"][0]["byte_count"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SourceValidationError):
        load_export(directory)


def test_failed_run_is_not_comparable(completed_export, monkeypatch):
    import epsa_rag.evaluation.retrieval.exports as exports

    directory, summary = completed_export
    monkeypatch.setattr(
        exports, "load_export", lambda path: (summary.model_copy(update={"status": "failed"}), ())
    )
    with pytest.raises(ValueError, match="completed runs"):
        compare_exports(directory, directory)
