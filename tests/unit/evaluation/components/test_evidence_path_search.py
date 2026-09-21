"""Behavioral coverage for the inference-only Component 06 evaluator."""

from __future__ import annotations

import pytest

from epsa_rag.core.exceptions import FrozenArtifactError, QuestionAnalysisError
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, Sentence
from epsa_rag.data.models import QuestionInput
from epsa_rag.evaluation.components import evidence_path_search as component
from epsa_rag.evaluation.components.chunk_analyzer import InferenceRetrieval


def _inputs() -> tuple[InferenceRetrieval, ...]:
    text = "Inception was directed by Christopher Nolan."
    return (
        InferenceRetrieval(
            question=QuestionInput(question_id="component06-q", text="Who directed Inception?"),
            chunks=(
                RankedParagraphChunk(
                    chunk=ParagraphChunk(
                        chunk_id="component06-inception",
                        title="Inception",
                        paragraph_text=text,
                        sentences=(Sentence(index=0, text=text),),
                    ),
                    rank=1,
                    score=0.9,
                ),
            ),
            retriever_version="retriever-v1",
        ),
    )


def test_evaluate_evidence_paths_returns_inference_only_completed_trace() -> None:
    summary, traces, events = component.evaluate_evidence_paths(
        _inputs(), run_id="component06-eval"
    )

    assert summary.completed_questions == 1
    assert summary.failed_questions == 0
    assert summary.diagnostics.graphs == 1
    assert len(traces) == 1
    assert traces[0].status == "completed"
    assert traces[0].evidence_graph is not None
    assert "supporting" not in traces[0].model_dump_json()
    assert events


def test_evaluate_evidence_paths_rejects_empty_inputs_and_records_component_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="requires inference inputs"):
        component.evaluate_evidence_paths((), run_id="empty")

    def fail_analysis(*_args: object, **_kwargs: object) -> object:
        raise QuestionAnalysisError("forced analysis failure")

    monkeypatch.setattr(component.RuleBasedQuestionAnalyzer, "analyze", fail_analysis)
    summary, traces, events = component.evaluate_evidence_paths(
        _inputs(),
        run_id="component06-failure",
        config=component.EvidencePathSearchEvaluationConfig(retain_instrumentation_events=False),
    )

    assert summary.completed_questions == 0
    assert summary.failed_questions == 1
    assert traces[0].status == "failed"
    assert traces[0].error_type == "QuestionAnalysisError"
    assert events == ()


def test_component06_export_and_markdown_reports_are_immutable(tmp_path) -> None:
    summary, traces, events = component.evaluate_evidence_paths(
        _inputs(), run_id="component06-export"
    )
    export_directory = tmp_path / "export"
    manifest = component.write_evidence_path_evaluation(export_directory, summary, traces, events)

    assert {item.relative_path for item in manifest.files} == {
        "run.json",
        "traces.jsonl",
        "events.jsonl",
    }
    with pytest.raises(FrozenArtifactError):
        component.write_evidence_path_evaluation(export_directory, summary, traces, events)

    report = tmp_path / "report.md"
    component.write_evidence_path_markdown_report(report, summary, traces)
    assert "Candidate paths" in report.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        component.write_evidence_path_markdown_report(report, summary, traces)


def test_component06_streaming_reports_validate_and_append(tmp_path) -> None:
    report = tmp_path / "streamed.md"
    with pytest.raises(ValueError, match="batch_size"):
        component.write_streaming_evidence_path_markdown_report(
            report, _inputs(), run_id="stream", batch_size=0
        )
    with pytest.raises(FileNotFoundError):
        component.append_streaming_evidence_path_markdown_report(
            tmp_path / "missing.md", _inputs(), run_id="append"
        )

    counts = component.write_streaming_evidence_path_markdown_report(
        report, _inputs(), run_id="stream", batch_size=1
    )
    appended = component.append_streaming_evidence_path_markdown_report(
        report, _inputs(), run_id="append", batch_size=1
    )

    assert counts["completed_questions"] == 1
    assert appended["completed_questions"] == 1
    assert report.read_text(encoding="utf-8").count("## component06-q") == 2
    with pytest.raises(FileExistsError):
        component.write_streaming_evidence_path_markdown_report(report, _inputs(), run_id="again")
