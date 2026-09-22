"""Behavioral coverage for the inference-only Component 09 evaluator."""

from __future__ import annotations

import pytest

from epsa_rag.core.exceptions import FrozenArtifactError, QuestionAnalysisError
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, Sentence
from epsa_rag.data.models import QuestionInput
from epsa_rag.evaluation.components import next_hop_query as component
from epsa_rag.evaluation.components.chunk_analyzer import InferenceRetrieval


def _inputs() -> tuple[InferenceRetrieval, ...]:
    text = "Inception was directed by Christopher Nolan."
    return (
        InferenceRetrieval(
            question=QuestionInput(question_id="component09-q", text="Who directed Inception?"),
            chunks=(
                RankedParagraphChunk(
                    chunk=ParagraphChunk(
                        chunk_id="component09-inception",
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


def test_evaluate_next_hop_queries_returns_inference_only_trace_without_retrieval() -> None:
    summary, traces, events = component.evaluate_next_hop_queries(
        _inputs(), run_id="component09-eval"
    )

    assert summary.completed_questions == 1
    assert summary.failed_questions == 0
    assert summary.diagnostics.queries + summary.diagnostics.no_queries == 1
    assert traces[0].status == "completed"
    assert traces[0].next_hop_query is not None
    assert "supporting_facts" not in traces[0].model_dump_json()
    assert "gold_answer" not in traces[0].model_dump_json()
    assert any(event.event_type == "epsa.next_hop_query.completed" for event in events)


def test_evaluator_records_failure_and_rejects_empty_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="requires inference inputs"):
        component.evaluate_next_hop_queries((), run_id="empty")

    def fail_analysis(*_args: object, **_kwargs: object) -> object:
        raise QuestionAnalysisError("forced analysis failure")

    monkeypatch.setattr(component.RuleBasedQuestionAnalyzer, "analyze", fail_analysis)
    summary, traces, events = component.evaluate_next_hop_queries(
        _inputs(),
        run_id="component09-failure",
        config=component.NextHopQueryEvaluationConfig(retain_instrumentation_events=False),
    )

    assert summary.completed_questions == 0
    assert summary.failed_questions == 1
    assert traces[0].error_type == "QuestionAnalysisError"
    assert events == ()


def test_next_hop_query_export_is_immutable_and_checksummed(tmp_path) -> None:
    summary, traces, events = component.evaluate_next_hop_queries(
        _inputs(), run_id="component09-export"
    )
    directory = tmp_path / "export"
    manifest = component.write_next_hop_query_evaluation(directory, summary, traces, events)

    assert {item.relative_path for item in manifest.files} == {
        "run.json",
        "traces.jsonl",
        "events.jsonl",
    }
    assert all(len(item.sha256) == 64 for item in manifest.files)
    with pytest.raises(FrozenArtifactError):
        component.write_next_hop_query_evaluation(directory, summary, traces, events)
