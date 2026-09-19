"""Tests for Component 01 diagnostic evaluation and inspectable local exports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from epsa_rag.core.exceptions import FrozenArtifactError
from epsa_rag.data.io import sha256_file
from epsa_rag.data.models import QuestionInput
from epsa_rag.evaluation.components.question_analyzer import (
    QuestionAnalyzerEvaluationConfig,
    evaluate_questions,
    load_inference_questions,
    render_development_report,
    write_export,
)
from epsa_rag.evaluation.components.question_analyzer_cli import main


def test_evaluation_reports_descriptive_diagnostics_and_structured_events() -> None:
    questions = (
        QuestionInput(question_id="q-1", text="Where was the director of Inception born?"),
        QuestionInput(question_id="q-2", text="Which of Ada and Grace was born earlier?"),
        QuestionInput(question_id="q-3", text="explain it."),
    )

    summary, traces, events = evaluate_questions(
        questions,
        run_id="question-analyzer-unit-v1",
        dataset_version="fixture-v1",
        dataset_manifest_sha256="a" * 64,
    )

    assert summary.completed_questions == 3
    assert summary.failed_questions == 0
    assert summary.diagnostics.question_type_counts == {
        "bridge": 1,
        "comparison": 1,
        "factoid": 1,
    }
    assert summary.diagnostics.questions_without_seed_entities == 1
    assert len(traces) == len(events) == 3
    assert all(event.event_type == "epsa.question_analysis.completed" for event in events)
    assert all(trace.analysis is not None for trace in traces)
    assert "accuracy" not in summary.model_dump_json().lower()
    assert summary.configuration_fingerprint == QuestionAnalyzerEvaluationConfig().fingerprint()


def test_evaluation_rejects_an_empty_fixed_question_collection() -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate_questions(
            (),
            run_id="question-analyzer-empty-v1",
            dataset_version="fixture-v1",
            dataset_manifest_sha256="a" * 64,
        )


def test_development_report_uses_the_manual_review_format() -> None:
    _, traces, _ = evaluate_questions(
        (QuestionInput(question_id="q-1", text="Who designed the theater where LPO plays?"),),
        run_id="question-analyzer-report-v1",
        dataset_version="fixture-v1",
        dataset_manifest_sha256="a" * 64,
    )

    report = render_development_report(
        traces, run_id="question-analyzer-report-v1", dataset_version="fixture-v1"
    )

    assert "Question type: bridge" in report
    assert "Expected answer type: PERSON" in report
    assert "Seed entities: LPO" in report
    assert "Comparison targets: none" in report


def test_loader_uses_only_inference_inputs_and_export_is_immutable(
    evaluation_data: tuple[object, object, Path], tmp_path: Path
) -> None:
    _, benchmark, dataset_directory = evaluation_data
    manifest, questions = load_inference_questions(dataset_directory)

    assert tuple(question.question_id for question in questions) == manifest.question_ids
    assert all(question.text for question in questions)
    assert all("SECRET GOLD" not in question.model_dump_json() for question in questions)

    summary, traces, events = evaluate_questions(
        questions,
        run_id="question-analyzer-export-v1",
        dataset_version=manifest.version,
        dataset_manifest_sha256=sha256_file(dataset_directory / "manifest.json"),
    )
    directory = write_export(tmp_path, summary=summary, traces=traces, events=events)

    assert (directory / "run.json").is_file()
    assert (directory / "traces.jsonl").is_file()
    assert (directory / "events.jsonl").is_file()
    assert (directory / "manifest.json").is_file()
    with pytest.raises(FrozenArtifactError, match="refusing to overwrite"):
        write_export(tmp_path, summary=summary, traces=traces, events=events)
    assert len(benchmark.examples) == len(questions)


def test_cli_runs_and_inspects_a_fixed_export(
    evaluation_data: tuple[object, object, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, benchmark, dataset_directory = evaluation_data
    export_root = tmp_path / "exports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "epsa-evaluate-question-analyzer",
            "run",
            "--run-id",
            "question-analyzer-cli-v1",
            "--dataset-directory",
            str(dataset_directory),
            "--export-root",
            str(export_root),
        ],
    )

    assert main() == 0
    assert "Diagnostics are not accuracy metrics" in capsys.readouterr().out
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "epsa-evaluate-question-analyzer",
            "inspect",
            str(export_root / "question-analyzer-cli-v1"),
            "--question-id",
            benchmark.examples[0].inference.question_id,
        ],
    )

    assert main() == 0
    assert benchmark.examples[0].inference.question_id in capsys.readouterr().out
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "epsa-evaluate-question-analyzer",
            "inspect",
            str(export_root / "question-analyzer-cli-v1"),
        ],
    )

    assert main() == 0
    assert "question-analyzer-run-v1" in capsys.readouterr().out
