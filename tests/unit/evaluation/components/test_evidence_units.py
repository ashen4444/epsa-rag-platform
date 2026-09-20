"""Component 03 diagnostic export and evaluation-only boundary tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from epsa_rag.core.exceptions import FrozenArtifactError, SourceValidationError
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, Sentence
from epsa_rag.data.models import (
    BenchmarkExample,
    EvaluationLabels,
    QuestionInput,
    SupportingFactLabel,
)
from epsa_rag.evaluation.components import evidence_units as unit_evaluation
from epsa_rag.evaluation.components import evidence_units_cli
from epsa_rag.evaluation.components.chunk_analyzer import InferenceRetrieval
from epsa_rag.evaluation.components.evidence_units import (
    evaluate_evidence_units,
    gold_coverage_diagnostics,
    inspect_trace,
    load_export,
    write_export,
)


def _inputs() -> tuple[InferenceRetrieval, ...]:
    sentence = Sentence(index=0, text="Inception was directed by Christopher Nolan.")
    chunk = RankedParagraphChunk(
        chunk=ParagraphChunk(
            chunk_id="chunk:inception",
            title="Inception",
            paragraph_text=sentence.text,
            sentences=(sentence,),
        ),
        rank=1,
        score=0.8,
    )
    return (
        InferenceRetrieval(
            question=QuestionInput(question_id="q-1", text="Who directed Inception?"),
            chunks=(chunk,),
            retriever_version="hybrid-retriever-v2",
        ),
    )


def _result():
    return evaluate_evidence_units(
        _inputs(),
        run_id="evidence-units-test-v1",
        dataset_version="hotpotqa_1000_v1",
        dataset_manifest_sha256="a" * 64,
        corpus_version="hotpotqa_10000_v1",
        retrieval_run_id="retrieval-fixture-v1",
        git_commit_sha="b" * 40,
        git_dirty=True,
        source_sha256={"src/example.py": "c" * 64},
        runtime={"python": "3.12.10"},
    )


def test_export_round_trip_inspection_and_integrity(tmp_path: Path) -> None:
    summary, traces, events = _result()
    directory = write_export(tmp_path, summary=summary, traces=traces, events=events)
    loaded_summary, loaded_traces = load_export(directory)
    assert loaded_summary == summary
    assert loaded_traces == traces
    assert inspect_trace(directory, "q-1") == traces[0]
    with pytest.raises(ValueError, match="not present"):
        inspect_trace(directory, "missing")
    with pytest.raises(FrozenArtifactError, match="overwrite"):
        write_export(tmp_path, summary=summary, traces=traces, events=events)
    trace_path = directory / "traces.jsonl"
    trace_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SourceValidationError, match="integrity"):
        load_export(directory)


def test_gold_coverage_is_a_post_inference_evaluation_join(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, traces, _ = _result()
    question = traces[0].question
    label = SupportingFactLabel(
        chunk_id="chunk:inception",
        title="Inception",
        sentence_index=0,
        evidence_unit_id="chunk:inception::s0",
    )
    example = BenchmarkExample(
        inference=question,
        evaluation=EvaluationLabels(
            answer="Christopher Nolan",
            question_type="bridge",
            difficulty="easy",
            supporting_facts=(label,),
        ),
    )
    monkeypatch.setattr(
        unit_evaluation,
        "load_inference_questions",
        lambda _: (SimpleNamespace(version="hotpotqa_1000_v1"), (question,)),
    )
    monkeypatch.setattr(unit_evaluation, "read_jsonl", lambda _: (example,))
    diagnostics = gold_coverage_diagnostics(traces, tmp_path)
    assert diagnostics[0].retrieved_gold_sentence_count == 1
    assert diagnostics[0].missing_gold_evidence_unit_ids == ()
    assert "is_supporting_sentence" not in traces[0].model_dump_json()


def test_cli_run_inspect_and_gold_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    summary, traces, events = _result()
    monkeypatch.setattr(
        evidence_units_cli,
        "load_development_inputs",
        lambda *_args, **_kwargs: (
            SimpleNamespace(version="hotpotqa_1000_v1"),
            "hotpotqa_10000_v1",
            "retrieval-fixture-v1",
            "hybrid-retriever-v2",
            _inputs(),
        ),
    )
    monkeypatch.setattr(
        evidence_units_cli,
        "evaluate_evidence_units",
        lambda *_args, **_kwargs: (summary, traces, events),
    )
    monkeypatch.setattr(evidence_units_cli, "sha256_file", lambda _: "a" * 64)
    monkeypatch.setattr(
        evidence_units_cli,
        "code_provenance",
        lambda *_args, **_kwargs: ("b" * 40, True, {}),
    )
    monkeypatch.setattr(evidence_units_cli, "runtime_provenance", lambda: {"python": "3.12.10"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "epsa-evaluate-evidence-units",
            "run",
            "--run-id",
            "evidence-units-test-v1",
            "--dataset-directory",
            str(tmp_path / "dataset"),
            "--retrieval-export-directory",
            str(tmp_path / "retrieval"),
            "--export-root",
            str(tmp_path / "exports"),
        ],
    )
    assert evidence_units_cli.main() == 0
    assert "not Component 03 accuracy metrics" in capsys.readouterr().out
    directory = tmp_path / "exports" / "evidence-units-test-v1"
    monkeypatch.setattr(
        sys,
        "argv",
        ["epsa-evaluate-evidence-units", "inspect", str(directory), "--question-id", "q-1"],
    )
    assert evidence_units_cli.main() == 0
    assert "chunk:inception" in capsys.readouterr().out
    monkeypatch.setattr(
        evidence_units_cli,
        "gold_coverage_diagnostics",
        lambda *_args: (),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "epsa-evaluate-evidence-units",
            "gold-coverage",
            str(directory),
            "--dataset-directory",
            str(tmp_path / "dataset"),
        ],
    )
    assert evidence_units_cli.main() == 0
    assert "Evaluation-only retrieved gold sentence coverage" in capsys.readouterr().out
