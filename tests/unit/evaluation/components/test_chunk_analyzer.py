"""Component 02 diagnostic evaluation is inference-only and reproducible."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from epsa_rag.core.exceptions import FrozenArtifactError
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, Sentence
from epsa_rag.data.models import QuestionInput
from epsa_rag.evaluation.components import chunk_analyzer_cli
from epsa_rag.evaluation.components.chunk_analyzer import (
    InferenceRetrieval,
    evaluate_chunks,
    inspect_trace,
    load_development_inputs,
    write_export,
)


def _inputs() -> tuple[InferenceRetrieval, ...]:
    paragraph = ParagraphChunk(
        chunk_id="chunk:inception",
        title="Inception",
        paragraph_text="Inception was directed by Christopher Nolan.",
        sentences=(Sentence(index=0, text="Inception was directed by Christopher Nolan."),),
    )
    return (
        InferenceRetrieval(
            question=QuestionInput(
                question_id="q-1", text="Where was the director of Inception born?"
            ),
            chunks=(RankedParagraphChunk(chunk=paragraph, rank=1, score=0.4),),
            retriever_version="hybrid-retriever-v2",
        ),
    )


def test_evaluation_produces_inference_only_question_traces_and_component_events() -> None:
    summary, traces, events = evaluate_chunks(
        _inputs(),
        run_id="chunk-analyzer-unit-v1",
        dataset_version="hotpotqa_1000_v1",
        dataset_manifest_sha256="a" * 64,
        corpus_version="hotpotqa_10000_v1",
        retrieval_run_id="retrieval-fixture-v1",
        git_commit_sha="a" * 40,
        git_dirty=True,
        source_sha256={"src/example.py": "b" * 64},
        runtime={"python": "3.12.10"},
    )

    assert summary.research_role == "development"
    assert summary.completed_questions == 1
    assert summary.failed_questions == 0
    assert summary.diagnostics.analyzed_chunks == 1
    assert summary.diagnostics.chunks_with_title_match == 1
    assert traces[0].evidence[0].chunk_id == "chunk:inception"
    assert [event.event_type for event in events] == [
        "epsa.question_analysis.completed", "epsa.chunk_analysis.completed"
    ]
    assert "accuracy" not in summary.model_dump_json().lower()
    assert "gold" not in traces[0].model_dump_json().lower()


def test_exclusive_export_has_inspectable_trace_and_checksums(tmp_path: Path) -> None:
    summary, traces, events = evaluate_chunks(
        _inputs(),
        run_id="chunk-analyzer-export-v1",
        dataset_version="hotpotqa_1000_v1",
        dataset_manifest_sha256="a" * 64,
        corpus_version="hotpotqa_10000_v1",
        retrieval_run_id="retrieval-fixture-v1",
        git_commit_sha="a" * 40,
        git_dirty=True,
        source_sha256={"src/example.py": "b" * 64},
        runtime={"python": "3.12.10"},
    )
    directory = write_export(tmp_path, summary=summary, traces=traces, events=events)

    assert (directory / "run.json").is_file()
    assert (directory / "traces.jsonl").is_file()
    assert (directory / "events.jsonl").is_file()
    assert (directory / "manifest.json").is_file()
    assert inspect_trace(directory, "q-1") == traces[0]
    with pytest.raises(FrozenArtifactError):
        write_export(tmp_path, summary=summary, traces=traces, events=events)


def test_empty_evaluation_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate_chunks(
            (),
            run_id="chunk-analyzer-empty-v1",
            dataset_version="hotpotqa_1000_v1",
            dataset_manifest_sha256="a" * 64,
            corpus_version="hotpotqa_10000_v1",
            retrieval_run_id="retrieval-fixture-v1",
            git_commit_sha="a" * 40,
            git_dirty=True,
            source_sha256={"src/example.py": "b" * 64},
            runtime={"python": "3.12.10"},
        )


def test_evaluation_records_a_failed_chunk_without_fabricating_success() -> None:
    failed = InferenceRetrieval(
        question=QuestionInput(question_id="q-2", text="Where is Paris?"),
        chunks=(
            RankedParagraphChunk(
                chunk=ParagraphChunk(
                    chunk_id="chunk:paris",
                    title="Paris",
                    paragraph_text="Paris is in France.",
                    sentences=(Sentence(index=0, text="Paris is in France."),),
                ),
                rank=1,
                score=0.3,
            ),
        ),
        retriever_version="hybrid-retriever-v2",
    )
    broken = failed.model_copy(
        update={"chunks": (failed.chunks[0].model_copy(update={"rank": 0}),)}
    )
    summary, traces, _ = evaluate_chunks(
        (broken,),
        run_id="chunk-analyzer-failed-v1",
        dataset_version="hotpotqa_1000_v1",
        dataset_manifest_sha256="a" * 64,
        corpus_version="hotpotqa_10000_v1",
        retrieval_run_id="retrieval-fixture-v1",
        git_commit_sha="a" * 40,
        git_dirty=True,
        source_sha256={"src/example.py": "b" * 64},
        runtime={"python": "3.12.10"},
    )

    assert summary.completed_questions == 0
    assert summary.failed_questions == 1
    assert traces[0].status == "failed"
    assert traces[0].evidence == ()
    assert traces[0].error_type == "ChunkAnalysisError"


def test_loader_discards_evaluation_labels_and_validates_retrieval_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    question = inputs[0].question
    manifest = SimpleNamespace(version="hotpotqa_1000_v1")
    metadata = SimpleNamespace(
        dataset_version="hotpotqa_1000_v1",
        dataset_manifest_sha256="a" * 64,
        question_ids=(question.question_id,),
        corpus_version="hotpotqa_10000_v1",
        run_id="retrieval-fixture-v1",
    )
    trace = SimpleNamespace(
        status="completed",
        question=question,
        retrieval=SimpleNamespace(
            results=inputs[0].chunks, retriever_version="hybrid-retriever-v2"
        ),
        evaluation={"answer": "SECRET GOLD"},
    )
    monkeypatch.setattr(
        "epsa_rag.evaluation.components.chunk_analyzer.load_inference_questions",
        lambda _: (manifest, (question,)),
    )
    monkeypatch.setattr(
        "epsa_rag.evaluation.components.chunk_analyzer.sha256_file", lambda _: "a" * 64
    )
    monkeypatch.setattr(
        "epsa_rag.evaluation.components.chunk_analyzer.load_export",
        lambda _: (
            SimpleNamespace(status="completed", full_benchmark=True, metadata=metadata),
            (trace,),
        ),
    )

    _, corpus_version, run_id, retriever_version, loaded = load_development_inputs(
        Path("dataset"), Path("retrieval"), depth=1
    )

    assert (corpus_version, run_id, retriever_version) == (
        "hotpotqa_10000_v1", "retrieval-fixture-v1", "hybrid-retriever-v2"
    )
    assert loaded[0].question == question
    assert "SECRET GOLD" not in loaded[0].model_dump_json()


def test_cli_runs_and_inspects_an_exclusive_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    summary, traces, events = evaluate_chunks(
        _inputs(),
        run_id="chunk-analyzer-cli-v1",
        dataset_version="hotpotqa_1000_v1",
        dataset_manifest_sha256="a" * 64,
        corpus_version="hotpotqa_10000_v1",
        retrieval_run_id="retrieval-fixture-v1",
        git_commit_sha="a" * 40,
        git_dirty=True,
        source_sha256={"src/example.py": "b" * 64},
        runtime={"python": "3.12.10"},
    )
    monkeypatch.setattr(
        chunk_analyzer_cli,
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
        chunk_analyzer_cli,
        "evaluate_chunks",
        lambda *_args, **_kwargs: (summary, traces, events),
    )
    monkeypatch.setattr(chunk_analyzer_cli, "sha256_file", lambda _: "a" * 64)
    monkeypatch.setattr(
        chunk_analyzer_cli, "code_provenance", lambda *_args, **_kwargs: ("a" * 40, True, {})
    )
    monkeypatch.setattr(chunk_analyzer_cli, "runtime_provenance", lambda: {"python": "3.12.10"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "epsa-evaluate-chunk-analyzer", "run", "--run-id", "chunk-analyzer-cli-v1",
            "--dataset-directory", str(tmp_path / "dataset"),
            "--retrieval-export-directory", str(tmp_path / "retrieval"),
            "--export-root", str(tmp_path / "exports"),
        ],
    )
    assert chunk_analyzer_cli.main() == 0
    assert "Diagnostics are not Component 02 accuracy metrics" in capsys.readouterr().out
    directory = tmp_path / "exports" / "chunk-analyzer-cli-v1"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "epsa-evaluate-chunk-analyzer",
            "inspect",
            str(directory),
            "--question-id",
            "q-1",
        ],
    )
    assert chunk_analyzer_cli.main() == 0
    assert "chunk:inception" in capsys.readouterr().out
    monkeypatch.setattr(
        sys, "argv", ["epsa-evaluate-chunk-analyzer", "inspect", str(directory)]
    )
    assert chunk_analyzer_cli.main() == 0
    assert "chunk-analyzer-run-v1" in capsys.readouterr().out
