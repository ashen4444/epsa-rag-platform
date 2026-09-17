from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from epsa_rag.core.exceptions import SourceValidationError
from epsa_rag.data import pipeline
from epsa_rag.data.config import HardTestPreparationConfig


def test_main_downloads_prepares_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    output_root = tmp_path / "output"
    calls: list[str] = []

    def fake_download_source(**kwargs: object) -> Path:
        calls.append("download")
        destination = kwargs["destination"]
        assert isinstance(destination, Path)
        destination.write_text("[]", encoding="utf-8")
        return destination

    def fake_prepare_benchmark(**kwargs: object) -> SimpleNamespace:
        calls.append("prepare")
        return SimpleNamespace(
            dataset_directory=output_root / "datasets" / "dataset-v1",
            corpus_directory=output_root / "corpus" / "corpus-v1",
            dataset_manifest=SimpleNamespace(
                eligible_question_count=3,
                selected_question_count=3,
            ),
            corpus_manifest=SimpleNamespace(
                duplicate_paragraph_count=2,
                unique_paragraph_count=4,
            ),
        )

    monkeypatch.setattr(pipeline, "download_source", fake_download_source)
    monkeypatch.setattr(pipeline, "prepare_benchmark", fake_prepare_benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "epsa-prepare-hotpotqa",
            "--download",
            "--source-path",
            str(source_path),
            "--output-root",
            str(output_root),
            "--question-count",
            "3",
            "--dataset-version",
            "dataset-v1",
            "--corpus-version",
            "corpus-v1",
        ],
    )

    assert pipeline.main() == 0
    assert calls == ["download", "prepare"]
    assert '"selected_questions": 3' in capsys.readouterr().out


def test_main_rejects_a_missing_source_without_download(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["epsa-prepare-hotpotqa", "--source-path", str(tmp_path / "missing.json")],
    )

    with pytest.raises(SystemExit, match="2"):
        pipeline.main()

    assert "pass --download" in capsys.readouterr().err


def test_main_resolves_the_hard_test_profile(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "train.json"
    source_path.write_text("[]", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_prepare_benchmark(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            dataset_directory=tmp_path / "data" / "datasets" / "test-v1",
            corpus_directory=tmp_path / "data" / "corpus" / "corpus-v1",
            dataset_manifest=SimpleNamespace(
                eligible_question_count=15_661,
                selected_question_count=10_000,
            ),
            corpus_manifest=SimpleNamespace(
                duplicate_paragraph_count=5,
                unique_paragraph_count=99_000,
            ),
        )

    monkeypatch.setattr(pipeline, "prepare_benchmark", fake_prepare_benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "epsa-prepare-hotpotqa",
            "--profile",
            "hard-test",
            "--source-path",
            str(source_path),
        ],
    )

    assert pipeline.main() == 0
    config = captured["config"]
    assert isinstance(config, HardTestPreparationConfig)
    assert config.question_count == 10_000
    assert config.difficulty_filter == "hard"
    assert '"eligible_questions": 15661' in capsys.readouterr().out


def test_main_reports_preparation_errors_as_cli_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text("[]", encoding="utf-8")

    def fail_preparation(**kwargs: object) -> None:
        raise SourceValidationError("invalid fixture")

    monkeypatch.setattr(pipeline, "prepare_benchmark", fail_preparation)
    monkeypatch.setattr(
        sys,
        "argv",
        ["epsa-prepare-hotpotqa", "--source-path", str(source_path)],
    )

    with pytest.raises(SystemExit, match="2"):
        pipeline.main()

    assert "invalid fixture" in capsys.readouterr().err

