"""Command-line tests for permanent system evaluation without live API calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai import APIConnectionError

from epsa_rag.evaluation.system import cli
from epsa_rag.evaluation.system.models import SystemEvaluationConfig
from epsa_rag.retrieval.config import HybridRetrieverConfig, RRFConfig


def _config() -> SystemEvaluationConfig:
    return SystemEvaluationConfig(
        retriever=HybridRetrieverConfig(fusion=RRFConfig(result_k=30))
    )


def test_run_command_normalizes_top_k_and_prints_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_default(**kwargs: object) -> SystemEvaluationConfig:
        captured.update(kwargs)
        return _config()

    def fake_run(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        progress = kwargs["progress"]
        assert callable(progress)
        progress(25, 25)
        return SimpleNamespace(
            metadata=SimpleNamespace(run_id="paired-eval-01"),
            status="completed",
            completed_traces=12,
            failed_traces=0,
            full_benchmark=False,
        )

    monkeypatch.setattr(cli, "default_system_config", fake_default)
    monkeypatch.setattr(cli, "run_system_evaluation", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "epsa-evaluate-systems",
            "run",
            "--run-id",
            "paired-eval-01",
            "--top-k",
            "10",
            "5",
            "10",
            "--question-limit",
            "1",
            "--allow-dirty-dev-run",
        ],
    )

    assert cli.main() == 0
    output = capsys.readouterr()
    assert captured["top_ks"] == (5, 10)
    assert captured["allow_dirty"] is True
    assert json.loads(output.out)["completed_traces"] == 12
    assert "Evaluated 25/25" in output.err


def test_inspect_command_prints_summary_or_selected_question(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    summary = SimpleNamespace(model_dump_json=lambda indent: '{"status":"completed"}')
    trace = SimpleNamespace(
        question=SimpleNamespace(question_id="q1"),
        model_dump=lambda mode: {"question_id": "q1"},
    )
    monkeypatch.setattr(cli, "load_system_export", lambda _: (summary, (trace,)))
    monkeypatch.setattr(sys, "argv", ["epsa-evaluate-systems", "inspect", str(tmp_path)])
    assert cli.main() == 0
    assert "completed" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["epsa-evaluate-systems", "inspect", str(tmp_path), "--question-id", "q1"],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)[0]["question_id"] == "q1"

    monkeypatch.setattr(
        sys,
        "argv",
        ["epsa-evaluate-systems", "inspect", str(tmp_path), "--question-id", "missing"],
    )
    with pytest.raises(SystemExit, match="2"):
        cli.main()


def test_cli_converts_openai_errors_to_actionable_parser_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "default_system_config", lambda **_: _config())
    monkeypatch.setattr(
        cli,
        "run_system_evaluation",
        lambda **_: (_ for _ in ()).throw(APIConnectionError(request=None)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["epsa-evaluate-systems", "run", "--run-id", "unavailable"],
    )
    with pytest.raises(SystemExit, match="2"):
        cli.main()

