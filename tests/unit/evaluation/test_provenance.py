from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from epsa_rag.core.exceptions import ConfigurationError
from epsa_rag.evaluation.retrieval import provenance


@pytest.mark.parametrize("dirty", [False, True])
def test_git_source_hashes_and_dirty_flag(tmp_path, monkeypatch, dirty):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "example.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]", encoding="utf-8")
    responses = [
        SimpleNamespace(stdout="a" * 40),
        SimpleNamespace(stdout="?? src/example.py" if dirty else ""),
    ]
    monkeypatch.setattr(provenance.subprocess, "run", MagicMock(side_effect=responses))
    commit, detected, hashes = provenance.code_provenance(tmp_path, allow_dirty=True)
    assert commit == "a" * 40
    assert detected == dirty
    assert set(hashes) == {"src/example.py", "pyproject.toml"}
    assert all(len(digest) == 64 for digest in hashes.values())


def test_dirty_research_runs_require_explicit_development_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        MagicMock(
            side_effect=[SimpleNamespace(stdout="a" * 40), SimpleNamespace(stdout=" M file.py")]
        ),
    )
    with pytest.raises(ConfigurationError, match="commit the implementation"):
        provenance.code_provenance(tmp_path, allow_dirty=False)


@pytest.mark.parametrize("error", [OSError(), subprocess.CalledProcessError(1, "git")])
def test_missing_git_provenance_is_not_fabricated(tmp_path, monkeypatch, error):
    monkeypatch.setattr(provenance.subprocess, "run", MagicMock(side_effect=error))
    with pytest.raises(ConfigurationError, match="Git repository"):
        provenance.code_provenance(tmp_path, allow_dirty=False)


def test_runtime_records_versions_without_environment(monkeypatch):
    monkeypatch.setattr(
        provenance,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": "numpy"}, version="2.4"),
            SimpleNamespace(metadata={}, version="unused"),
        ],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "NEVER RECORD ME")
    result = provenance.runtime_provenance()
    assert result["package:numpy"] == "2.4"
    assert result["python"]
    assert "NEVER RECORD ME" not in str(result)
