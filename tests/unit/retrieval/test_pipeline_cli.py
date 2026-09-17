from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from epsa_rag.core.exceptions import IndexIntegrityError
from epsa_rag.retrieval import cli, pipeline
from epsa_rag.retrieval.config import BM25Config, DenseConfig


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("all", ["bm25", "provider", "dense"]), ("bm25", ["bm25"]), ("dense", ["provider", "dense"])],
)
def test_build_indexes_selects_requested_backends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
    expected: list[str],
) -> None:
    calls: list[str] = []
    corpus = object()
    monkeypatch.setattr(pipeline.FrozenCorpus, "load", lambda path: corpus)

    def fake_bm25(**kwargs: Any) -> str:
        calls.append("bm25")
        return "bm25-result"

    def fake_provider(config: DenseConfig, **kwargs: Any) -> str:
        calls.append("provider")
        assert kwargs["progress_callback"] is not None
        return "provider"

    def fake_dense(**kwargs: Any) -> str:
        calls.append("dense")
        assert kwargs["embedding_provider"] == "provider"
        return "dense-result"

    monkeypatch.setattr(pipeline, "build_bm25_index", fake_bm25)
    monkeypatch.setattr(pipeline, "OpenAIEmbeddingProvider", fake_provider)
    monkeypatch.setattr(pipeline, "build_dense_index", fake_dense)

    results = pipeline.build_indexes(
        corpus_directory=tmp_path,
        index_root=tmp_path / "indexes",
        kind=kind,  # type: ignore[arg-type]
        bm25_config=BM25Config(),
        dense_config=DenseConfig(),
    )

    assert calls == expected
    assert len(results) == expected.count("bm25") + expected.count("dense")


def test_index_pipeline_main_prints_built_index_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    built = SimpleNamespace(
        directory=tmp_path / "index",
        manifest=SimpleNamespace(
            configuration_fingerprint="abc",
            index_kind="bm25",
            index_version="bm25-v1",
        ),
    )
    monkeypatch.setattr(pipeline, "build_indexes", lambda **kwargs: (built,))
    monkeypatch.setattr(
        sys,
        "argv",
        ["epsa-build-retrieval-indexes", "--kind", "bm25", "--candidate-k", "10"],
    )

    assert pipeline.main() == 0
    output = capsys.readouterr().out
    assert '"index_kind": "bm25"' in output
    assert '"configuration_fingerprint": "abc"' in output


def test_index_pipeline_main_reports_project_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(**kwargs: Any) -> tuple[()]:
        raise IndexIntegrityError("broken index")

    monkeypatch.setattr(pipeline, "build_indexes", fail)
    monkeypatch.setattr(sys, "argv", ["epsa-build-retrieval-indexes"])
    with pytest.raises(SystemExit, match="2"):
        pipeline.main()
    assert "broken index" in capsys.readouterr().err


def test_retrieve_cli_loads_indexes_and_prints_canonical_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = SimpleNamespace(manifest=SimpleNamespace(version="corpus-v1"))
    manifests = [
        SimpleNamespace(configuration=BM25Config()),
        SimpleNamespace(configuration=DenseConfig()),
    ]
    monkeypatch.setattr(cli.FrozenCorpus, "load", lambda path: corpus)
    monkeypatch.setattr(cli, "load_index_manifest", lambda path: manifests.pop(0))
    monkeypatch.setattr(cli, "load_bm25_index", lambda *args, **kwargs: "bm25")
    monkeypatch.setattr(cli, "load_dense_index", lambda *args, **kwargs: "dense-index")
    monkeypatch.setattr(cli, "OpenAIEmbeddingProvider", lambda config: "provider")
    monkeypatch.setattr(cli, "DenseRetriever", lambda index, provider: "dense")

    class FakeHybridRetriever:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["bm25"] == "bm25"
            assert kwargs["dense"] == "dense"
            assert kwargs["config"].retriever_version == "hybrid-retriever-v2"
            assert kwargs["config"].fusion.bm25_weight == 0.3
            assert kwargs["config"].fusion.dense_weight == 0.7

        def retrieve(self, query: Any) -> SimpleNamespace:
            assert query.text == "Who wrote it?"
            return SimpleNamespace(
                model_dump_json=lambda **kwargs: '{"retriever_version":"hybrid-retriever-v2"}'
            )

    monkeypatch.setattr(cli, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(
        sys,
        "argv",
        ["epsa-retrieve", "Who wrote it?", "--question-id", "q1", "--top-k", "5"],
    )

    assert cli.main() == 0
    assert '"retriever_version"' in capsys.readouterr().out


def test_retrieve_cli_reports_loading_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(path: Path) -> None:
        raise IndexIntegrityError("missing index")

    monkeypatch.setattr(cli.FrozenCorpus, "load", fail)
    monkeypatch.setattr(sys, "argv", ["epsa-retrieve", "question"])
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    assert "missing index" in capsys.readouterr().err
