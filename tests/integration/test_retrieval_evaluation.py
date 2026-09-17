from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest
from openai import OpenAIError

from epsa_rag.core.exceptions import (
    ConfigurationError,
    FrozenArtifactError,
    QueryEmbeddingCacheError,
    SourceValidationError,
)
from epsa_rag.evaluation.retrieval import cli, pipeline
from epsa_rag.evaluation.retrieval.exports import compare_exports, load_export
from epsa_rag.evaluation.retrieval.models import EvaluationConfig
from epsa_rag.retrieval.config import BM25Config, DenseConfig, RRFConfig
from epsa_rag.retrieval.persistence import build_bm25_index, build_dense_index


@pytest.fixture
def evaluation_runner(evaluation_data, evaluation_embeddings, tmp_path, monkeypatch):
    corpus, _, directory = evaluation_data
    indexes = tmp_path / "indexes"
    build_bm25_index(corpus=corpus, index_root=indexes, config=BM25Config())
    build_dense_index(
        corpus=corpus,
        index_root=indexes,
        config=DenseConfig(),
        embedding_provider=evaluation_embeddings,
    )
    monkeypatch.setattr(pipeline, "code_provenance", lambda *a, **k: ("a" * 40, False, {}))
    monkeypatch.setattr(pipeline, "runtime_provenance", lambda: {"python": "3.12"})
    client = MagicMock()
    monkeypatch.setattr(pipeline, "OpenAI", lambda **k: client)
    monkeypatch.setattr(pipeline, "OpenAIEmbeddingProvider", lambda *a, **k: evaluation_embeddings)
    return dict(
        dataset_directory=directory,
        corpus_directory=corpus.directory,
        index_root=indexes,
        export_root=tmp_path / "exports",
        repository_root=tmp_path,
    )


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_real_indexes_to_verified_diagnostic_exports(evaluation_runner, mode):
    config = EvaluationConfig(mode=mode)
    summary = pipeline.run_benchmark(run_id=f"test-{mode}", config=config, **evaluation_runner)
    directory = evaluation_runner["export_root"] / f"test-{mode}"
    loaded, traces = load_export(directory)
    assert loaded == summary
    assert summary.full_benchmark
    assert summary.completed_questions == 3
    assert len(traces) == 3
    assert all(trace.retrieval.results for trace in traces)
    assert len(summary.metadata.index_manifests) == (2 if mode == "hybrid" else 1)
    with pytest.raises(FrozenArtifactError):
        pipeline.run_benchmark(run_id=f"test-{mode}", config=config, **evaluation_runner)


def test_subset_and_warmup_are_explicit(evaluation_runner):
    summary = pipeline.run_benchmark(
        run_id="subset",
        **evaluation_runner,
        config=EvaluationConfig(mode="bm25", question_limit=1, warmup_questions=1),
    )
    assert not summary.full_benchmark
    assert summary.completed_questions == summary.warmup_completed == 1
    assert len(summary.metadata.question_ids) == 1


def test_build_frozen_query_cache_then_evaluate_read_only(
    evaluation_runner, evaluation_embeddings, monkeypatch, tmp_path
):
    cache_root = tmp_path / "query-cache"
    manifest = pipeline.build_benchmark_query_cache(
        dataset_directory=evaluation_runner["dataset_directory"],
        corpus_directory=evaluation_runner["corpus_directory"],
        index_root=evaluation_runner["index_root"],
        repository_root=evaluation_runner["repository_root"],
        query_cache_root=cache_root,
        cache_version="test-cache-v1",
        dense_index_version="dense-openai-small-faiss-flatip-v1",
    )
    assert manifest.question_count == 3
    monkeypatch.setattr(
        pipeline,
        "OpenAI",
        MagicMock(side_effect=AssertionError("read-only mode must not initialize OpenAI")),
    )
    summary = pipeline.run_benchmark(
        run_id="cached",
        config=EvaluationConfig(
            mode="hybrid",
            query_embedding_cache="read-only",
            query_embedding_cache_version="test-cache-v1",
        ),
        query_cache_root=cache_root,
        **evaluation_runner,
    )
    _, traces = load_export(evaluation_runner["export_root"] / "cached")
    assert summary.metadata.query_embedding_collection == manifest
    assert all(trace.query_embedding is not None for trace in traces)
    assert all(trace.query_embedding.source == "cache" for trace in traces if trace.query_embedding)
    assert summary.retrieval_core_latency_p50_ms is not None


def test_read_only_cache_is_preflighted_before_export_creation(evaluation_runner, tmp_path):
    with pytest.raises(QueryEmbeddingCacheError, match="frozen query collection"):
        pipeline.run_benchmark(
            run_id="missing-cache",
            config=EvaluationConfig(
                mode="dense",
                query_embedding_cache="read-only",
                query_embedding_cache_version="missing-cache-v1",
            ),
            query_cache_root=tmp_path / "missing",
            **evaluation_runner,
        )
    assert not (evaluation_runner["export_root"] / "missing-cache").exists()


@pytest.mark.parametrize("change", [{"question_limit": 4}, {"warmup_questions": 4}])
def test_limits_validated_before_run_creation(evaluation_runner, change):
    with pytest.raises(ConfigurationError):
        pipeline.run_benchmark(
            run_id="invalid", **evaluation_runner, config=EvaluationConfig(mode="bm25", **change)
        )
    assert not (evaluation_runner["export_root"] / "invalid").exists()


@pytest.mark.parametrize("branch", ["bm25", "dense"])
def test_config_index_mismatch_rejected(evaluation_runner, branch):
    config = EvaluationConfig(mode=branch)
    changed_branch = getattr(config.retriever, branch).model_copy(update={"candidate_k": 99})
    changed = config.model_copy(
        update={"retriever": config.retriever.model_copy(update={branch: changed_branch})}
    )
    with pytest.raises(ConfigurationError, match="configuration differs"):
        pipeline.run_benchmark(run_id="invalid-config", config=changed, **evaluation_runner)


def test_config_loaded_from_frozen_manifests(evaluation_runner):
    config = pipeline.configuration_from_indexes(
        corpus_directory=evaluation_runner["corpus_directory"],
        index_root=evaluation_runner["index_root"],
        bm25_version="bm25-v1",
        dense_version="dense-openai-small-faiss-flatip-v1",
        mode="hybrid",
        fusion=RRFConfig(),
    )
    assert config == EvaluationConfig().retriever
    assert config.retriever_version == "hybrid-retriever-v2"
    assert config.fusion.bm25_weight == 0.3
    assert config.fusion.dense_weight == 0.7
    v1 = pipeline.configuration_from_indexes(
        corpus_directory=evaluation_runner["corpus_directory"],
        index_root=evaluation_runner["index_root"],
        bm25_version="bm25-v1",
        dense_version="dense-openai-small-faiss-flatip-v1",
        mode="hybrid",
        retriever_version="hybrid-retriever-v1",
        fusion=RRFConfig(bm25_weight=1.0, dense_weight=1.0),
    )
    assert v1.retriever_version == "hybrid-retriever-v1"
    assert v1.fusion.bm25_weight == v1.fusion.dense_weight == 1.0
    config = pipeline.configuration_from_indexes(
        corpus_directory=evaluation_runner["corpus_directory"],
        index_root=evaluation_runner["index_root"],
        bm25_version="bm25-v1",
        dense_version="not-built",
        mode="bm25",
        fusion=RRFConfig(),
    )
    assert config.dense.index_version == "not-built"


def test_export_tampering_detected(evaluation_runner):
    pipeline.run_benchmark(
        run_id="tamper", config=EvaluationConfig(mode="bm25"), **evaluation_runner
    )
    directory = evaluation_runner["export_root"] / "tamper"
    with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(SourceValidationError, match="integrity"):
        load_export(directory)


def test_compare_reports_paired_changes(evaluation_runner):
    for name, mode in (("before", "bm25"), ("after", "dense")):
        pipeline.run_benchmark(run_id=name, config=EvaluationConfig(mode=mode), **evaluation_runner)
    root = evaluation_runner["export_root"]
    compared = compare_exports(root / "before", root / "after", metric="recall@1")
    assert sum(len(ids) for ids in compared["question_ids"].values()) == 3
    missing = compare_exports(
        root / "before", root / "after", metric="missing_gold_document_rate@1"
    )
    assert missing["question_ids"] == compared["question_ids"]
    with pytest.raises(ValueError, match="unavailable"):
        compare_exports(root / "before", root / "after", metric="unknown")
    pipeline.run_benchmark(
        run_id="subset", config=EvaluationConfig(mode="bm25", question_limit=1), **evaluation_runner
    )
    with pytest.raises(ValueError, match="identical benchmark"):
        compare_exports(root / "before", root / "subset")


def test_cli_run_inspect_failures_question_and_compare(evaluation_runner, monkeypatch, capsys):
    args = ["epsa-evaluate-retriever", "run", "--run-id", "cli-test", "--mode", "bm25"]
    for key, value in evaluation_runner.items():
        args.extend(["--" + key.replace("_", "-"), str(value)])
    monkeypatch.setattr(sys, "argv", args)
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["completed_questions"] == 3
    directory = evaluation_runner["export_root"] / "cli-test"
    _, traces = load_export(directory)
    for extra in ([], ["--failures"], ["--question-id", traces[0].question.question_id]):
        monkeypatch.setattr(
            sys, "argv", ["epsa-evaluate-retriever", "inspect", str(directory), *extra]
        )
        assert cli.main() == 0
        assert json.loads(capsys.readouterr().out) is not None
    monkeypatch.setattr(
        sys, "argv", ["epsa-evaluate-retriever", "compare", str(directory), str(directory)]
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["delta"] == 0
    monkeypatch.setattr(
        sys,
        "argv",
        ["epsa-evaluate-retriever", "inspect", str(directory), "--question-id", "absent"],
    )
    with pytest.raises(SystemExit):
        cli.main()


def test_cli_builds_frozen_query_cache(evaluation_runner, monkeypatch, capsys, tmp_path):
    cache_root = tmp_path / "cli-query-cache"
    args = ["epsa-evaluate-retriever", "cache-queries"]
    for key in ("dataset_directory", "corpus_directory", "index_root", "repository_root"):
        args.extend(["--" + key.replace("_", "-"), str(evaluation_runner[key])])
    args.extend(
        [
            "--query-cache-root",
            str(cache_root),
            "--query-embedding-cache-version",
            "cli-cache-v1",
        ]
    )
    monkeypatch.setattr(sys, "argv", args)

    assert cli.main() == 0

    result = json.loads(capsys.readouterr().out)
    assert result["question_count"] == 3
    assert result["cache_version"] == "cli-cache-v1"
    assert (cache_root / "cli-cache-v1" / "collections" / "eval-dataset-v1").is_dir()


def test_cli_missing_key_and_failed_run(evaluation_runner, monkeypatch, capsys):
    args = ["cmd", "run", "--run-id", "cli-failed", "--mode", "hybrid"]
    for key, value in evaluation_runner.items():
        args.extend(["--" + key.replace("_", "-"), str(value)])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(pipeline, "OpenAI", MagicMock(side_effect=OpenAIError("missing")))
    with pytest.raises(SystemExit):
        cli.main()
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    monkeypatch.setattr(pipeline, "OpenAI", MagicMock())
    monkeypatch.setattr(
        pipeline.SingleBranchRetriever, "retrieve", MagicMock(side_effect=RuntimeError)
    )
    args[args.index("hybrid")] = "bm25"
    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_single_branch_default_depth(evaluation_data):
    corpus, _, _ = evaluation_data
    from epsa_rag.core.models import RetrievalQuery
    from epsa_rag.retrieval.bm25.index import BM25Index

    adapter = pipeline.SingleBranchRetriever(
        corpus, BM25Index.build(corpus.chunks, BM25Config()), "bm25-v1"
    )
    assert adapter.retrieve(RetrievalQuery(text="Alpha")).results
