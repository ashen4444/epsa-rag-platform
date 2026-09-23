"""Offline tests for the live system-evaluation composition root."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from epsa_rag.evaluation.system import pipeline as orchestration
from epsa_rag.evaluation.system.models import SystemEvaluationConfig, SystemVariant
from epsa_rag.retrieval.config import HybridRetrieverConfig, RRFConfig
from epsa_rag.retrieval.dense.query_cache import QueryEmbeddingObservation

from .test_system_evaluation import _example, _fixed_trace


class _Delegate:
    def __init__(self, observations: list[QueryEmbeddingObservation]) -> None:
        self.observations = observations

    def run(self, *, question_id: str, question: str):  # type: ignore[no-untyped-def]
        self.observations.append(
            QueryEmbeddingObservation(
                cache_mode="read-write",
                source="cache",
                cache_version="query-cache-v1",
                cache_key="a" * 64,
                query_text_sha256="b" * 64,
                model="text-embedding-3-small",
                dimensions=1536,
                vector_sha256="c" * 64,
                cache_entry="entries/a.json",
                latency_ms=1,
            )
        )
        return _fixed_trace()


def _config(*, question_limit: int | None = None) -> SystemEvaluationConfig:
    return SystemEvaluationConfig(
        top_ks=(1,),
        question_limit=question_limit,
        retriever=HybridRetrieverConfig(fusion=RRFConfig(result_k=1)),
    )


def test_observed_pipeline_scopes_and_drains_embedding_events() -> None:
    observations: list[QueryEmbeddingObservation] = []
    observed = orchestration._ObservedPipeline(_Delegate(observations), observations)  # type: ignore[arg-type]

    assert observed.run(question_id="q1", question="Where was the director born?")
    assert observations == []
    assert len(observed.take_query_embedding_observations()) == 1
    assert observed.take_query_embedding_observations() == ()


def test_default_config_uses_frozen_hybrid_v2_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_configuration(**kwargs: object) -> HybridRetrieverConfig:
        captured.update(kwargs)
        return HybridRetrieverConfig(fusion=kwargs["fusion"])

    monkeypatch.setattr(orchestration, "configuration_from_indexes", fake_configuration)
    config = orchestration.default_system_config(
        corpus_directory=tmp_path,
        index_root=tmp_path,
        top_ks=(5, 10),
        question_limit=1,
    )
    assert config.retriever.fusion.result_k == 10
    assert config.question_limit == 1
    assert captured["mode"] == "hybrid"
    with pytest.raises(ValueError, match="nonempty"):
        orchestration.default_system_config(
            corpus_directory=tmp_path, index_root=tmp_path, top_ks=()
        )


def test_run_composes_all_three_systems_and_restores_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    example = _example()
    corpus = SimpleNamespace(
        manifest=SimpleNamespace(version="corpus-v1"),
        manifest_sha256="1" * 64,
        corpus_file_sha256="2" * 64,
    )
    benchmark = SimpleNamespace(
        examples=(example,),
        manifest=SimpleNamespace(
            version="dataset-v1", files=(SimpleNamespace(sha256="3" * 64),)
        ),
        manifest_sha256="4" * 64,
    )
    config = _config()
    index = SimpleNamespace(config=config.retriever.bm25)
    dense_index = SimpleNamespace(config=config.retriever.dense)
    closed: list[bool] = []
    threads: list[int] = []
    factories: list[SystemVariant] = []

    class FakeCache:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def load_collection(self, **kwargs: object) -> None:
            assert len(kwargs["queries"]) == 1  # type: ignore[arg-type]

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["base_url"] == "https://api.openai.com/v1"

        def close(self) -> None:
            closed.append(True)

    class FakeStore:
        def __init__(self, root: Path, metadata: object) -> None:
            self.directory = root / "paired-eval"

    metadata = SimpleNamespace(run_id="paired-eval")
    summary = SimpleNamespace(status="completed")
    monkeypatch.setattr(orchestration, "code_provenance", lambda *_a, **_k: ("a" * 40, False, {}))
    monkeypatch.setattr(orchestration, "runtime_provenance", lambda: {})
    monkeypatch.setattr(orchestration.FrozenCorpus, "load", lambda _: corpus)
    monkeypatch.setattr(orchestration.FrozenBenchmark, "load", lambda *_: benchmark)
    monkeypatch.setattr(orchestration, "index_directory", lambda *_a, **_k: tmp_path)
    monkeypatch.setattr(orchestration, "load_bm25_index", lambda *_a, **_k: index)
    monkeypatch.setattr(orchestration, "load_dense_index", lambda *_a, **_k: dense_index)
    monkeypatch.setattr(orchestration, "QueryEmbeddingCache", FakeCache)
    monkeypatch.setattr(orchestration, "load_index_manifest", lambda _: "manifest")
    monkeypatch.setattr(orchestration, "SystemRunMetadata", lambda **_: metadata)
    monkeypatch.setattr(orchestration, "ResumableSystemRunStore", FakeStore)
    monkeypatch.setattr(orchestration, "OpenAI", FakeClient)
    monkeypatch.setattr(orchestration.faiss, "omp_get_max_threads", lambda: 7)
    monkeypatch.setattr(orchestration.faiss, "omp_set_num_threads", threads.append)
    for name in (
        "OpenAIEmbeddingProvider",
        "CachedQueryEmbeddingProvider",
        "DenseRetriever",
        "HybridRetriever",
        "OpenAIFinalAnswerGenerator",
        "OpenAIAdaptiveRetrievalController",
        "TikTokenCounter",
    ):
        monkeypatch.setattr(orchestration, name, lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(
        orchestration.EPSAController, "research_v1", lambda: SimpleNamespace()
    )
    monkeypatch.setattr(
        orchestration, "FixedBaselinePipeline", lambda **_: SimpleNamespace()
    )
    monkeypatch.setattr(
        orchestration, "AdaptiveBaselinePipeline", lambda **_: SimpleNamespace()
    )
    monkeypatch.setattr(orchestration, "EPSAPipeline", lambda **_: SimpleNamespace())

    def fake_evaluate(**kwargs: object):  # type: ignore[no-untyped-def]
        factory = kwargs["pipeline_factory"]
        for condition in config.conditions():
            assert isinstance(factory(condition), orchestration._ObservedPipeline)  # type: ignore[operator]
            factories.append(condition.system)
        return summary

    monkeypatch.setattr(orchestration, "evaluate_systems", fake_evaluate)
    monkeypatch.setattr(orchestration, "load_system_export", lambda _: (summary, ()))

    result = orchestration.run_system_evaluation(
        run_id="paired-eval",
        dataset_directory=tmp_path,
        corpus_directory=tmp_path,
        index_root=tmp_path,
        export_root=tmp_path,
        repository_root=tmp_path,
        config=config,
    )

    assert result is summary
    assert factories == [SystemVariant.FIXED, SystemVariant.ADAPTIVE, SystemVariant.EPSA]
    assert threads == [1, 7]
    assert closed == [True]


def test_run_rejects_invalid_limit_and_index_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    corpus = SimpleNamespace(manifest=SimpleNamespace(version="corpus-v1"))
    benchmark = SimpleNamespace(examples=(_example(),))
    monkeypatch.setattr(orchestration, "code_provenance", lambda *_a, **_k: ("a", False, {}))
    monkeypatch.setattr(orchestration.FrozenCorpus, "load", lambda _: corpus)
    monkeypatch.setattr(orchestration.FrozenBenchmark, "load", lambda *_: benchmark)
    common = dict(
        run_id="bad",
        dataset_directory=tmp_path,
        corpus_directory=tmp_path,
        index_root=tmp_path,
        export_root=tmp_path,
        repository_root=tmp_path,
    )
    with pytest.raises(Exception, match="question_limit exceeds"):
        orchestration.run_system_evaluation(config=_config(question_limit=2), **common)

    monkeypatch.setattr(orchestration, "index_directory", lambda *_a, **_k: tmp_path)
    monkeypatch.setattr(
        orchestration,
        "load_bm25_index",
        lambda *_a, **_k: SimpleNamespace(config=SimpleNamespace()),
    )
    monkeypatch.setattr(
        orchestration,
        "load_dense_index",
        lambda *_a, **_k: SimpleNamespace(config=_config().retriever.dense),
    )
    with pytest.raises(Exception, match="differs from frozen indexes"):
        orchestration.run_system_evaluation(config=_config(), **common)
