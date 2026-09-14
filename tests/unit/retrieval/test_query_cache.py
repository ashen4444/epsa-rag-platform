from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from epsa_rag.core.exceptions import QueryEmbeddingCacheError
from epsa_rag.core.models import ParagraphChunk, RetrievalQuery
from epsa_rag.retrieval.config import DenseConfig
from epsa_rag.retrieval.dense.query_cache import (
    CachedQueryEmbeddingProvider,
    QueryEmbeddingCache,
    QueryEmbeddingObservation,
    QueryEmbeddingObservationBuffer,
    query_embedding_cache_key,
    query_text_sha256,
    vector_sha256,
)
from epsa_rag.retrieval.interfaces import FloatMatrix, FloatVector


class FakeProvider:
    model = "text-embedding-3-small"
    dimensions = 1536

    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.queries: list[RetrievalQuery] = []
        self.batches: list[tuple[RetrievalQuery, ...]] = []

    def embed_documents(self, chunks: Sequence[ParagraphChunk]) -> FloatMatrix:
        return np.full((len(chunks), self.dimensions), self.value, dtype=np.float32)

    def embed_query(self, query: RetrievalQuery) -> FloatVector:
        self.queries.append(query)
        return np.full(self.dimensions, self.value, dtype=np.float32)

    def embed_queries(self, queries: Sequence[RetrievalQuery]) -> FloatMatrix:
        batch = tuple(queries)
        self.batches.append(batch)
        return np.full((len(batch), self.dimensions), self.value, dtype=np.float32)


def test_cache_key_uses_exact_text_model_dimensions_and_logical_version() -> None:
    plain = RetrievalQuery(text="Question?", question_id="q1")
    same_text = RetrievalQuery(text="Question?", question_id="q2")
    spaced = RetrievalQuery(text=" Question? ", question_id="q1")
    arguments = dict(model="text-embedding-3-small", dimensions=1536, cache_version="cache-v1")

    assert query_embedding_cache_key(plain, **arguments) == query_embedding_cache_key(
        same_text, **arguments
    )
    assert query_embedding_cache_key(plain, **arguments) != query_embedding_cache_key(
        spaced, **arguments
    )
    assert query_text_sha256(plain) != query_text_sha256(spaced)
    assert query_embedding_cache_key(plain, **arguments) != query_embedding_cache_key(
        plain, **(arguments | {"cache_version": "cache-v2"})
    )


def test_cache_round_trip_is_immutable_and_integrity_checked(tmp_path: Path) -> None:
    config = DenseConfig()
    cache = QueryEmbeddingCache(tmp_path, config=config, cache_version="cache-v1")
    query = RetrievalQuery(text="Exact question?", question_id="q1")
    vector = np.arange(1, config.dimensions + 1, dtype=np.float32)
    generated_at = datetime(2026, 1, 2, tzinfo=UTC)

    manifest = cache.store(query, vector, generated_at=generated_at)
    loaded = cache.load(query)

    assert loaded is not None
    assert np.array_equal(loaded[0], vector)
    assert loaded[1] == manifest
    assert loaded[1].generated_at == generated_at
    assert loaded[1].vector_sha256 == vector_sha256(vector)
    assert cache.relative_entry(cache.key(query)).endswith(cache.key(query))
    assert cache.store(query, vector) == manifest
    with pytest.raises(QueryEmbeddingCacheError, match="conflicting"):
        cache.store(query, vector + 1)

    vector_path = cache.entry_directory(cache.key(query)) / "vector.npy"
    vector_path.write_bytes(b"corrupt")
    with pytest.raises(QueryEmbeddingCacheError, match="checksum"):
        cache.load(query)


def test_cache_rejects_invalid_version_vector_and_manifest(tmp_path: Path) -> None:
    config = DenseConfig()
    for version in ("bad version", "bad:version"):
        with pytest.raises(QueryEmbeddingCacheError, match="version"):
            QueryEmbeddingCache(tmp_path, config=config, cache_version=version)
    cache = QueryEmbeddingCache(tmp_path, config=config, cache_version="cache-v1")
    query = RetrievalQuery(text="Question?")
    with pytest.raises(QueryEmbeddingCacheError, match="float32 shape"):
        cache.store(query, np.ones(config.dimensions, dtype=np.float64))
    with pytest.raises(QueryEmbeddingCacheError, match="invalid"):
        cache.store(query, np.zeros(config.dimensions, dtype=np.float32))

    directory = cache.entry_directory(cache.key(query))
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"invalid": True}), encoding="utf-8")
    with pytest.raises(QueryEmbeddingCacheError, match="manifest"):
        cache.load(query)


def test_provider_modes_apply_live_hit_miss_and_observation_rules(tmp_path: Path) -> None:
    config = DenseConfig()
    query = RetrievalQuery(text="Question?", question_id="q1")
    delegate = FakeProvider()
    cache = QueryEmbeddingCache(tmp_path, config=config, cache_version="cache-v1")
    observations: list[QueryEmbeddingObservation] = []
    read_write = CachedQueryEmbeddingProvider(
        config=config,
        mode="read-write",
        cache=cache,
        delegate=delegate,
        observer=observations.append,
    )

    first = read_write.embed_query(query)
    second = read_write.embed_query(query)

    assert np.array_equal(first, second)
    assert delegate.queries == [query]
    assert [item.source for item in observations] == ["live", "cache"]
    assert observations[0].cache_key == cache.key(query)
    assert observations[0].query_text_sha256 == query_text_sha256(query)
    assert observations[0].vector_sha256 == vector_sha256(first)

    read_only = CachedQueryEmbeddingProvider(config=config, mode="read-only", cache=cache)
    assert np.array_equal(read_only.embed_query(query), first)
    with pytest.raises(QueryEmbeddingCacheError, match="cache miss"):
        read_only.embed_query(RetrievalQuery(text="New question?"))
    with pytest.raises(QueryEmbeddingCacheError, match="cannot embed documents"):
        read_only.embed_documents(())

    disabled_observations: list[QueryEmbeddingObservation] = []
    disabled = CachedQueryEmbeddingProvider(
        config=config,
        mode="disabled",
        delegate=delegate,
        observer=disabled_observations.append,
    )
    disabled.embed_query(query)
    assert delegate.queries == [query, query]
    assert disabled_observations[0].source == "live"
    assert disabled_observations[0].cache_version is None
    assert disabled_observations[0].cache_entry is None

    buffer = QueryEmbeddingObservationBuffer()
    buffer.record(observations[0])
    assert buffer.take(query) == observations[0]
    assert buffer.take(query) is None
    buffer.require_empty()
    buffer.record(observations[0])
    with pytest.raises(QueryEmbeddingCacheError, match="order mismatch"):
        buffer.take(RetrievalQuery(text="Different?"))


def test_frozen_collection_batches_misses_and_validates_exact_benchmark(tmp_path: Path) -> None:
    config = DenseConfig()
    cache = QueryEmbeddingCache(tmp_path, config=config, cache_version="cache-v1")
    provider = FakeProvider()
    queries = (
        RetrievalQuery(text="First?", question_id="q1"),
        RetrievalQuery(text="Second?", question_id="q2"),
    )
    arguments = dict(
        queries=queries,
        dataset_version="dataset-v1",
        dataset_manifest_sha256="a" * 64,
        dataset_file_sha256="b" * 64,
    )

    manifest = cache.build_collection(
        **arguments,
        git_commit_sha="c" * 40,
        provider=provider,
        generated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert provider.batches == [queries]
    assert manifest.question_count == 2
    assert cache.load_collection(**arguments) == manifest
    assert (
        cache.build_collection(**arguments, git_commit_sha="c" * 40, provider=provider) == manifest
    )
    assert provider.batches == [queries]
    with pytest.raises(QueryEmbeddingCacheError, match="identity mismatch"):
        cache.load_collection(**(arguments | {"dataset_file_sha256": "d" * 64}))
    with pytest.raises(QueryEmbeddingCacheError, match="entry mismatch"):
        cache.load_collection(**(arguments | {"queries": tuple(reversed(queries))}))


def test_frozen_collection_requires_identified_queries_and_valid_batch(tmp_path: Path) -> None:
    config = DenseConfig()
    cache = QueryEmbeddingCache(tmp_path, config=config, cache_version="cache-v1")
    provider = FakeProvider()
    arguments = dict(
        dataset_version="dataset-v1",
        dataset_manifest_sha256="a" * 64,
        dataset_file_sha256="b" * 64,
        git_commit_sha="c" * 40,
        provider=provider,
    )
    with pytest.raises(QueryEmbeddingCacheError, match="identified"):
        cache.build_collection(queries=(RetrievalQuery(text="Question?"),), **arguments)

    provider.embed_queries = lambda queries: np.ones((1, 2), dtype=np.float32)  # type: ignore[method-assign]
    with pytest.raises(QueryEmbeddingCacheError, match="expected float32"):
        cache.build_collection(
            queries=(RetrievalQuery(text="Question?", question_id="q1"),), **arguments
        )


@pytest.mark.parametrize(
    ("mode", "with_cache", "with_delegate", "message"),
    [
        ("disabled", False, False, "live provider"),
        ("read-only", False, False, "requires a cache"),
        ("read-write", True, False, "requires a live provider"),
    ],
)
def test_provider_rejects_incomplete_mode_configuration(
    tmp_path: Path,
    mode: str,
    with_cache: bool,
    with_delegate: bool,
    message: str,
) -> None:
    config = DenseConfig()
    cache = (
        QueryEmbeddingCache(tmp_path, config=config, cache_version="cache-v1")
        if with_cache
        else None
    )
    delegate = FakeProvider() if with_delegate else None
    with pytest.raises(QueryEmbeddingCacheError, match=message):
        CachedQueryEmbeddingProvider(
            config=config,
            mode=mode,  # type: ignore[arg-type]
            cache=cache,
            delegate=delegate,
        )
