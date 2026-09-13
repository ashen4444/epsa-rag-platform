from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from epsa_rag.core.exceptions import FrozenArtifactError, IndexIntegrityError
from epsa_rag.core.models import ParagraphChunk, RetrievalQuery
from epsa_rag.data.config import PreparationConfig
from epsa_rag.data.pipeline import prepare_benchmark
from epsa_rag.retrieval.config import BM25Config, DenseConfig
from epsa_rag.retrieval.corpus import FrozenCorpus
from epsa_rag.retrieval.persistence import (
    build_bm25_index,
    build_dense_index,
    load_bm25_index,
    load_dense_index,
    load_index_manifest,
)


class FakeEmbeddingProvider:
    model = "text-embedding-3-small"
    dimensions = 1536

    def embed_documents(self, chunks: tuple[ParagraphChunk, ...]) -> np.ndarray:
        matrix = np.zeros((len(chunks), self.dimensions), dtype=np.float32)
        for index in range(len(chunks)):
            matrix[index, index] = 1
        return matrix

    def embed_query(self, query: RetrievalQuery) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        vector[0] = 1
        return vector


def prepare_corpus(tmp_path: Path) -> FrozenCorpus:
    records = []
    for index in range(3):
        records.append(
            {
                "_id": f"q{index}",
                "question": f"Question {index}?",
                "answer": f"Answer {index}",
                "type": "bridge",
                "level": "hard",
                "supporting_facts": [[f"Title {index}", 0]],
                "context": [[f"Title {index}", [f"Sentence {index}."]]],
            }
        )
    source = tmp_path / "source.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    config = PreparationConfig(
        dataset_version="dataset-retrieval-test-v1",
        corpus_version="corpus-retrieval-test-v1",
        question_count=3,
        expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    result = prepare_benchmark(
        source_path=source,
        output_root=tmp_path / "data",
        config=config,
        generated_at=datetime(2026, 9, 13, tzinfo=UTC),
    )
    return FrozenCorpus.load(result.corpus_directory)


def test_builds_loads_and_freezes_bm25_and_dense_indexes(tmp_path: Path) -> None:
    corpus = prepare_corpus(tmp_path)
    index_root = tmp_path / "indexes"
    generated_at = datetime(2026, 9, 13, 12, tzinfo=UTC)
    bm25_config = BM25Config()
    dense_config = DenseConfig()

    bm25_result = build_bm25_index(
        corpus=corpus,
        index_root=index_root,
        config=bm25_config,
        generated_at=generated_at,
    )
    dense_result = build_dense_index(
        corpus=corpus,
        index_root=index_root,
        config=dense_config,
        embedding_provider=FakeEmbeddingProvider(),
        generated_at=generated_at,
    )

    assert (bm25_result.directory / "index.json").is_file()
    assert (dense_result.directory / "index.faiss").is_file()
    assert (dense_result.directory / "chunk_ids.json").is_file()
    assert bm25_result.manifest.corpus_file_sha256 == corpus.corpus_file_sha256
    assert dense_result.manifest.backend == "faiss.IndexFlatIP"
    assert dense_result.manifest.chunk_count == len(corpus.chunks)
    assert {item.name for item in dense_result.manifest.dependencies} == {
        "faiss-cpu",
        "numpy",
        "openai",
        "python",
    }

    bm25 = load_bm25_index(bm25_result.directory, corpus=corpus)
    dense = load_dense_index(dense_result.directory, corpus=corpus)
    assert bm25.chunk_ids == dense.chunk_ids
    assert bm25.search(RetrievalQuery(text="Sentence 0"), top_k=1)
    query_vector = FakeEmbeddingProvider().embed_query(RetrievalQuery(text="q"))
    assert dense.search_vector(query_vector, top_k=1)

    with pytest.raises(FrozenArtifactError, match="overwrite"):
        build_bm25_index(corpus=corpus, index_root=index_root, config=bm25_config)
    with pytest.raises(FrozenArtifactError, match="overwrite"):
        build_dense_index(
            corpus=corpus,
            index_root=index_root,
            config=dense_config,
            embedding_provider=FakeEmbeddingProvider(),
        )


def test_index_manifest_detects_tampering_and_provider_mismatch(tmp_path: Path) -> None:
    corpus = prepare_corpus(tmp_path)
    index_root = tmp_path / "indexes"
    result = build_bm25_index(corpus=corpus, index_root=index_root, config=BM25Config())
    manifest_path = result.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corpus_file_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IndexIntegrityError, match="corpus file hash"):
        load_bm25_index(result.directory, corpus=corpus)

    bad_provider = FakeEmbeddingProvider()
    bad_provider.model = "different-model"
    with pytest.raises(IndexIntegrityError, match="model"):
        build_dense_index(
            corpus=corpus,
            index_root=index_root,
            config=DenseConfig(),
            embedding_provider=bad_provider,
        )


def test_load_index_manifest_rejects_missing_or_invalid_manifest(tmp_path: Path) -> None:
    with pytest.raises(IndexIntegrityError, match="unable to load"):
        load_index_manifest(tmp_path)

    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(IndexIntegrityError, match="unable to load"):
        load_index_manifest(tmp_path)
