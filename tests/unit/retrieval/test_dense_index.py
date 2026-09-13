from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pytest

from epsa_rag.core.exceptions import ConfigurationError, FrozenArtifactError, IndexIntegrityError
from epsa_rag.core.models import RetrievalQuery
from epsa_rag.retrieval.config import DenseConfig
from epsa_rag.retrieval.dense.index import FaissFlatIPIndex, normalize_rows
from epsa_rag.retrieval.dense.retriever import DenseRetriever


def vectors() -> np.ndarray:
    matrix = np.zeros((3, 1536), dtype=np.float32)
    matrix[0, 0] = 1
    matrix[1, 1] = 1
    matrix[2, 0] = 1
    matrix[2, 1] = 1
    return matrix


class FakeProvider:
    model = "text-embedding-3-small"
    dimensions = 1536

    def embed_query(self, query: RetrievalQuery) -> np.ndarray:
        return vectors()[0]

    def embed_documents(self, chunks: object) -> np.ndarray:
        return vectors()


def test_normalization_returns_float32_unit_vectors_without_mutating_input() -> None:
    original = vectors()
    normalized = normalize_rows(original, dimensions=1536)

    assert normalized.dtype == np.float32
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1)
    assert original[2, 0] == 1


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.ones((2, 3), dtype=np.float32), "shape"),
        (np.zeros((1, 1536), dtype=np.float32), "zero"),
        (np.full((1, 1536), np.nan, dtype=np.float32), "non-finite"),
    ],
)
def test_normalization_rejects_invalid_matrices(matrix: np.ndarray, message: str) -> None:
    with pytest.raises(IndexIntegrityError, match=message):
        normalize_rows(matrix, dimensions=1536)


def test_faiss_flat_ip_search_is_exact_and_deterministic() -> None:
    index = FaissFlatIPIndex.build(
        vectors=vectors(),
        chunk_ids=("chunk:a", "chunk:b", "chunk:c"),
        config=DenseConfig(),
    )

    hits = index.search_vector(vectors()[0], top_k=3)

    assert [hit.chunk_id for hit in hits] == ["chunk:a", "chunk:c", "chunk:b"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(2**-0.5)
    with pytest.raises(ValueError, match="positive"):
        index.search_vector(vectors()[0], top_k=0)
    with pytest.raises(IndexIntegrityError, match="shape"):
        index.search_vector(np.ones(2, dtype=np.float32), top_k=1)


def test_faiss_index_persistence_round_trip(tmp_path: Path) -> None:
    config = DenseConfig()
    original = FaissFlatIPIndex.build(
        vectors=vectors(), chunk_ids=("chunk:a", "chunk:b", "chunk:c"), config=config
    )
    index_path = tmp_path / "index.faiss"
    ids_path = tmp_path / "chunk_ids.json"
    original.save(index_path, ids_path)
    loaded = FaissFlatIPIndex.load(index_path, ids_path, config=config)

    assert loaded.chunk_ids == original.chunk_ids
    assert loaded.search_vector(vectors()[1], top_k=2)[0].chunk_id == "chunk:b"
    with pytest.raises(FrozenArtifactError, match="overwrite"):
        original.save(index_path, tmp_path / "other.json")


def test_faiss_index_rejects_misaligned_or_wrong_metric_indexes() -> None:
    config = DenseConfig()
    with pytest.raises(IndexIntegrityError, match="empty"):
        FaissFlatIPIndex.build(
            vectors=np.empty((0, 1536), dtype=np.float32), chunk_ids=(), config=config
        )
    with pytest.raises(IndexIntegrityError, match="align"):
        FaissFlatIPIndex.build(
            vectors=vectors(), chunk_ids=("chunk:a",), config=config
        )
    l2_index = faiss.IndexFlatL2(1536)
    l2_index.add(vectors()[:1])
    with pytest.raises(IndexIntegrityError, match="inner-product"):
        FaissFlatIPIndex(index=l2_index, chunk_ids=("chunk:a",), config=config)


def test_faiss_load_rejects_missing_and_invalid_id_mappings(tmp_path: Path) -> None:
    config = DenseConfig()
    with pytest.raises(IndexIntegrityError, match="unable to read FAISS"):
        FaissFlatIPIndex.load(tmp_path / "missing.faiss", tmp_path / "ids.json", config=config)

    original = FaissFlatIPIndex.build(
        vectors=vectors(), chunk_ids=("chunk:a", "chunk:b", "chunk:c"), config=config
    )
    index_path = tmp_path / "index.faiss"
    ids_path = tmp_path / "chunk_ids.json"
    original.save(index_path, ids_path)
    ids_path.write_text('{"schema_version":"9.0","chunk_ids":[]}', encoding="utf-8")
    with pytest.raises(IndexIntegrityError, match="invalid dense"):
        FaissFlatIPIndex.load(index_path, ids_path, config=config)


def test_dense_retriever_validates_provider_and_searches() -> None:
    index = FaissFlatIPIndex.build(
        vectors=vectors(),
        chunk_ids=("chunk:a", "chunk:b", "chunk:c"),
        config=DenseConfig(),
    )
    retriever = DenseRetriever(index, FakeProvider())

    assert retriever.chunk_ids == index.chunk_ids
    assert retriever.search(RetrievalQuery(text="alpha"), top_k=1)[0].chunk_id == "chunk:a"

    bad_provider = FakeProvider()
    bad_provider.model = "wrong-model"
    with pytest.raises(ConfigurationError, match="model"):
        DenseRetriever(index, bad_provider)
    bad_provider.model = "text-embedding-3-small"
    bad_provider.dimensions = 5
    with pytest.raises(ConfigurationError, match="dimensions"):
        DenseRetriever(index, bad_provider)
