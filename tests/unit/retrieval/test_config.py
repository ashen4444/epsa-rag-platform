from __future__ import annotations

import pytest
from pydantic import ValidationError

from epsa_rag.retrieval.config import BM25Config, DenseConfig, HybridRetrieverConfig, RRFConfig


def test_locked_retrieval_defaults_are_explicit_and_fingerprintable() -> None:
    config = HybridRetrieverConfig()

    assert config.bm25.backend == "epsa-okapi-bm25"
    assert config.dense.embedding_model == "text-embedding-3-small"
    assert config.dense.dimensions == 1536
    assert config.dense.backend == "faiss.IndexFlatIP"
    assert config.fusion.rank_constant == 60
    assert config.fingerprint() == HybridRetrieverConfig().fingerprint()


@pytest.mark.parametrize(
    ("bm25_k", "dense_k", "result_k"),
    [(5, 100, 10), (100, 5, 10)],
)
def test_hybrid_config_rejects_insufficient_candidate_depth(
    bm25_k: int, dense_k: int, result_k: int
) -> None:
    with pytest.raises(ValidationError, match="candidate_k"):
        HybridRetrieverConfig(
            bm25=BM25Config(candidate_k=bm25_k),
            dense=DenseConfig(candidate_k=dense_k),
            fusion=RRFConfig(result_k=result_k),
        )


def test_retrieval_config_rejects_unlocked_or_invalid_settings() -> None:
    with pytest.raises(ValidationError):
        DenseConfig(dimensions=3)
    with pytest.raises(ValidationError):
        BM25Config(b=1.1)
    with pytest.raises(ValidationError):
        RRFConfig(bm25_weight=0)
