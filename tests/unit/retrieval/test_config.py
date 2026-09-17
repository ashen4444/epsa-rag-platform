from __future__ import annotations

import pytest
from pydantic import ValidationError

from epsa_rag.retrieval.config import (
    HYBRID_RETRIEVER_V1,
    HYBRID_RETRIEVER_V2,
    HYBRID_V1_BM25_WEIGHT,
    HYBRID_V1_DENSE_WEIGHT,
    HYBRID_V1_RRF_RANK_CONSTANT,
    BM25Config,
    DenseConfig,
    HybridRetrieverConfig,
    RRFConfig,
)


def test_locked_retrieval_defaults_are_explicit_and_fingerprintable() -> None:
    config = HybridRetrieverConfig()

    assert config.bm25.backend == "epsa-okapi-bm25"
    assert config.dense.embedding_model == "text-embedding-3-small"
    assert config.dense.dimensions == 1536
    assert config.dense.backend == "faiss.IndexFlatIP"
    assert config.retriever_version == HYBRID_RETRIEVER_V2
    assert config.fusion.rank_constant == 60
    assert config.fusion.bm25_weight == 0.3
    assert config.fusion.dense_weight == 0.7
    assert config.fingerprint() == HybridRetrieverConfig().fingerprint()


def test_v1_fusion_configuration_remains_reconstructible() -> None:
    v1 = HybridRetrieverConfig(
        retriever_version=HYBRID_RETRIEVER_V1,
        fusion=RRFConfig(
            bm25_weight=HYBRID_V1_BM25_WEIGHT,
            dense_weight=HYBRID_V1_DENSE_WEIGHT,
        ),
    )

    assert v1.retriever_version == "hybrid-retriever-v1"
    assert v1.fusion.rank_constant == HYBRID_V1_RRF_RANK_CONSTANT
    assert v1.fusion.bm25_weight == 1.0
    assert v1.fusion.dense_weight == 1.0
    assert v1.fingerprint() != HybridRetrieverConfig().fingerprint()


@pytest.mark.parametrize(
    ("version", "rank_constant", "weights"),
    [
        (HYBRID_RETRIEVER_V1, 60, (0.3, 0.7)),
        (HYBRID_RETRIEVER_V2, 60, (1.0, 1.0)),
        (HYBRID_RETRIEVER_V2, 30, (0.3, 0.7)),
    ],
)
def test_named_frozen_retrievers_reject_mismatched_fusion_configuration(
    version: str, rank_constant: int, weights: tuple[float, float]
) -> None:
    with pytest.raises(ValidationError, match="requires RRF constant"):
        HybridRetrieverConfig(
            retriever_version=version,
            fusion=RRFConfig(
                rank_constant=rank_constant,
                bm25_weight=weights[0],
                dense_weight=weights[1],
            ),
        )


def test_explicit_development_version_can_use_experimental_weights() -> None:
    config = HybridRetrieverConfig(
        retriever_version="hybrid-retriever-development-80dense",
        fusion=RRFConfig(rank_constant=30, bm25_weight=0.2, dense_weight=0.8),
    )

    assert config.fusion.rank_constant == 30
    assert config.fusion.bm25_weight == 0.2
    assert config.fusion.dense_weight == 0.8


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
