from __future__ import annotations

import pytest

from epsa_rag.core.exceptions import ContractError
from epsa_rag.retrieval.config import RRFConfig
from epsa_rag.retrieval.fusion.rrf import reciprocal_rank_fusion
from epsa_rag.retrieval.models import BackendHit


def hit(chunk_id: str, rank: int, score: float) -> BackendHit:
    return BackendHit(chunk_id=chunk_id, rank=rank, score=score)


def test_rrf_combines_ranks_and_preserves_branch_provenance() -> None:
    fused = reciprocal_rank_fusion(
        {
            "bm25": (hit("chunk:a", 1, 8.0), hit("chunk:b", 2, 4.0)),
            "dense": (hit("chunk:b", 1, 0.9), hit("chunk:c", 2, 0.8)),
        },
        weights={"bm25": 1.0, "dense": 1.0},
        config=RRFConfig(rank_constant=60, result_k=3),
    )

    assert [item.chunk_id for item in fused] == ["chunk:b", "chunk:a", "chunk:c"]
    assert fused[0].source_scores == {"bm25": 4.0, "dense": 0.9}
    assert fused[0].source_ranks == {"bm25": 2, "dense": 1}
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_applies_weights_limits_and_stable_ties() -> None:
    fused = reciprocal_rank_fusion(
        {
            "bm25": (hit("chunk:b", 1, 1.0),),
            "dense": (hit("chunk:a", 1, 1.0),),
        },
        weights={"bm25": 1.0, "dense": 1.0},
        config=RRFConfig(result_k=2),
        top_k=1,
    )
    assert [item.chunk_id for item in fused] == ["chunk:a"]


def test_rrf_rejects_invalid_sources_rankings_and_limits() -> None:
    config = RRFConfig()
    with pytest.raises(ContractError, match="identical"):
        reciprocal_rank_fusion(
            {"bm25": ()}, weights={"dense": 1.0}, config=config
        )
    duplicate = (hit("chunk:a", 1, 1), hit("chunk:a", 2, 0.5))
    with pytest.raises(ContractError, match="duplicate"):
        reciprocal_rank_fusion(
            {"bm25": duplicate}, weights={"bm25": 1.0}, config=config
        )
    unordered = (hit("chunk:a", 2, 1), hit("chunk:b", 1, 0.5))
    with pytest.raises(ContractError, match="increasing"):
        reciprocal_rank_fusion(
            {"bm25": unordered}, weights={"bm25": 1.0}, config=config
        )
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion(
            {"bm25": ()}, weights={"bm25": 1.0}, config=config, top_k=0
        )
