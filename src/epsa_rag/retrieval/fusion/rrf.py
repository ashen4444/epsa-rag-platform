"""Deterministic weighted Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from epsa_rag.core.exceptions import ContractError
from epsa_rag.retrieval.config import RRFConfig
from epsa_rag.retrieval.models import BackendHit, FusedHit


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[BackendHit]],
    *,
    weights: Mapping[str, float],
    config: RRFConfig,
    top_k: int | None = None,
) -> tuple[FusedHit, ...]:
    """Fuse unique branch rankings while preserving ranks and raw scores."""

    if set(rankings) != set(weights):
        raise ContractError("RRF rankings and weights must have identical source names")
    limit = config.result_k if top_k is None else top_k
    if limit < 1:
        raise ValueError("top_k must be positive")

    fused_scores: dict[str, float] = {}
    source_scores: dict[str, dict[str, float]] = {}
    source_ranks: dict[str, dict[str, int]] = {}
    for source, hits in rankings.items():
        seen: set[str] = set()
        previous_rank = 0
        for hit in hits:
            if hit.chunk_id in seen:
                raise ContractError(f"RRF source {source!r} contains duplicate chunk IDs")
            if hit.rank <= previous_rank:
                raise ContractError(f"RRF source {source!r} ranks must be strictly increasing")
            seen.add(hit.chunk_id)
            previous_rank = hit.rank
            fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + (
                weights[source] / (config.rank_constant + hit.rank)
            )
            source_scores.setdefault(hit.chunk_id, {})[source] = hit.score
            source_ranks.setdefault(hit.chunk_id, {})[source] = hit.rank

    ordered_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (
            -fused_scores[chunk_id],
            min(source_ranks[chunk_id].values()),
            chunk_id,
        ),
    )[:limit]
    return tuple(
        FusedHit(
            chunk_id=chunk_id,
            score=fused_scores[chunk_id],
            source_scores=source_scores[chunk_id],
            source_ranks=source_ranks[chunk_id],
        )
        for chunk_id in ordered_ids
    )
