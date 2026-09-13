"""Serializable backend and hybrid retrieval results."""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import Field, field_validator

from epsa_rag.core.ids import Identifier
from epsa_rag.core.models import ContractModel, RankedParagraphChunk, RetrievalQuery


class BackendHit(ContractModel):
    """Minimal backend-neutral hit used at the fusion boundary."""

    chunk_id: Identifier
    rank: Annotated[int, Field(ge=1)]
    score: float

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        """Reject backend scores that cannot be serialized or compared."""

        if not math.isfinite(value):
            raise ValueError("backend score must be finite")
        return value


class FusedHit(ContractModel):
    """Fused identity, score, and complete branch provenance."""

    chunk_id: Identifier
    score: float
    source_scores: dict[Identifier, float]
    source_ranks: dict[Identifier, Annotated[int, Field(ge=1)]]

    @field_validator("score")
    @classmethod
    def require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fusion score must be finite")
        return value


class RetrievalResult(ContractModel):
    """Canonical inspectable output from one hybrid retrieval call."""

    query: RetrievalQuery
    retriever_version: Identifier
    results: tuple[RankedParagraphChunk, ...]
