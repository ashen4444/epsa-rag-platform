"""Versioned immutable configuration for Component 05 research-v1."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from epsa_rag.core.config import ConfigModel


class EvidenceGraphBuilderV1Config(ConfigModel):
    """Frozen configuration for the documented thesis-compatible graph shape."""

    mode: Literal["research_v1"] = "research_v1"
    schema_version: Literal["evidence-graph-v1"] = "evidence-graph-v1"
    minimum_anchor_weight: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def require_frozen_anchor_weight(self) -> EvidenceGraphBuilderV1Config:
        if self.minimum_anchor_weight != 0.1:
            raise ValueError("research_v1 minimum anchor weight must remain 0.1")
        return self
