"""Immutable configuration for the frozen Component 04 research scorer."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from epsa_rag.core.config import ConfigModel


class EvidenceScorerV1Config(ConfigModel):
    mode: Literal["research_v1"] = "research_v1"
    entity_match_weight: float = Field(default=0.25, ge=0, le=1)
    relation_match_weight: float = Field(default=0.20, ge=0, le=1)
    answer_type_match_weight: float = Field(default=0.15, ge=0, le=1)
    token_overlap_weight: float = Field(default=0.15, ge=0, le=1)
    title_match_weight: float = Field(default=0.10, ge=0, le=1)
    retrieval_component_weight: float = Field(default=0.05, ge=0, le=1)
    bridge_entity_weight: float = Field(default=0.10, ge=0, le=1)
    short_content_penalty: float = Field(default=0.20, ge=0, le=1)
    long_content_penalty: float = Field(default=0.10, ge=0, le=1)
    no_structure_penalty: float = Field(default=0.15, ge=0, le=1)
    maximum_noise_penalty: float = Field(default=0.35, ge=0, le=1)

    @model_validator(mode="after")
    def require_frozen_weights(self) -> EvidenceScorerV1Config:
        if (
            sum(
                (
                    self.entity_match_weight,
                    self.relation_match_weight,
                    self.answer_type_match_weight,
                    self.token_overlap_weight,
                    self.title_match_weight,
                    self.retrieval_component_weight,
                    self.bridge_entity_weight,
                )
            )
            != 1.0
        ):
            raise ValueError("research_v1 positive feature weights must sum to 1.0")
        return self
