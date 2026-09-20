"""Versioned constants for deterministic Component 02 rules."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from epsa_rag.core.config import ConfigModel


class _ChunkRuleConfig(ConfigModel):
    """Shared score contract; versioned modes retain separate configuration types."""

    mode: str
    title_entity_score: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    quoted_entity_score: float = Field(default=0.98, ge=0, le=1, allow_inf_nan=False)
    multi_token_entity_score: float = Field(default=0.9, ge=0, le=1, allow_inf_nan=False)
    single_token_entity_score: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    relation_score: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    full_date_score: float = Field(default=0.95, ge=0, le=1, allow_inf_nan=False)
    year_score: float = Field(default=0.9, ge=0, le=1, allow_inf_nan=False)
    number_score: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    location_score: float = Field(default=0.65, ge=0, le=1, allow_inf_nan=False)
    generic_entity_score: float = Field(default=0.55, ge=0, le=1, allow_inf_nan=False)
    minimum_bridge_length: int = Field(default=3, ge=1)


class ChunkAnalyzerConfig(_ChunkRuleConfig):
    """Historical rule scores; none are calibrated probabilities."""

    mode: Literal["rule_based_v1"] = "rule_based_v1"


class RuleBasedV2ChunkAnalyzerConfig(_ChunkRuleConfig):
    """Versioned deterministic rules for grounded, cross-chunk candidates."""

    mode: Literal["rule_based_v2"] = "rule_based_v2"
    maximum_entity_length: int = Field(default=120, ge=20)
    maximum_entity_tokens: int = Field(default=8, ge=1)
    relation_window_chars: int = Field(default=120, ge=1)
    association_score: float = Field(default=0.55, ge=0, le=1, allow_inf_nan=False)
    person_score: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    organization_score: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    title_or_work_score: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    allow_body_mention_links: bool = True
