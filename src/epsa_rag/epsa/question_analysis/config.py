"""Explicit, immutable configuration for the frozen rule-based analyzer."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from epsa_rag.core.config import ConfigModel


class QuestionAnalyzerConfig(ConfigModel):
    """Research-critical constants for the reproducible Component 01 rule set.

    Scores are deterministic rule scores, never calibrated probabilities.
    """

    version: Literal["rule_based_v1"] = "rule_based_v1"
    quoted_entity_score: float = Field(default=0.98, ge=0, le=1)
    multi_token_entity_score: float = Field(default=0.90, ge=0, le=1)
    single_token_entity_score: float = Field(default=0.70, ge=0, le=1)
    relation_hint_score: float = Field(default=0.80, ge=0, le=1)
    known_answer_type_score: float = Field(default=0.75, ge=0, le=1)
    unknown_answer_type_score: float = Field(default=0.35, ge=0, le=1)
    which_of_comparison_score: float = Field(default=0.80, ge=0, le=1)
    between_comparison_score: float = Field(default=0.65, ge=0, le=1)
