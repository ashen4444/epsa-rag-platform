"""Versioned immutable configuration for Component 06 research-v1."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from epsa_rag.core.config import ConfigModel


class EvidencePathSearcherV1Config(ConfigModel):
    """Frozen constants for the documented thesis-compatible template search."""

    mode: Literal["research_v1"] = "research_v1"
    schema_version: Literal["evidence-path-v1"] = "evidence-path-v1"
    expected_answer_type_weight: float = Field(default=0.25, ge=0, le=1)
    relation_match_weight: float = Field(default=0.20, ge=0, le=1)
    path_length_weight: float = Field(default=0.10, ge=0, le=1)
    specific_bridge_bonus: float = Field(default=0.20, ge=0, le=1)
    generic_bridge_bonus: float = Field(default=0.05, ge=0, le=1)
    answer_specificity_bonus: float = Field(default=0.15, ge=0, le=1)
    retrieval_quality_weight: float = Field(default=0.05, ge=0, le=1)
    yes_no_connection_bonus: float = Field(default=0.15, ge=0, le=1)
    possible_answer_candidate_bonus: float = Field(default=0.40, ge=0, le=1)
    sentence_answer_type_bonus: float = Field(default=0.25, ge=0, le=1)
    specific_candidate_bonus: float = Field(default=0.10, ge=0, le=1)
    generic_answer_type_penalty: float = Field(default=0.20, ge=0, le=1)

    @model_validator(mode="after")
    def require_frozen_constants(self) -> EvidencePathSearcherV1Config:
        expected = {
            "expected_answer_type_weight": 0.25,
            "relation_match_weight": 0.20,
            "path_length_weight": 0.10,
            "specific_bridge_bonus": 0.20,
            "generic_bridge_bonus": 0.05,
            "answer_specificity_bonus": 0.15,
            "retrieval_quality_weight": 0.05,
            "yes_no_connection_bonus": 0.15,
            "possible_answer_candidate_bonus": 0.40,
            "sentence_answer_type_bonus": 0.25,
            "specific_candidate_bonus": 0.10,
            "generic_answer_type_penalty": 0.20,
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("research_v1 path-search constants must remain frozen")
        return self
