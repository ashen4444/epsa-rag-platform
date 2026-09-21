"""Frozen configuration for Component 07 research-v1 decisions."""

from __future__ import annotations

from typing import Literal

from epsa_rag.core.config import ConfigModel


class SufficiencyDecisionV1Config(ConfigModel):
    """Explicit version identity and historical confidence constants."""

    mode: Literal["research_v1"] = "research_v1"
    schema_version: Literal["sufficiency-decision-v1"] = "sufficiency-decision-v1"
    bridge_confidence_base: float = 0.70
    factoid_confidence_base: float = 0.68
    yes_no_confidence_base: float = 0.62
    insufficient_confidence_base: float = 0.25
    path_score_contribution_cap: float = 1.0
    path_score_contribution_weight: float = 0.25

    def model_post_init(self, __context: object) -> None:
        expected = (0.70, 0.68, 0.62, 0.25, 1.0, 0.25)
        actual = (
            self.bridge_confidence_base,
            self.factoid_confidence_base,
            self.yes_no_confidence_base,
            self.insufficient_confidence_base,
            self.path_score_contribution_cap,
            self.path_score_contribution_weight,
        )
        if actual != expected:
            raise ValueError("research_v1 sufficiency constants must remain frozen")
