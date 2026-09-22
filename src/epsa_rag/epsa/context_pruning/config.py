"""Frozen configuration for Component 08 research-v1 context pruning."""

from __future__ import annotations

from typing import Literal

from epsa_rag.core.config import ConfigModel


class ResearchContextPrunerV1Config(ConfigModel):
    """Historical constants retained for deterministic research-v1 compatibility."""

    mode: Literal["research_v1"] = "research_v1"
    schema_version: Literal["pruned-context-v1"] = "pruned-context-v1"
    neighbor_distance: Literal[1] = 1
    max_expanded_units: Literal[6] = 6
