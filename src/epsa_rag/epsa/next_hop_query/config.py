"""Frozen configurations for Component 09 query proposals."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from epsa_rag.core.config import ConfigModel


class ReconstructedNextHopQueryGeneratorConfig(ConfigModel):
    """Explicit constants for the source-unavailable compatibility policy.

    The historical Component 09 query-builder source was not recovered. These
    constants therefore identify a reconstructed policy, not ``research_v1``.
    """

    mode: Literal["research_v1_reconstructed"] = "research_v1_reconstructed"
    schema_version: Literal["next-hop-query-v1-reconstructed"] = "next-hop-query-v1-reconstructed"
    query_separator: Literal[" "] = " "
    bridge_query_confidence: float = Field(default=0.65, ge=0, le=1)
    comparison_query_confidence: float = Field(default=0.50, ge=0, le=1)
    seed_query_confidence: float = Field(default=0.40, ge=0, le=1)

    @model_validator(mode="after")
    def require_reconstructed_constants(self) -> ReconstructedNextHopQueryGeneratorConfig:
        if (
            self.bridge_query_confidence != 0.65
            or self.comparison_query_confidence != 0.50
            or self.seed_query_confidence != 0.40
        ):
            raise ValueError("reconstructed confidence constants are fixed")
        return self


class HistoricalAdaptedNextHopQueryGeneratorConfig(ConfigModel):
    """Recovered historical rules adapted only at the current contract boundary."""

    mode: Literal["research_v1_historical_adapted"] = "research_v1_historical_adapted"
    schema_version: Literal["next-hop-query-v1-historical-adapted"] = (
        "next-hop-query-v1-historical-adapted"
    )
    relation_query_terms: dict[str, str] = Field(
        default_factory=lambda: {
            "born": "born birthplace",
            "birthplace": "born birthplace",
            "directed": "directed director",
            "written": "written author",
            "author": "author written",
            "located": "located location",
            "capital": "capital location",
            "population": "population number",
            "length": "length",
            "released": "released date",
            "published": "published date",
            "founded": "founded founder",
            "starring": "starring cast",
            "member": "member",
            "genre": "genre",
            "occupation": "occupation profession",
            "spouse": "spouse married",
            "parent": "parent",
            "child": "child",
            "educated": "educated alma mater",
            "discovered": "discovered discovery",
            "capacity": "capacity seats",
        }
    )
    answer_type_keywords: dict[str, str] = Field(
        default_factory=lambda: {
            "LOCATION": "location",
            "DATE": "date",
            "NUMBER": "number",
            "PERSON": "person",
            "ORGANIZATION": "organization",
            "TITLE_OR_WORK": "title",
            "ENTITY": "entity",
            "BOOLEAN": "evidence",
            "UNKNOWN": "evidence",
        }
    )
