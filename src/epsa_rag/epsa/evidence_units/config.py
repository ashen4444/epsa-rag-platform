"""Immutable, versioned Component 03 configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from epsa_rag.core.config import ConfigModel


class EvidenceUnitExtractorConfig(ConfigModel):
    mode: Literal["rule_based_v1", "rule_based_v2"] = "rule_based_v1"
    segmenter_version: Literal["regex-v1", "regex-v2"] = "regex-v1"
    resolver_version: str = Field(default="title-v1", min_length=1)


class EvidenceUnitExtractorV2Config(EvidenceUnitExtractorConfig):
    mode: Literal["rule_based_v2"] = "rule_based_v2"
    segmenter_version: Literal["regex-v2"] = "regex-v2"
    resolver_version: str = Field(default="conservative-title-v2", min_length=1)
