"""Strict inference-safe Component 04 input-derived outputs."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from epsa_rag.core.models import ContractModel
from epsa_rag.epsa.evidence_units.models import EvidenceUnit

Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class EvidenceFeatureVector(ContractModel):
    entity_match_score: Score
    relation_match_score: Score
    answer_type_match_score: Score
    token_overlap_score: Score
    title_match_score: Score
    retrieval_score_component: Score
    bridge_entity_score: Score
    noise_penalty: Score


class EvidenceScoreMetadata(ContractModel):
    scorer: Literal["RuleBasedEvidenceScorerV1"] = "RuleBasedEvidenceScorerV1"
    version: Literal["research_v1"] = "research_v1"
    configuration_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    feature_provider_version: Literal["research_v1"] = "research_v1"
    raw_weighted_score: float


class ScoredEvidenceUnit(ContractModel):
    """A scored view of an unchanged Component 03 evidence unit."""

    evidence_unit: EvidenceUnit
    final_score: Score
    score_breakdown: EvidenceFeatureVector
    metadata: EvidenceScoreMetadata

    @model_validator(mode="after")
    def require_rounded_scores(self) -> ScoredEvidenceUnit:
        if self.final_score != round(self.final_score, 6):
            raise ValueError("final score must be rounded to six decimals")
        if any(value != round(value, 6) for value in self.score_breakdown.model_dump().values()):
            raise ValueError("score breakdown values must be rounded to six decimals")
        return self
