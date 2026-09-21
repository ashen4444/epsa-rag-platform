"""Exact deterministic ``research_v1`` feature extraction."""

from __future__ import annotations

import re

from epsa_rag.epsa.evidence_scoring.config import EvidenceScorerV1Config
from epsa_rag.epsa.evidence_scoring.models import EvidenceFeatureVector
from epsa_rag.epsa.evidence_units.models import EvidenceUnit
from epsa_rag.epsa.question_analysis.models import AnswerType, QuestionAnalysis

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "there",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "with",
    }
)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _normalized_set(values: tuple[str, ...]) -> set[str]:
    return {value.casefold().strip() for value in values if value.casefold().strip()}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_PATTERN.findall(text.casefold())
        if token not in _STOPWORDS and len(token) > 1
    }


def _clip(value: float, upper: float = 1.0) -> float:
    return max(0.0, min(upper, value))


class ResearchV1EvidenceFeatureExtractor:
    version = "research_v1"

    def __init__(self, config: EvidenceScorerV1Config) -> None:
        self._config = config

    def extract(
        self, evidence_unit: EvidenceUnit, question_analysis: QuestionAnalysis
    ) -> EvidenceFeatureVector:
        seeds = tuple(item.text for item in question_analysis.seed_entities)
        seed_set = _normalized_set(seeds)
        unit_entities = _normalized_set(evidence_unit.entities)
        required = _normalized_set(
            tuple(item.relation for item in question_analysis.required_relation_hints)
        )
        present = _normalized_set(evidence_unit.relation_hints)
        tokens = _tokens(evidence_unit.resolved_text)
        entity = (
            _clip(
                sum(
                    seed in unit_entities or seed in evidence_unit.resolved_text.casefold()
                    for seed in seed_set
                )
                / len(seeds)
            )
            if seeds
            else 0.0
        )
        relation = _clip(len(required & present) / len(required)) if required else 0.0
        expected = question_analysis.expected_answer_type
        types = set(evidence_unit.answer_type_candidates)
        answer = (
            0.0
            if expected is AnswerType.UNKNOWN
            else (
                1.0
                if expected in types
                else (0.6 if expected is AnswerType.ENTITY and types else 0.0)
            )
        )
        qtokens = _tokens(question_analysis.normalized_question)
        overlap = _clip(len(qtokens & tokens) / len(qtokens)) if qtokens else 0.0
        title = evidence_unit.doc_title.casefold().strip()
        title_match = (
            1.0
            if title and title in seed_set
            else (
                0.8 if title and title in question_analysis.normalized_question.casefold() else 0.0
            )
        )
        retrieval = (
            _clip(1.0 / evidence_unit.retrieval_rank)
            if evidence_unit.retrieval_rank is not None
            else _clip(evidence_unit.retrieval_score or 0.0)
        )
        bridges = unit_entities - seed_set - ({title} if title else set())
        bridge = 1.0 if bridges and present else (0.5 if bridges else 0.0)
        noise = (
            (self._config.short_content_penalty if len(tokens) < 4 else 0.0)
            + (self._config.long_content_penalty if len(tokens) > 60 else 0.0)
            + (self._config.no_structure_penalty if not unit_entities and not present else 0.0)
        )
        return EvidenceFeatureVector(
            entity_match_score=round(entity, 6),
            relation_match_score=round(relation, 6),
            answer_type_match_score=round(answer, 6),
            token_overlap_score=round(overlap, 6),
            title_match_score=round(title_match, 6),
            retrieval_score_component=round(retrieval, 6),
            bridge_entity_score=round(bridge, 6),
            noise_penalty=round(_clip(noise, self._config.maximum_noise_penalty), 6),
        )
