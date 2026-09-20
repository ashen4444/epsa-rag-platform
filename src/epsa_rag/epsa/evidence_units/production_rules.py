"""Inference-safe v2 features using Component 02's established semantics."""

from __future__ import annotations

import re

from epsa_rag.core.models import Sentence
from epsa_rag.epsa.chunk_analysis.config import RuleBasedV2ChunkAnalyzerConfig
from epsa_rag.epsa.chunk_analysis.models import ChunkEntityMentionV2
from epsa_rag.epsa.chunk_analysis.rules import token_overlap
from epsa_rag.epsa.chunk_analysis.v2_rules import (
    contextual_answers,
    contextual_entities,
    contextual_relations,
)
from epsa_rag.epsa.evidence_units.models import SentenceEntity, StructuredAnswerCandidate
from epsa_rag.epsa.question_analysis.models import QuestionAnalysis
from epsa_rag.epsa.question_analysis.normalizer import normalize_entity

_CHUNK_RULES = RuleBasedV2ChunkAnalyzerConfig()
_NAME_PREFIXES = frozenset(
    {
        "a", "an", "the", "dr", "general", "mr", "mrs", "ms", "pope",
        "president", "professor", "senator", "sir",
    }
)


def _literal_seed_mention(sentence: str, seed: str) -> bool:
    """Recover a whole named mention missed by long-span entity extraction.

    Only visibly capitalized names with at least two normalized tokens qualify.
    Short punctuation variants are tolerated, but a bare ``York`` inside
    ``New York`` or ``York City`` is not the same entity. This intentionally
    does not infer aliases or coreference.
    """

    tokens = normalize_entity(seed).split()
    if len(tokens) < 2:
        return False
    phrase = r"[\W_]{1,3}".join(re.escape(token) for token in tokens)
    pattern = re.compile(rf"(?<!\w)(?:{re.escape(seed)}|{phrase})(?!\w)", re.IGNORECASE)
    for match in pattern.finditer(sentence):
        if sum(word[0].isupper() for word in re.findall(r"\w+", match.group())) < 2:
            continue
        before = sentence[: match.start()]
        after = sentence[match.end() :]
        left = re.search(r"([A-Z][\w'-]*)\s+$", before)
        right = re.match(r"\s+([A-Z][\w'-]*)\b", after)
        if left and left.group(1).casefold() not in _NAME_PREFIXES:
            continue
        if right:
            continue
        return True
    return False


def features(
    sentence_text: str,
    title: str,
    analysis: QuestionAnalysis,
) -> tuple[
    tuple[str, ...],
    tuple[SentenceEntity, ...],
    tuple[str, ...],
    tuple[StructuredAnswerCandidate, ...],
    tuple[str, ...],
    float,
]:
    """Extract only source-grounded spans; title anchors have a separate scope."""

    body = contextual_entities(sentence_text, _CHUNK_RULES)
    title_entity = (
        ChunkEntityMentionV2(
            text=title,
            normalized=normalize_entity(title),
            source="doc_title",
            span_scope="doc_title",
            confidence=_CHUNK_RULES.title_entity_score,
        )
        if title.strip()
        else None
    )
    all_entities = (*((title_entity,) if title_entity else ()), *body)
    entity_features = tuple(
        SentenceEntity(
            text=item.text,
            normalized=item.normalized,
            source="doc_title" if item.span_scope == "doc_title" else "sentence_text",
            start_char=item.start_char,
            end_char=item.end_char,
        )
        for item in all_entities
    )
    relation_records = contextual_relations(
        title,
        sentence_text,
        (Sentence(index=0, text=sentence_text),),
        body,
        _CHUNK_RULES,
    )
    relation_hints = tuple(dict.fromkeys(item.relation for item in relation_records))
    candidates = tuple(
        StructuredAnswerCandidate(
            text=item.text,
            normalized=normalize_entity(item.text),
            answer_type=item.answer_type,
            source=item.source,
            span_scope="doc_title" if item.span_scope == "doc_title" else "sentence_text",
            start_char=item.start_char,
            end_char=item.end_char,
            rule_score=item.confidence,
        )
        for item in contextual_answers(title, sentence_text, all_entities, _CHUNK_RULES)
        if (
            item.start_char is not None
            and item.end_char is not None
            and sentence_text[item.start_char : item.end_char] == item.text
        )
    )
    normalized_body_entities = {
        item.normalized for item in entity_features if item.source == "sentence_text"
    }
    overlap = tuple(
        dict.fromkeys(
            mention.text
            for mention in analysis.seed_entities
            if mention.normalized in normalized_body_entities
            or _literal_seed_mention(sentence_text, mention.text)
        )
    )
    _, token_score = token_overlap(analysis.normalized_question, sentence_text)
    return (
        tuple(dict.fromkeys(item.text for item in entity_features)),
        entity_features,
        relation_hints,
        candidates,
        overlap,
        token_score,
    )
