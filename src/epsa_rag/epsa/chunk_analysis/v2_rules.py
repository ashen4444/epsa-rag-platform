"""Deterministic contextual features for the versioned Component 02 successor."""

from __future__ import annotations

import re

from epsa_rag.core.models import Sentence
from epsa_rag.epsa.chunk_analysis.config import RuleBasedV2ChunkAnalyzerConfig
from epsa_rag.epsa.chunk_analysis.models import (
    ChunkAnswerCandidateV2,
    ChunkEntityMention,
    ChunkEntityMentionV2,
    GroundedChunkRelationHint,
)
from epsa_rag.epsa.chunk_analysis.rules import extract_answers
from epsa_rag.epsa.question_analysis.config import QuestionAnalyzerConfig
from epsa_rag.epsa.question_analysis.entity_extractor import extract_seed_entities
from epsa_rag.epsa.question_analysis.models import AnswerType
from epsa_rag.epsa.question_analysis.normalizer import normalize_entity

_NON_SPECIFIC = frozenset(
    "a an and during he her his it its she since that the their them they this those we you"
    .split()
)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_RELATIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("born", re.compile(r"\b(?:born(?:\s+in)?|birthplace)\b", re.I)),
    ("directed", re.compile(r"\b(?:directed|director)\b", re.I)),
    ("written", re.compile(r"\b(?:written|writer|author|novelist)\b", re.I)),
    ("located", re.compile(r"\b(?:located\s+(?:in|at)|based\s+(?:in|at)|situated\s+in)\b", re.I)),
    ("headquarters", re.compile(r"\b(?:head office|headquartered|headquarters)\b", re.I)),
    ("named", re.compile(r"\b(?:named\s+(?:after|for)|named)\b", re.I)),
    ("nationality", re.compile(r"\b(?:nationality|national|citizen)\b", re.I)),
    ("founded", re.compile(r"\b(?:founded|founder|established|started)\b", re.I)),
    ("published", re.compile(r"\b(?:published|publisher)\b", re.I)),
    ("released", re.compile(r"\b(?:released|release date)\b", re.I)),
    ("starring", re.compile(r"\b(?:starring|starred|cast)\b", re.I)),
    ("member", re.compile(r"\b(?:member|part of|belongs to)\b", re.I)),
    ("capital", re.compile(r"\bcapital\b", re.I)),
    ("population", re.compile(r"\bpopulation\b", re.I)),
    ("genre", re.compile(r"\bgenre\b", re.I)),
    ("occupation", re.compile(r"\b(?:occupation|profession)\b", re.I)),
    ("spouse", re.compile(r"\b(?:spouse|married|wife|husband)\b", re.I)),
    ("parent", re.compile(r"\b(?:parent|father|mother)\b", re.I)),
    ("child", re.compile(r"\b(?:child|son|daughter)\b", re.I)),
    ("educated", re.compile(r"\b(?:educated|alma mater|attended)\b", re.I)),
)


def contextual_entities(
    body: str, config: RuleBasedV2ChunkAnalyzerConfig
) -> tuple[ChunkEntityMentionV2, ...]:
    """Reuse Component 01's guarded mention boundaries and reject non-specific spans."""

    question_config = QuestionAnalyzerConfig(
        quoted_entity_score=config.quoted_entity_score,
        multi_token_entity_score=config.multi_token_entity_score,
        single_token_entity_score=config.single_token_entity_score,
    )
    result: list[ChunkEntityMentionV2] = []
    starts = [0, *(match.end() for match in _SENTENCE_BREAK.finditer(body))]
    ends = [*(match.start() for match in _SENTENCE_BREAK.finditer(body)), len(body)]
    for start, end in zip(starts, ends, strict=True):
        for mention in extract_seed_entities(body[start:end], question_config):
            normalized = normalize_entity(mention.text)
            if (
                normalized in _NON_SPECIFIC
                or len(mention.text) > config.maximum_entity_length
                or len(normalized.split()) > config.maximum_entity_tokens
            ):
                continue
            result.append(
                ChunkEntityMentionV2(
                    text=mention.text,
                    normalized=normalized,
                    source="chunk",
                    start_char=start + mention.start,
                    end_char=start + mention.end,
                    span_scope="paragraph_text",
                    confidence=mention.confidence,
                )
            )
    return tuple(result)


def is_specific_entity(entity: ChunkEntityMention, config: RuleBasedV2ChunkAnalyzerConfig) -> bool:
    normalized = entity.normalized
    return (
        normalized not in _NON_SPECIFIC
        and len(normalized) >= config.minimum_bridge_length
        and len(entity.text) <= config.maximum_entity_length
        and len(normalized.split()) <= config.maximum_entity_tokens
        and (len(normalized.split()) > 1 or len(normalized) >= 5 or entity.text.isupper())
    )


def sentence_bounds(body: str, sentences: tuple[Sentence, ...]) -> tuple[tuple[int, int, int], ...]:
    if sentences and "".join(sentence.text for sentence in sentences) == body:
        bounds: list[tuple[int, int, int]] = []
        position = 0
        for sentence in sentences:
            end = position + len(sentence.text)
            bounds.append((position, end, sentence.index))
            position = end
        return tuple(bounds)
    return ((0, len(body), 0),)


def _title_before(body: str, title: str, sentence_start: int, position: int) -> bool:
    if not title.strip():
        return False
    return re.search(
        rf"(?<!\w){re.escape(title.strip())}(?!\w)",
        body[sentence_start:position],
        re.I,
    ) is not None


def contextual_relations(
    title: str,
    body: str,
    sentences: tuple[Sentence, ...],
    mentions: tuple[ChunkEntityMention, ...],
    config: RuleBasedV2ChunkAnalyzerConfig,
) -> tuple[GroundedChunkRelationHint, ...]:
    """Ground lexical hints in source offsets and nearby sentence-level mentions."""

    bounds = sentence_bounds(body, sentences)
    detected: list[GroundedChunkRelationHint] = []
    for relation, pattern in _RELATIONS:
        for match in pattern.finditer(body):
            start, end, sentence_index = next(
                (lo, hi, index) for lo, hi, index in bounds if lo <= match.start() < hi
            )
            nearby = [
                mention for mention in mentions
                if mention.start_char is not None and mention.end_char is not None
                and start <= mention.start_char < end
            ]
            left = min(
                (item for item in nearby if item.end_char is not None
                 and item.end_char <= match.start()),
                key=lambda item: match.start() - (item.end_char or 0),
                default=None,
            )
            right = min(
                (item for item in nearby if item.start_char is not None
                 and item.start_char >= match.end()),
                key=lambda item: (item.start_char or 0) - match.end(),
                default=None,
            )
            if (
                left is not None
                and match.start() - (left.end_char or 0) > config.relation_window_chars
            ):
                left = None
            if (
                right is not None
                and (right.start_char or 0) - match.end() > config.relation_window_chars
            ):
                right = None
            subject: str | None = None
            object_: str | None = None
            title_on_left = _title_before(body, title, start, match.start())
            passive_by = re.match(r"\s+by\b", body[match.end():], re.I) is not None
            if relation in {"directed", "written", "founded", "released", "published"}:
                if passive_by:
                    subject = right.text if right is not None else None
                    object_ = title if title_on_left else None
                elif match.group(0).lower() in {
                    "directed", "written", "founded", "released", "published"
                }:
                    subject = left.text if left is not None else None
                    object_ = right.text if right is not None else None
            elif relation == "born":
                subject = left.text if left is not None else None
                if " in" in match.group(0).lower():
                    object_ = right.text if right is not None else None
            elif relation == "located":
                subject = left.text if left is not None else None
                object_ = right.text if right is not None else None
            elif relation == "starring":
                if match.group(0).lower() == "starring":
                    subject = right.text if right is not None else None
                    object_ = title if title_on_left else None
                elif match.group(0).lower() == "starred":
                    subject = left.text if left is not None else None
                    object_ = right.text if right is not None else None
            detected.append(
                GroundedChunkRelationHint(
                    relation=relation,
                    matched_text=match.group(0),
                    start_char=match.start(),
                    end_char=match.end(),
                    span_scope="paragraph_text",
                    confidence=config.relation_score,
                    sentence_index=sentence_index,
                    subject_entity=subject,
                    object_entity=object_,
                )
            )
    for match in re.finditer(r"\bby\b", body, re.I):
        start, end, sentence_index = next(
            (lo, hi, index) for lo, hi, index in bounds if lo <= match.start() < hi
        )
        prefix = body[max(start, match.start() - 24):match.start()]
        if re.search(
            r"\b(?:directed|written|founded|produced|published|released|filmed)\s+$",
            prefix, re.I,
        ):
            continue
        title_before = _title_before(body, title, start, match.start())
        right = min(
            (item for item in mentions if item.start_char is not None
             and match.end() <= item.start_char < end
             and item.start_char - match.end() <= 12
             and is_specific_entity(item, config)),
            key=lambda item: item.start_char or 0,
            default=None,
        )
        if not title_before or right is None:
            continue
        detected.append(
            GroundedChunkRelationHint(
                relation="associated",
                matched_text=match.group(0),
                start_char=match.start(),
                end_char=match.end(),
                span_scope="paragraph_text",
                confidence=config.association_score,
                sentence_index=sentence_index,
                subject_entity=right.text,
                object_entity=title,
            )
        )
    for match in re.finditer(r"\b([A-Z]{2,})\s+(raid|operation|mission)\b", body):
        start, _, sentence_index = next(
            (lo, hi, index) for lo, hi, index in bounds if lo <= match.start() < hi
        )
        title_before = _title_before(body, title, start, match.start())
        acronym = next(
            (item for item in mentions if item.start_char == match.start()
             and item.text == match.group(1)),
            None,
        )
        if not title_before or acronym is None:
            continue
        detected.append(
            GroundedChunkRelationHint(
                relation="associated",
                matched_text=match.group(0),
                start_char=match.start(),
                end_char=match.end(),
                span_scope="paragraph_text",
                confidence=config.association_score,
                sentence_index=sentence_index,
                subject_entity=acronym.text,
                object_entity=title,
            )
        )
    return tuple(sorted(detected, key=lambda hint: (hint.start_char, hint.relation)))


def contextual_answers(
    title: str,
    body: str,
    entities: tuple[ChunkEntityMentionV2, ...],
    config: RuleBasedV2ChunkAnalyzerConfig,
) -> tuple[ChunkAnswerCandidateV2, ...]:
    """Type explicit names conservatively; keep legacy numeric rules on body text."""

    candidates = [
        ChunkAnswerCandidateV2(
            answer_type=item.answer_type,
            text=item.text,
            source=item.source,
            start_char=item.start_char,
            end_char=item.end_char,
            span_scope="paragraph_text",
            confidence=item.confidence,
        )
        for item in extract_answers(body, config)
        if item.answer_type in {AnswerType.DATE, AnswerType.NUMBER}
    ]
    seen: set[tuple[AnswerType, str]] = {
        (item.answer_type, normalize_entity(item.text)) for item in candidates
    }
    for entity in entities:
        is_title = entity.span_scope == "doc_title"
        start = entity.start_char
        end = entity.end_char
        prefix = body[max(0, (start or 0) - 40):(start or 0)] if not is_title else ""
        suffix = body[(end or 0):(end or 0) + 50] if not is_title else ""
        normalized = entity.normalized
        answer_type = AnswerType.ENTITY
        title_context = (
            re.search(rf"(?<!\w){re.escape(title)}(?!\w)\s+(.{{0,80}})", body[:150], re.I)
            if is_title and title else None
        )
        title_suffix = title_context.group(1) if title_context else ""
        if is_title and re.match(
            r"(?:(?:was|is)\s+born|directed|wrote|founded)\b", title_suffix, re.I
        ):
            answer_type = AnswerType.PERSON
        elif is_title and re.match(
            r"(?:was|is)\s+(?:an?\s+)?(?:company|studio|university|bank)\b",
            title_suffix, re.I,
        ):
            answer_type = AnswerType.ORGANIZATION
        elif is_title and re.match(
            r"(?:was|is)\s+(?:an?\s+)?(?:film|novel|song|album|book|"
            r"series|play|movie)\b", title_suffix, re.I,
        ):
            answer_type = AnswerType.TITLE_OR_WORK
        elif re.search(r"\b(?:born|located|based|lived)\s+in\s+$", prefix, re.I):
            answer_type = AnswerType.LOCATION
        elif re.search(r"\b(?:directed|written|founded|produced)\s+by\s+$", prefix, re.I):
            answer_type = AnswerType.PERSON
        elif re.match(r"\s+(?:was|is)\s+born\b", suffix, re.I):
            answer_type = AnswerType.PERSON
        elif (len(normalized.split()) > 1 and re.search(
            r"\b(?:bros|records|studios|university|corporation|company|inc|ltd|"
            r"agency|committee|association|bank)\b", normalized, re.I
        )) or re.match(
            r"\s+(?:is|was)\s+(?:an?\s+)?(?:company|studio|university|bank)\b",
            suffix, re.I,
        ):
            answer_type = AnswerType.ORGANIZATION
        elif re.search(
            r"\b(?:film|novel|song|album|book|series|play|movie)\s+$", prefix, re.I
        ):
            answer_type = AnswerType.TITLE_OR_WORK
        elif is_title and re.search(
            r"\b(?:directed|released|published|starring)\b", body[:100], re.I
        ) and re.search(rf"(?<!\w){re.escape(title)}(?!\w)", body[:100], re.I):
            answer_type = AnswerType.TITLE_OR_WORK
        key = (answer_type, normalized)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            ChunkAnswerCandidateV2(
                answer_type=answer_type,
                text=entity.text,
                source="chunk_entity_like_span",
                start_char=start,
                end_char=end,
                span_scope=entity.span_scope,
                confidence=(
                    config.location_score if answer_type is AnswerType.LOCATION
                    else config.person_score if answer_type is AnswerType.PERSON
                    else config.organization_score if answer_type is AnswerType.ORGANIZATION
                    else config.title_or_work_score
                    if answer_type is AnswerType.TITLE_OR_WORK
                    else config.generic_entity_score
                ),
            )
        )
    return tuple(candidates)
