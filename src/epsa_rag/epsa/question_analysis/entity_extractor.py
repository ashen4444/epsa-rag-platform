"""Unicode-safe deterministic extraction of clean seed-entity candidates."""

from __future__ import annotations

import re

from epsa_rag.epsa.question_analysis.config import QuestionAnalyzerConfig
from epsa_rag.epsa.question_analysis.models import EntityMention
from epsa_rag.epsa.question_analysis.normalizer import normalize_entity

_STANDALONE_ARTIFACTS = frozenset(
    {
        "a",
        "an",
        "are",
        "because",
        "between",
        "did",
        "do",
        "does",
        "during",
        "from",
        "how",
        "in",
        "is",
        "the",
        "this",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
    }
)
_CONNECTORS = frozenset(
    {
        "at",
        "da",
        "de",
        "del",
        "der",
        "di",
        "du",
        "for",
        "in",
        "la",
        "le",
        "of",
        "on",
        "the",
        "van",
        "von",
    }
)
_QUOTED = re.compile(
    r'"(?P<double_text>[^\"]+)"|(?<![\w])\'(?P<single_text>[^\']+)\'(?![\w])'
)
_WORD = re.compile(r"[^\W\d_][\w\u2019'-]*", flags=re.UNICODE)


def _make_mention(
    text: str, source: str, start: int, end: int, confidence: float
) -> EntityMention | None:
    first, separator, remainder = text.partition(" ")
    if separator and first.casefold() in _STANDALONE_ARTIFACTS:
        text, start = remainder, start + len(first) + len(separator)
    if text.casefold() in _STANDALONE_ARTIFACTS:
        return None
    if text.casefold().endswith("'s") or text.casefold().endswith("\u2019s"):
        text, end = text[:-2], end - 2
    normalized = normalize_entity(text)
    if not normalized or normalized in _STANDALONE_ARTIFACTS:
        return None
    return EntityMention(
        text=text, normalized=normalized, source=source, start=start, end=end, confidence=confidence
    )


def _capitalized_phrases(text: str) -> list[tuple[str, int, int]]:
    tokens = [(match.group(0), match.start(), match.end()) for match in _WORD.finditer(text)]
    phrases: list[tuple[str, int, int]] = []
    index = 0
    while index < len(tokens):
        token, start, end = tokens[index]
        if not token[0].isupper():
            index += 1
            continue
        finish = index + 1
        while finish < len(tokens):
            next_token, _, next_end = tokens[finish]
            if next_token[0].isupper():
                end = next_end
                finish += 1
            elif (
                next_token.casefold() in _CONNECTORS
                and finish + 1 < len(tokens)
                and tokens[finish + 1][0][0].isupper()
            ):
                end = tokens[finish + 1][2]
                finish += 2
            else:
                break
        if finish > index + 1:
            phrases.append((text[start:end], start, end))
        index = finish
    return phrases


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < other_end and other_start < end for other_start, other_end in spans)


def extract_seed_entities(
    matching_question: str, config: QuestionAnalyzerConfig
) -> tuple[EntityMention, ...]:
    """Prefer full quoted/multiword names and suppress their redundant token fragments."""

    detected: list[EntityMention] = []
    protected_spans: list[tuple[int, int]] = []
    for match in _QUOTED.finditer(matching_question):
        text_group = "double_text" if match.group("double_text") is not None else "single_text"
        mention = _make_mention(
            match.group(text_group),
            "quoted_string",
            match.start(text_group),
            match.end(text_group),
            config.quoted_entity_score,
        )
        if mention is not None:
            detected.append(mention)
            protected_spans.append((mention.start, mention.end))
    for phrase, start, end in _capitalized_phrases(matching_question):
        if _overlaps(start, end, protected_spans):
            continue
        mention = _make_mention(
            phrase, "capitalized_phrase", start, end, config.multi_token_entity_score
        )
        if mention is not None:
            detected.append(mention)
            protected_spans.append((mention.start, mention.end))
    for match in _WORD.finditer(matching_question):
        if not match.group(0)[0].isupper() or _overlaps(
            match.start(), match.end(), protected_spans
        ):
            continue
        mention = _make_mention(
            match.group(0),
            "title_like_token",
            match.start(),
            match.end(),
            config.single_token_entity_score,
        )
        if mention is not None:
            detected.append(mention)
    unique: list[EntityMention] = []
    seen: set[str] = set()
    for mention in detected:
        if mention.normalized not in seen:
            seen.add(mention.normalized)
            unique.append(mention)
    return tuple(unique)
