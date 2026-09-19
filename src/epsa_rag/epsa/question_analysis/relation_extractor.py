"""Contextual lexical relation hints, deliberately separate from proven evidence."""

from __future__ import annotations

import re

from epsa_rag.epsa.question_analysis.config import QuestionAnalyzerConfig
from epsa_rag.epsa.question_analysis.models import RelationHint

_RELATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("born", re.compile(r"\b(?:born|birthplace)\b")),
    ("directed", re.compile(r"\b(?:directed|director)\b")),
    ("written", re.compile(r"\b(?:written|writer|author|novelist)\b")),
    (
        "located",
        re.compile(r"\b(?:located\s+(?:in|at)|based\s+(?:in|at)|situated\s+in|home city)\b"),
    ),
    ("headquarters", re.compile(r"\b(?:head office|headquartered|headquarters)\b")),
    ("named", re.compile(r"\b(?:named after|named for|named)\b")),
    ("nationality", re.compile(r"\b(?:nationality|national|citizen)\b")),
    ("founded", re.compile(r"\b(?:founded|founder|established|started)\b")),
    ("published", re.compile(r"\b(?:published|publisher)\b")),
    ("released", re.compile(r"\b(?:released|release date)\b")),
    ("starring", re.compile(r"\b(?:starring|starred|cast)\b")),
    ("member", re.compile(r"\b(?:member|part of|belongs to)\b")),
    ("capital", re.compile(r"\bcapital\b")),
    ("population", re.compile(r"\bpopulation\b")),
    ("genre", re.compile(r"\bgenre\b")),
    ("occupation", re.compile(r"\b(?:occupation|profession)\b")),
    ("spouse", re.compile(r"\b(?:spouse|married|wife|husband)\b")),
    ("parent", re.compile(r"\b(?:parent|father|mother)\b")),
    ("child", re.compile(r"\b(?:child|son|daughter)\b")),
    ("educated", re.compile(r"\b(?:educated|alma mater|attended)\b")),
)
_QUOTED = re.compile(r'"[^\"]+"|(?<![\w])\'[^\']+\'(?![\w])')
_IDIOMATIC_PARENT = re.compile(r"\bfather of the atomic bomb\b|\blove child\b")
_NATIONAL_COUNCIL = re.compile(r"\bnational council\b")
_PARENT_OF_SURFACE = re.compile(r"\b(?:father|mother)\s+of\s+(?:the\s+)?(?P<subject>\w+)")
_TITLE_LIKE_SURFACE = re.compile(r"\b(?:[A-Z][\w'-]*\s+){1,5}[A-Z][\w'-]*\b")


def _protected_spans(question: str) -> tuple[tuple[int, int], ...]:
    return tuple((match.start(), match.end()) for match in _QUOTED.finditer(question))


def _is_protected(
    relation: str,
    start: int,
    end: int,
    question: str,
    quoted_spans: tuple[tuple[int, int], ...],
    surface_question: str | None,
) -> bool:
    if any(start < quote_end and quote_start < end for quote_start, quote_end in quoted_spans):
        return True
    if relation == "parent" and _IDIOMATIC_PARENT.search(question[max(0, start - 20) : end + 20]):
        return True
    if relation == "parent" and surface_question is not None:
        parent_phrase = _PARENT_OF_SURFACE.search(surface_question, max(0, start - 5))
        if parent_phrase is not None and parent_phrase.start() <= start < parent_phrase.end():
            # A lowercase noun after "father/mother of" is an idiomatic or descriptive phrase;
            # a capitalized name remains a usable kinship relation hint.
            return parent_phrase.group("subject")[0].islower()
    if relation in {"parent", "child"} and surface_question is not None:
        for title_span in _TITLE_LIKE_SURFACE.finditer(surface_question):
            if title_span.start() <= start < title_span.end():
                return True
    return (
        relation == "nationality"
        and _NATIONAL_COUNCIL.search(question[max(0, start - 5) : end + 12]) is not None
    )


def extract_relation_hints(
    normalized_question: str, config: QuestionAnalyzerConfig, *, surface_question: str | None = None
) -> tuple[RelationHint, ...]:
    """Emit contextual relation hypotheses, deduplicated by normalized relation in text order."""

    quoted_spans = _protected_spans(normalized_question)
    detected: list[tuple[int, int, RelationHint]] = []
    for rule_order, (relation, pattern) in enumerate(_RELATION_PATTERNS):
        for match in pattern.finditer(normalized_question):
            if _is_protected(
                relation,
                match.start(),
                match.end(),
                normalized_question,
                quoted_spans,
                surface_question,
            ):
                continue
            detected.append(
                (
                    match.start(),
                    rule_order,
                    RelationHint(
                        relation=relation,
                        matched_text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        confidence=config.relation_hint_score,
                    ),
                )
            )
    unique: list[RelationHint] = []
    seen: set[str] = set()
    for _, _, hint in sorted(detected, key=lambda item: (item[0], item[1])):
        if hint.relation not in seen:
            seen.add(hint.relation)
            unique.append(hint)
    return tuple(unique)
