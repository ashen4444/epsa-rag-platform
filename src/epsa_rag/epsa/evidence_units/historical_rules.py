"""Frozen sentence feature rules from the documented thesis extractor."""

from __future__ import annotations

import re

_STOPWORDS = frozenset(
    "a an and are as at be by for from in into is it its of on or that the their "
    "then there this to was were which who whom whose with what when where why how "
    "did does do had has have".split()
)
_RELATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("born", (r"\bborn\b", r"\bbirthplace\b", r"\bborn in\b")),
    ("birthplace", (r"\bbirthplace\b", r"\bplace of birth\b")),
    ("directed", (r"\bdirected\b", r"\bdirector\b", r"\bdirected by\b")),
    ("written", (r"\bwritten\b", r"\bwriter\b", r"\bauthor\b", r"\bnovelist\b")),
    ("author", (r"\bauthor\b", r"\bwritten by\b")),
    ("located", (r"\blocated\b", r"\blocated in\b", r"\bbased in\b")),
    ("headquarters", (r"\bhead office\b", r"\bheadquartered\b", r"\bheadquarters\b")),
    ("named", (r"\bnamed after\b", r"\bnamed for\b", r"\bnamed\b")),
    ("nationality", (r"\bnationality\b", r"\bnational\b", r"\bcitizen\b")),
    ("founded", (r"\bfounded\b", r"\bfounder\b", r"\bestablished\b")),
    ("published", (r"\bpublished\b", r"\bpublisher\b")),
    ("released", (r"\breleased\b", r"\brelease date\b")),
    ("starring", (r"\bstarring\b", r"\bstarred\b", r"\bcast\b")),
    ("member", (r"\bmember\b", r"\bmembers\b", r"\bpart of\b")),
    ("capital", (r"\bcapital\b",)),
    ("population", (r"\bpopulation\b",)),
    ("genre", (r"\bgenre\b",)),
    ("occupation", (r"\boccupation\b", r"\bprofession\b")),
    ("spouse", (r"\bspouse\b", r"\bmarried\b", r"\bwife\b", r"\bhusband\b")),
    ("parent", (r"\bparent\b", r"\bfather\b", r"\bmother\b")),
    ("child", (r"\bchild\b", r"\bson\b", r"\bdaughter\b")),
    ("educated", (r"\beducated\b", r"\bstudied\b", r"\battended\b")),
    ("alma mater", (r"\balma mater\b",)),
    ("discovered", (r"\bdiscovered\b", r"\bdiscovery\b")),
    ("capacity", (r"\bcapacity\b", r"\bseats\b", r"\bseat\b")),
)
_ORG_SUFFIX = re.compile(
    r"\b(?:University|College|Institute|School|Hospital|Bank|Company|"
    r"Corporation|Corp\.?|Inc\.?|Ltd\.?|Association|Club|FC|Agency|"
    r"Ministry|Department|Committee|Council|League|Museum|Theatre|"
    r"Center|Centre)\b"
)
_DATE = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{4}\b"
)
_NUMBER = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
_QUOTED = re.compile(r"""["“”'‘’]([^"“”'‘’]{2,})["“”'‘’]""")
_CAPITALIZED = re.compile(
    r"\b(?:[A-Z][\w.&'’-]*)(?:\s+(?:of|the|and|de|da|del|van|von|la|le|"
    r"[A-Z][\w.&'’-]*))*"
)


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip(" ,.;:()[]{}")
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return tuple(result)


def entities(text: str, title: str) -> tuple[str, ...]:
    values = [title] if title else []
    values.extend(match.group(1) for match in _QUOTED.finditer(text))
    for match in _CAPITALIZED.finditer(text):
        phrase = match.group(0).strip()
        if (
            phrase.casefold() in _STOPWORDS
            or phrase.casefold()
            in {"he", "she", "it", "they", "his", "her", "its", "their", "him", "them"}
            or len(phrase) <= 1
        ):
            continue
        values.append(phrase)
    return _dedupe(values)


def relation_hints(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    return tuple(
        name
        for name, patterns in _RELATIONS
        if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in patterns)
    )


def answer_types(
    text: str, entity_values: tuple[str, ...], relations: tuple[str, ...]
) -> tuple[str, ...]:
    result: list[str] = []
    if _DATE.search(text):
        result.append("DATE")
    if _NUMBER.search(text):
        result.append("NUMBER")
    if {"born", "birthplace", "located", "capital"}.intersection(relations):
        result.append("LOCATION")
    if any(_ORG_SUFFIX.search(item) for item in entity_values):
        result.append("ORGANIZATION")
    people = [item for item in entity_values if len(item.split()) >= 2]
    if people and not any(_ORG_SUFFIX.search(item) for item in people):
        result.append("PERSON")
    if _QUOTED.search(text) or {
        "released",
        "published",
        "written",
        "directed",
        "starring",
        "genre",
    }.intersection(relations):
        result.append("TITLE_OR_WORK")
    if entity_values:
        result.append("ENTITY")
    return _dedupe(result)


def question_overlap(
    question: str, question_entities: tuple[str, ...], resolved_text: str
) -> tuple[tuple[str, ...], float]:
    entity_overlap = _dedupe(
        [
            entity
            for entity in question_entities
            if entity and entity.casefold() in resolved_text.casefold()
        ]
    )

    def tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.casefold())
            if token not in _STOPWORDS and len(token) > 1
        }

    question_tokens = tokens(question)
    score = (
        round(len(question_tokens & tokens(resolved_text)) / len(question_tokens), 6)
        if question_tokens
        else 0.0
    )
    return entity_overlap, score
