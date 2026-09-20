"""Historical deterministic Component 02 feature rules, isolated from Component 01."""

from __future__ import annotations

import re

from epsa_rag.epsa.chunk_analysis.config import ChunkAnalyzerConfig, _ChunkRuleConfig
from epsa_rag.epsa.chunk_analysis.models import (
    ChunkAnswerCandidate,
    ChunkEntityMention,
    ChunkRelationHint,
)
from epsa_rag.epsa.question_analysis.models import AnswerType

TOKEN_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how in is it of on or "
    "that the their this to was were what when where which who whose with".split()
)
QUESTION_LEADS = frozenset(
    "what which who whom whose where when why how is are was were do does did can could "
    "has have had".split()
)
LEXICAL_EQUIVALENTS: dict[str, frozenset[str]] = {
    "director": frozenset(("directed", "director")),
    "writer": frozenset(("written", "writer", "author")),
    "author": frozenset(("written", "writer", "author")),
    "birthplace": frozenset(("born", "birthplace")),
}
RELATION_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("born", (r"\bborn\b", r"\bbirthplace\b", r"\bborn in\b")),
    ("directed", (r"\bdirected\b", r"\bdirector\b")),
    ("written", (r"\bwritten\b", r"\bwriter\b", r"\bauthor\b", r"\bnovelist\b")),
    ("located", (r"\blocated\b", r"\bbased in\b", r"\bin\b", r"\bat\b")),
    ("headquarters", (r"\bhead office\b", r"\bheadquartered\b", r"\bheadquarters\b")),
    ("named", (r"\bnamed after\b", r"\bnamed for\b", r"\bnamed\b")),
    ("nationality", (r"\bnationality\b", r"\bnational\b", r"\bcitizen\b")),
    ("founded", (r"\bfounded\b", r"\bfounder\b", r"\bestablished\b", r"\bstarted\b")),
    ("published", (r"\bpublished\b", r"\bpublisher\b")),
    ("released", (r"\breleased\b", r"\brelease date\b")),
    ("starring", (r"\bstarring\b", r"\bstarred\b", r"\bcast\b")),
    ("member", (r"\bmember\b", r"\bpart of\b", r"\bbelongs to\b")),
    ("capital", (r"\bcapital\b",)),
    ("population", (r"\bpopulation\b",)),
    ("genre", (r"\bgenre\b",)),
    ("occupation", (r"\boccupation\b", r"\bprofession\b")),
    ("spouse", (r"\bspouse\b", r"\bmarried\b", r"\bwife\b", r"\bhusband\b")),
    ("parent", (r"\bparent\b", r"\bfather\b", r"\bmother\b")),
    ("child", (r"\bchild\b", r"\bson\b", r"\bdaughter\b")),
    ("educated", (r"\beducated\b", r"\balma mater\b", r"\battended\b")),
)


def normalize_text(text: str) -> str:
    text = text.strip().replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    return re.sub(r"\s+", " ", text).lower()


def normalize_entity(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", normalize_text(text))
    return re.sub(r"\s+", " ", text).strip()


def _clean_entity(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t\n\r,.;:!?()[]{}")


def _add_entity(
    mentions: list[ChunkEntityMention],
    seen: set[str],
    text: str,
    start: int,
    end: int,
    score: float,
) -> None:
    text = _clean_entity(text)
    normalized = normalize_entity(text)
    if text and normalized and normalized not in seen:
        seen.add(normalized)
        mentions.append(
            ChunkEntityMention(
                text=text,
                normalized=normalized,
                source="chunk",
                start_char=start,
                end_char=end,
                confidence=score,
            )
        )


def extract_entities(text: str, config: _ChunkRuleConfig) -> tuple[ChunkEntityMention, ...]:
    """Preserve the legacy quote, capitalized phrase, and token pass order."""

    mentions: list[ChunkEntityMention] = []
    seen: set[str] = set()
    for pattern in (r'"([^"]+)"', r"'([^']+)'"):
        for match in re.finditer(pattern, text):
            _add_entity(
                mentions, seen, match.group(1).strip(), match.start(1), match.end(1),
                config.quoted_entity_score,
            )
    word = r"(?:[A-Z][A-Za-z0-9'&-]*(?:\.[A-Z][A-Za-z0-9'&-]*)*|[A-Z]{2,})"
    connector = r"(?:of|the|and|for|in|on|at|de|la|le|du|&)"
    multi = re.compile(rf"\b{word}(?:[ \t]+(?:{connector}|{word}))+")
    for match in multi.finditer(text):
        entity = _clean_entity(match.group(0))
        if normalize_entity(entity).split(" ", maxsplit=1)[0] not in QUESTION_LEADS:
            _add_entity(
                mentions, seen, entity, match.start(), match.end(), config.multi_token_entity_score
            )
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9'&.-]{2,}\b", text):
        entity = _clean_entity(match.group(0))
        if normalize_entity(entity).split(" ", maxsplit=1)[0] not in QUESTION_LEADS:
            _add_entity(
                mentions, seen, entity, match.start(), match.end(), config.single_token_entity_score
            )
    return tuple(mentions)


def extract_relations(text: str, config: ChunkAnalyzerConfig) -> tuple[ChunkRelationHint, ...]:
    """Retain historical lexical patterns, including their known broad location triggers."""

    normalized = normalize_text(text)
    seen: set[tuple[str, str, int]] = set()
    hints: list[ChunkRelationHint] = []
    for relation, patterns in RELATION_PATTERNS:
        for pattern in patterns:
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                key = (relation, match.group(0), match.start())
                if key not in seen:
                    seen.add(key)
                    hints.append(
                        ChunkRelationHint(
                            relation=relation,
                            matched_text=match.group(0),
                            start_char=match.start(),
                            end_char=match.end(),
                            confidence=config.relation_score,
                        )
                    )
    return tuple(hints)


def extract_answers(text: str, config: _ChunkRuleConfig) -> tuple[ChunkAnswerCandidate, ...]:
    """Extract historical DATE, NUMBER, LOCATION and generic ENTITY candidates."""

    candidates: list[ChunkAnswerCandidate] = []
    seen: set[tuple[AnswerType, str]] = set()
    patterns: tuple[tuple[AnswerType, str, float], ...] = (
        (AnswerType.DATE, r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b", config.year_score),
        (
            AnswerType.DATE,
            r"\b(?:January|February|March|April|May|June|July|August|September|October|"
            r"November|December)\s+\d{1,2},\s+\d{4}\b",
            config.full_date_score,
        ),
        (AnswerType.NUMBER, r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", config.number_score),
        (
            AnswerType.LOCATION,
            r"\b(?:in|at|from|located in|based in)\s+"
            r"([A-Z][A-Za-z'&.-]+(?:\s+[A-Z][A-Za-z'&.-]+){0,4})",
            config.location_score,
        ),
    )
    for answer_type, pattern, score in patterns:
        for match in re.finditer(pattern, text):
            candidate = match.group(1) if match.lastindex else match.group(0)
            candidate = candidate.strip(" ,.;:!?()[]{}")
            key = (answer_type, candidate.lower())
            if candidate and key not in seen:
                seen.add(key)
                candidates.append(
                    ChunkAnswerCandidate(
                        answer_type=answer_type,
                        text=candidate,
                        source="chunk_pattern",
                        start_char=match.start(1) if match.lastindex else match.start(),
                        end_char=match.end(1) if match.lastindex else match.end(),
                        confidence=score,
                    )
                )
    for mention in extract_entities(text, config):
        key = (AnswerType.ENTITY, mention.normalized)
        if key not in seen:
            seen.add(key)
            candidates.append(
                ChunkAnswerCandidate(
                    answer_type=AnswerType.ENTITY,
                    text=mention.text,
                    source="chunk_entity_like_span",
                    start_char=mention.start_char or 0,
                    end_char=mention.end_char or 0,
                    confidence=config.generic_entity_score,
                )
            )
    return tuple(candidates)


def entity_matches(left: str, right: str) -> bool:
    return bool(left and right and (left == right or left in right or right in left))


def content_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in TOKEN_STOPWORDS and len(token) > 1
    )


def token_overlap(question: str, analysis_text: str) -> tuple[tuple[str, ...], float]:
    question_tokens = content_tokens(question)
    text_tokens = set(content_tokens(analysis_text))
    overlap = [token for token in question_tokens if token in text_tokens]
    lowered_text = analysis_text.lower()
    for question_token, variants in LEXICAL_EQUIVALENTS.items():
        if question_token in question_tokens and any(
            variant in lowered_text for variant in variants
        ):
            overlap.append(question_token)
    unique = tuple(dict.fromkeys(overlap))
    return unique, round(len(set(unique)) / max(len(set(question_tokens)), 1), 4)
