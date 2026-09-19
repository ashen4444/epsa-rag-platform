"""Answer-slot-aware, deterministic expected-answer-type inference."""

from __future__ import annotations

import re

from epsa_rag.epsa.question_analysis.models import AnswerType

_DATE_SLOT = re.compile(
    r"\b(?:what|which|in what|in which|founding)\s+(?:year|date)\b|\bfounding year\b|\bbirthdate\b|"
    r"\bborn when\b|\bsince when\b|\bwhen\s+(?:was|were|did|does|do|is|are)\b",
    flags=re.IGNORECASE,
)
_LOCATION_SLOT = re.compile(
    r"^\s*where\b|\b(?:what|which|in what|in which)\s+"
    r"(?:city|country|state|county|province|town)\b",
    flags=re.IGNORECASE,
)
_NUMBER_SLOT = re.compile(
    r"\b(?:how many|how much|how long|how far|what number|which number|"
    r"what (?:is )?(?:the )?population|which population|"
    r"what age|which age|what height|which height|what distance|which distance|"
    r"what percentage|which percentage)\b",
    flags=re.IGNORECASE,
)
_REQUESTED_POPULATION = re.compile(
    r"\bwhat\s+(?:is|was)\b[^?]{0,100}\bpopulation\b|\bwhat\s+population\b|"
    r"\bpopulation\s+of\s+(?:what|which)\b|"
    r"\bpopulation\b[^?]{0,80}\b(?:is|was)\s+what\b",
    flags=re.IGNORECASE,
)
_PERSON_TYPED_SLOT = re.compile(
    r"\b(?:which|what)\s+"
    r"(?:(?!(?:is|was|were|did|does|do|has|have|had|the|a|an|that|who|where|"
    r"of|in|with|alongside|about|featuring|starring|by|for|from|on|at|and)\b)"
    r"[\w-]+\s+){0,4}"
    r"(?:director|actor|actress|singer-songwriter|singer|writer|author|manager|coach|"
    r"composer|engineer|scholar|goalkeeper|politician|journalist)\b",
    flags=re.IGNORECASE,
)
_TITLE_SLOT = re.compile(
    r"\b(?:which|what)\s+(?:film|movie|book|novel|album|song|series|game)\b",
    flags=re.IGNORECASE,
)
_ORGANIZATION_SLOT = re.compile(
    r"\b(?:which|what)\s+(?:company|organization|university|team|band|network|publisher)\b",
    flags=re.IGNORECASE,
)
_PERSON_SLOT = re.compile(r"^\s*(?:who|whom|whose|which person|which member)\b", re.IGNORECASE)
_GENERIC_SLOT = re.compile(r"^\s*(?:which|what)\b", re.IGNORECASE)
_BOOLEAN_LEAD = re.compile(
    r"^\s*(?:is|are|was|were|do|does|did|can|could|has|have|had)\b", re.IGNORECASE
)


def infer_answer_type(normalized_question: str) -> AnswerType:
    """Prefer explicit answer slots over incidental words in descriptive clauses."""

    if _PERSON_SLOT.search(normalized_question):
        return AnswerType.PERSON
    if _DATE_SLOT.search(normalized_question):
        return AnswerType.DATE
    if _NUMBER_SLOT.search(normalized_question):
        return AnswerType.NUMBER
    if _REQUESTED_POPULATION.search(normalized_question):
        return AnswerType.NUMBER
    if _LOCATION_SLOT.search(normalized_question):
        return AnswerType.LOCATION
    if _PERSON_TYPED_SLOT.search(normalized_question):
        return AnswerType.PERSON
    if _TITLE_SLOT.search(normalized_question):
        return AnswerType.TITLE_OR_WORK
    if _ORGANIZATION_SLOT.search(normalized_question):
        return AnswerType.ORGANIZATION
    if _BOOLEAN_LEAD.search(normalized_question):
        return AnswerType.BOOLEAN
    if _GENERIC_SLOT.search(normalized_question):
        return AnswerType.ENTITY
    return AnswerType.UNKNOWN
