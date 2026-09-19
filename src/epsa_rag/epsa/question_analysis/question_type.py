"""Ordered operational reasoning-type rules for Component 01."""

from __future__ import annotations

import re

from epsa_rag.epsa.question_analysis.models import AnswerType, QuestionType

_NESTED_ROLE = re.compile(
    r"\b(?:director|writer|author|novelist|actor|coach|player|stadium|theater|theatre|"
    r"venue|founder|publisher|member|spouse|parent|father|mother|child|son|daughter)\s+of\b",
    re.IGNORECASE,
)
_RELATIVE_STRUCTURE = re.compile(
    r"\b(?:actor|actress|coach|player|stadium|theater|theatre|venue|university|organization|team|unit)\s+"
    r"(?:who|which|where|that)\b",
    re.IGNORECASE,
)
_ROLE_RELATIVE_ATTRIBUTE = re.compile(
    r"\b(?:actor|actress|coach|player|stadium|theater|theatre|venue)\b.*\b(?:who|which)\b",
    re.IGNORECASE,
)
_ROLE_FROM = re.compile(
    r"\b(?:actor|actress|coach|player|director|author|writer)\s+(?:from|for|in)\b",
    re.IGNORECASE,
)
_EVENT_TO_ENTITY = re.compile(
    r"\b(?:by|from|for)\s+(?:a|the)\s+(?:[a-z-]+\s+){0,3}"
    r"(?:unit|team|organization|organisation|company|university)\b.*\b(?:founded|born|died|located)\b",
    re.IGNORECASE,
)


def infer_question_type(
    normalized_question: str,
    *,
    expected_answer_type: AnswerType,
    relation_hint_count: int,
    comparison_intent: bool = False,
) -> QuestionType:
    """Apply scalar comparison → boolean → compositional bridge → factoid precedence."""

    if comparison_intent:
        return QuestionType.COMPARISON
    if expected_answer_type is AnswerType.BOOLEAN:
        return QuestionType.YES_NO
    if _NESTED_ROLE.search(normalized_question) is not None and (
        relation_hint_count >= 2 or expected_answer_type is not AnswerType.PERSON
    ):
        return QuestionType.BRIDGE
    if (
        _RELATIVE_STRUCTURE.search(normalized_question) is not None
        or _ROLE_RELATIVE_ATTRIBUTE.search(normalized_question) is not None
        or _ROLE_FROM.search(normalized_question) is not None
        or _EVENT_TO_ENTITY.search(normalized_question) is not None
    ):
        return QuestionType.BRIDGE
    return QuestionType.FACTOID
