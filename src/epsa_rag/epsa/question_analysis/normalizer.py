"""Stable normalization shared by deterministic Component 01 rules."""

from __future__ import annotations

import re

_QUOTATION_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
    }
)
_WHITESPACE = re.compile(r"\s+")
_ENTITY_SEPARATORS = re.compile(r"[\W_]+", flags=re.UNICODE)


def normalize_matching_text(question: str) -> str:
    """Normalize quote marks and whitespace while preserving case for source mentions."""

    return _WHITESPACE.sub(" ", question.translate(_QUOTATION_TRANSLATION).strip())


def normalize_question(question: str) -> str:
    """Return the documented lowercase matching form of a question."""

    return normalize_matching_text(question).casefold()


def normalize_entity(text: str) -> str:
    """Normalize entity-like text for deterministic exact deduplication."""

    return _WHITESPACE.sub(" ", _ENTITY_SEPARATORS.sub(" ", text.casefold())).strip()
