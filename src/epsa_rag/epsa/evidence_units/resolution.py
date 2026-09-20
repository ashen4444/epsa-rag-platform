"""Transparent, replaceable pronoun handling."""

from __future__ import annotations

import re
from typing import Protocol

from epsa_rag.epsa.evidence_units.models import ResolvedSentence

_PRONOUN = r"he|she|it|they|his|her|its|their|him|them"
_DIRECT = re.compile(rf"^(?P<pronoun>{_PRONOUN})\b", re.IGNORECASE)
_PREFIXED = re.compile(
    rf"^(?P<prefix>(?:in|during|after|before|later|then|also|however),?\s+)"
    rf"(?P<pronoun>{_PRONOUN})\b",
    re.IGNORECASE,
)
_OTHER_NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


class CoreferenceResolver(Protocol):
    version: str

    def resolve(
        self, sentence: str, document_title: str, context: tuple[str, ...]
    ) -> ResolvedSentence: ...


def _replace(sentence: str, document_title: str) -> str:
    text = sentence.strip()
    if not text or not document_title:
        return text
    match = _DIRECT.match(text) or _PREFIXED.match(text)
    if match is None:
        return text
    replacement = document_title
    if match.group("pronoun").casefold() in {"his", "her", "its", "their"}:
        replacement += "'s"
    start, end = match.span("pronoun")
    return f"{text[:start]}{replacement}{text[end:]}".strip()


class TitlePronounResolver:
    version = "title-v1"

    def resolve(
        self, sentence: str, document_title: str, context: tuple[str, ...]
    ) -> ResolvedSentence:
        del context
        derived = _replace(sentence, document_title)
        return ResolvedSentence(
            original_text=sentence,
            resolved_text=derived,
            method="title_pronoun" if derived != sentence else "unchanged",
            changed=derived != sentence,
            reason="leading_pronoun_to_title" if derived != sentence else "no_substitution",
            provider_version=self.version,
        )


class ConservativeTitlePronounResolver:
    version = "conservative-title-v2"

    def resolve(
        self, sentence: str, document_title: str, context: tuple[str, ...]
    ) -> ResolvedSentence:
        if not document_title.strip() or (
            _DIRECT.match(sentence.strip()) is None and _PREFIXED.match(sentence.strip()) is None
        ):
            return ResolvedSentence(
                original_text=sentence,
                resolved_text=sentence,
                method="unchanged",
                changed=False,
                reason="no_substitution",
                provider_version=self.version,
            )
        candidate = _replace(sentence, document_title)
        if candidate == sentence:
            return ResolvedSentence(
                original_text=sentence,
                resolved_text=sentence,
                method="unchanged",
                changed=False,
                reason="no_substitution",
                provider_version=self.version,
            )
        previous = context[-1] if context else ""
        leading = _DIRECT.match(sentence.strip()) or _PREFIXED.match(sentence.strip())
        neutral_pronoun = leading is not None and leading.group("pronoun").casefold() in {
            "it", "its"
        }
        recent_title_subject = bool(
            neutral_pronoun
            and re.search(
                rf"(?<!\w){re.escape(document_title.strip())}(?!\w)\s+"
                r"(?:was|is|has|had|became|began)\b",
                previous,
                re.IGNORECASE,
            )
        )
        other_names = [
            match.group(0)
            for match in _OTHER_NAME.finditer(previous)
            if match.group(0).casefold() != document_title.casefold()
            and match.group(0).casefold() not in {"he", "she", "it", "they"}
        ]
        if other_names and not recent_title_subject:
            return ResolvedSentence(
                original_text=sentence,
                resolved_text=sentence,
                method="ambiguous_context",
                changed=False,
                reason="competing_capitalized_mention",
                provider_version=self.version,
            )
        return ResolvedSentence(
            original_text=sentence,
            resolved_text=candidate,
            method="title_pronoun",
            changed=True,
            reason=(
                "recent_title_subject" if recent_title_subject else "unambiguous_title_fallback"
            ),
            provider_version=self.version,
        )
