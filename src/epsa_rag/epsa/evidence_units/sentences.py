"""Preserve native sentence indices; segment only when metadata is absent."""

from __future__ import annotations

import re
from itertools import pairwise
from typing import Literal

from pydantic import Field

from epsa_rag.core.models import ContractModel
from epsa_rag.epsa.chunk_analysis.models import CanonicalRetrievedChunk

SentenceSource = Literal["metadata", "paragraph_fallback", "segmented"]
_ENDING = frozenset(".!?")
_ABBREVIATIONS = frozenset({"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "fig"})


class NormalizedSentence(ContractModel):
    sentence_id: int = Field(ge=0)
    text: str
    source: SentenceSource
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


def _metadata_sentences(
    chunk: CanonicalRetrievedChunk, *, historical: bool
) -> tuple[NormalizedSentence, ...]:
    if not chunk.sentences:
        return ()
    result: list[NormalizedSentence] = []
    cursor = 0
    for sentence in chunk.sentences:
        original = sentence.text
        text = original.strip() if historical else original
        if not text.strip():
            cursor += len(original)
            continue
        start: int | None = None
        end: int | None = None
        if chunk.paragraph_text[cursor : cursor + len(original)] == original:
            start = cursor + (len(original) - len(original.lstrip()) if historical else 0)
            end = start + len(text)
        result.append(
            NormalizedSentence(
                sentence_id=sentence.index,
                text=text,
                source="metadata",
                start_char=start,
                end_char=end,
            )
        )
        cursor += len(original)
    return tuple(result)


def _can_split(text: str, punctuation_index: int) -> bool:
    if text[punctuation_index] != ".":
        return True
    if (punctuation_index > 0 and text[punctuation_index - 1] == ".") or (
        punctuation_index + 1 < len(text) and text[punctuation_index + 1] == "."
    ):
        return False
    prefix = text[:punctuation_index]
    match = re.search(r"([A-Za-z]+)$", prefix)
    if match is None:
        return True
    token = match.group(1)
    if token.casefold() in _ABBREVIATIONS or len(token) == 1:
        return False
    return True


def segment_sentences(text: str) -> tuple[NormalizedSentence, ...]:
    """Small deterministic English fallback retaining source offsets."""

    if not text.strip():
        return ()
    boundaries = [0]
    for index, character in enumerate(text):
        if character not in _ENDING or not _can_split(text, index):
            continue
        next_index = index + 1
        while next_index < len(text) and text[next_index] in "\"'”’)]}":
            next_index += 1
        if next_index >= len(text) or not text[next_index].isspace():
            continue
        next_nonspace = next_index
        while next_nonspace < len(text) and text[next_nonspace].isspace():
            next_nonspace += 1
        if next_nonspace < len(text) and text[next_nonspace] in "\"'“‘([":
            next_nonspace += 1
        if next_nonspace < len(text) and text[next_nonspace].isupper():
            boundaries.append(next_index)
    boundaries.append(len(text))
    result: list[NormalizedSentence] = []
    for lo, hi in pairwise(boundaries):
        segment = text[lo:hi]
        if not segment.strip():
            continue
        start = lo + len(segment) - len(segment.lstrip())
        end = hi - (len(segment) - len(segment.rstrip()))
        result.append(
            NormalizedSentence(
                sentence_id=len(result),
                text=text[start:end],
                source="segmented",
                start_char=start,
                end_char=end,
            )
        )
    return tuple(result)


def normalize_sentences(
    chunk: CanonicalRetrievedChunk, *, historical: bool
) -> tuple[NormalizedSentence, ...]:
    metadata = _metadata_sentences(chunk, historical=historical)
    if metadata:
        return metadata
    paragraph = chunk.paragraph_text or chunk.chunk_text
    if not paragraph.strip():
        return ()
    if historical:
        text = paragraph.strip()
        start = paragraph.find(text)
        return (
            NormalizedSentence(
                sentence_id=0,
                text=text,
                source="paragraph_fallback",
                start_char=start,
                end_char=start + len(text),
            ),
        )
    return segment_sentences(paragraph)
