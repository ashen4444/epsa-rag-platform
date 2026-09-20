"""Convert supported retrieved-result shapes to one inference-safe chunk contract."""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from epsa_rag.core.exceptions import ChunkAnalysisError
from epsa_rag.core.models import RankedParagraphChunk, Sentence
from epsa_rag.epsa.chunk_analysis.models import CanonicalRetrievedChunk

_TITLE_PREFIX = re.compile(
    r"^\s*Title:\s*.*?\n\s*Paragraph:\s*", flags=re.IGNORECASE | re.DOTALL
)


def _value(obj: object, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _first(obj: object, *names: str) -> Any:
    # Match the historical nested-field precedence while keeping gold fields out.
    for nested_name in ("chunk", "document", "metadata"):
        nested = _value(obj, nested_name)
        if nested is not None and nested is not obj:
            for name in names:
                found = _value(nested, name)
                if found is not None:
                    return found
    for name in names:
        found = _value(obj, name)
        if found is not None:
            return found
    return None


def _sentences(value: object) -> tuple[Sentence, ...]:
    if value is None:
        return ()
    if not isinstance(value, (tuple, list)):
        raise ChunkAnalysisError("sentences must be a sequence")
    result: list[Sentence] = []
    for position, item in enumerate(value):
        if isinstance(item, Sentence):
            result.append(item)
        elif isinstance(item, str):
            result.append(Sentence(index=position, text=item))
        else:
            index = _value(item, "index")
            if index is None:
                index = _value(item, "sentence_id")
            if index is None:
                index = position
            result.append(Sentence(index=index, text=_value(item, "text")))
    return tuple(result)


def adapt_retrieved_chunk(
    chunk: object,
    *,
    retrieval_rank: int | None = None,
    retrieval_score: float | None = None,
) -> CanonicalRetrievedChunk:
    """Normalize canonical results and documented legacy aliases, rejecting invalid provenance."""

    if isinstance(chunk, CanonicalRetrievedChunk):
        return CanonicalRetrievedChunk.model_validate(
            {
                **chunk.model_dump(mode="json"),
                "retrieval_rank": (
                    retrieval_rank if retrieval_rank is not None else chunk.retrieval_rank
                ),
                "retrieval_score": (
                    retrieval_score if retrieval_score is not None else chunk.retrieval_score
                ),
            }
        )
    if isinstance(chunk, RankedParagraphChunk):
        return CanonicalRetrievedChunk(
            chunk_id=chunk.chunk.chunk_id,
            doc_title=chunk.chunk.title,
            paragraph_text=chunk.chunk.paragraph_text,
            chunk_text=f"Title: {chunk.chunk.title}\nParagraph: {chunk.chunk.paragraph_text}",
            sentences=chunk.chunk.sentences,
            retrieval_rank=retrieval_rank if retrieval_rank is not None else chunk.rank,
            retrieval_score=retrieval_score if retrieval_score is not None else chunk.score,
        )
    chunk_id = _first(chunk, "chunk_id", "id")
    if chunk_id is None:
        raise ChunkAnalysisError("chunk_id is required")
    title = _first(chunk, "doc_title", "title")
    paragraph = _first(chunk, "paragraph_text")
    chunk_text = _first(chunk, "chunk_text", "text")
    title_text = "" if title is None else str(title)
    body = "" if paragraph is None else str(paragraph)
    rendered = "" if chunk_text is None else str(chunk_text)
    if not body and rendered:
        body = _TITLE_PREFIX.sub("", rendered).strip()
    if not rendered:
        rendered = (
            f"Title: {title_text}\nParagraph: {body}"
            if title_text and body
            else body or title_text
        )
    if not body:
        raise ChunkAnalysisError("chunk must contain paragraph or chunk text")
    try:
        return CanonicalRetrievedChunk(
            chunk_id=str(chunk_id),
            doc_id=_first(chunk, "doc_id"),
            doc_title=title_text,
            paragraph_index=_first(chunk, "paragraph_index", "paragraph_id"),
            paragraph_text=body,
            chunk_text=rendered,
            sentences=_sentences(_first(chunk, "sentences")),
            retrieval_rank=(
                retrieval_rank
                if retrieval_rank is not None
                else _first(chunk, "rank", "retrieval_rank")
            ),
            retrieval_score=(
                retrieval_score
                if retrieval_score is not None
                else _first(chunk, "score", "retrieval_score", "fusion_score")
            ),
            source_question_id=_first(chunk, "question_id", "source_question_id"),
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise ChunkAnalysisError("invalid retrieved chunk fields") from error
