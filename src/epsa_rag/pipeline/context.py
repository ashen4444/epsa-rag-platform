"""Deterministic context rendering outside retrieval and EPSA algorithms."""

from __future__ import annotations

from collections.abc import Sequence

from epsa_rag.core.models import RankedParagraphChunk
from epsa_rag.pipeline.models import RenderedContext


def render_full_paragraph_context(
    results: Sequence[RankedParagraphChunk],
) -> RenderedContext:
    """Render ranked full paragraphs without exposing internal IDs to the answer model."""

    text = "\n\n".join(
        f"[Document {index}]\nTitle: {result.chunk.title}\nText: {result.chunk.paragraph_text}"
        for index, result in enumerate(results, start=1)
    )
    return RenderedContext(
        format_version="ranked-full-paragraph-v1",
        context_kind="full_paragraphs",
        chunk_ids=tuple(result.chunk.chunk_id for result in results),
        text=text,
        character_count=len(text),
    )
