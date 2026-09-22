"""Deterministic context rendering outside retrieval and EPSA algorithms."""

from __future__ import annotations

from collections.abc import Sequence

from epsa_rag.core.models import RankedParagraphChunk
from epsa_rag.epsa.context_pruning.models import PrunedContext
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


def render_epsa_pruned_context(context: PrunedContext) -> RenderedContext:
    """Adapt Component 08 output to the shared final-answer context boundary."""

    return RenderedContext(
        format_version="epsa-pruned-sentences-v1",
        context_kind="pruned_sentences",
        chunk_ids=context.selected_chunk_ids,
        text=context.selected_context_text,
        character_count=len(context.selected_context_text),
    )
