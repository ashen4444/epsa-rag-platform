from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, RetrievalQuery, Sentence


def make_chunk() -> ParagraphChunk:
    return ParagraphChunk(
        chunk_id="chunk:alpha",
        title="Alpha",
        paragraph_text="First. Second.",
        sentences=(Sentence(index=0, text="First."), Sentence(index=1, text=" Second.")),
    )


def test_paragraph_chunk_round_trips_as_json_without_gold_labels() -> None:
    chunk = make_chunk()
    restored = ParagraphChunk.model_validate_json(chunk.model_dump_json())

    assert restored == chunk
    assert restored.sentences[1].index == 1
    assert "supporting" not in chunk.model_dump()


@pytest.mark.parametrize(
    "sentences",
    [
        (Sentence(index=1, text="Second."), Sentence(index=0, text="First.")),
        (Sentence(index=0, text="First."), Sentence(index=0, text="Again.")),
    ],
)
def test_paragraph_chunk_rejects_invalid_sentence_order(
    sentences: tuple[Sentence, ...],
) -> None:
    with pytest.raises(ValidationError, match="contiguous and zero-based"):
        ParagraphChunk(
            chunk_id="chunk:alpha",
            title="Alpha",
            paragraph_text="Text.",
            sentences=sentences,
        )


def test_paragraph_chunk_rejects_reconstructed_or_missing_sentence_structure() -> None:
    with pytest.raises(ValidationError, match="exact ordered sentence"):
        ParagraphChunk(
            chunk_id="chunk:alpha",
            title="Alpha",
            paragraph_text="First. Second.",
            sentences=(Sentence(index=0, text="First."), Sentence(index=1, text="Second.")),
        )

    with pytest.raises(ValidationError, match="at least 1"):
        ParagraphChunk(
            chunk_id="chunk:alpha",
            title="Alpha",
            paragraph_text="Text.",
            sentences=(),
        )


def test_contracts_preserve_text_and_reject_unknown_fields() -> None:
    query = RetrievalQuery(text="  Who wrote it?  ", question_id="  q-1  ")

    assert query.text == "  Who wrote it?  "
    assert query.question_id == "q-1"

    with pytest.raises(ValidationError):
        RetrievalQuery(text="Question?", backend_hint="dense")


def test_contracts_reject_blank_queries_but_preserve_empty_native_sentences() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        RetrievalQuery(text="   ")

    assert Sentence(index=0, text="").text == ""


def test_ranked_chunk_serializes_backend_independent_scores() -> None:
    result = RankedParagraphChunk(
        chunk=make_chunk(),
        rank=1,
        score=0.75,
        source_scores={"bm25": 2.5, "dense": 0.8},
        source_ranks={"bm25": 1, "dense": 2},
    )
    restored = RankedParagraphChunk.model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.source_scores == {"bm25": 2.5, "dense": 0.8}
    assert restored.source_ranks == {"bm25": 1, "dense": 2}


@pytest.mark.parametrize("score", [math.inf, -math.inf, math.nan])
def test_ranked_chunk_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        RankedParagraphChunk(chunk=make_chunk(), rank=1, score=score)


def test_ranked_chunk_rejects_non_finite_source_scores() -> None:
    with pytest.raises(ValidationError, match="finite"):
        RankedParagraphChunk(
            chunk=make_chunk(),
            rank=1,
            score=1.0,
            source_scores={"dense": math.nan},
        )


def test_ranked_chunk_rejects_invalid_source_ranks() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        RankedParagraphChunk(
            chunk=make_chunk(),
            rank=1,
            score=1.0,
            source_ranks={"dense": 0},
        )
