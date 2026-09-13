from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, RetrievalQuery, Sentence
from epsa_rag.retrieval.models import BackendHit, FusedHit, RetrievalResult


def chunk() -> ParagraphChunk:
    return ParagraphChunk(
        chunk_id="chunk:a",
        title="Alpha",
        paragraph_text="Alpha text.",
        sentences=(Sentence(index=0, text="Alpha text."),),
    )


def test_backend_and_fused_hits_are_serializable() -> None:
    backend = BackendHit(chunk_id="chunk:a", rank=1, score=2.5)
    fused = FusedHit(
        chunk_id="chunk:a",
        score=0.03,
        source_scores={"bm25": backend.score},
        source_ranks={"bm25": backend.rank},
    )

    assert BackendHit.model_validate_json(backend.model_dump_json()) == backend
    assert FusedHit.model_validate_json(fused.model_dump_json()) == fused


@pytest.mark.parametrize("model", [BackendHit, FusedHit])
def test_hits_reject_non_finite_scores(model: type[BackendHit] | type[FusedHit]) -> None:
    kwargs: dict[str, object] = {"chunk_id": "chunk:a", "score": math.nan}
    if model is BackendHit:
        kwargs["rank"] = 1
    else:
        kwargs.update(source_scores={}, source_ranks={})
    with pytest.raises(ValidationError, match="finite"):
        model(**kwargs)


def test_retrieval_result_wraps_canonical_ranked_chunks() -> None:
    query = RetrievalQuery(text="Alpha?")
    ranked = RankedParagraphChunk(chunk=chunk(), rank=1, score=0.5)
    result = RetrievalResult(
        query=query,
        retriever_version="hybrid-retriever-v1",
        results=(ranked,),
    )

    assert RetrievalResult.model_validate_json(result.model_dump_json()) == result
