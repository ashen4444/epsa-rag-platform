from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from epsa_rag.core.exceptions import ConfigurationError, ContractError
from epsa_rag.core.models import ParagraphChunk, RetrievalQuery, Sentence
from epsa_rag.retrieval.config import HybridRetrieverConfig
from epsa_rag.retrieval.corpus import FrozenCorpus
from epsa_rag.retrieval.hybrid_retriever import HybridRetriever
from epsa_rag.retrieval.models import BackendHit


def chunk(chunk_id: str, title: str) -> ParagraphChunk:
    text = f"{title} text."
    return ParagraphChunk(
        chunk_id=chunk_id,
        title=title,
        paragraph_text=text,
        sentences=(Sentence(index=0, text=text),),
    )


class FakeBackend:
    def __init__(self, chunk_ids: tuple[str, ...], hits: tuple[BackendHit, ...]) -> None:
        self._chunk_ids = chunk_ids
        self.hits = hits
        self.requested_top_k: int | None = None

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return self._chunk_ids

    def search(self, query: RetrievalQuery, *, top_k: int) -> tuple[BackendHit, ...]:
        self.requested_top_k = top_k
        return self.hits[:top_k]


def corpus() -> FrozenCorpus:
    chunks = (chunk("chunk:a", "Alpha"), chunk("chunk:b", "Beta"))
    return cast(
        FrozenCorpus,
        SimpleNamespace(chunks=chunks, by_id={item.chunk_id: item for item in chunks}),
    )


def test_hybrid_retriever_returns_canonical_chunks_and_branch_provenance() -> None:
    loaded_corpus = corpus()
    ids = tuple(item.chunk_id for item in loaded_corpus.chunks)
    bm25 = FakeBackend(ids, (BackendHit(chunk_id="chunk:a", rank=1, score=4.0),))
    dense = FakeBackend(
        ids,
        (
            BackendHit(chunk_id="chunk:b", rank=1, score=0.9),
            BackendHit(chunk_id="chunk:a", rank=2, score=0.8),
        ),
    )
    config = HybridRetrieverConfig()
    retriever = HybridRetriever(
        corpus=loaded_corpus, bm25=bm25, dense=dense, config=config
    )

    result = retriever.retrieve(RetrievalQuery(text="Alpha?"), top_k=2)

    assert [item.chunk.chunk_id for item in result.results] == ["chunk:a", "chunk:b"]
    assert result.results[0].source_ranks == {"bm25": 1, "dense": 2}
    assert result.results[0].source_scores == {"bm25": 4.0, "dense": 0.8}
    assert bm25.requested_top_k == config.bm25.candidate_k
    assert dense.requested_top_k == config.dense.candidate_k


def test_hybrid_retriever_rejects_mismatched_indexes_and_unknown_hits() -> None:
    loaded_corpus = corpus()
    ids = tuple(item.chunk_id for item in loaded_corpus.chunks)
    valid = FakeBackend(ids, ())
    invalid = FakeBackend(tuple(reversed(ids)), ())
    with pytest.raises(ConfigurationError, match="BM25"):
        HybridRetriever(
            corpus=loaded_corpus,
            bm25=invalid,
            dense=valid,
            config=HybridRetrieverConfig(),
        )
    with pytest.raises(ConfigurationError, match="dense"):
        HybridRetriever(
            corpus=loaded_corpus,
            bm25=valid,
            dense=invalid,
            config=HybridRetrieverConfig(),
        )

    unknown = FakeBackend(
        ids, (BackendHit(chunk_id="chunk:unknown", rank=1, score=1.0),)
    )
    retriever = HybridRetriever(
        corpus=loaded_corpus,
        bm25=valid,
        dense=unknown,
        config=HybridRetrieverConfig(),
    )
    with pytest.raises(ContractError, match="unknown"):
        retriever.retrieve(RetrievalQuery(text="question"))
    with pytest.raises(ValueError, match="positive"):
        retriever.retrieve(RetrievalQuery(text="question"), top_k=0)
