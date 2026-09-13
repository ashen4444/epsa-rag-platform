"""Shared BM25 plus dense retriever with canonical fused output."""

from __future__ import annotations

from epsa_rag.core.exceptions import ConfigurationError, ContractError
from epsa_rag.core.models import RankedParagraphChunk, RetrievalQuery
from epsa_rag.retrieval.config import HybridRetrieverConfig
from epsa_rag.retrieval.corpus import FrozenCorpus
from epsa_rag.retrieval.fusion.rrf import reciprocal_rank_fusion
from epsa_rag.retrieval.interfaces import RetrievalBackend
from epsa_rag.retrieval.models import RetrievalResult


class HybridRetriever:
    """Backend-neutral orchestration for the shared project retriever."""

    def __init__(
        self,
        *,
        corpus: FrozenCorpus,
        bm25: RetrievalBackend,
        dense: RetrievalBackend,
        config: HybridRetrieverConfig,
    ) -> None:
        expected_ids = tuple(chunk.chunk_id for chunk in corpus.chunks)
        if bm25.chunk_ids != expected_ids:
            raise ConfigurationError("BM25 index does not match the loaded corpus order")
        if dense.chunk_ids != expected_ids:
            raise ConfigurationError("dense index does not match the loaded corpus order")
        self.corpus = corpus
        self.bm25 = bm25
        self.dense = dense
        self.config = config

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Execute both branches, fuse them, and resolve canonical corpus chunks."""

        result_k = self.config.fusion.result_k if top_k is None else top_k
        if result_k < 1:
            raise ValueError("top_k must be positive")
        bm25_hits = self.bm25.search(query, top_k=self.config.bm25.candidate_k)
        dense_hits = self.dense.search(query, top_k=self.config.dense.candidate_k)
        fused = reciprocal_rank_fusion(
            {"bm25": bm25_hits, "dense": dense_hits},
            weights={
                "bm25": self.config.fusion.bm25_weight,
                "dense": self.config.fusion.dense_weight,
            },
            config=self.config.fusion,
            top_k=result_k,
        )
        results: list[RankedParagraphChunk] = []
        for rank, hit in enumerate(fused, start=1):
            chunk = self.corpus.by_id.get(hit.chunk_id)
            if chunk is None:
                raise ContractError(f"retrieval hit references unknown chunk {hit.chunk_id}")
            results.append(
                RankedParagraphChunk(
                    chunk=chunk,
                    rank=rank,
                    score=hit.score,
                    source_scores=hit.source_scores,
                    source_ranks=hit.source_ranks,
                )
            )
        return RetrievalResult(
            query=query,
            retriever_version=self.config.retriever_version,
            results=tuple(results),
        )
