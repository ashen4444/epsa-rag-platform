"""Explicit, fingerprintable configuration for Phase 3 retrieval."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from epsa_rag.core.config import ConfigModel
from epsa_rag.core.ids import Identifier


class BM25Config(ConfigModel):
    """Deterministic Okapi BM25 configuration."""

    index_version: Identifier = "bm25-v1"
    backend: Literal["epsa-okapi-bm25"] = "epsa-okapi-bm25"
    tokenizer: Literal["unicode-word-casefold-v1"] = "unicode-word-casefold-v1"
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    index_title: Literal[True] = True
    index_paragraph_text: Literal[True] = True
    candidate_k: int = Field(default=100, ge=1)


class DenseConfig(ConfigModel):
    """OpenAI embedding and exact FAISS index configuration."""

    index_version: Identifier = "dense-openai-small-faiss-flatip-v1"
    provider: Literal["openai"] = "openai"
    embedding_model: Literal["text-embedding-3-small"] = "text-embedding-3-small"
    dimensions: Literal[1536] = 1536
    document_template: Literal["Title: {title}\nText: {paragraph_text}"] = (
        "Title: {title}\nText: {paragraph_text}"
    )
    query_template: Literal["{query_text}"] = "{query_text}"
    encoding_format: Literal["float"] = "float"
    batch_size: int = Field(default=128, ge=1, le=2048)
    backend: Literal["faiss.IndexFlatIP"] = "faiss.IndexFlatIP"
    metric: Literal["cosine-via-l2-normalized-inner-product"] = (
        "cosine-via-l2-normalized-inner-product"
    )
    vector_dtype: Literal["float32"] = "float32"
    candidate_k: int = Field(default=100, ge=1)


class RRFConfig(ConfigModel):
    """Reciprocal Rank Fusion configuration."""

    implementation_version: Identifier = "rrf-v1"
    rank_constant: int = Field(default=60, ge=1)
    bm25_weight: float = Field(default=1.0, gt=0)
    dense_weight: float = Field(default=1.0, gt=0)
    result_k: int = Field(default=10, ge=1)


class HybridRetrieverConfig(ConfigModel):
    """Complete research-critical configuration for the shared retriever."""

    retriever_version: Identifier = "hybrid-retriever-v1"
    bm25: BM25Config = Field(default_factory=BM25Config)
    dense: DenseConfig = Field(default_factory=DenseConfig)
    fusion: RRFConfig = Field(default_factory=RRFConfig)

    @model_validator(mode="after")
    def validate_candidate_depths(self) -> HybridRetrieverConfig:
        """Ensure both branches can supply the requested fused result depth."""

        if self.bm25.candidate_k < self.fusion.result_k:
            raise ValueError("BM25 candidate_k must be at least fusion result_k")
        if self.dense.candidate_k < self.fusion.result_k:
            raise ValueError("dense candidate_k must be at least fusion result_k")
        return self
