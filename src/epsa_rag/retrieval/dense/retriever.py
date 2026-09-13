"""Query embedding plus exact FAISS search."""

from __future__ import annotations

from epsa_rag.core.exceptions import ConfigurationError
from epsa_rag.core.models import RetrievalQuery
from epsa_rag.retrieval.dense.index import FaissFlatIPIndex
from epsa_rag.retrieval.interfaces import EmbeddingProvider
from epsa_rag.retrieval.models import BackendHit


class DenseRetriever:
    """Dense retrieval adapter that keeps embedding and index concerns separate."""

    def __init__(self, index: FaissFlatIPIndex, embedding_provider: EmbeddingProvider) -> None:
        if embedding_provider.model != index.config.embedding_model:
            raise ConfigurationError("embedding provider model does not match dense index")
        if embedding_provider.dimensions != index.config.dimensions:
            raise ConfigurationError("embedding provider dimensions do not match dense index")
        self.index = index
        self.embedding_provider = embedding_provider

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return self.index.chunk_ids

    def search(self, query: RetrievalQuery, *, top_k: int) -> tuple[BackendHit, ...]:
        """Embed the exact query text and search the dense index."""

        return self.index.search_vector(self.embedding_provider.embed_query(query), top_k=top_k)
