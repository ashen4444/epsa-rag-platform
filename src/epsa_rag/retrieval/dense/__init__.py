"""OpenAI embeddings and exact FAISS dense retrieval."""

from epsa_rag.retrieval.dense.embeddings import OpenAIEmbeddingProvider
from epsa_rag.retrieval.dense.index import FaissFlatIPIndex
from epsa_rag.retrieval.dense.retriever import DenseRetriever

__all__ = ["DenseRetriever", "FaissFlatIPIndex", "OpenAIEmbeddingProvider"]
