"""Shared orchestration contracts for baseline and EPSA pipelines."""

from epsa_rag.pipeline.merge import merge_retrieval_hops
from epsa_rag.pipeline.models import (
    HopMergeDiagnostics,
    HopMergeResult,
    MergedChunkProvenance,
    RetrievalOccurrence,
)
from epsa_rag.pipeline.protocols import HybridRetrieverProtocol

__all__ = [
    "HopMergeDiagnostics",
    "HopMergeResult",
    "HybridRetrieverProtocol",
    "MergedChunkProvenance",
    "RetrievalOccurrence",
    "merge_retrieval_hops",
]
