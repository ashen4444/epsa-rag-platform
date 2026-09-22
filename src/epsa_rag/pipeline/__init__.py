"""Shared orchestration contracts for baseline and EPSA pipelines."""

from epsa_rag.pipeline.answering import OpenAIFinalAnswerGenerator
from epsa_rag.pipeline.baseline_pipeline import FixedBaselinePipeline
from epsa_rag.pipeline.config import FinalAnswerConfig, FixedBaselineConfig
from epsa_rag.pipeline.context import render_full_paragraph_context
from epsa_rag.pipeline.merge import merge_retrieval_hops
from epsa_rag.pipeline.models import (
    AnswerGenerationRequest,
    FinalAnswer,
    FinalAnswerPayload,
    FixedBaselineTrace,
    HopMergeDiagnostics,
    HopMergeResult,
    LLMTokenUsage,
    MergedChunkProvenance,
    RenderedContext,
    RetrievalOccurrence,
)
from epsa_rag.pipeline.protocols import FinalAnswerGeneratorProtocol, HybridRetrieverProtocol

__all__ = [
    "AnswerGenerationRequest",
    "FinalAnswer",
    "FinalAnswerConfig",
    "FinalAnswerGeneratorProtocol",
    "FinalAnswerPayload",
    "FixedBaselineConfig",
    "FixedBaselinePipeline",
    "FixedBaselineTrace",
    "HopMergeDiagnostics",
    "HopMergeResult",
    "HybridRetrieverProtocol",
    "LLMTokenUsage",
    "MergedChunkProvenance",
    "OpenAIFinalAnswerGenerator",
    "RenderedContext",
    "RetrievalOccurrence",
    "merge_retrieval_hops",
    "render_full_paragraph_context",
]
