"""Shared orchestration contracts for baseline and EPSA pipelines."""

from epsa_rag.pipeline.adaptive_controller import OpenAIAdaptiveRetrievalController
from epsa_rag.pipeline.answering import OpenAIFinalAnswerGenerator
from epsa_rag.pipeline.baseline_pipeline import AdaptiveBaselinePipeline, FixedBaselinePipeline
from epsa_rag.pipeline.config import (
    AdaptiveBaselineConfig,
    AdaptiveControllerConfig,
    FinalAnswerConfig,
    FixedBaselineConfig,
)
from epsa_rag.pipeline.context import render_full_paragraph_context
from epsa_rag.pipeline.merge import merge_retrieval_hops
from epsa_rag.pipeline.models import (
    AdaptiveBaselineTrace,
    AdaptiveControlDecision,
    AdaptiveControlOutput,
    AdaptiveControlRequest,
    AdaptiveReasonCode,
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
from epsa_rag.pipeline.protocols import (
    AdaptiveRetrievalControllerProtocol,
    FinalAnswerGeneratorProtocol,
    HybridRetrieverProtocol,
)

__all__ = [
    "AdaptiveBaselineConfig",
    "AdaptiveBaselinePipeline",
    "AdaptiveBaselineTrace",
    "AdaptiveControlDecision",
    "AdaptiveControlOutput",
    "AdaptiveControlRequest",
    "AdaptiveControllerConfig",
    "AdaptiveReasonCode",
    "AdaptiveRetrievalControllerProtocol",
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
    "OpenAIAdaptiveRetrievalController",
    "OpenAIFinalAnswerGenerator",
    "RenderedContext",
    "RetrievalOccurrence",
    "merge_retrieval_hops",
    "render_full_paragraph_context",
]
