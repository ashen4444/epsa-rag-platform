"""Permanent fixed one-hop RAG baseline orchestration."""

from __future__ import annotations

from time import perf_counter

from epsa_rag.core.models import RetrievalQuery
from epsa_rag.pipeline.config import FixedBaselineConfig
from epsa_rag.pipeline.context import render_full_paragraph_context
from epsa_rag.pipeline.merge import merge_retrieval_hops
from epsa_rag.pipeline.models import AnswerGenerationRequest, FixedBaselineTrace
from epsa_rag.pipeline.protocols import FinalAnswerGeneratorProtocol, HybridRetrieverProtocol


class FixedBaselinePipeline:
    """Retrieve once and send the complete ranked paragraph context for answering."""

    def __init__(
        self,
        *,
        retriever: HybridRetrieverProtocol,
        answer_generator: FinalAnswerGeneratorProtocol,
        config: FixedBaselineConfig,
    ) -> None:
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.config = config

    def run(self, *, question_id: str, question: str) -> FixedBaselineTrace:
        """Execute one inspectable fixed-depth baseline question."""

        started = perf_counter()
        query = RetrievalQuery(text=question, question_id=question_id)
        hop1 = self.retriever.retrieve(query, top_k=self.config.top_k)
        merged = merge_retrieval_hops(hop1)
        context = render_full_paragraph_context(merged.results)
        request = AnswerGenerationRequest(
            question_id=question_id,
            question=question,
            context=context,
        )
        answer = self.answer_generator.generate(request)
        return FixedBaselineTrace(
            question_id=question_id,
            top_k=self.config.top_k,
            hop1=hop1,
            merged_retrieval=merged,
            final_answer_request=request,
            final_answer=answer,
            latency_ms=(perf_counter() - started) * 1000,
        )
