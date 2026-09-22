"""Strict, bounded two-hop EPSA RAG system orchestration."""

from __future__ import annotations

from time import perf_counter

from epsa_rag.core.models import RetrievalQuery
from epsa_rag.instrumentation import TraceContext
from epsa_rag.pipeline.config import EPSAPipelineConfig
from epsa_rag.pipeline.context import render_epsa_pruned_context
from epsa_rag.pipeline.epsa_models import EPSAPassResult, EPSAPipelineTrace, EPSATerminalState
from epsa_rag.pipeline.merge import merge_retrieval_hops
from epsa_rag.pipeline.models import AnswerGenerationRequest
from epsa_rag.pipeline.protocols import (
    EPSAControllerProtocol,
    FinalAnswerGeneratorProtocol,
    HybridRetrieverProtocol,
)


class EPSAPipeline:
    """Run EPSA, at most one rule-based retrieval hop, and final answering."""

    def __init__(
        self,
        *,
        retriever: HybridRetrieverProtocol,
        controller: EPSAControllerProtocol,
        answer_generator: FinalAnswerGeneratorProtocol,
        config: EPSAPipelineConfig,
    ) -> None:
        self.retriever = retriever
        self.controller = controller
        self.answer_generator = answer_generator
        self.config = config

    def run(
        self,
        *,
        question_id: str,
        question: str,
        trace_context: TraceContext | None = None,
    ) -> EPSAPipelineTrace:
        """Execute strict EPSA using the original question for both evidence passes."""

        started = perf_counter()
        original_query = RetrievalQuery(text=question, question_id=question_id)
        hop1 = self.retriever.retrieve(original_query, top_k=self.config.top_k)
        pass1 = self.controller.run_pass(
            question=question,
            ranked_chunks=hop1.results,
            pass_number=1,
            generate_next_hop=True,
            trace_context=trace_context,
        )
        proposal = pass1.next_hop_query
        query_text = (
            proposal.query
            if not pass1.sufficiency_decision.sufficient and proposal is not None
            else None
        )
        hop2 = (
            self.retriever.retrieve(
                RetrievalQuery(text=query_text, question_id=question_id),
                top_k=self.config.top_k,
            )
            if query_text is not None
            else None
        )
        merged = merge_retrieval_hops(hop1, hop2)
        pass2 = (
            self.controller.run_pass(
                question=question,
                ranked_chunks=merged.results,
                pass_number=2,
                generate_next_hop=False,
                trace_context=trace_context,
            )
            if hop2 is not None
            else None
        )
        final_pass = pass2 or pass1
        terminal_state = _terminal_state(pass1.sufficiency_decision.sufficient, pass2)
        final_context = render_epsa_pruned_context(final_pass.pruned_context)
        answer_request = AnswerGenerationRequest(
            question_id=question_id,
            question=question,
            context=final_context,
        )
        answer = self.answer_generator.generate(answer_request)
        return EPSAPipelineTrace(
            question_id=question_id,
            top_k=self.config.top_k,
            hop1=hop1,
            epsa_pass1=pass1,
            hop2=hop2,
            merged_retrieval=merged,
            epsa_pass2=pass2,
            terminal_state=terminal_state,
            final_answer_request=answer_request,
            final_answer=answer,
            latency_ms=(perf_counter() - started) * 1000,
        )


def _terminal_state(
    first_sufficient: bool, second: EPSAPassResult | None
) -> EPSATerminalState:
    if first_sufficient:
        return EPSATerminalState.SUFFICIENT_HOP1
    if second is None:
        return EPSATerminalState.INSUFFICIENT_NO_QUERY
    if second.sufficiency_decision.sufficient:
        return EPSATerminalState.SUFFICIENT_HOP2
    return EPSATerminalState.INSUFFICIENT_AFTER_HOP2
