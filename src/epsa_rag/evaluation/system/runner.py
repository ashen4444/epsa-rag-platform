"""Paired, resumable execution of fixed, adaptive, and EPSA RAG conditions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from epsa_rag.data.models import BenchmarkExample
from epsa_rag.evaluation.system.metrics import (
    answer_exact_match,
    answer_token_f1,
    mean,
    percentile,
)
from epsa_rag.evaluation.system.models import (
    AnswerQuality,
    ConditionSummary,
    PipelineTrace,
    RetrievalQuality,
    SystemCondition,
    SystemEfficiency,
    SystemEvaluationConfig,
    SystemQuestionTrace,
    SystemRunMetadata,
    SystemRunSummary,
    SystemVariant,
    utc_now,
)
from epsa_rag.evaluation.system.storage import ResumableSystemRunStore
from epsa_rag.evaluation.system.token_accounting import TokenCounter
from epsa_rag.pipeline.context import render_full_paragraph_context
from epsa_rag.pipeline.epsa_models import EPSAPipelineTrace, EPSATerminalState
from epsa_rag.pipeline.models import AdaptiveBaselineTrace, FixedBaselineTrace, LLMTokenUsage
from epsa_rag.retrieval.dense.query_cache import QueryEmbeddingObservation


class EvaluationPipeline(Protocol):
    def run(self, *, question_id: str, question: str) -> PipelineTrace: ...

    def take_query_embedding_observations(self) -> tuple[QueryEmbeddingObservation, ...]: ...


PipelineFactory = Callable[[SystemCondition], EvaluationPipeline]
ProgressCallback = Callable[[int, int], None]


def evaluate_systems(
    *,
    examples: Sequence[BenchmarkExample],
    metadata: SystemRunMetadata,
    store: ResumableSystemRunStore,
    pipeline_factory: PipelineFactory,
    token_counter: TokenCounter,
    progress: ProgressCallback | None = None,
) -> SystemRunSummary:
    """Run every question-condition pair, resuming already persisted traces."""

    started_at = utc_now()
    conditions = metadata.configuration.conditions()
    expected_ids = tuple(item.inference.question_id for item in examples)
    if expected_ids != metadata.question_ids:
        raise ValueError("evaluation examples do not match run metadata order")
    pipelines = {condition.key: pipeline_factory(condition) for condition in conditions}
    traces: list[SystemQuestionTrace] = []
    total = len(examples) * len(conditions)
    done = 0
    for example in examples:
        for condition in conditions:
            existing = store.load_trace(condition, example.inference.question_id)
            if existing is not None:
                trace = existing
            else:
                trace = _evaluate_one(
                    example,
                    condition,
                    pipelines[condition.key],
                    token_counter,
                )
                store.write_trace(trace)
            traces.append(trace)
            done += 1
            if progress is not None:
                progress(done, total)
    _validate_paired_hop1(tuple(traces))
    summary = _summarize(metadata, tuple(traces), started_at)
    store.finalize(summary)
    return summary


def _evaluate_one(
    example: BenchmarkExample,
    condition: SystemCondition,
    pipeline: EvaluationPipeline,
    token_counter: TokenCounter,
) -> SystemQuestionTrace:
    try:
        trace = pipeline.run(
            question_id=example.inference.question_id,
            question=example.inference.text,
        )
        observations = pipeline.take_query_embedding_observations()
        answer = trace.final_answer.answer
        quality = AnswerQuality(
            exact_match=answer_exact_match(answer, example.evaluation.answer),
            token_f1=answer_token_f1(answer, example.evaluation.answer),
            generation_failed=False,
        )
        efficiency = _efficiency(trace, token_counter)
        retrieval_quality = _retrieval_quality(trace, example)
        return SystemQuestionTrace(
            condition=condition,
            question=example.inference,
            evaluation_labels=example.evaluation,
            status="completed",
            pipeline_trace=trace,
            query_embeddings=observations,
            generated_answer=answer,
            quality=quality,
            efficiency=efficiency,
            retrieval_quality=retrieval_quality,
        )
    except Exception as error:
        pipeline.take_query_embedding_observations()
        return SystemQuestionTrace(
            condition=condition,
            question=example.inference,
            evaluation_labels=example.evaluation,
            status="failed",
            error_type=type(error).__name__,
            error_message=str(error),
        )


def _efficiency(trace: PipelineTrace, counter: TokenCounter) -> SystemEfficiency:
    final_tokens = counter.count(trace.final_answer_request.context.text)
    candidate_text = render_full_paragraph_context(trace.merged_retrieval.results).text
    candidate_tokens = counter.count(candidate_text)
    usages = _llm_usages(trace)
    pruning = (
        (candidate_tokens - final_tokens) / candidate_tokens * 100
        if isinstance(trace, EPSAPipelineTrace) and candidate_tokens > 0
        else None
    )
    return SystemEfficiency(
        final_context_tokens=final_tokens,
        candidate_context_tokens=candidate_tokens,
        pruning_reduction_percent=pruning,
        llm_input_tokens=sum(item.input_tokens for item in usages),
        llm_output_tokens=sum(item.output_tokens for item in usages),
        llm_cached_input_tokens=sum(item.cached_input_tokens for item in usages),
        hop2_activated=(
            trace.hop2 is not None if not isinstance(trace, FixedBaselineTrace) else False
        ),
        insufficient_no_query=_insufficient_no_query(trace),
        latency_ms=trace.latency_ms,
    )


def _llm_usages(trace: PipelineTrace) -> tuple[LLMTokenUsage, ...]:
    if isinstance(trace, AdaptiveBaselineTrace):
        return (trace.controller_decision.usage, trace.final_answer.usage)
    return (trace.final_answer.usage,)


def _insufficient_no_query(trace: PipelineTrace) -> bool:
    if isinstance(trace, AdaptiveBaselineTrace):
        output = trace.controller_decision.output
        return not output.sufficient and output.next_hop_query is None
    return (
        isinstance(trace, EPSAPipelineTrace)
        and trace.terminal_state is EPSATerminalState.INSUFFICIENT_NO_QUERY
    )


def _retrieval_quality(
    trace: PipelineTrace, example: BenchmarkExample
) -> RetrievalQuality:
    gold = tuple(dict.fromkeys(item.chunk_id for item in example.evaluation.supporting_facts))
    hop1_ids = {item.chunk.chunk_id for item in trace.hop1.results}
    final_ids = {item.chunk.chunk_id for item in trace.merged_retrieval.results}
    hop1_found = tuple(chunk_id for chunk_id in gold if chunk_id in hop1_ids)
    final_found = tuple(chunk_id for chunk_id in gold if chunk_id in final_ids)
    return RetrievalQuality(
        gold_chunk_ids=gold,
        hop1_found_chunk_ids=hop1_found,
        final_found_chunk_ids=final_found,
        all_gold_available_hop1=len(hop1_found) == len(gold),
        all_gold_available_final=len(final_found) == len(gold),
        hop2_evidence_gain_count=len(final_found) - len(hop1_found),
    )


def _summarize(
    metadata: SystemRunMetadata,
    traces: tuple[SystemQuestionTrace, ...],
    started_at: object,
) -> SystemRunSummary:
    summaries = tuple(
        _condition_summary(condition, traces)
        for condition in metadata.configuration.conditions()
    )
    failed = sum(trace.status == "failed" for trace in traces)
    return SystemRunSummary(
        metadata=metadata,
        started_at=started_at,
        ended_at=utc_now(),
        status="completed_with_failures" if failed else "completed",
        full_benchmark=(
            metadata.configuration.question_limit is None
            and len(metadata.question_ids) == metadata.full_dataset_question_count
        ),
        completed_traces=len(traces) - failed,
        failed_traces=failed,
        condition_summaries=summaries,
        paired_context_reductions=_paired_reductions(traces, metadata.configuration),
    )


def _condition_summary(
    condition: SystemCondition, traces: tuple[SystemQuestionTrace, ...]
) -> ConditionSummary:
    selected = tuple(trace for trace in traces if trace.condition == condition)
    completed = tuple(trace for trace in selected if trace.status == "completed")
    qualities = tuple(trace.quality for trace in completed if trace.quality is not None)
    efficiencies = tuple(
        trace.efficiency for trace in completed if trace.efficiency is not None
    )
    retrieval = tuple(
        trace.retrieval_quality
        for trace in completed
        if trace.retrieval_quality is not None
    )
    metrics = {
        "exact_match": mean(item.exact_match for item in qualities),
        "token_f1": mean(item.token_f1 for item in qualities),
        "answer_failure_rate": (len(selected) - len(completed)) / len(selected),
        "hop2_activation_rate": mean(float(item.hop2_activated) for item in efficiencies),
        "insufficient_no_query_rate": mean(
            float(item.insufficient_no_query) for item in efficiencies
        ),
        "all_gold_available_hop1_rate": mean(
            float(item.all_gold_available_hop1) for item in retrieval
        ),
        "all_gold_available_final_rate": mean(
            float(item.all_gold_available_final) for item in retrieval
        ),
        "mean_hop2_evidence_gain_count": mean(
            float(item.hop2_evidence_gain_count) for item in retrieval
        ),
        "mean_final_context_tokens": mean(
            float(item.final_context_tokens) for item in efficiencies
        ),
        "mean_candidate_context_tokens": mean(
            float(item.candidate_context_tokens) for item in efficiencies
        ),
        "mean_llm_input_tokens": mean(float(item.llm_input_tokens) for item in efficiencies),
        "mean_llm_output_tokens": mean(float(item.llm_output_tokens) for item in efficiencies),
        "latency_p50_ms": percentile((item.latency_ms for item in efficiencies), 0.50),
        "latency_p95_ms": percentile((item.latency_ms for item in efficiencies), 0.95),
    }
    pruning = tuple(
        item.pruning_reduction_percent
        for item in efficiencies
        if item.pruning_reduction_percent is not None
    )
    if pruning:
        metrics["mean_pruning_reduction_percent"] = mean(pruning)
    return ConditionSummary(
        condition=condition,
        planned_questions=len(selected),
        completed_questions=len(completed),
        failed_questions=len(selected) - len(completed),
        metrics=metrics,
    )


def _paired_reductions(
    traces: tuple[SystemQuestionTrace, ...], config: SystemEvaluationConfig
) -> dict[str, float]:
    index = {
        (trace.question.question_id, trace.condition.system, trace.condition.top_k): trace
        for trace in traces
        if trace.status == "completed" and trace.efficiency is not None
    }
    output: dict[str, float] = {}
    for top_k in config.top_ks:
        for baseline in (SystemVariant.FIXED, SystemVariant.ADAPTIVE):
            reductions = []
            for question_id in {trace.question.question_id for trace in traces}:
                base = index.get((question_id, baseline, top_k))
                epsa = index.get((question_id, SystemVariant.EPSA, top_k))
                if (
                    base is None
                    or epsa is None
                    or base.efficiency is None
                    or epsa.efficiency is None
                ):
                    continue
                denominator = base.efficiency.final_context_tokens
                if denominator > 0:
                    reductions.append(
                        (denominator - epsa.efficiency.final_context_tokens)
                        / denominator
                        * 100
                    )
            output[f"epsa_vs_{baseline.value}_top_{top_k}"] = mean(reductions)
    return output


def _validate_paired_hop1(traces: tuple[SystemQuestionTrace, ...]) -> None:
    """Prove all completed systems saw identical Hop-1 results for each question/K."""

    expected: dict[tuple[str, int], object] = {}
    for trace in traces:
        if trace.pipeline_trace is None:
            continue
        key = (trace.question.question_id, trace.condition.top_k)
        hop1 = trace.pipeline_trace.hop1
        previous = expected.setdefault(key, hop1)
        if previous != hop1:
            raise ValueError("paired systems produced different Hop-1 retrieval results")
