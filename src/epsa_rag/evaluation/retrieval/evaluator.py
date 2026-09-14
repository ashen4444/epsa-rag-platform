"""Serial benchmark runner using canonical queries and the existing instrumentation boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from time import perf_counter
from typing import Protocol

from pydantic import JsonValue

from epsa_rag.core.exceptions import ContractError
from epsa_rag.core.models import RetrievalQuery
from epsa_rag.data.models import BenchmarkExample
from epsa_rag.evaluation.retrieval.metrics import mean_metrics, percentile, retrieval_metrics
from epsa_rag.evaluation.retrieval.models import (
    QuestionTrace,
    RunMetadata,
    RunSummary,
    utc_now,
)
from epsa_rag.instrumentation.context import TraceContext
from epsa_rag.instrumentation.events import InstrumentationEvent
from epsa_rag.instrumentation.sinks import InstrumentationSink
from epsa_rag.retrieval.corpus import FrozenCorpus
from epsa_rag.retrieval.dense.query_cache import QueryEmbeddingObservationBuffer
from epsa_rag.retrieval.models import RetrievalResult


class CanonicalRetriever(Protocol):
    def retrieve(self, query: RetrievalQuery, *, top_k: int | None = None) -> RetrievalResult: ...


def _validate_result(
    result: RetrievalResult,
    query: RetrievalQuery,
    corpus: FrozenCorpus,
    top_k: int,
    expected_version: str,
) -> None:
    if result.query != query or result.retriever_version != expected_version:
        raise ContractError("retrieval result query/version does not match the run")
    if len(result.results) > top_k:
        raise ContractError("retrieval returned more than requested top_k")
    seen: set[str] = set()
    for rank, hit in enumerate(result.results, 1):
        if hit.rank != rank or hit.chunk.chunk_id in seen:
            raise ContractError("retrieval ranks must be contiguous and chunks unique")
        if corpus.by_id.get(hit.chunk.chunk_id) != hit.chunk:
            raise ContractError("retrieval returned an unknown or modified corpus chunk")
        seen.add(hit.chunk.chunk_id)


def evaluate(
    *,
    examples: Sequence[BenchmarkExample],
    corpus: FrozenCorpus,
    retriever: CanonicalRetriever,
    metadata: RunMetadata,
    sink: InstrumentationSink,
    clock: Callable[[], float] = perf_counter,
    wall_clock: Callable[[], datetime] = utc_now,
    progress: Callable[[int, int], None] | None = None,
    embedding_observations: QueryEmbeddingObservationBuffer | None = None,
) -> RunSummary:
    """Fail fast with a structured failed run; never silently drop a failed question.

    Warmups are explicit and excluded from timing/quality. Query timing includes the entire
    retriever call (including a live embedding or cache lookup), but excludes validation and event
    writes. Retrieval-core timing subtracts the observed query-embedding boundary. Run throughput
    additionally includes scoring and per-question event delivery.
    """

    config = metadata.configuration
    if tuple(item.inference.question_id for item in examples) != metadata.question_ids:
        raise ContractError("run question order does not match provenance")
    if not examples or config.warmup_questions > len(examples):
        raise ContractError("benchmark must be nonempty and contain enough warmup questions")
    if any(not item.evaluation.supporting_facts for item in examples):
        raise ContractError("every evaluated question requires gold documents")
    run_context = TraceContext.start(run_id=metadata.run_id)

    def emit(kind: str, payload: dict[str, JsonValue], context: TraceContext = run_context) -> None:
        sink.emit(
            InstrumentationEvent(
                context=context,
                event_type=kind,
                source="retrieval-evaluation",
                source_version=config.evaluator_version,
                payload=payload,
            )
        )

    started_at = wall_clock()
    emit("evaluation.run.started", metadata.model_dump(mode="json"))
    traces: list[QuestionTrace] = []
    error_type: str | None = None
    warmup_completed = 0
    top_k = config.retriever.fusion.result_k
    expected_version = (
        config.retriever.retriever_version
        if config.mode == "hybrid"
        else getattr(config.retriever, config.mode).index_version
    )
    try:
        for example in examples[: config.warmup_questions]:
            query = RetrievalQuery(
                text=example.inference.text, question_id=example.inference.question_id
            )
            warmup_result = retriever.retrieve(query, top_k=top_k)
            if embedding_observations is not None:
                embedding_observations.take(query)
            _validate_result(warmup_result, query, corpus, top_k, expected_version)
            warmup_completed += 1
    except Exception as error:
        error_type = type(error).__name__
    loop_start = clock()
    if error_type is None:
        for example in examples:
            query = RetrievalQuery(
                text=example.inference.text, question_id=example.inference.question_id
            )
            gold = frozenset(label.chunk_id for label in example.evaluation.supporting_facts)
            result: RetrievalResult | None = None
            question_error = None
            request_start = clock()
            try:
                result = retriever.retrieve(query, top_k=top_k)
            except Exception as error:
                question_error = type(error).__name__
            elapsed = clock() - request_start
            embedding = (
                embedding_observations.take(query)
                if embedding_observations is not None
                else None
            )
            elapsed_ms = elapsed * 1000
            core_latency_ms = (
                max(0.0, elapsed_ms - embedding.latency_ms)
                if embedding is not None
                else elapsed_ms if embedding_observations is None else None
            )
            if result is not None:
                try:
                    _validate_result(result, query, corpus, top_k, expected_version)
                except ContractError as error:
                    question_error = type(error).__name__
            # Invalid/failed output receives zero retrieval credit and is retained for diagnosis.
            identities = (
                tuple(hit.chunk.chunk_id for hit in result.results)
                if result is not None and question_error is None
                else ()
            )
            trace = QuestionTrace(
                question=example.inference,
                question_type=example.evaluation.question_type,
                difficulty=example.evaluation.difficulty,
                gold_supporting_facts=example.evaluation.supporting_facts,
                gold_identities=tuple(sorted(gold)),
                missing_gold_identities={
                    str(k): tuple(sorted(gold - set(identities[:k]))) for k in config.cutoffs
                },
                status="completed" if question_error is None else "failed",
                error_type=question_error,
                retrieval=result,
                query_embedding=embedding,
                latency_ms=elapsed_ms,
                retrieval_core_latency_ms=core_latency_ms,
                metrics=retrieval_metrics(identities, gold, config.cutoffs),
            )
            traces.append(trace)
            emit(
                "evaluation.question.completed",
                trace.model_dump(mode="json"),
                TraceContext.start(run_id=metadata.run_id, question_id=query.question_id),
            )
            if progress is not None:
                progress(len(traces), len(examples))
            if question_error is not None:
                error_type = question_error
                break
    elapsed_wall = clock() - loop_start
    metrics, denominators = mean_metrics([trace.metrics for trace in traces])
    latencies = [trace.latency_ms for trace in traces]
    successful = [trace.latency_ms for trace in traces if trace.status == "completed"]
    core_latencies = [
        trace.retrieval_core_latency_ms
        for trace in traces
        if trace.retrieval_core_latency_ms is not None
    ]
    retrieval_seconds = sum(latencies) / 1000
    retrieval_core_seconds = sum(core_latencies) / 1000
    summary = RunSummary(
        metadata=metadata,
        started_at=started_at,
        ended_at=wall_clock(),
        status="completed" if error_type is None else "failed",
        full_benchmark=(error_type is None and len(traces) == metadata.full_dataset_question_count),
        planned_questions=len(examples),
        completed_questions=len(successful),
        failed_questions=len(traces) - len(successful),
        unattempted_questions=len(examples) - len(traces),
        warmup_completed=warmup_completed,
        error_type=error_type,
        metrics=metrics,
        metric_denominators=denominators,
        latency_p50_ms=percentile(latencies, 0.5),
        latency_p95_ms=percentile(latencies, 0.95),
        successful_latency_p50_ms=percentile(successful, 0.5),
        successful_latency_p95_ms=percentile(successful, 0.95),
        retrieval_core_latency_p50_ms=percentile(core_latencies, 0.5),
        retrieval_core_latency_p95_ms=percentile(core_latencies, 0.95),
        retrieval_seconds=retrieval_seconds,
        retrieval_core_seconds=retrieval_core_seconds,
        evaluation_wall_seconds=elapsed_wall,
        throughput_questions_per_second=len(successful) / elapsed_wall if elapsed_wall else None,
        retrieval_throughput_questions_per_second=(
            len(successful) / retrieval_seconds if retrieval_seconds else None
        ),
        retrieval_core_throughput_questions_per_second=(
            len(successful) / retrieval_core_seconds if retrieval_core_seconds else None
        ),
    )
    if embedding_observations is not None:
        embedding_observations.require_empty()
    emit("evaluation.run.finished", summary.model_dump(mode="json"))
    return summary
