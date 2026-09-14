from __future__ import annotations

import pytest

from epsa_rag.core.exceptions import ContractError, EmbeddingError
from epsa_rag.core.models import RankedParagraphChunk, RetrievalQuery
from epsa_rag.evaluation.retrieval.evaluator import evaluate
from epsa_rag.evaluation.retrieval.models import QuestionTrace
from epsa_rag.instrumentation.sinks import InMemoryInstrumentationSink
from epsa_rag.retrieval.models import RetrievalResult


class FixtureRetriever:
    def __init__(self, corpus, examples, *, failure=None, invalid=None):
        self.corpus, self.examples = corpus, examples
        self.failure, self.invalid = failure, invalid
        self.calls = []

    def retrieve(self, query, *, top_k=None):
        # Fixtures deliberately return known rankings to check evaluator math and isolation.
        self.calls.append(query)
        if self.failure:
            raise self.failure("Sensitive details must not appear in an export")
        example = next(e for e in self.examples if e.inference.question_id == query.question_id)
        gold_ids = list(dict.fromkeys(f.chunk_id for f in example.evaluation.supporting_facts))
        chunks = [self.corpus.by_id[c] for c in gold_ids]
        if self.invalid == "same_title":
            chunks[0] = next(
                c
                for c in self.corpus.chunks
                if c.title == chunks[0].title and c.chunk_id != chunks[0].chunk_id
            )
        hits = tuple(
            RankedParagraphChunk(chunk=c, rank=i, score=1 / i) for i, c in enumerate(chunks, 1)
        )
        result = RetrievalResult(query=query, retriever_version="hybrid-retriever-v1", results=hits)
        if self.invalid == "query":
            result = result.model_copy(update={"query": RetrievalQuery(text="wrong")})
        if self.invalid == "rank":
            result = result.model_copy(
                update={"results": (hits[0].model_copy(update={"rank": 2}),)}
            )
        if self.invalid == "duplicate":
            result = result.model_copy(update={"results": (hits[0], hits[0])})
        if self.invalid == "too_many":
            result = result.model_copy(update={"results": hits * 10})
        if self.invalid == "chunk":
            chunk = chunks[0].model_copy(update={"title": "corrupted"})
            result = result.model_copy(
                update={"results": (hits[0].model_copy(update={"chunk": chunk}),)}
            )
        return result


def test_success_is_observable_and_gold_never_enters_query(evaluation_data, evaluation_metadata):
    corpus, benchmark, _ = evaluation_data
    retriever = FixtureRetriever(corpus, benchmark.examples)
    sink = InMemoryInstrumentationSink()
    ticks = iter([0, 1, 1.1, 2, 2.2, 3, 3.3, 4])
    summary = evaluate(
        examples=benchmark.examples,
        corpus=corpus,
        retriever=retriever,
        metadata=evaluation_metadata,
        sink=sink,
        clock=lambda: next(ticks),
    )
    assert summary.status == "completed"
    assert summary.full_benchmark
    assert summary.metrics["recall@1"] == 0.5
    assert summary.metrics["recall@10"] == 1
    assert summary.latency_p50_ms == pytest.approx(200)
    assert summary.latency_p95_ms == pytest.approx(290)
    assert summary.retrieval_seconds == pytest.approx(0.6)
    assert summary.throughput_questions_per_second == 0.75
    assert summary.retrieval_throughput_questions_per_second == pytest.approx(5)
    assert len(sink.events) == 5
    for query, example, event in zip(
        retriever.calls, benchmark.examples, sink.events[1:-1], strict=True
    ):
        assert query.model_dump() == example.inference.model_dump()
        assert "SECRET GOLD" not in query.model_dump_json()
        assert event.context.question_id == query.question_id
        assert event.context.run_id == evaluation_metadata.run_id
        assert QuestionTrace.model_validate(event.payload).retrieval is not None


def test_same_title_different_chunk_gets_no_credit(evaluation_data, evaluation_metadata):
    corpus, benchmark, _ = evaluation_data
    summary = evaluate(
        examples=benchmark.examples,
        corpus=corpus,
        retriever=FixtureRetriever(corpus, benchmark.examples, invalid="same_title"),
        metadata=evaluation_metadata,
        sink=InMemoryInstrumentationSink(),
    )
    assert summary.metrics["recall@10"] == 0.5
    assert summary.metrics["any_gold_missing@10"] == 1
    assert summary.metrics["both_supporting_documents_found@10"] == 0


@pytest.mark.parametrize("invalid", ["query", "rank", "duplicate", "too_many", "chunk"])
def test_invalid_results_fail_and_are_not_silently_excluded(
    evaluation_data, evaluation_metadata, invalid
):
    corpus, benchmark, _ = evaluation_data
    summary = evaluate(
        examples=benchmark.examples,
        corpus=corpus,
        retriever=FixtureRetriever(corpus, benchmark.examples, invalid=invalid),
        metadata=evaluation_metadata,
        sink=InMemoryInstrumentationSink(),
    )
    assert summary.status == "failed"
    assert not summary.full_benchmark
    assert summary.failed_questions == 1
    assert summary.unattempted_questions == 2
    assert summary.metrics["recall@10"] == 0
    assert summary.metric_denominators["recall@10"] == 1


def test_request_failure_redacts_exception_details(evaluation_data, evaluation_metadata):
    corpus, benchmark, _ = evaluation_data
    sink = InMemoryInstrumentationSink()
    summary = evaluate(
        examples=benchmark.examples,
        corpus=corpus,
        retriever=FixtureRetriever(corpus, benchmark.examples, failure=EmbeddingError),
        metadata=evaluation_metadata,
        sink=sink,
    )
    assert summary.error_type == "EmbeddingError"
    assert summary.successful_latency_p50_ms is None
    assert "Sensitive" not in "".join(event.model_dump_json() for event in sink.events)


@pytest.mark.parametrize("fail", [False, True])
def test_warmup_is_excluded_and_failures_are_marked(evaluation_data, evaluation_metadata, fail):
    corpus, benchmark, _ = evaluation_data
    config = evaluation_metadata.configuration.model_copy(update={"warmup_questions": 1})
    metadata = evaluation_metadata.model_copy(update={"configuration": config})
    retriever = FixtureRetriever(
        corpus, benchmark.examples, failure=EmbeddingError if fail else None
    )
    summary = evaluate(
        examples=benchmark.examples,
        corpus=corpus,
        retriever=retriever,
        metadata=metadata,
        sink=InMemoryInstrumentationSink(),
    )
    assert summary.warmup_completed == (0 if fail else 1)
    assert len(retriever.calls) == (1 if fail else 4)
    assert summary.completed_questions == (0 if fail else 3)
    if fail:
        assert summary.metrics == {}
        assert summary.unattempted_questions == 3


@pytest.mark.parametrize("problem", ["order", "empty", "warmup", "gold"])
def test_invalid_inputs_fail_before_retrieval(evaluation_data, evaluation_metadata, problem):
    corpus, benchmark, _ = evaluation_data
    examples = benchmark.examples
    metadata = evaluation_metadata
    if problem == "order":
        examples = examples[::-1]
    elif problem == "empty":
        examples = ()
        metadata = metadata.model_copy(update={"question_ids": ()})
    elif problem == "warmup":
        metadata = metadata.model_copy(
            update={
                "configuration": metadata.configuration.model_copy(update={"warmup_questions": 10})
            }
        )
    else:
        examples = (
            examples[0].model_copy(
                update={
                    "evaluation": examples[0].evaluation.model_copy(update={"supporting_facts": ()})
                }
            ),
            *examples[1:],
        )
    retriever = FixtureRetriever(corpus, examples)
    with pytest.raises(ContractError):
        evaluate(
            examples=examples,
            corpus=corpus,
            retriever=retriever,
            metadata=metadata,
            sink=InMemoryInstrumentationSink(),
        )
    assert not retriever.calls
