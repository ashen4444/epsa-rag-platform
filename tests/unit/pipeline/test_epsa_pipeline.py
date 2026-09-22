"""Tests for deterministic EPSA controller and bounded system orchestration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from epsa_rag.core.models import (
    ParagraphChunk,
    RankedParagraphChunk,
    RetrievalQuery,
    Sentence,
)
from epsa_rag.epsa.next_hop_query import (
    RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1,
)
from epsa_rag.instrumentation import InMemoryInstrumentationSink, TraceContext
from epsa_rag.pipeline import (
    AnswerGenerationRequest,
    EPSAController,
    EPSAPassResult,
    EPSAPipeline,
    EPSAPipelineConfig,
    EPSAPipelineTrace,
    EPSATerminalState,
    FinalAnswer,
    FinalAnswerConfig,
    LLMTokenUsage,
)
from epsa_rag.retrieval.models import RetrievalResult


def _hit(chunk_id: str, title: str, text: str, rank: int = 1) -> RankedParagraphChunk:
    return RankedParagraphChunk(
        chunk=ParagraphChunk(
            chunk_id=chunk_id,
            title=title,
            paragraph_text=text,
            sentences=(Sentence(index=0, text=text),),
        ),
        rank=rank,
        score=1 / rank,
    )


def _answer() -> FinalAnswer:
    config = FinalAnswerConfig()
    return FinalAnswer(
        answer="London",
        response_id="resp_epsa_answer",
        model=config.model,
        generator_version=config.generator_version,
        prompt_version=config.prompt_version,
        configuration_fingerprint=config.fingerprint(),
        usage=LLMTokenUsage(input_tokens=20, output_tokens=1, total_tokens=21),
        latency_ms=1.0,
    )


class _Retriever:
    def __init__(self, responses: tuple[tuple[RankedParagraphChunk, ...], ...]) -> None:
        self.responses = responses
        self.calls: list[tuple[RetrievalQuery, int | None]] = []

    def retrieve(
        self, query: RetrievalQuery, *, top_k: int | None = None
    ) -> RetrievalResult:
        self.calls.append((query, top_k))
        return RetrievalResult(
            query=query,
            retriever_version="hybrid-retriever-v2",
            results=self.responses[len(self.calls) - 1],
        )


class _AnswerGenerator:
    def __init__(self) -> None:
        self.requests: list[AnswerGenerationRequest] = []

    def generate(self, request: AnswerGenerationRequest) -> FinalAnswer:
        self.requests.append(request)
        return _answer()


def _pipeline(
    responses: tuple[tuple[RankedParagraphChunk, ...], ...],
) -> tuple[EPSAPipeline, _Retriever, _AnswerGenerator]:
    retriever = _Retriever(responses)
    answers = _AnswerGenerator()
    pipeline = EPSAPipeline(
        retriever=retriever,
        controller=EPSAController.research_v1(),
        answer_generator=answers,
        config=EPSAPipelineConfig(top_k=5),
    )
    return pipeline, retriever, answers


def _bridge_trace() -> EPSAPipelineTrace:
    first = "Inception was directed by Christopher Nolan."
    second = "Christopher Nolan was born in London."
    pipeline, _, _ = _pipeline(
        (
            (_hit("inception", "Inception", first),),
            (_hit("nolan", "Christopher Nolan", second),),
        )
    )
    return pipeline.run(
        question_id="q-contract",
        question="Where was the director of Inception born?",
    )


def test_epsa_controller_runs_all_components_with_observable_provenance() -> None:
    sink = InMemoryInstrumentationSink()
    controller = EPSAController.research_v1(instrumentation_sink=sink)
    text = "Paris is the capital of France."

    result = controller.run_pass(
        question="What is the capital of France?",
        ranked_chunks=(_hit("paris", "Paris", text),),
        pass_number=1,
        generate_next_hop=True,
        trace_context=TraceContext.start(run_id="epsa-controller-test", question_id="q1"),
    )

    assert result.sufficiency_decision.sufficient
    assert result.input_chunk_ids == ("paris",)
    assert result.next_hop_query is not None
    assert result.next_hop_query.query is None
    assert result.pruned_context.selected_context_text.endswith(text)
    assert len(sink.events) == 9


def test_epsa_pipeline_stops_after_sufficient_hop1() -> None:
    text = "Paris is the capital of France."
    pipeline, retriever, _ = _pipeline(((_hit("paris", "Paris", text),),))

    trace = pipeline.run(question_id="q1", question="What is the capital of France?")

    assert trace.terminal_state is EPSATerminalState.SUFFICIENT_HOP1
    assert len(retriever.calls) == 1
    assert trace.hop2 is None
    assert trace.epsa_pass2 is None
    assert trace.final_answer_request.context.context_kind == "pruned_sentences"
    assert trace.final_answer_request.context.text.endswith(text)


def test_epsa_pipeline_uses_rule_query_and_reanalyzes_merged_evidence() -> None:
    first = "Inception was directed by Christopher Nolan."
    second = "Christopher Nolan was born in London."
    pipeline, retriever, _ = _pipeline(
        (
            (_hit("inception", "Inception", first),),
            (_hit("nolan", "Christopher Nolan", second),),
        )
    )

    trace = pipeline.run(
        question_id="q-bridge",
        question="Where was the director of Inception born?",
    )

    assert trace.terminal_state is EPSATerminalState.SUFFICIENT_HOP2
    assert len(retriever.calls) == 2
    assert retriever.calls[1][0].text == trace.epsa_pass1.next_hop_query.query
    assert trace.epsa_pass2 is not None
    assert trace.epsa_pass2.next_hop_query is None
    assert trace.epsa_pass2.question_analysis.raw_question == retriever.calls[0][0].text
    assert trace.final_answer_request.context.chunk_ids == ("inception", "nolan")
    assert "[Title: Inception" in trace.final_answer_request.context.text
    assert "[Document" not in trace.final_answer_request.context.text


def test_epsa_pipeline_records_insufficient_no_query_without_fallback_context() -> None:
    pipeline, retriever, _ = _pipeline(((),))

    trace = pipeline.run(question_id="q-no-query", question="Is it?")

    assert trace.terminal_state is EPSATerminalState.INSUFFICIENT_NO_QUERY
    assert len(retriever.calls) == 1
    assert trace.epsa_pass1.next_hop_query is not None
    assert trace.epsa_pass1.next_hop_query.query is None
    assert trace.final_answer_request.context.text == ""
    assert trace.final_answer_request.context.chunk_ids == ()


def test_epsa_pipeline_keeps_strict_pruned_context_after_failed_hop2() -> None:
    first = "Inception was directed by Christopher Nolan."
    irrelevant = "The Moon is Earth's natural satellite."
    pipeline, retriever, _ = _pipeline(
        (
            (_hit("inception", "Inception", first),),
            (_hit("moon", "Moon", irrelevant),),
        )
    )

    trace = pipeline.run(
        question_id="q-failed-hop2",
        question="Where was the director of Inception born?",
    )

    assert trace.terminal_state is EPSATerminalState.INSUFFICIENT_AFTER_HOP2
    assert len(retriever.calls) == 2
    assert trace.epsa_pass2 is not None
    assert trace.final_answer_request.context.text == (
        trace.epsa_pass2.pruned_context.selected_context_text
    )
    assert trace.final_answer_request.context.context_kind == "pruned_sentences"
    assert "[Document" not in trace.final_answer_request.context.text


def test_epsa_pass_contract_rejects_broken_component_provenance() -> None:
    trace = _bridge_trace()
    result = trace.epsa_pass2
    assert result is not None

    invalid_updates = [
        (
            {"input_chunk_ids": ("inception", "inception")},
            "input chunks must be unique",
        ),
        (
            {"input_chunk_ids": tuple(reversed(result.input_chunk_ids))},
            "chunk evidence must align",
        ),
        (
            {
                "evidence_units": (
                    result.evidence_units[0].model_copy(
                        update={
                            "chunk_id": "unknown",
                            "evidence_unit_id": "unknown::s0",
                        }
                    ),
                    *result.evidence_units[1:],
                )
            },
            "evidence units must originate",
        ),
        (
            {"scored_evidence_units": tuple(reversed(result.scored_evidence_units))},
            "scored evidence must align",
        ),
        (
            {
                "evidence_graph": result.evidence_graph.model_copy(
                    update={
                        "metadata": result.evidence_graph.metadata.model_copy(
                            update={"num_scored_evidence_units": 999}
                        )
                    }
                )
            },
            "graph must account",
        ),
        (
            {
                "sufficiency_decision": result.sufficiency_decision.model_copy(
                    update={
                        "metadata": result.sufficiency_decision.metadata.model_copy(
                            update={
                                "source_graph": result.evidence_graph.metadata.model_copy(
                                    update={"required_relation_hints": ()}
                                )
                            }
                        )
                    }
                )
            },
            "decision must originate",
        ),
        (
            {
                "pruned_context": result.pruned_context.model_copy(
                    update={
                        "metadata": result.pruned_context.metadata.model_copy(
                            update={
                                "source_sufficiency_decision": (
                                    result.sufficiency_decision.metadata.model_copy(
                                        update={"candidate_path_ids": ()}
                                    )
                                )
                            }
                        )
                    }
                )
            },
            "pruned context must originate",
        ),
    ]
    for updates, message in invalid_updates:
        invalid = result.model_copy(update=updates)
        with pytest.raises(ValidationError, match=message):
            EPSAPassResult.model_validate(invalid.model_dump())

    first_pass = trace.epsa_pass1
    assert first_pass.next_hop_query is not None
    invalid_query = first_pass.next_hop_query.model_copy(
        update={
            "metadata": first_pass.next_hop_query.metadata.model_copy(
                update={
                    "source_question_analysis": first_pass.question_analysis.metadata.model_copy(
                        update={"configuration_fingerprint": "0" * 64}
                    )
                }
            )
        }
    )
    invalid = first_pass.model_copy(update={"next_hop_query": invalid_query})
    with pytest.raises(ValidationError, match="query must originate"):
        EPSAPassResult.model_validate(invalid.model_dump())


def test_epsa_trace_contract_rejects_broken_hop_and_context_flow() -> None:
    trace = _bridge_trace()
    assert trace.hop2 is not None
    assert trace.epsa_pass2 is not None

    wrong_query = trace.hop2.query.model_copy(update={"text": "wrong query"})
    wrong_hop2 = trace.hop2.model_copy(update={"query": wrong_query})
    wrong_pass_number = trace.epsa_pass1.model_copy(update={"pass_number": 2})
    wrong_first_analysis = trace.epsa_pass1.question_analysis.model_copy(
        update={"raw_question": "different question"}
    )
    wrong_first_question = trace.epsa_pass1.model_copy(
        update={"question_analysis": wrong_first_analysis}
    )
    wrong_second_analysis = trace.epsa_pass2.question_analysis.model_copy(
        update={"raw_question": "different question"}
    )
    wrong_second_question = trace.epsa_pass2.model_copy(
        update={"question_analysis": wrong_second_analysis}
    )
    forbidden_query = RuleBasedNextHopQueryGeneratorHistoricalAdaptedV1().generate(
        trace.epsa_pass2.question_analysis,
        trace.epsa_pass2.sufficiency_decision,
        trace.epsa_pass2.evidence_graph,
        trace.epsa_pass2.candidate_paths,
    )
    forbidden_third_hop = trace.epsa_pass2.model_copy(
        update={"next_hop_query": forbidden_query}
    )
    wrong_state = trace.model_copy(
        update={"terminal_state": EPSATerminalState.INSUFFICIENT_AFTER_HOP2}
    )
    wrong_request = trace.final_answer_request.model_copy(update={"question_id": "other"})
    full_context = trace.final_answer_request.context.model_copy(
        update={"context_kind": "full_paragraphs"}
    )
    wrong_kind_request = trace.final_answer_request.model_copy(update={"context": full_context})
    wrong_text_context = trace.final_answer_request.context.model_copy(
        update={"text": "wrong", "character_count": 5}
    )
    wrong_text_request = trace.final_answer_request.model_copy(
        update={"context": wrong_text_context}
    )

    cases = [
        (trace.model_copy(update={"question_id": "other"}), "question ID must match"),
        (trace.model_copy(update={"epsa_pass1": wrong_pass_number}), "pass 1 must consume"),
        (
            trace.model_copy(update={"epsa_pass1": wrong_first_question}),
            "pass 1 must analyze the original question",
        ),
        (trace.model_copy(update={"hop2": wrong_hop2}), "must execute EPSA's proposed"),
        (trace.model_copy(update={"epsa_pass2": None}), "requires EPSA pass 2"),
        (
            trace.model_copy(update={"epsa_pass2": wrong_second_question}),
            "pass 2 must reanalyze the original question",
        ),
        (
            trace.model_copy(update={"epsa_pass2": forbidden_third_hop}),
            "cannot propose another retrieval",
        ),
        (wrong_state, "terminal state does not match"),
        (trace.model_copy(update={"final_answer_request": wrong_request}), "original question"),
        (
            trace.model_copy(update={"final_answer_request": wrong_kind_request}),
            "strict pruned sentence",
        ),
        (
            trace.model_copy(update={"final_answer_request": wrong_text_request}),
            "must equal final EPSA pruned context",
        ),
    ]
    for invalid, message in cases:
        with pytest.raises(ValidationError, match=message):
            EPSAPipelineTrace.model_validate(invalid.model_dump())
