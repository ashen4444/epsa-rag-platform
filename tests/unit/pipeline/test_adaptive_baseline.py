"""Tests for the adaptive LLM two-hop baseline."""

from __future__ import annotations

from typing import cast
from unittest.mock import Mock

import pytest
from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from epsa_rag.core.exceptions import AdaptiveControllerError
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, RetrievalQuery, Sentence
from epsa_rag.pipeline import (
    AdaptiveBaselineConfig,
    AdaptiveBaselinePipeline,
    AdaptiveControlDecision,
    AdaptiveControllerConfig,
    AdaptiveControlOutput,
    AdaptiveControlRequest,
    AdaptiveReasonCode,
    AnswerGenerationRequest,
    FinalAnswer,
    FinalAnswerConfig,
    LLMTokenUsage,
    OpenAIAdaptiveRetrievalController,
    render_full_paragraph_context,
)
from epsa_rag.retrieval.models import RetrievalResult


def _hit(chunk_id: str, rank: int) -> RankedParagraphChunk:
    text = f"Evidence from {chunk_id}."
    return RankedParagraphChunk(
        chunk=ParagraphChunk(
            chunk_id=chunk_id,
            title=f"Title {chunk_id}",
            paragraph_text=text,
            sentences=(Sentence(index=0, text=text),),
        ),
        rank=rank,
        score=1 / rank,
    )


def _usage() -> LLMTokenUsage:
    return LLMTokenUsage(input_tokens=20, output_tokens=5, total_tokens=25)


def _decision(output: AdaptiveControlOutput) -> AdaptiveControlDecision:
    config = AdaptiveControllerConfig()
    return AdaptiveControlDecision(
        output=output,
        response_id="resp_controller",
        model=config.model,
        controller_version=config.controller_version,
        prompt_version=config.prompt_version,
        configuration_fingerprint=config.fingerprint(),
        usage=_usage(),
        latency_ms=1.0,
    )


def _answer() -> FinalAnswer:
    config = FinalAnswerConfig()
    return FinalAnswer(
        answer="London",
        response_id="resp_answer",
        model=config.model,
        generator_version=config.generator_version,
        prompt_version=config.prompt_version,
        configuration_fingerprint=config.fingerprint(),
        usage=_usage(),
        latency_ms=1.0,
    )


class _Retriever:
    def __init__(self) -> None:
        self.calls: list[tuple[RetrievalQuery, int | None]] = []

    def retrieve(
        self, query: RetrievalQuery, *, top_k: int | None = None
    ) -> RetrievalResult:
        self.calls.append((query, top_k))
        hits = (
            (_hit("a", 1), _hit("b", 2))
            if len(self.calls) == 1
            else (_hit("b", 1), _hit("c", 2))
        )
        return RetrievalResult(
            query=query,
            retriever_version="hybrid-retriever-v2",
            results=hits,
        )


class _AnswerGenerator:
    def __init__(self) -> None:
        self.requests: list[AnswerGenerationRequest] = []

    def generate(self, request: AnswerGenerationRequest) -> FinalAnswer:
        self.requests.append(request)
        return _answer()


@pytest.mark.parametrize(
    ("output", "expected_calls", "expected_ids"),
    [
        (
            AdaptiveControlOutput(
                sufficient=True,
                reason_code=AdaptiveReasonCode.SUFFICIENT,
                missing_evidence=None,
                evidence_document_numbers=(1, 2),
                next_hop_query=None,
            ),
            1,
            ("a", "b"),
        ),
        (
            AdaptiveControlOutput(
                sufficient=False,
                reason_code=AdaptiveReasonCode.MISSING_BRIDGE_EVIDENCE,
                missing_evidence="The bridge location is missing.",
                evidence_document_numbers=(1,),
                next_hop_query="Beta bridge location",
            ),
            2,
            ("a", "b", "c"),
        ),
        (
            AdaptiveControlOutput(
                sufficient=False,
                reason_code=AdaptiveReasonCode.INSUFFICIENT_NO_GROUNDED_QUERY,
                missing_evidence="No grounded target is available.",
                evidence_document_numbers=(),
                next_hop_query=None,
            ),
            1,
            ("a", "b"),
        ),
    ],
)
def test_adaptive_pipeline_executes_only_the_approved_optional_hop(
    output: AdaptiveControlOutput,
    expected_calls: int,
    expected_ids: tuple[str, ...],
) -> None:
    class Controller:
        def decide(self, request: AdaptiveControlRequest) -> AdaptiveControlDecision:
            assert request.hop1_context.chunk_ids == ("a", "b")
            return _decision(output)

    retriever = _Retriever()
    answer_generator = _AnswerGenerator()
    pipeline = AdaptiveBaselinePipeline(
        retriever=retriever,
        controller=Controller(),
        answer_generator=answer_generator,
        config=AdaptiveBaselineConfig(top_k=5),
    )

    trace = pipeline.run(question_id="q1", question="Where was Beta?")

    assert len(retriever.calls) == expected_calls
    assert all(top_k == 5 for _, top_k in retriever.calls)
    assert trace.final_answer_request.context.chunk_ids == expected_ids
    assert trace.merged_retrieval.diagnostics.unique_chunks == len(expected_ids)
    if output.next_hop_query is not None:
        assert retriever.calls[1][0].text == output.next_hop_query


def test_openai_controller_uses_the_approved_five_field_contract() -> None:
    parsed = Mock(
        sufficient=False,
        reason_code=AdaptiveReasonCode.MISSING_BRIDGE_EVIDENCE,
        missing_evidence="Birthplace evidence is missing.",
        evidence_document_numbers=[1],
        next_hop_query="Author birthplace",
    )
    usage = Mock(
        input_tokens=30,
        output_tokens=10,
        total_tokens=40,
        input_tokens_details=Mock(cached_tokens=4),
    )
    response = Mock(
        output_parsed=parsed,
        usage=usage,
        id="resp_1",
        model="gpt-4o-mini-2024-07-18",
    )
    client = Mock()
    client.responses.parse.return_value = response
    controller = OpenAIAdaptiveRetrievalController(client=cast(OpenAI, client))
    request = AdaptiveControlRequest(
        question_id="q1",
        question="Where was the author born?",
        hop1_context=render_full_paragraph_context((_hit("a", 1),)),
    )

    decision = controller.decide(request)

    assert decision.output.next_hop_query == "Author birthplace"
    assert decision.output.evidence_document_numbers == (1,)
    assert decision.usage.cached_input_tokens == 4
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_output_tokens"] == 256
    assert kwargs["text_format"].__name__ == "_AdaptiveControlSchema"
    assert "Hop-1 context:\n[Document 1]" in kwargs["input"]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (Mock(output_parsed=None, usage=Mock()), "no parsed output"),
        (
            Mock(
                output_parsed=Mock(
                    sufficient=True,
                    reason_code=AdaptiveReasonCode.MISSING_ANSWER_EVIDENCE,
                    missing_evidence=None,
                    evidence_document_numbers=[],
                    next_hop_query=None,
                ),
                usage=Mock(),
            ),
            "violated the decision contract",
        ),
        (
            Mock(
                output_parsed=Mock(
                    sufficient=True,
                    reason_code=AdaptiveReasonCode.SUFFICIENT,
                    missing_evidence=None,
                    evidence_document_numbers=[],
                    next_hop_query=None,
                ),
                usage=None,
            ),
            "no token usage",
        ),
    ],
)
def test_openai_controller_rejects_invalid_responses(response: Mock, message: str) -> None:
    client = Mock()
    client.responses.parse.return_value = response
    controller = OpenAIAdaptiveRetrievalController(client=cast(OpenAI, client))
    request = AdaptiveControlRequest(
        question_id="q1",
        question="Question?",
        hop1_context=render_full_paragraph_context(()),
    )

    with pytest.raises(AdaptiveControllerError, match=message):
        controller.decide(request)


def test_openai_controller_rejects_out_of_range_document_and_wraps_sdk_error() -> None:
    parsed = Mock(
        sufficient=False,
        reason_code=AdaptiveReasonCode.MISSING_ANSWER_EVIDENCE,
        missing_evidence="Missing answer.",
        evidence_document_numbers=[2],
        next_hop_query="answer relation",
    )
    client = Mock()
    client.responses.parse.return_value = Mock(output_parsed=parsed, usage=Mock())
    controller = OpenAIAdaptiveRetrievalController(client=cast(OpenAI, client))
    request = AdaptiveControlRequest(
        question_id="q1",
        question="Question?",
        hop1_context=render_full_paragraph_context((_hit("a", 1),)),
    )
    with pytest.raises(AdaptiveControllerError, match="outside the Hop-1 context"):
        controller.decide(request)

    client.responses.parse.side_effect = OpenAIError("failure")
    with pytest.raises(AdaptiveControllerError, match="request failed"):
        controller.decide(request)


def test_adaptive_contracts_reject_inconsistent_states() -> None:
    with pytest.raises(ValidationError, match="unique and sorted"):
        AdaptiveControlOutput(
            sufficient=True,
            reason_code=AdaptiveReasonCode.SUFFICIENT,
            missing_evidence=None,
            evidence_document_numbers=(2, 1),
            next_hop_query=None,
        )
    with pytest.raises(ValidationError, match="cannot request"):
        AdaptiveControlOutput(
            sufficient=True,
            reason_code=AdaptiveReasonCode.MISSING_ANSWER_EVIDENCE,
            missing_evidence=None,
            evidence_document_numbers=(),
            next_hop_query=None,
        )
    with pytest.raises(ValidationError, match="require missing evidence"):
        AdaptiveControlOutput(
            sufficient=False,
            reason_code=AdaptiveReasonCode.SUFFICIENT,
            missing_evidence=None,
            evidence_document_numbers=(),
            next_hop_query=None,
        )
    with pytest.raises(ValidationError, match="no-grounded-query"):
        AdaptiveControlOutput(
            sufficient=False,
            reason_code=AdaptiveReasonCode.MISSING_BRIDGE_EVIDENCE,
            missing_evidence="Missing bridge.",
            evidence_document_numbers=(),
            next_hop_query=None,
        )
    with pytest.raises(ValidationError, match="cannot contain"):
        AdaptiveControlOutput(
            sufficient=False,
            reason_code=AdaptiveReasonCode.INSUFFICIENT_NO_GROUNDED_QUERY,
            missing_evidence="Missing bridge.",
            evidence_document_numbers=(),
            next_hop_query="bridge",
        )


def test_adaptive_contract_normalizes_optional_text() -> None:
    output = AdaptiveControlOutput(
        sufficient=False,
        reason_code=AdaptiveReasonCode.INSUFFICIENT_NO_GROUNDED_QUERY,
        missing_evidence="  Missing evidence.  ",
        evidence_document_numbers=(),
        next_hop_query="   ",
    )

    assert output.missing_evidence == "Missing evidence."
    assert output.next_hop_query is None


def test_adaptive_request_requires_question_and_full_paragraph_context() -> None:
    context = render_full_paragraph_context((_hit("a", 1),))
    with pytest.raises(ValidationError, match="question must not be blank"):
        AdaptiveControlRequest(question_id="q1", question=" ", hop1_context=context)

    pruned_context = context.model_copy(update={"context_kind": "pruned_sentences"})
    with pytest.raises(ValidationError, match="requires full Hop-1 paragraphs"):
        AdaptiveControlRequest(
            question_id="q1",
            question="Question?",
            hop1_context=pruned_context,
        )
