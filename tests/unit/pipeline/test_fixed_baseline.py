"""Tests for final-answer generation and the fixed one-hop baseline."""

from __future__ import annotations

from typing import cast
from unittest.mock import Mock

import pytest
from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from epsa_rag.core.exceptions import AnswerGenerationError
from epsa_rag.core.models import ParagraphChunk, RankedParagraphChunk, RetrievalQuery, Sentence
from epsa_rag.pipeline import (
    AnswerGenerationRequest,
    FinalAnswer,
    FinalAnswerConfig,
    FinalAnswerGeneratorProtocol,
    FixedBaselineConfig,
    FixedBaselinePipeline,
    LLMTokenUsage,
    OpenAIFinalAnswerGenerator,
    RenderedContext,
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


def _answer(value: str = "London") -> FinalAnswer:
    config = FinalAnswerConfig()
    return FinalAnswer(
        answer=value,
        response_id="resp_1",
        model=config.model,
        generator_version=config.generator_version,
        prompt_version=config.prompt_version,
        configuration_fingerprint=config.fingerprint(),
        usage=LLMTokenUsage(input_tokens=20, output_tokens=1, total_tokens=21),
        latency_ms=1.0,
    )


def _request() -> AnswerGenerationRequest:
    return AnswerGenerationRequest(
        question_id="q1",
        question="Where was Beta?",
        context=render_full_paragraph_context((_hit("a", 1),)),
    )


def test_render_full_paragraph_context_is_deterministic_and_inference_safe() -> None:
    context = render_full_paragraph_context((_hit("a", 1), _hit("b", 2)))

    assert context.chunk_ids == ("a", "b")
    assert context.context_kind == "full_paragraphs"
    assert context.text == (
        "[Document 1]\nTitle: Title a\nText: Evidence from a.\n\n"
        "[Document 2]\nTitle: Title b\nText: Evidence from b."
    )
    assert "Chunk:" not in context.text
    assert context.character_count == len(context.text)


def test_fixed_baseline_retrieves_once_and_sends_full_context() -> None:
    class Retriever:
        def __init__(self) -> None:
            self.calls: list[tuple[RetrievalQuery, int | None]] = []

        def retrieve(
            self, query: RetrievalQuery, *, top_k: int | None = None
        ) -> RetrievalResult:
            self.calls.append((query, top_k))
            return RetrievalResult(
                query=query,
                retriever_version="hybrid-retriever-v2",
                results=(_hit("a", 1), _hit("b", 2)),
            )

    class Generator:
        def __init__(self) -> None:
            self.requests: list[AnswerGenerationRequest] = []

        def generate(self, request: AnswerGenerationRequest) -> FinalAnswer:
            self.requests.append(request)
            return _answer()

    retriever = Retriever()
    generator = Generator()
    pipeline = FixedBaselinePipeline(
        retriever=retriever,
        answer_generator=generator,
        config=FixedBaselineConfig(top_k=5),
    )

    trace = pipeline.run(question_id="q1", question="Where was Beta?")

    assert retriever.calls[0][1] == 5
    assert len(retriever.calls) == 1
    assert len(generator.requests) == 1
    assert trace.final_answer.answer == "London"
    assert trace.final_answer_request.context.chunk_ids == ("a", "b")
    assert trace.merged_retrieval.diagnostics.hop2_input_chunks == 0
    assert isinstance(generator, FinalAnswerGeneratorProtocol)


def test_openai_generator_uses_frozen_structured_response_settings() -> None:
    parsed = Mock(answer=" London ")
    details = Mock(cached_tokens=3)
    usage = Mock(input_tokens=20, output_tokens=1, total_tokens=21, input_tokens_details=details)
    response = Mock(
        output_parsed=parsed,
        usage=usage,
        id="resp_1",
        model="gpt-4o-mini-2024-07-18",
    )
    client = Mock()
    client.responses.parse.return_value = response
    config = FinalAnswerConfig()
    generator = OpenAIFinalAnswerGenerator(config=config, client=cast(OpenAI, client))

    answer = generator.generate(_request())

    assert answer.answer == "London"
    assert answer.usage.cached_input_tokens == 3
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini-2024-07-18"
    assert kwargs["temperature"] == 0.0
    assert kwargs["store"] is False
    assert kwargs["truncation"] == "disabled"
    assert kwargs["max_output_tokens"] == 64
    assert kwargs["text_format"].__name__ == "_OpenAIAnswerSchema"
    assert "Original question:\nWhere was Beta?" in kwargs["input"]
    assert "Context:\n[Document 1]" in kwargs["input"]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (Mock(output_parsed=None, usage=Mock()), "no parsed output"),
        (Mock(output_parsed=Mock(answer=" "), usage=Mock()), "response was invalid"),
        (Mock(output_parsed=Mock(answer="London"), usage=None), "no token usage"),
    ],
)
def test_openai_generator_rejects_invalid_responses(response: Mock, message: str) -> None:
    client = Mock()
    client.responses.parse.return_value = response
    generator = OpenAIFinalAnswerGenerator(client=cast(OpenAI, client))

    with pytest.raises(AnswerGenerationError, match=message):
        generator.generate(_request())


def test_openai_generator_wraps_sdk_errors() -> None:
    client = Mock()
    client.responses.parse.side_effect = OpenAIError("failure")
    generator = OpenAIFinalAnswerGenerator(client=cast(OpenAI, client))

    with pytest.raises(AnswerGenerationError, match="request failed"):
        generator.generate(_request())


def test_final_answer_contracts_reject_inconsistent_values() -> None:
    with pytest.raises(ValidationError, match="character count"):
        RenderedContext(
            format_version="v1",
            context_kind="full_paragraphs",
            chunk_ids=("a",),
            text="text",
            character_count=3,
        )
    with pytest.raises(ValidationError, match="must be unique"):
        RenderedContext(
            format_version="v1",
            context_kind="full_paragraphs",
            chunk_ids=("a", "a"),
            text="text",
            character_count=4,
        )
    with pytest.raises(ValidationError, match="question must not be blank"):
        AnswerGenerationRequest(
            question_id="q1",
            question=" ",
            context=render_full_paragraph_context(()),
        )
    with pytest.raises(ValidationError, match="input plus output"):
        LLMTokenUsage(input_tokens=3, output_tokens=2, total_tokens=4)
    with pytest.raises(ValidationError, match="cannot exceed"):
        LLMTokenUsage(
            input_tokens=3,
            output_tokens=2,
            total_tokens=5,
            cached_input_tokens=4,
        )
    with pytest.raises(ValidationError, match="must not be blank"):
        _answer(" ")
