"""OpenAI final-answer adapter with frozen structured-output behavior."""

from __future__ import annotations

from time import perf_counter

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict

from epsa_rag.core.exceptions import AnswerGenerationError
from epsa_rag.pipeline.config import FinalAnswerConfig
from epsa_rag.pipeline.models import (
    AnswerGenerationRequest,
    FinalAnswer,
    FinalAnswerPayload,
    LLMTokenUsage,
)


class _OpenAIAnswerSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


class OpenAIFinalAnswerGenerator:
    """Generate one concise answer using the frozen GPT-4o-mini snapshot."""

    def __init__(
        self,
        *,
        config: FinalAnswerConfig | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.config = config or FinalAnswerConfig()
        try:
            self._client = client or OpenAI()
        except OpenAIError as error:
            raise AnswerGenerationError(
                f"unable to initialize OpenAI final-answer client: {error}"
            ) from error

    def generate(self, request: AnswerGenerationRequest) -> FinalAnswer:
        """Call the Responses API and preserve exact usage and model provenance."""

        started = perf_counter()
        try:
            response = self._client.responses.parse(
                model=self.config.model,
                instructions=self.config.instructions,
                input=_format_input(request),
                text_format=_OpenAIAnswerSchema,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                store=self.config.store,
                truncation=self.config.truncation,
            )
        except OpenAIError as error:
            raise AnswerGenerationError(f"OpenAI final-answer request failed: {error}") from error
        parsed = response.output_parsed
        if parsed is None:
            raise AnswerGenerationError("OpenAI final-answer response had no parsed output")
        try:
            payload = FinalAnswerPayload(answer=parsed.answer)
        except ValueError as error:
            raise AnswerGenerationError("OpenAI final-answer response was invalid") from error
        usage = response.usage
        if usage is None:
            raise AnswerGenerationError("OpenAI final-answer response had no token usage")
        cached_tokens = (
            usage.input_tokens_details.cached_tokens
            if usage.input_tokens_details is not None
            else 0
        )
        return FinalAnswer(
            answer=payload.answer,
            response_id=response.id,
            model=response.model,
            generator_version=self.config.generator_version,
            prompt_version=self.config.prompt_version,
            configuration_fingerprint=self.config.fingerprint(),
            usage=LLMTokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                cached_input_tokens=cached_tokens,
            ),
            latency_ms=(perf_counter() - started) * 1000,
        )


def _format_input(request: AnswerGenerationRequest) -> str:
    return f"Original question:\n{request.question}\n\nContext:\n{request.context.text}"
