"""Structured OpenAI controller for the adaptive LLM retrieval baseline."""

from __future__ import annotations

from time import perf_counter

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from epsa_rag.core.exceptions import AdaptiveControllerError
from epsa_rag.pipeline.config import AdaptiveControllerConfig
from epsa_rag.pipeline.models import (
    AdaptiveControlDecision,
    AdaptiveControlOutput,
    AdaptiveControlRequest,
    AdaptiveReasonCode,
    LLMTokenUsage,
)


class _AdaptiveControlSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sufficient: bool
    reason_code: AdaptiveReasonCode
    missing_evidence: str | None
    evidence_document_numbers: list[int] = Field(default_factory=list)
    next_hop_query: str | None


class OpenAIAdaptiveRetrievalController:
    """Use GPT-4o-mini for one inspectable sufficiency/query decision."""

    def __init__(
        self,
        *,
        config: AdaptiveControllerConfig | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.config = config or AdaptiveControllerConfig()
        try:
            self._client = client or OpenAI()
        except OpenAIError as error:
            raise AdaptiveControllerError(
                f"unable to initialize OpenAI adaptive controller: {error}"
            ) from error

    def decide(self, request: AdaptiveControlRequest) -> AdaptiveControlDecision:
        """Return a validated five-field decision and complete API provenance."""

        started = perf_counter()
        try:
            response = self._client.responses.parse(
                model=self.config.model,
                instructions=self.config.instructions,
                input=_format_controller_input(request),
                text_format=_AdaptiveControlSchema,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                store=self.config.store,
                truncation=self.config.truncation,
            )
        except OpenAIError as error:
            raise AdaptiveControllerError(
                f"OpenAI adaptive-controller request failed: {error}"
            ) from error
        parsed = response.output_parsed
        if parsed is None:
            raise AdaptiveControllerError(
                "OpenAI adaptive-controller response had no parsed output"
            )
        try:
            output = AdaptiveControlOutput(
                sufficient=parsed.sufficient,
                reason_code=parsed.reason_code,
                missing_evidence=parsed.missing_evidence,
                evidence_document_numbers=tuple(parsed.evidence_document_numbers),
                next_hop_query=parsed.next_hop_query,
            )
        except ValueError as error:
            raise AdaptiveControllerError(
                "OpenAI adaptive-controller response violated the decision contract"
            ) from error
        document_count = len(request.hop1_context.chunk_ids)
        if any(number > document_count for number in output.evidence_document_numbers):
            raise AdaptiveControllerError(
                "OpenAI adaptive-controller cited a document outside the Hop-1 context"
            )
        usage = response.usage
        if usage is None:
            raise AdaptiveControllerError("OpenAI adaptive-controller response had no token usage")
        cached_tokens = (
            usage.input_tokens_details.cached_tokens
            if usage.input_tokens_details is not None
            else 0
        )
        return AdaptiveControlDecision(
            output=output,
            response_id=response.id,
            model=response.model,
            controller_version=self.config.controller_version,
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


def _format_controller_input(request: AdaptiveControlRequest) -> str:
    return (
        f"Original question:\n{request.question}\n\n"
        f"Hop-1 context:\n{request.hop1_context.text}"
    )
