"""Serializable structured instrumentation events."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from epsa_rag.core.ids import Identifier
from epsa_rag.instrumentation.context import TraceContext


class InstrumentationEvent(BaseModel):
    """A storage-neutral diagnostic event emitted by research code."""

    # Identifier fields normalize themselves. Payload strings must retain exact source text,
    # including the native whitespace in HotPotQA sentences and complete queries.
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: Identifier = Field(default_factory=lambda: f"event:{uuid4().hex}")
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context: TraceContext
    event_type: Identifier
    source: Identifier
    source_version: Identifier | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Require an unambiguous instant for reproducible traces."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value

    @field_validator("payload")
    @classmethod
    def detach_payload(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Detach the event's top-level payload mapping from caller-owned state."""

        return value.copy()
