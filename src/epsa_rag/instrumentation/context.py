"""Correlation context for research traces."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from epsa_rag.core.ids import Identifier


class TraceContext(BaseModel):
    """Identifiers that correlate events without depending on a storage backend."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    run_id: Identifier
    trace_id: Identifier
    question_id: Identifier | None = None

    @classmethod
    def start(cls, *, run_id: str, question_id: str | None = None) -> TraceContext:
        """Start a trace with a locally generated correlation identifier."""

        return cls(run_id=run_id, trace_id=f"trace:{uuid4().hex}", question_id=question_id)

