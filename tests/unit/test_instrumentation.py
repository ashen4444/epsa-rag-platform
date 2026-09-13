from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from epsa_rag.instrumentation import (
    InMemoryInstrumentationSink,
    InstrumentationEvent,
    InstrumentationSink,
    NoOpInstrumentationSink,
    TraceContext,
)


def make_event() -> InstrumentationEvent:
    return InstrumentationEvent(
        event_id="event:test",
        occurred_at=datetime.fromisoformat("2026-09-13T09:00:00+00:00"),
        context=TraceContext(run_id="foundation-test", trace_id="trace:test", question_id="q-1"),
        event_type="contract.validated",
        source="tests",
        source_version="foundation-v1",
        payload={"valid": True, "count": 1},
    )


def test_event_round_trips_with_trace_context() -> None:
    event = make_event()
    restored = InstrumentationEvent.model_validate_json(event.model_dump_json())

    assert restored == event
    assert restored.context.question_id == "q-1"


def test_event_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        InstrumentationEvent(
            occurred_at=datetime(2026, 9, 13, 9, 0),
            context=TraceContext(run_id="run-1", trace_id="trace:test"),
            event_type="test.event",
            source="tests",
        )


def test_trace_context_factory_generates_valid_unique_trace_ids() -> None:
    first = TraceContext.start(run_id="run-1", question_id="q-1")
    second = TraceContext.start(run_id="run-1", question_id="q-1")

    assert first.trace_id.startswith("trace:")
    assert first.trace_id != second.trace_id


def test_in_memory_sink_preserves_order_and_returns_snapshot() -> None:
    first = make_event()
    second = first.model_copy(update={"event_id": "event:second"})
    sink = InMemoryInstrumentationSink()

    sink.emit(first)
    snapshot = sink.events
    sink.emit(second)

    assert snapshot == (first,)
    assert sink.events == (first, second)
    assert isinstance(sink, InstrumentationSink)

    sink.clear()
    assert sink.events == ()


def test_no_op_sink_satisfies_protocol_and_discards_event() -> None:
    sink = NoOpInstrumentationSink()

    assert isinstance(sink, InstrumentationSink)
    assert sink.emit(make_event()) is None

