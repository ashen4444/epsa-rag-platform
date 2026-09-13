"""Instrumentation sink protocol and foundation implementations."""

from __future__ import annotations

from threading import Lock
from typing import Protocol, runtime_checkable

from epsa_rag.instrumentation.events import InstrumentationEvent


@runtime_checkable
class InstrumentationSink(Protocol):
    """Boundary through which research code emits structured events."""

    def emit(self, event: InstrumentationEvent) -> None:
        """Accept one event or raise an instrumentation-specific error."""


class NoOpInstrumentationSink:
    """Discard events when instrumentation is deliberately disabled."""

    def emit(self, event: InstrumentationEvent) -> None:
        """Accept and discard an event."""


class InMemoryInstrumentationSink:
    """Thread-safe event collector intended for unit tests and local diagnostics."""

    def __init__(self) -> None:
        self._events: list[InstrumentationEvent] = []
        self._lock = Lock()

    def emit(self, event: InstrumentationEvent) -> None:
        """Append an event to the collector."""

        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> tuple[InstrumentationEvent, ...]:
        """Return an immutable snapshot in emission order."""

        with self._lock:
            return tuple(self._events)

    def clear(self) -> None:
        """Remove collected events without replacing the sink."""

        with self._lock:
            self._events.clear()

