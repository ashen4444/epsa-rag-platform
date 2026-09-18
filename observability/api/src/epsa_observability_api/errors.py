"""Sanitized domain errors exposed through the HTTP application."""

from __future__ import annotations


class ObservabilityApiError(Exception):
    """Expected API-layer failure that must not reveal storage internals."""


class NotFoundError(ObservabilityApiError):
    """The requested immutable run or question does not exist."""


class IncompatibleRunsError(ObservabilityApiError):
    """Two completed runs cannot be compared scientifically."""

    def __init__(self, criteria: tuple[str, ...]) -> None:
        super().__init__("Runs are not compatible for summary comparison.")
        self.criteria = criteria


class DatabaseUnavailableError(ObservabilityApiError):
    """Database connectivity or Phase 5A schema compatibility failed."""


class StoredRecordError(ObservabilityApiError):
    """A stored record contradicts the immutable Phase 5A contract."""
