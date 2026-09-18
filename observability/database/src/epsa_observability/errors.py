"""Safe errors: never expose SQL parameters, connection strings, or payloads."""

from epsa_rag.core.exceptions import InstrumentationError


class StorageError(InstrumentationError):
    """Persistence or validation failed; the operation was not acknowledged."""


class ConflictError(StorageError):
    """An existing immutable identity was reused with different content."""
