"""Project exception hierarchy."""


class EpsaRagError(Exception):
    """Base class for errors intentionally exposed by the research package."""


class ConfigurationError(EpsaRagError):
    """Raised when configuration cannot support a requested operation."""


class ContractError(EpsaRagError):
    """Raised when data violates a cross-component project contract."""


class InstrumentationError(EpsaRagError):
    """Raised when an instrumentation sink cannot accept an event."""


class DataPreparationError(EpsaRagError):
    """Base class for deterministic dataset and corpus preparation failures."""


class SourceValidationError(DataPreparationError):
    """Raised when source data violates the expected HotPotQA contract."""


class FrozenArtifactError(DataPreparationError):
    """Raised when preparation would overwrite an existing versioned artifact."""


class RetrievalError(EpsaRagError):
    """Base class for retrieval, embedding, and retrieval-index failures."""


class IndexIntegrityError(RetrievalError):
    """Raised when a persisted retrieval index fails validation."""


class EmbeddingError(RetrievalError):
    """Raised when an embedding provider violates the project contract."""


class QueryEmbeddingCacheError(EmbeddingError):
    """Raised when a query-embedding cache entry is missing or fails integrity checks."""
