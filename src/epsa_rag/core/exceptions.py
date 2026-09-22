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


class QuestionAnalysisError(EpsaRagError):
    """Raised when Question Analyzer input or deterministic parsing is invalid."""


class ChunkAnalysisError(EpsaRagError):
    """Raised when a retrieved chunk cannot be analyzed by Component 02."""


class EvidenceUnitExtractionError(EpsaRagError):
    """Raised when Component 03 input or sentence extraction is invalid."""


class EvidenceScoringError(EpsaRagError):
    """Raised when Component 04 input or score calculation is invalid."""


class EvidenceGraphBuildError(EpsaRagError):
    """Raised when Component 05 cannot construct a valid evidence graph."""


class EvidencePathSearchError(EpsaRagError):
    """Raised when Component 06 cannot produce valid candidate evidence paths."""


class SufficiencyDecisionError(EpsaRagError):
    """Raised when Component 07 cannot make a valid sufficiency decision."""


class ContextPruningError(EpsaRagError):
    """Raised when Component 08 cannot produce a valid pruned context."""
