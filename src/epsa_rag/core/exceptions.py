"""Project exception hierarchy."""


class EpsaRagError(Exception):
    """Base class for errors intentionally exposed by the research package."""


class ConfigurationError(EpsaRagError):
    """Raised when configuration cannot support a requested operation."""


class ContractError(EpsaRagError):
    """Raised when data violates a cross-component project contract."""


class InstrumentationError(EpsaRagError):
    """Raised when an instrumentation sink cannot accept an event."""

