from epsa_rag.core.exceptions import (
    ConfigurationError,
    ContractError,
    EpsaRagError,
    InstrumentationError,
)


def test_public_exceptions_share_one_catchable_root() -> None:
    errors = (ConfigurationError, ContractError, InstrumentationError)

    assert all(issubclass(error, EpsaRagError) for error in errors)

