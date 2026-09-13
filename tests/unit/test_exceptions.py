from epsa_rag.core.exceptions import (
    ConfigurationError,
    ContractError,
    DataPreparationError,
    EmbeddingError,
    EpsaRagError,
    FrozenArtifactError,
    IndexIntegrityError,
    InstrumentationError,
    RetrievalError,
    SourceValidationError,
)


def test_public_exceptions_share_one_catchable_root() -> None:
    errors = (
        ConfigurationError,
        ContractError,
        DataPreparationError,
        InstrumentationError,
        RetrievalError,
    )

    assert all(issubclass(error, EpsaRagError) for error in errors)
    assert issubclass(SourceValidationError, DataPreparationError)
    assert issubclass(FrozenArtifactError, DataPreparationError)
    assert issubclass(IndexIntegrityError, RetrievalError)
    assert issubclass(EmbeddingError, RetrievalError)
