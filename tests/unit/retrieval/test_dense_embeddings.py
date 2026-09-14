from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from openai import OpenAI, OpenAIError

from epsa_rag.core.exceptions import EmbeddingError
from epsa_rag.core.models import ParagraphChunk, RetrievalQuery, Sentence
from epsa_rag.retrieval.config import DenseConfig
from epsa_rag.retrieval.dense.embeddings import OpenAIEmbeddingProvider, format_document


def chunk() -> ParagraphChunk:
    return ParagraphChunk(
        chunk_id="chunk:a",
        title="Alpha",
        paragraph_text="First. Second.",
        sentences=(Sentence(index=0, text="First."), Sentence(index=1, text=" Second.")),
    )


class FakeEmbeddingsResource:
    def __init__(self, *, dimensions: int = 1536) -> None:
        self.dimensions = dimensions
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        data = [
            SimpleNamespace(index=index, embedding=[float(index + 1)] * self.dimensions)
            for index in reversed(range(len(inputs)))
        ]
        return SimpleNamespace(data=data)


def provider(
    resource: FakeEmbeddingsResource,
    *,
    batch_size: int = 128,
    progress_callback: Any = None,
) -> OpenAIEmbeddingProvider:
    client = SimpleNamespace(embeddings=resource)
    return OpenAIEmbeddingProvider(
        DenseConfig(batch_size=batch_size),
        client=cast(OpenAI, client),
        progress_callback=progress_callback,
    )


def test_document_and_query_embedding_inputs_obey_locked_boundary() -> None:
    resource = FakeEmbeddingsResource()
    adapter = provider(resource)
    item = chunk()

    document_vector = adapter.embed_documents((item,))
    query_vector = adapter.embed_query(RetrievalQuery(text="  Original question?  "))

    assert format_document(item) == "Title: Alpha\nText: First. Second."
    assert document_vector.shape == (1, 1536)
    assert query_vector.shape == (1536,)
    assert adapter.model == "text-embedding-3-small"
    assert adapter.dimensions == 1536
    assert resource.calls[0] == {
        "input": ["Title: Alpha\nText: First. Second."],
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "encoding_format": "float",
    }
    assert resource.calls[1]["input"] == ["  Original question?  "]


def test_query_embedding_batch_preserves_exact_input_order() -> None:
    resource = FakeEmbeddingsResource()
    adapter = provider(resource)
    queries = (RetrievalQuery(text=" First? "), RetrievalQuery(text="Second?"))

    vectors = adapter.embed_queries(queries)

    assert vectors.shape == (2, 1536)
    assert resource.calls[0]["input"] == [" First? ", "Second?"]


def test_embedding_adapter_batches_and_restores_response_order() -> None:
    resource = FakeEmbeddingsResource()
    progress: list[tuple[int, int]] = []
    adapter = provider(
        resource,
        batch_size=2,
        progress_callback=lambda *args: progress.append(args),
    )

    matrix = adapter._embed_texts(["one", "two", "three"])

    assert len(resource.calls) == 2
    assert matrix.shape == (3, 1536)
    assert matrix[0, 0] == 1
    assert matrix[1, 0] == 2
    assert matrix[2, 0] == 1
    assert progress == [(2, 3), (3, 3)]
    assert adapter._embed_texts([]).shape == (0, 1536)


@pytest.mark.parametrize(
    ("resource", "message"),
    [
        (FakeEmbeddingsResource(dimensions=2), "shape"),
        (FakeEmbeddingsResource(dimensions=1535), "shape"),
    ],
)
def test_embedding_adapter_rejects_wrong_dimensions(
    resource: FakeEmbeddingsResource, message: str
) -> None:
    with pytest.raises(EmbeddingError, match=message):
        provider(resource).embed_query(RetrievalQuery(text="question"))


def test_embedding_adapter_rejects_invalid_response_indices() -> None:
    resource = FakeEmbeddingsResource()

    def invalid_create(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(data=[SimpleNamespace(index=2, embedding=[1.0] * 1536)])

    resource.create = invalid_create  # type: ignore[method-assign]
    with pytest.raises(EmbeddingError, match="indices"):
        provider(resource).embed_query(RetrievalQuery(text="question"))


def test_embedding_adapter_rejects_empty_input_and_wraps_api_errors() -> None:
    resource = FakeEmbeddingsResource()
    with pytest.raises(EmbeddingError, match="must not be empty"):
        provider(resource)._embed_texts([""])

    def failing_create(**kwargs: Any) -> SimpleNamespace:
        raise OpenAIError("api unavailable")

    resource.create = failing_create  # type: ignore[method-assign]
    with pytest.raises(EmbeddingError, match="request failed"):
        provider(resource).embed_query(RetrievalQuery(text="question"))


def test_embedding_adapter_rejects_zero_and_non_finite_vectors() -> None:
    resource = FakeEmbeddingsResource()

    def zero_create(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.0] * 1536)])

    resource.create = zero_create  # type: ignore[method-assign]
    with pytest.raises(EmbeddingError, match="zero"):
        provider(resource).embed_query(RetrievalQuery(text="question"))

    def nan_create(**kwargs: Any) -> SimpleNamespace:
        values = np.ones(1536, dtype=np.float32)
        values[0] = np.nan
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=values.tolist())])

    resource.create = nan_create  # type: ignore[method-assign]
    with pytest.raises(EmbeddingError, match="non-finite"):
        provider(resource).embed_query(RetrievalQuery(text="question"))
