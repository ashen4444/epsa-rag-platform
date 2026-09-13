from __future__ import annotations

import json
from pathlib import Path

import pytest

from epsa_rag.core.exceptions import FrozenArtifactError, IndexIntegrityError
from epsa_rag.core.models import ParagraphChunk, RetrievalQuery, Sentence
from epsa_rag.retrieval.bm25.index import BM25Index, chunk_search_text, tokenize
from epsa_rag.retrieval.config import BM25Config


def make_chunk(chunk_id: str, title: str, text: str) -> ParagraphChunk:
    return ParagraphChunk(
        chunk_id=chunk_id,
        title=title,
        paragraph_text=text,
        sentences=(Sentence(index=0, text=text),),
    )


def chunks() -> tuple[ParagraphChunk, ...]:
    return (
        make_chunk("chunk:a", "Alpha", "Red apple fruit."),
        make_chunk("chunk:b", "Beta", "Blue ocean water."),
        make_chunk("chunk:c", "Gamma", "Apple pie dessert."),
    )


def test_tokenizer_and_document_text_are_explicit() -> None:
    item = chunks()[0]
    assert chunk_search_text(item) == "Alpha Red apple fruit."
    assert tokenize("CAFÉ, Apple_2!") == ("café", "apple_2")


def test_bm25_ranks_matches_and_handles_unknown_terms() -> None:
    index = BM25Index.build(chunks(), BM25Config())

    hits = index.search(RetrievalQuery(text="red apple"), top_k=3)

    assert [hit.chunk_id for hit in hits] == ["chunk:a", "chunk:c"]
    assert hits[0].rank == 1
    assert hits[0].score > hits[1].score > 0
    assert index.search(RetrievalQuery(text="unknown"), top_k=3) == ()
    with pytest.raises(ValueError, match="positive"):
        index.search(RetrievalQuery(text="apple"), top_k=0)


def test_bm25_persistence_round_trip_and_overwrite_protection(tmp_path: Path) -> None:
    original = BM25Index.build(chunks(), BM25Config(k1=1.2))
    path = tmp_path / "index.json"
    original.save(path)
    loaded = BM25Index.load(path, expected_config=original.config)

    query = RetrievalQuery(text="ocean")
    assert loaded.chunk_ids == original.chunk_ids
    assert loaded.search(query, top_k=2) == original.search(query, top_k=2)
    with pytest.raises(FrozenArtifactError, match="overwrite"):
        original.save(path)


def test_bm25_load_rejects_invalid_schema_or_configuration(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"schema_version": "9.0"}), encoding="utf-8")
    with pytest.raises(IndexIntegrityError, match="unsupported"):
        BM25Index.load(path)

    valid_path = tmp_path / "valid.json"
    BM25Index.build(chunks(), BM25Config()).save(valid_path)
    with pytest.raises(IndexIntegrityError, match="configuration"):
        BM25Index.load(valid_path, expected_config=BM25Config(k1=1.1))


@pytest.mark.parametrize(
    ("chunk_ids", "lengths", "postings", "message"),
    [
        ((), (), {}, "at least one"),
        (("chunk:a", "chunk:a"), (1, 1), {}, "unique"),
        (("chunk:a",), (), {}, "align"),
        (("chunk:a",), (0,), {}, "at least one token"),
        (("chunk:a",), (1,), {"": ((0, 1),)}, "tokens"),
        (("chunk:a",), (1,), {"a": ((1, 1),)}, "indices"),
        (("chunk:a",), (1,), {"a": ((0, 0),)}, "frequencies"),
    ],
)
def test_bm25_constructor_rejects_corrupt_structures(
    chunk_ids: tuple[str, ...],
    lengths: tuple[int, ...],
    postings: dict[str, tuple[tuple[int, int], ...]],
    message: str,
) -> None:
    with pytest.raises(IndexIntegrityError, match=message):
        BM25Index(
            config=BM25Config(),
            chunk_ids=chunk_ids,
            document_lengths=lengths,
            postings=postings,
        )
