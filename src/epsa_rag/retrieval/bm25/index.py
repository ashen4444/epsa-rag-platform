"""Small deterministic Okapi BM25 index with transparent persistence."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from epsa_rag.core.exceptions import IndexIntegrityError
from epsa_rag.core.models import ParagraphChunk, RetrievalQuery
from epsa_rag.retrieval.config import BM25Config
from epsa_rag.retrieval.io import read_json_object, write_json_artifact_exclusive
from epsa_rag.retrieval.models import BackendHit

_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_SCHEMA_VERSION = "1.0"


def tokenize(text: str) -> tuple[str, ...]:
    """Apply the versioned Unicode-word, case-folding tokenizer."""

    return tuple(match.group(0).casefold() for match in _WORD_PATTERN.finditer(text))


def chunk_search_text(chunk: ParagraphChunk) -> str:
    """Build the explicit lexical document from title and paragraph text."""

    return f"{chunk.title} {chunk.paragraph_text}"


class BM25Index:
    """In-memory and persistable exact Okapi BM25 index."""

    def __init__(
        self,
        *,
        config: BM25Config,
        chunk_ids: Sequence[str],
        document_lengths: Sequence[int],
        postings: dict[str, tuple[tuple[int, int], ...]],
    ) -> None:
        self.config = config
        self._chunk_ids = tuple(chunk_ids)
        self._document_lengths = tuple(document_lengths)
        self._postings = postings
        self._validate()
        self._average_document_length = sum(self._document_lengths) / len(
            self._document_lengths
        )

    @classmethod
    def build(cls, chunks: Sequence[ParagraphChunk], config: BM25Config) -> BM25Index:
        """Build postings from a stable corpus order."""

        if not chunks:
            raise IndexIntegrityError("cannot build BM25 index from an empty corpus")
        chunk_ids: list[str] = []
        document_lengths: list[int] = []
        mutable_postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        for document_index, chunk in enumerate(chunks):
            tokens = tokenize(chunk_search_text(chunk))
            chunk_ids.append(chunk.chunk_id)
            document_lengths.append(len(tokens))
            for token, frequency in sorted(Counter(tokens).items()):
                mutable_postings[token].append((document_index, frequency))
        postings = {
            token: tuple(entries) for token, entries in sorted(mutable_postings.items())
        }
        return cls(
            config=config,
            chunk_ids=chunk_ids,
            document_lengths=document_lengths,
            postings=postings,
        )

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return self._chunk_ids

    def search(self, query: RetrievalQuery, *, top_k: int) -> tuple[BackendHit, ...]:
        """Score all lexical matches and return deterministic top hits."""

        if top_k < 1:
            raise ValueError("top_k must be positive")
        scores: defaultdict[int, float] = defaultdict(float)
        document_count = len(self._chunk_ids)
        for token in tokenize(query.text):
            entries = self._postings.get(token)
            if entries is None:
                continue
            document_frequency = len(entries)
            inverse_document_frequency = math.log(
                1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for document_index, term_frequency in entries:
                document_length = self._document_lengths[document_index]
                length_normalization = 1 - self.config.b + self.config.b * (
                    document_length / self._average_document_length
                )
                numerator = term_frequency * (self.config.k1 + 1)
                denominator = term_frequency + self.config.k1 * length_normalization
                scores[document_index] += inverse_document_frequency * numerator / denominator

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self._chunk_ids[item[0]]),
        )[: min(top_k, len(scores))]
        return tuple(
            BackendHit(chunk_id=self._chunk_ids[index], rank=rank, score=score)
            for rank, (index, score) in enumerate(ranked, start=1)
        )

    def save(self, path: Path) -> None:
        """Persist a deterministic transparent JSON representation."""

        payload = {
            "schema_version": _SCHEMA_VERSION,
            "configuration": self.config.model_dump(mode="json"),
            "chunk_ids": self._chunk_ids,
            "document_lengths": self._document_lengths,
            "postings": {
                token: entries for token, entries in sorted(self._postings.items())
            },
        }
        write_json_artifact_exclusive(
            path,
            payload,
            relative_path=path.name,
            record_count=len(self._chunk_ids),
        )

    @classmethod
    def load(cls, path: Path, *, expected_config: BM25Config | None = None) -> BM25Index:
        """Load and validate the transparent JSON representation."""

        payload = read_json_object(path)
        try:
            if payload.get("schema_version") != _SCHEMA_VERSION:
                raise ValueError("unsupported BM25 index schema")
            config = BM25Config.model_validate(payload["configuration"])
            if expected_config is not None and config != expected_config:
                raise ValueError("BM25 index configuration does not match its manifest")
            chunk_ids = tuple(str(value) for value in payload["chunk_ids"])
            document_lengths = tuple(int(value) for value in payload["document_lengths"])
            raw_postings = payload["postings"]
            if not isinstance(raw_postings, dict):
                raise ValueError("BM25 postings must be an object")
            postings = {
                str(token): tuple((int(index), int(frequency)) for index, frequency in entries)
                for token, entries in raw_postings.items()
            }
            return cls(
                config=config,
                chunk_ids=chunk_ids,
                document_lengths=document_lengths,
                postings=postings,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IndexIntegrityError(f"invalid BM25 index {path}: {error}") from error

    def _validate(self) -> None:
        if not self._chunk_ids:
            raise IndexIntegrityError("BM25 index must contain at least one chunk")
        if len(self._chunk_ids) != len(set(self._chunk_ids)):
            raise IndexIntegrityError("BM25 chunk IDs must be unique")
        if len(self._document_lengths) != len(self._chunk_ids):
            raise IndexIntegrityError("BM25 document lengths do not align with chunk IDs")
        if any(length <= 0 for length in self._document_lengths):
            raise IndexIntegrityError("BM25 documents must contain at least one token")
        previous_token: str | None = None
        for token, entries in self._postings.items():
            if not token or (previous_token is not None and token <= previous_token):
                raise IndexIntegrityError("BM25 posting tokens must be unique and sorted")
            previous_token = token
            previous_index = -1
            for document_index, frequency in entries:
                if document_index <= previous_index or document_index >= len(self._chunk_ids):
                    raise IndexIntegrityError("BM25 posting document indices are invalid")
                if frequency <= 0:
                    raise IndexIntegrityError("BM25 term frequencies must be positive")
                previous_index = document_index
