"""Validated access to one frozen Phase 2 corpus."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from pydantic import ValidationError

from epsa_rag.core.exceptions import IndexIntegrityError
from epsa_rag.core.models import ParagraphChunk
from epsa_rag.data.io import read_jsonl, sha256_file
from epsa_rag.data.manifests import CorpusManifest
from epsa_rag.data.validation import validate_corpus_artifact


@dataclass(frozen=True)
class FrozenCorpus:
    """Corpus records and their verified Phase 2 provenance."""

    directory: Path
    manifest: CorpusManifest
    chunks: tuple[ParagraphChunk, ...]
    by_id: Mapping[str, ParagraphChunk]
    manifest_sha256: str

    @classmethod
    def load(cls, directory: Path) -> FrozenCorpus:
        """Load a corpus only after manifest and artifact validation."""

        manifest_path = directory / "manifest.json"
        corpus_path = directory / "corpus.jsonl"
        try:
            with manifest_path.open(encoding="utf-8") as stream:
                manifest = CorpusManifest.model_validate(json.load(stream))
            validate_corpus_artifact(corpus_path, manifest)
            chunks = tuple(
                ParagraphChunk.model_validate(item) for item in read_jsonl(corpus_path)
            )
        except (OSError, ValueError, ValidationError) as error:
            message = f"unable to load frozen corpus {directory}: {error}"
            raise IndexIntegrityError(message) from error

        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(by_id) != len(chunks):
            raise IndexIntegrityError("frozen corpus contains duplicate chunk IDs")
        return cls(
            directory=directory,
            manifest=manifest,
            chunks=chunks,
            by_id=MappingProxyType(by_id),
            manifest_sha256=sha256_file(manifest_path),
        )

    @property
    def corpus_file_sha256(self) -> str:
        """Return the corpus JSONL hash frozen in its manifest."""

        return self.manifest.files[0].sha256
