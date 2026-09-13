"""Post-write validation for frozen Phase 2 artifacts."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from epsa_rag.core.exceptions import SourceValidationError
from epsa_rag.core.ids import make_evidence_unit_id
from epsa_rag.core.models import ParagraphChunk
from epsa_rag.data.io import read_jsonl, sha256_file
from epsa_rag.data.manifests import CorpusManifest, DatasetManifest
from epsa_rag.data.models import BenchmarkExample


def validate_dataset_artifact(path: Path, manifest: DatasetManifest) -> None:
    """Verify dataset checksum, count, schema, ordering, and question identity."""

    artifact = manifest.files[0]
    _require_file_integrity(path, artifact.sha256, artifact.byte_count)
    try:
        records = tuple(BenchmarkExample.model_validate(item) for item in read_jsonl(path))
    except (ValidationError, ValueError) as error:
        raise SourceValidationError(f"invalid dataset artifact {path}: {error}") from error
    question_ids = tuple(record.inference.question_id for record in records)
    if len(records) != artifact.record_count or question_ids != manifest.question_ids:
        raise SourceValidationError("dataset artifact does not match its manifest")
    if len(question_ids) != len(set(question_ids)):
        raise SourceValidationError("dataset question ids must be unique")


def validate_corpus_artifact(path: Path, manifest: CorpusManifest) -> None:
    """Verify corpus checksum, count, schema, ordering, and chunk identity."""

    artifact = manifest.files[0]
    _require_file_integrity(path, artifact.sha256, artifact.byte_count)
    try:
        records = tuple(ParagraphChunk.model_validate(item) for item in read_jsonl(path))
    except (ValidationError, ValueError) as error:
        raise SourceValidationError(f"invalid corpus artifact {path}: {error}") from error
    chunk_ids = tuple(record.chunk_id for record in records)
    if len(records) != artifact.record_count or chunk_ids != manifest.chunk_ids:
        raise SourceValidationError("corpus artifact does not match its manifest")
    if chunk_ids != tuple(sorted(set(chunk_ids))):
        raise SourceValidationError("corpus chunk ids must be unique and sorted")


def validate_benchmark_links(dataset_path: Path, corpus_path: Path) -> None:
    """Ensure every gold supporting label resolves to its exact corpus sentence."""

    try:
        corpus = {
            chunk.chunk_id: chunk
            for chunk in (ParagraphChunk.model_validate(item) for item in read_jsonl(corpus_path))
        }
        examples = (
            BenchmarkExample.model_validate(item) for item in read_jsonl(dataset_path)
        )
        for example in examples:
            for label in example.evaluation.supporting_facts:
                chunk = corpus.get(label.chunk_id)
                if chunk is None:
                    raise SourceValidationError(
                        f"supporting chunk {label.chunk_id} is absent from the corpus"
                    )
                if label.title != chunk.title or label.sentence_index >= len(chunk.sentences):
                    raise SourceValidationError(
                        f"supporting label does not resolve: {label.evidence_unit_id}"
                    )
                expected_id = make_evidence_unit_id(chunk.chunk_id, label.sentence_index)
                if label.evidence_unit_id != expected_id:
                    raise SourceValidationError(
                        f"invalid evidence unit id: {label.evidence_unit_id}"
                    )
    except (ValidationError, ValueError) as error:
        raise SourceValidationError(f"unable to validate benchmark links: {error}") from error


def _require_file_integrity(path: Path, expected_sha256: str, expected_bytes: int) -> None:
    if not path.is_file():
        raise SourceValidationError(f"artifact file does not exist: {path}")
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
        raise SourceValidationError(f"artifact integrity check failed: {path}")
