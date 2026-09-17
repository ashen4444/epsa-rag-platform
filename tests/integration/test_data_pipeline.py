from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from epsa_rag.core.exceptions import FrozenArtifactError, SourceValidationError
from epsa_rag.data.config import HardTestPreparationConfig, PreparationConfig
from epsa_rag.data.io import canonical_json, read_jsonl, sha256_file
from epsa_rag.data.manifests import ArtifactFile
from epsa_rag.data.pipeline import prepare_benchmark
from epsa_rag.data.validation import (
    validate_benchmark_links,
    validate_corpus_artifact,
    validate_dataset_artifact,
)


def write_source(path: Path) -> str:
    records = []
    for index in range(4):
        records.append(
            {
                "_id": f"q{index}",
                "question": f"Question {index}?",
                "answer": f"Answer {index}",
                "type": "comparison" if index % 2 else "bridge",
                "level": "hard",
                "supporting_facts": [["Shared", 0], [f"Unique {index}", 0]],
                "context": [
                    ["Shared", ["Shared sentence."]],
                    [f"Unique {index}", [f"Unique sentence {index}."]],
                ],
            }
        )
    path.write_text(json.dumps(records), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_publishes_validated_immutable_artifacts(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_hash = write_source(source_path)
    output_root = tmp_path / "output"
    config = PreparationConfig(
        dataset_version="dataset-test-v1",
        corpus_version="corpus-test-v1",
        question_count=3,
        selection_seed=7,
        source_uri="https://example.test/source.json",
        expected_source_sha256=source_hash,
    )
    generated_at = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)

    result = prepare_benchmark(
        source_path=source_path,
        output_root=output_root,
        config=config,
        generated_at=generated_at,
    )

    dataset_file = result.dataset_directory / "dataset.jsonl"
    corpus_file = result.corpus_directory / "corpus.jsonl"
    assert dataset_file.is_file()
    assert corpus_file.is_file()
    assert (result.dataset_directory / "manifest.json").is_file()
    assert result.dataset_manifest.selected_question_count == 3
    assert result.corpus_manifest.candidate_paragraph_count == 6
    assert result.corpus_manifest.unique_paragraph_count == 4
    assert result.corpus_manifest.duplicate_paragraph_count == 2
    assert len(tuple(read_jsonl(dataset_file))) == 3
    assert len(tuple(read_jsonl(corpus_file))) == 4
    validate_benchmark_links(dataset_file, corpus_file)

    with pytest.raises(FrozenArtifactError, match="refusing to overwrite"):
        prepare_benchmark(
            source_path=source_path,
            output_root=output_root,
            config=config,
            generated_at=generated_at,
        )


def test_pipeline_publishes_a_filtered_hard_test_benchmark(tmp_path: Path) -> None:
    source_path = tmp_path / "train.json"
    records = []
    for index, level in enumerate(("hard", "medium", "hard", "easy", "hard")):
        records.append(
            {
                "_id": f"q{index}",
                "question": f"Question {index}?",
                "answer": f"Answer {index}",
                "type": "bridge",
                "level": level,
                "supporting_facts": [[f"Title {index}", 0]],
                "context": [[f"Title {index}", [f"Sentence {index}."]]],
            }
        )
    source_path.write_text(json.dumps(records), encoding="utf-8")
    config = HardTestPreparationConfig(
        dataset_version="hard-test-dataset-v1",
        corpus_version="hard-test-corpus-v1",
        question_count=3,
        expected_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )

    result = prepare_benchmark(source_path=source_path, output_root=tmp_path / "out", config=config)

    dataset_records = tuple(read_jsonl(result.dataset_directory / "dataset.jsonl"))
    assert result.dataset_manifest.source.split == "train"
    assert result.dataset_manifest.schema_version == "1.1"
    assert result.corpus_manifest.schema_version == "1.1"
    assert result.dataset_manifest.eligible_question_count == 3
    assert result.dataset_manifest.selected_question_count == 3
    assert result.dataset_manifest.generation.generator_version == "hotpotqa-preparation-v2"
    assert all(record["evaluation"]["difficulty"] == "hard" for record in dataset_records)


def test_validation_detects_artifact_tampering(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    config = PreparationConfig(
        dataset_version="dataset-test-v1",
        corpus_version="corpus-test-v1",
        question_count=2,
        expected_source_sha256=write_source(source_path),
    )
    result = prepare_benchmark(source_path=source_path, output_root=tmp_path / "out", config=config)
    dataset_file = result.dataset_directory / "dataset.jsonl"
    dataset_file.write_text("{}\n", encoding="utf-8")

    with pytest.raises(SourceValidationError, match="integrity check failed"):
        validate_dataset_artifact(dataset_file, result.dataset_manifest)


def test_validation_detects_schema_and_identity_mismatches(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    config = PreparationConfig(
        dataset_version="dataset-test-v1",
        corpus_version="corpus-test-v1",
        question_count=3,
        expected_source_sha256=write_source(source_path),
    )
    result = prepare_benchmark(source_path=source_path, output_root=tmp_path / "out", config=config)
    dataset_file = result.dataset_directory / "dataset.jsonl"
    corpus_file = result.corpus_directory / "corpus.jsonl"

    mismatched_dataset_manifest = result.dataset_manifest.model_copy(
        update={"question_ids": tuple(reversed(result.dataset_manifest.question_ids))}
    )
    with pytest.raises(SourceValidationError, match="does not match"):
        validate_dataset_artifact(dataset_file, mismatched_dataset_manifest)

    dataset_file.write_text("{}\n", encoding="utf-8")
    invalid_dataset_file = ArtifactFile(
        relative_path="dataset.jsonl",
        sha256=sha256_file(dataset_file),
        byte_count=dataset_file.stat().st_size,
        record_count=1,
    )
    invalid_dataset_manifest = result.dataset_manifest.model_copy(
        update={"files": (invalid_dataset_file,)}
    )
    with pytest.raises(SourceValidationError, match="invalid dataset artifact"):
        validate_dataset_artifact(dataset_file, invalid_dataset_manifest)

    corpus_records = list(read_jsonl(corpus_file))
    corpus_file.write_text(
        "".join(f"{canonical_json(record)}\n" for record in reversed(corpus_records)),
        encoding="utf-8",
    )
    reversed_chunk_ids = tuple(reversed(result.corpus_manifest.chunk_ids))
    reversed_corpus_file = ArtifactFile(
        relative_path="corpus.jsonl",
        sha256=sha256_file(corpus_file),
        byte_count=corpus_file.stat().st_size,
        record_count=len(corpus_records),
    )
    reversed_corpus_manifest = result.corpus_manifest.model_copy(
        update={"chunk_ids": reversed_chunk_ids, "files": (reversed_corpus_file,)}
    )
    with pytest.raises(SourceValidationError, match="unique and sorted"):
        validate_corpus_artifact(corpus_file, reversed_corpus_manifest)
