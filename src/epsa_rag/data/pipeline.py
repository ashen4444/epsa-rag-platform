"""Orchestration and CLI for the deterministic Phase 2 preparation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn
from uuid import uuid4

from epsa_rag.core.exceptions import DataPreparationError, FrozenArtifactError
from epsa_rag.data.builder import build_artifacts
from epsa_rag.data.config import (
    HardTestPreparationConfig,
    PreparationConfig,
    PreparationConfiguration,
)
from epsa_rag.data.hotpotqa import download_source, load_selected_source
from epsa_rag.data.io import sha256_file, write_json_exclusive, write_jsonl_exclusive
from epsa_rag.data.manifests import (
    CorpusManifest,
    DatasetManifest,
    GenerationManifest,
    SourceManifest,
)
from epsa_rag.data.validation import (
    validate_benchmark_links,
    validate_corpus_artifact,
    validate_dataset_artifact,
)

DEFAULT_SOURCE_PATH = Path("data/raw/hotpotqa/hotpot_dev_distractor_v1.json")
HARD_TEST_SOURCE_PATH = Path("data/raw/hotpotqa/hotpot_train_v1.1.json")
DEFAULT_OUTPUT_ROOT = Path("data")


@dataclass(frozen=True)
class PreparationResult:
    """Published paths and validated manifests from one preparation run."""

    dataset_directory: Path
    corpus_directory: Path
    dataset_manifest: DatasetManifest
    corpus_manifest: CorpusManifest


def prepare_benchmark(
    *,
    source_path: Path,
    output_root: Path,
    config: PreparationConfiguration,
    generated_at: datetime | None = None,
) -> PreparationResult:
    """Create and validate frozen dataset and corpus versions without overwrite."""

    dataset_directory = output_root / "datasets" / config.dataset_version
    corpus_directory = output_root / "corpus" / config.corpus_version
    _require_new_targets(dataset_directory, corpus_directory)

    selected_source = load_selected_source(source_path, config=config)
    prepared = build_artifacts(selected_source.examples)
    source_manifest = SourceManifest(
        dataset=config.source_dataset,
        configuration=config.source_configuration,
        split=config.source_split,
        uri=config.source_uri,
        retrieved_from_uri=config.download_uri,
        filename=source_path.name,
        sha256=sha256_file(source_path),
        byte_count=source_path.stat().st_size,
        record_count=selected_source.source_record_count,
        fully_validated_record_count=(
            selected_source.source_record_count
            - len(selected_source.invalid_unselected_question_ids)
        ),
        invalid_unselected_question_ids=selected_source.invalid_unselected_question_ids,
    )
    generation = GenerationManifest(
        generated_at=generated_at or datetime.now(UTC),
        configuration_fingerprint=config.fingerprint(),
        generator_version=(
            "hotpotqa-preparation-v2"
            if isinstance(config, HardTestPreparationConfig)
            else "hotpotqa-preparation-v1"
        ),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / f".epsa-phase2-{uuid4().hex}"
    stage_root.mkdir()
    try:
        stage_dataset = stage_root / "dataset"
        stage_corpus = stage_root / "corpus"
        dataset_file = stage_dataset / "dataset.jsonl"
        corpus_file = stage_corpus / "corpus.jsonl"

        dataset_artifact = write_jsonl_exclusive(
            dataset_file,
            prepared.examples,
            relative_path="dataset.jsonl",
        )
        corpus_artifact = write_jsonl_exclusive(
            corpus_file,
            prepared.corpus,
            relative_path="corpus.jsonl",
        )
        dataset_manifest = DatasetManifest(
            schema_version=(
                "1.1" if isinstance(config, HardTestPreparationConfig) else "1.0"
            ),
            version=config.dataset_version,
            source=source_manifest,
            generation=generation,
            configuration=config,
            source_question_count=selected_source.source_record_count,
            eligible_question_count=selected_source.eligible_record_count,
            selected_question_count=len(prepared.examples),
            question_ids=tuple(example.inference.question_id for example in prepared.examples),
            files=(dataset_artifact,),
        )
        corpus_manifest = CorpusManifest(
            schema_version=(
                "1.1" if isinstance(config, HardTestPreparationConfig) else "1.0"
            ),
            version=config.corpus_version,
            source_dataset_version=config.dataset_version,
            source=source_manifest,
            generation=generation,
            configuration=config,
            candidate_paragraph_count=prepared.candidate_paragraph_count,
            unique_paragraph_count=len(prepared.corpus),
            duplicate_paragraph_count=prepared.duplicate_paragraph_count,
            chunk_ids=tuple(chunk.chunk_id for chunk in prepared.corpus),
            files=(corpus_artifact,),
        )
        write_json_exclusive(stage_dataset / "manifest.json", dataset_manifest)
        write_json_exclusive(stage_corpus / "manifest.json", corpus_manifest)
        validate_dataset_artifact(dataset_file, dataset_manifest)
        validate_corpus_artifact(corpus_file, corpus_manifest)
        validate_benchmark_links(dataset_file, corpus_file)

        dataset_directory.parent.mkdir(parents=True, exist_ok=True)
        corpus_directory.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_dataset, dataset_directory)
        try:
            os.replace(stage_corpus, corpus_directory)
        except Exception:
            shutil.rmtree(dataset_directory)
            raise
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)

    return PreparationResult(
        dataset_directory=dataset_directory,
        corpus_directory=corpus_directory,
        dataset_manifest=dataset_manifest,
        corpus_manifest=corpus_manifest,
    )


def _require_new_targets(dataset_directory: Path, corpus_directory: Path) -> None:
    existing = [path for path in (dataset_directory, corpus_directory) if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FrozenArtifactError(f"refusing to overwrite existing artifact version(s): {joined}")


def _parser_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


def build_parser() -> argparse.ArgumentParser:
    """Create the Phase 2 command-line interface."""

    parser = argparse.ArgumentParser(
        description="Prepare the deterministic EPSA-RAG HotPotQA benchmark and global corpus."
    )
    parser.add_argument(
        "--profile",
        choices=("development", "hard-test"),
        default="development",
        help="Select the frozen source, filtering, and artifact defaults",
    )
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--question-count", type=int)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--dataset-version")
    parser.add_argument("--corpus-version")
    return parser


def main() -> int:
    """Run preparation from the command line and print an inspectable summary."""

    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.profile == "hard-test":
        defaults: PreparationConfiguration = HardTestPreparationConfig()
        config: PreparationConfiguration = HardTestPreparationConfig(
            dataset_version=(
                arguments.dataset_version
                if arguments.dataset_version is not None
                else defaults.dataset_version
            ),
            corpus_version=(
                arguments.corpus_version
                if arguments.corpus_version is not None
                else defaults.corpus_version
            ),
            question_count=(
                arguments.question_count
                if arguments.question_count is not None
                else defaults.question_count
            ),
            selection_seed=(
                arguments.selection_seed
                if arguments.selection_seed is not None
                else defaults.selection_seed
            ),
        )
        default_source_path = HARD_TEST_SOURCE_PATH
    else:
        defaults = PreparationConfig()
        config = PreparationConfig(
            dataset_version=(
                arguments.dataset_version
                if arguments.dataset_version is not None
                else defaults.dataset_version
            ),
            corpus_version=(
                arguments.corpus_version
                if arguments.corpus_version is not None
                else defaults.corpus_version
            ),
            question_count=(
                arguments.question_count
                if arguments.question_count is not None
                else defaults.question_count
            ),
            selection_seed=(
                arguments.selection_seed
                if arguments.selection_seed is not None
                else defaults.selection_seed
            ),
        )
        default_source_path = DEFAULT_SOURCE_PATH
    source_path: Path = arguments.source_path or default_source_path
    if arguments.download:
        download_source(
            uri=config.download_uri,
            destination=source_path,
            expected_sha256=config.expected_source_sha256,
        )
    elif not source_path.is_file():
        _parser_error(parser, f"source file does not exist: {source_path}; pass --download")

    try:
        result = prepare_benchmark(
            source_path=source_path,
            output_root=arguments.output_root,
            config=config,
        )
    except DataPreparationError as error:
        _parser_error(parser, str(error))

    summary = {
        "corpus_directory": str(result.corpus_directory),
        "dataset_directory": str(result.dataset_directory),
        "duplicate_paragraphs": result.corpus_manifest.duplicate_paragraph_count,
        "eligible_questions": result.dataset_manifest.eligible_question_count,
        "selected_questions": result.dataset_manifest.selected_question_count,
        "unique_paragraphs": result.corpus_manifest.unique_paragraph_count,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the installed entry point.
    raise SystemExit(main())
