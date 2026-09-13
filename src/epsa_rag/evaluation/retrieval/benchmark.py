"""Load and verify the frozen benchmark before any retrieval request."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from epsa_rag.core.exceptions import SourceValidationError
from epsa_rag.data.io import read_jsonl, sha256_file
from epsa_rag.data.manifests import DatasetManifest
from epsa_rag.data.models import BenchmarkExample
from epsa_rag.data.validation import validate_benchmark_links, validate_dataset_artifact
from epsa_rag.retrieval.corpus import FrozenCorpus


@dataclass(frozen=True)
class FrozenBenchmark:
    examples: tuple[BenchmarkExample, ...]
    manifest: DatasetManifest
    manifest_sha256: str

    @classmethod
    def load(cls, directory: Path, corpus: FrozenCorpus) -> FrozenBenchmark:
        manifest_path = directory / "manifest.json"
        path = directory / "dataset.jsonl"
        manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if len(manifest.files) != 1 or manifest.files[0].relative_path != "dataset.jsonl":
            raise SourceValidationError("expected exactly one dataset.jsonl artifact")
        validate_dataset_artifact(path, manifest)
        if manifest.version != corpus.manifest.source_dataset_version:
            raise SourceValidationError("dataset version does not match corpus source dataset")
        if manifest.source != corpus.manifest.source:
            raise SourceValidationError("dataset and corpus source provenance differ")
        if manifest.configuration != corpus.manifest.configuration:
            raise SourceValidationError("dataset and corpus preparation configurations differ")
        validate_benchmark_links(path, corpus.directory / "corpus.jsonl")
        examples = tuple(BenchmarkExample.model_validate(item) for item in read_jsonl(path))
        if not examples or len(examples) != manifest.selected_question_count:
            raise SourceValidationError("dataset selected count is empty or inconsistent")
        if any(not example.evaluation.supporting_facts for example in examples):
            raise SourceValidationError("every question must have gold supporting facts")
        return cls(examples, manifest, sha256_file(manifest_path))
