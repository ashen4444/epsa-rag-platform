"""Small deterministic two-support benchmark shared by Phase 4 unit/integration tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from epsa_rag.core.models import ParagraphChunk, RetrievalQuery
from epsa_rag.data.config import PreparationConfig
from epsa_rag.data.pipeline import prepare_benchmark
from epsa_rag.evaluation.retrieval.benchmark import FrozenBenchmark
from epsa_rag.evaluation.retrieval.models import EvaluationConfig, RunMetadata
from epsa_rag.retrieval.corpus import FrozenCorpus
from epsa_rag.retrieval.interfaces import FloatMatrix, FloatVector


class EvaluationEmbeddings:
    """Offline fixture only; never used to report real dense retrieval accuracy."""

    model = "text-embedding-3-small"
    dimensions = 1536

    def embed_documents(self, chunks: Sequence[ParagraphChunk]) -> FloatMatrix:
        result = np.zeros((len(chunks), self.dimensions), dtype=np.float32)
        result[:, 0] = 1
        return result

    def embed_query(self, query: RetrievalQuery) -> FloatVector:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        vector[0] = 1
        return vector

    def embed_queries(self, queries: Sequence[RetrievalQuery]) -> FloatMatrix:
        return np.vstack([self.embed_query(query) for query in queries]).astype(np.float32)


@pytest.fixture
def evaluation_embeddings() -> EvaluationEmbeddings:
    return EvaluationEmbeddings()


@pytest.fixture
def evaluation_data(tmp_path: Path) -> tuple[FrozenCorpus, FrozenBenchmark, Path]:
    source = tmp_path / "source.json"
    records = [
        {
            "_id": f"q{i}",
            "question": f"Where did Alpha {i} meet Beta?",
            "answer": "SECRET GOLD",
            "type": "bridge",
            "level": "hard",
            "supporting_facts": [["Alpha", 0], ["Beta", 0]],
            "context": [
                ["Alpha", [f" Alpha {i} met Beta. ", " Another native sentence.\n"]],
                ["Beta", ["Beta was in London."]],
                ["Distractor", ["Unrelated material."]],
            ],
        }
        for i in range(3)
    ]
    source.write_text(json.dumps(records), encoding="utf-8")
    config = PreparationConfig(
        dataset_version="eval-dataset-v1",
        corpus_version="eval-corpus-v1",
        question_count=3,
        expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    prepared = prepare_benchmark(source_path=source, output_root=tmp_path / "data", config=config)
    corpus = FrozenCorpus.load(prepared.corpus_directory)
    benchmark = FrozenBenchmark.load(prepared.dataset_directory, corpus)
    return corpus, benchmark, prepared.dataset_directory


@pytest.fixture
def evaluation_metadata(evaluation_data: tuple) -> RunMetadata:
    corpus, benchmark, _ = evaluation_data
    config = EvaluationConfig()
    return RunMetadata(
        run_id="test-evaluation-v1",
        git_commit_sha="a" * 40,
        git_dirty=False,
        source_sha256={"src/example.py": "b" * 64},
        runtime={"python": "3.12.10"},
        dataset_version=benchmark.manifest.version,
        corpus_version=corpus.manifest.version,
        dataset_manifest_sha256=benchmark.manifest_sha256,
        dataset_file_sha256=benchmark.manifest.files[0].sha256,
        corpus_manifest_sha256=corpus.manifest_sha256,
        corpus_file_sha256=corpus.corpus_file_sha256,
        index_manifests=(),
        question_ids=tuple(item.inference.question_id for item in benchmark.examples),
        full_dataset_question_count=len(benchmark.examples),
        configuration=config,
        configuration_fingerprint=config.fingerprint(),
    )
