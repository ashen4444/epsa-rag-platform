from __future__ import annotations

import json

import pytest

from epsa_rag.core.exceptions import SourceValidationError
from epsa_rag.data.io import sha256_file
from epsa_rag.evaluation.retrieval.benchmark import FrozenBenchmark


@pytest.mark.parametrize(
    "problem", ["files", "version", "source", "configuration", "count", "gold"]
)
def test_frozen_benchmark_rejects_inconsistent_artifacts(evaluation_data, problem):
    corpus, _, directory = evaluation_data
    path = directory / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if problem == "files":
        data["files"] = []
    elif problem == "version":
        data["version"] = "wrong-version"
    elif problem == "source":
        data["source"]["sha256"] = "a" * 64
    elif problem == "configuration":
        data["configuration"]["selection_seed"] = 99
    elif problem == "count":
        data["selected_question_count"] = 20
    else:
        dataset_path = directory / "dataset.jsonl"
        rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
        rows[0]["evaluation"]["supporting_facts"] = []
        dataset_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        data["files"][0]["sha256"] = sha256_file(dataset_path)
        data["files"][0]["byte_count"] = dataset_path.stat().st_size
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SourceValidationError):
        FrozenBenchmark.load(directory, corpus)
