from __future__ import annotations

from pathlib import Path

import pytest

from epsa_rag.core.exceptions import FrozenArtifactError
from epsa_rag.data.io import read_jsonl, sha256_file, write_json_exclusive, write_jsonl_exclusive
from epsa_rag.data.models import QuestionInput


def test_jsonl_helpers_write_canonical_integrity_metadata(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    artifact = write_jsonl_exclusive(
        path,
        (QuestionInput(question_id="q1", text="Question?"),),
        relative_path="records.jsonl",
    )

    assert artifact.sha256 == sha256_file(path)
    assert artifact.byte_count == path.stat().st_size
    assert artifact.record_count == 1
    assert tuple(read_jsonl(path)) == ({"question_id": "q1", "text": "Question?"},)

    with pytest.raises(FrozenArtifactError, match="refusing to overwrite"):
        write_jsonl_exclusive(path, (), relative_path="records.jsonl")


def test_manifest_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    model = QuestionInput(question_id="q1", text="Question?")
    write_json_exclusive(path, model)

    with pytest.raises(FrozenArtifactError, match="frozen manifest"):
        write_json_exclusive(path, model)


@pytest.mark.parametrize("content", ["not-json\n", "[]\n"])
def test_read_jsonl_rejects_invalid_records(tmp_path: Path, content: str) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        tuple(read_jsonl(path))
