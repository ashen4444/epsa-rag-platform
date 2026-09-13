from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from epsa_rag.core.exceptions import FrozenArtifactError, IndexIntegrityError
from epsa_rag.data.manifests import ArtifactFile
from epsa_rag.retrieval.config import BM25Config
from epsa_rag.retrieval.io import (
    read_json_object,
    validate_artifact,
    write_json_artifact_exclusive,
    write_manifest_exclusive,
)
from epsa_rag.retrieval.manifests import DependencyVersion, RetrievalIndexManifest


def manifest() -> RetrievalIndexManifest:
    config = BM25Config()
    return RetrievalIndexManifest(
        index_kind="bm25",
        index_version=config.index_version,
        backend=config.backend,
        generated_at=datetime(2026, 9, 13, tzinfo=UTC),
        corpus_version="corpus-v1",
        corpus_manifest_sha256="a" * 64,
        corpus_file_sha256="b" * 64,
        chunk_count=1,
        chunk_id_order_sha256="c" * 64,
        configuration=config,
        configuration_fingerprint=config.fingerprint(),
        dependencies=(DependencyVersion(name="python", version="3.12.10"),),
        files=(),
    )


def test_json_and_manifest_writes_are_exclusive(tmp_path: Path) -> None:
    artifact_path = tmp_path / "index.json"
    artifact = write_json_artifact_exclusive(
        artifact_path, {"answer": 42}, relative_path="index.json", record_count=1
    )
    assert read_json_object(artifact_path) == {"answer": 42}
    validate_artifact(artifact_path, artifact)
    with pytest.raises(FrozenArtifactError, match="overwrite"):
        write_json_artifact_exclusive(
            artifact_path, {}, relative_path="index.json", record_count=0
        )

    manifest_path = tmp_path / "manifest.json"
    write_manifest_exclusive(manifest_path, manifest())
    with pytest.raises(FrozenArtifactError, match="overwrite"):
        write_manifest_exclusive(manifest_path, manifest())


def test_json_read_and_artifact_validation_reject_corruption(tmp_path: Path) -> None:
    with pytest.raises(IndexIntegrityError, match="unable to read"):
        read_json_object(tmp_path / "missing.json")
    value_path = tmp_path / "value.json"
    value_path.write_text("[]", encoding="utf-8")
    with pytest.raises(IndexIntegrityError, match="not a JSON object"):
        read_json_object(value_path)

    missing = ArtifactFile(
        relative_path="missing", sha256="a" * 64, byte_count=1, record_count=1
    )
    with pytest.raises(IndexIntegrityError, match="does not exist"):
        validate_artifact(tmp_path / "missing", missing)
    value_path.write_text("{}", encoding="utf-8")
    with pytest.raises(IndexIntegrityError, match="integrity"):
        validate_artifact(value_path, missing)
