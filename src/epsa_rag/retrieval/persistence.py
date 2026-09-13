"""Build, publish, validate, and load immutable Phase 3 indexes."""

from __future__ import annotations

import json
import os
import platform
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from epsa_rag.core.exceptions import FrozenArtifactError, IndexIntegrityError
from epsa_rag.core.ids import stable_digest
from epsa_rag.data.manifests import ArtifactFile
from epsa_rag.retrieval.bm25.index import BM25Index
from epsa_rag.retrieval.config import BM25Config, DenseConfig
from epsa_rag.retrieval.corpus import FrozenCorpus
from epsa_rag.retrieval.dense.index import FaissFlatIPIndex
from epsa_rag.retrieval.interfaces import EmbeddingProvider
from epsa_rag.retrieval.io import describe_artifact, validate_artifact, write_manifest_exclusive
from epsa_rag.retrieval.manifests import DependencyVersion, RetrievalIndexManifest


@dataclass(frozen=True)
class BuiltIndex:
    """Published index location and validated manifest."""

    directory: Path
    manifest: RetrievalIndexManifest


def index_directory(
    index_root: Path,
    *,
    kind: str,
    corpus_version: str,
    index_version: str,
) -> Path:
    """Return the stable directory for an index version."""

    return index_root / kind / corpus_version / index_version


def build_bm25_index(
    *,
    corpus: FrozenCorpus,
    index_root: Path,
    config: BM25Config,
    generated_at: datetime | None = None,
) -> BuiltIndex:
    """Build, validate, and atomically publish one immutable BM25 index."""

    target = index_directory(
        index_root,
        kind="bm25",
        corpus_version=corpus.manifest.version,
        index_version=config.index_version,
    )
    _require_new_target(target)
    index = BM25Index.build(corpus.chunks, config)
    stage_root, stage = _new_stage(index_root)
    try:
        index_path = stage / "index.json"
        index.save(index_path)
        artifact = describe_artifact(
            index_path, relative_path="index.json", record_count=len(corpus.chunks)
        )
        manifest = _manifest(
            kind="bm25",
            config=config,
            corpus=corpus,
            files=(artifact,),
            dependencies=(
                DependencyVersion(name="python", version=platform.python_version()),
            ),
            generated_at=generated_at,
        )
        write_manifest_exclusive(stage / "manifest.json", manifest)
        loaded = load_bm25_index(stage, corpus=corpus)
        if loaded.chunk_ids != index.chunk_ids:
            raise IndexIntegrityError("published BM25 index failed its order validation")
        _publish(stage, target)
    finally:
        _remove_stage(stage_root)
    return BuiltIndex(directory=target, manifest=manifest)


def build_dense_index(
    *,
    corpus: FrozenCorpus,
    index_root: Path,
    config: DenseConfig,
    embedding_provider: EmbeddingProvider,
    generated_at: datetime | None = None,
) -> BuiltIndex:
    """Embed, build, validate, and publish an immutable exact FAISS index."""

    if embedding_provider.model != config.embedding_model:
        raise IndexIntegrityError("embedding provider model does not match dense configuration")
    if embedding_provider.dimensions != config.dimensions:
        raise IndexIntegrityError("embedding provider dimensions do not match dense configuration")
    target = index_directory(
        index_root,
        kind="dense",
        corpus_version=corpus.manifest.version,
        index_version=config.index_version,
    )
    _require_new_target(target)
    vectors = embedding_provider.embed_documents(corpus.chunks)
    index = FaissFlatIPIndex.build(
        vectors=vectors,
        chunk_ids=tuple(chunk.chunk_id for chunk in corpus.chunks),
        config=config,
    )
    stage_root, stage = _new_stage(index_root)
    try:
        index_path = stage / "index.faiss"
        ids_path = stage / "chunk_ids.json"
        index.save(index_path, ids_path)
        files = (
            describe_artifact(
                index_path, relative_path="index.faiss", record_count=len(corpus.chunks)
            ),
            describe_artifact(
                ids_path, relative_path="chunk_ids.json", record_count=len(corpus.chunks)
            ),
        )
        manifest = _manifest(
            kind="dense",
            config=config,
            corpus=corpus,
            files=files,
            dependencies=tuple(
                DependencyVersion(name=name, version=_package_version(package))
                for name, package in (
                    ("faiss-cpu", "faiss-cpu"),
                    ("numpy", "numpy"),
                    ("openai", "openai"),
                    ("python", "python"),
                )
            ),
            generated_at=generated_at,
        )
        write_manifest_exclusive(stage / "manifest.json", manifest)
        loaded = load_dense_index(stage, corpus=corpus)
        if loaded.chunk_ids != index.chunk_ids:
            raise IndexIntegrityError("published dense index failed its order validation")
        _publish(stage, target)
    finally:
        _remove_stage(stage_root)
    return BuiltIndex(directory=target, manifest=manifest)


def load_index_manifest(directory: Path) -> RetrievalIndexManifest:
    """Parse and validate a retrieval index manifest."""

    path = directory / "manifest.json"
    try:
        with path.open(encoding="utf-8") as stream:
            return RetrievalIndexManifest.model_validate(json.load(stream))
    except (OSError, ValueError, ValidationError) as error:
        raise IndexIntegrityError(f"unable to load index manifest {path}: {error}") from error


def load_bm25_index(directory: Path, *, corpus: FrozenCorpus) -> BM25Index:
    """Validate provenance and load a BM25 index."""

    manifest = load_index_manifest(directory)
    if manifest.index_kind != "bm25" or not isinstance(manifest.configuration, BM25Config):
        raise IndexIntegrityError("index manifest is not BM25")
    _validate_manifest(directory, manifest, corpus)
    index = BM25Index.load(
        directory / "index.json", expected_config=manifest.configuration
    )
    _require_corpus_order(index.chunk_ids, corpus)
    return index


def load_dense_index(directory: Path, *, corpus: FrozenCorpus) -> FaissFlatIPIndex:
    """Validate provenance and load an exact FAISS index."""

    manifest = load_index_manifest(directory)
    if manifest.index_kind != "dense" or not isinstance(manifest.configuration, DenseConfig):
        raise IndexIntegrityError("index manifest is not dense")
    _validate_manifest(directory, manifest, corpus)
    index = FaissFlatIPIndex.load(
        directory / "index.faiss",
        directory / "chunk_ids.json",
        config=manifest.configuration,
    )
    _require_corpus_order(index.chunk_ids, corpus)
    return index


def _manifest(
    *,
    kind: str,
    config: BM25Config | DenseConfig,
    corpus: FrozenCorpus,
    files: tuple[ArtifactFile, ...],
    dependencies: tuple[DependencyVersion, ...],
    generated_at: datetime | None,
) -> RetrievalIndexManifest:
    return RetrievalIndexManifest(
        index_kind=kind,
        index_version=config.index_version,
        backend=config.backend,
        generated_at=generated_at or datetime.now(UTC),
        corpus_version=corpus.manifest.version,
        corpus_manifest_sha256=corpus.manifest_sha256,
        corpus_file_sha256=corpus.corpus_file_sha256,
        chunk_count=len(corpus.chunks),
        chunk_id_order_sha256=stable_digest(
            *(chunk.chunk_id for chunk in corpus.chunks)
        ),
        configuration=config,
        configuration_fingerprint=config.fingerprint(),
        dependencies=dependencies,
        files=files,
    )


def _validate_manifest(
    directory: Path,
    manifest: RetrievalIndexManifest,
    corpus: FrozenCorpus,
) -> None:
    if manifest.corpus_version != corpus.manifest.version:
        raise IndexIntegrityError("index corpus version does not match loaded corpus")
    if manifest.corpus_manifest_sha256 != corpus.manifest_sha256:
        raise IndexIntegrityError("index corpus manifest hash does not match loaded corpus")
    if manifest.corpus_file_sha256 != corpus.corpus_file_sha256:
        raise IndexIntegrityError("index corpus file hash does not match loaded corpus")
    if manifest.chunk_count != len(corpus.chunks):
        raise IndexIntegrityError("index chunk count does not match loaded corpus")
    expected_order_hash = stable_digest(*(chunk.chunk_id for chunk in corpus.chunks))
    if manifest.chunk_id_order_sha256 != expected_order_hash:
        raise IndexIntegrityError("index chunk order hash does not match loaded corpus")
    if manifest.configuration.fingerprint() != manifest.configuration_fingerprint:
        raise IndexIntegrityError("index configuration fingerprint is invalid")
    for artifact in manifest.files:
        validate_artifact(directory / artifact.relative_path, artifact)


def _require_corpus_order(chunk_ids: tuple[str, ...], corpus: FrozenCorpus) -> None:
    expected = tuple(chunk.chunk_id for chunk in corpus.chunks)
    if chunk_ids != expected:
        raise IndexIntegrityError("index chunk IDs do not match the loaded corpus order")


def _new_stage(index_root: Path) -> tuple[Path, Path]:
    index_root.mkdir(parents=True, exist_ok=True)
    stage_root = index_root / f".epsa-phase3-{uuid4().hex}"
    stage = stage_root / "index"
    stage.mkdir(parents=True)
    return stage_root, stage


def _publish(stage: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(stage, target)
    except FileExistsError as error:
        raise FrozenArtifactError(f"refusing to overwrite retrieval index: {target}") from error


def _require_new_target(target: Path) -> None:
    if target.exists():
        raise FrozenArtifactError(f"refusing to overwrite retrieval index: {target}")


def _remove_stage(stage_root: Path) -> None:
    if stage_root.exists():
        shutil.rmtree(stage_root)


def _package_version(package: str) -> str:
    if package == "python":
        return platform.python_version()
    try:
        return version(package)
    except PackageNotFoundError as error:
        raise IndexIntegrityError(f"required package is not installed: {package}") from error
