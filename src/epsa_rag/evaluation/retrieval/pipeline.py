"""Compose existing frozen artifacts and retrievers for a Phase 4 benchmark."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import faiss
from openai import OpenAI

from epsa_rag.core.exceptions import ConfigurationError
from epsa_rag.core.models import RankedParagraphChunk, RetrievalQuery
from epsa_rag.evaluation.retrieval.benchmark import FrozenBenchmark
from epsa_rag.evaluation.retrieval.evaluator import CanonicalRetriever, evaluate
from epsa_rag.evaluation.retrieval.exports import DiagnosticExportSink, load_export
from epsa_rag.evaluation.retrieval.models import EvaluationConfig, RunMetadata, RunSummary
from epsa_rag.evaluation.retrieval.provenance import code_provenance, runtime_provenance
from epsa_rag.instrumentation.sinks import InstrumentationSink
from epsa_rag.retrieval.config import BM25Config, DenseConfig, HybridRetrieverConfig, RRFConfig
from epsa_rag.retrieval.corpus import FrozenCorpus
from epsa_rag.retrieval.dense.embeddings import OpenAIEmbeddingProvider
from epsa_rag.retrieval.dense.retriever import DenseRetriever
from epsa_rag.retrieval.hybrid_retriever import HybridRetriever
from epsa_rag.retrieval.interfaces import RetrievalBackend
from epsa_rag.retrieval.models import RetrievalResult
from epsa_rag.retrieval.persistence import (
    index_directory,
    load_bm25_index,
    load_dense_index,
    load_index_manifest,
)


class SingleBranchRetriever:
    """Canonical adapter for BM25/dense ablations using the exact same branch implementations."""

    def __init__(self, corpus: FrozenCorpus, backend: RetrievalBackend, version: str) -> None:
        self.corpus, self.backend, self.version = corpus, backend, version

    def retrieve(self, query: RetrievalQuery, *, top_k: int | None = None) -> RetrievalResult:
        hits = self.backend.search(query, top_k=10 if top_k is None else top_k)
        return RetrievalResult(
            query=query,
            retriever_version=self.version,
            results=tuple(
                RankedParagraphChunk(
                    chunk=self.corpus.by_id[hit.chunk_id],
                    rank=hit.rank,
                    score=hit.score,
                    source_scores={self.version: hit.score},
                    source_ranks={self.version: hit.rank},
                )
                for hit in hits
            ),
        )


def run_benchmark(
    *,
    run_id: str,
    dataset_directory: Path,
    corpus_directory: Path,
    index_root: Path,
    export_root: Path,
    repository_root: Path,
    config: EvaluationConfig,
    allow_dirty: bool = False,
    downstream: InstrumentationSink | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> RunSummary:
    """Validate all inputs before requests and reserve the immutable run before any API call."""

    commit, dirty, hashes = code_provenance(repository_root, allow_dirty=allow_dirty)
    corpus = FrozenCorpus.load(corpus_directory)
    benchmark = FrozenBenchmark.load(dataset_directory, corpus)
    if config.question_limit is not None and config.question_limit > len(benchmark.examples):
        raise ConfigurationError("question_limit exceeds the frozen benchmark size")
    examples = benchmark.examples[: config.question_limit]
    if config.warmup_questions > len(examples):
        raise ConfigurationError("warmup_questions exceeds the selected question count")
    bm25_directory = index_directory(
        index_root,
        kind="bm25",
        corpus_version=corpus.manifest.version,
        index_version=config.retriever.bm25.index_version,
    )
    dense_directory = index_directory(
        index_root,
        kind="dense",
        corpus_version=corpus.manifest.version,
        index_version=config.retriever.dense.index_version,
    )
    manifests = []
    bm25 = None
    dense_index = None
    if config.mode in {"hybrid", "bm25"}:
        bm25 = load_bm25_index(bm25_directory, corpus=corpus)
        manifests.append(load_index_manifest(bm25_directory))
        if bm25.config != config.retriever.bm25:
            raise ConfigurationError("BM25 run configuration differs from the frozen index")
    if config.mode in {"hybrid", "dense"}:
        dense_index = load_dense_index(dense_directory, corpus=corpus)
        manifests.append(load_index_manifest(dense_directory))
        if dense_index.config != config.retriever.dense:
            raise ConfigurationError("dense run configuration differs from the frozen index")
    metadata = RunMetadata(
        run_id=run_id,
        git_commit_sha=commit,
        git_dirty=dirty,
        source_sha256=hashes,
        runtime=runtime_provenance(),
        dataset_version=benchmark.manifest.version,
        corpus_version=corpus.manifest.version,
        dataset_manifest_sha256=benchmark.manifest_sha256,
        dataset_file_sha256=benchmark.manifest.files[0].sha256,
        corpus_manifest_sha256=corpus.manifest_sha256,
        corpus_file_sha256=corpus.corpus_file_sha256,
        index_manifests=tuple(manifests),
        question_ids=tuple(item.inference.question_id for item in examples),
        full_dataset_question_count=len(benchmark.examples),
        configuration=config,
        configuration_fingerprint=config.fingerprint(),
    )
    client = None
    previous_threads = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(config.faiss_threads)
        dense = None
        if dense_index is not None:
            client = OpenAI(
                timeout=config.openai_timeout_seconds,
                max_retries=config.openai_max_retries,
                base_url="https://api.openai.com/v1",
            )
            dense = DenseRetriever(
                dense_index, OpenAIEmbeddingProvider(config.retriever.dense, client=client)
            )
        retriever: CanonicalRetriever
        if bm25 is not None and dense is not None:
            retriever = HybridRetriever(
                corpus=corpus, bm25=bm25, dense=dense, config=config.retriever
            )
        elif bm25 is not None:
            retriever = SingleBranchRetriever(corpus, bm25, config.retriever.bm25.index_version)
        elif dense is not None:
            retriever = SingleBranchRetriever(corpus, dense, config.retriever.dense.index_version)
        else:  # pragma: no cover - mode validation guarantees a branch.
            raise ConfigurationError("no retrieval branch selected")
        with DiagnosticExportSink(export_root, metadata, downstream) as sink:
            summary = evaluate(
                examples=examples,
                corpus=corpus,
                retriever=retriever,
                metadata=metadata,
                sink=sink,
                progress=progress,
            )
            sink.finalize(summary)
        # Prove persisted traces round-trip before reporting successful completion to the CLI.
        load_export(sink.directory)
        return summary
    finally:
        faiss.omp_set_num_threads(previous_threads)
        if client is not None:
            client.close()


def configuration_from_indexes(
    *,
    corpus_directory: Path,
    index_root: Path,
    bm25_version: str,
    dense_version: str,
    mode: str,
    fusion: RRFConfig,
) -> HybridRetrieverConfig:
    """Use frozen index settings, avoiding accidental divergence from how indexes were built."""

    corpus = FrozenCorpus.load(corpus_directory)
    bm25 = BM25Config(index_version=bm25_version)
    dense = DenseConfig(index_version=dense_version)
    for kind, index_version in (("bm25", bm25_version), ("dense", dense_version)):
        if mode not in {"hybrid", kind}:
            continue
        manifest = load_index_manifest(
            index_directory(
                index_root,
                kind=kind,
                corpus_version=corpus.manifest.version,
                index_version=index_version,
            )
        )
        if kind == "bm25" and isinstance(manifest.configuration, BM25Config):
            bm25 = manifest.configuration
        elif kind == "dense" and isinstance(manifest.configuration, DenseConfig):
            dense = manifest.configuration
        else:
            raise ConfigurationError("index kind does not match its configuration")
    return HybridRetrieverConfig(bm25=bm25, dense=dense, fusion=fusion)
