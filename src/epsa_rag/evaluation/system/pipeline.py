"""Compose frozen artifacts and permanent RAG pipelines for paired evaluation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import faiss
from openai import OpenAI

from epsa_rag.core.exceptions import ConfigurationError
from epsa_rag.core.models import RetrievalQuery
from epsa_rag.evaluation.retrieval.benchmark import FrozenBenchmark
from epsa_rag.evaluation.retrieval.pipeline import (
    DEFAULT_QUERY_CACHE_ROOT,
    configuration_from_indexes,
)
from epsa_rag.evaluation.retrieval.provenance import code_provenance, runtime_provenance
from epsa_rag.evaluation.system.models import (
    PipelineTrace,
    SystemCondition,
    SystemEvaluationConfig,
    SystemRunMetadata,
    SystemRunSummary,
    SystemVariant,
)
from epsa_rag.evaluation.system.runner import EvaluationPipeline, evaluate_systems
from epsa_rag.evaluation.system.storage import (
    ResumableSystemRunStore,
    load_system_export,
)
from epsa_rag.evaluation.system.token_accounting import TikTokenCounter
from epsa_rag.pipeline import (
    AdaptiveBaselineConfig,
    AdaptiveBaselinePipeline,
    EPSAController,
    EPSAPipeline,
    EPSAPipelineConfig,
    FixedBaselineConfig,
    FixedBaselinePipeline,
    OpenAIAdaptiveRetrievalController,
    OpenAIFinalAnswerGenerator,
)
from epsa_rag.retrieval.config import (
    DEFAULT_HYBRID_RETRIEVER_VERSION,
    HYBRID_V2_BM25_WEIGHT,
    HYBRID_V2_DENSE_WEIGHT,
    RRFConfig,
)
from epsa_rag.retrieval.corpus import FrozenCorpus
from epsa_rag.retrieval.dense.embeddings import OpenAIEmbeddingProvider
from epsa_rag.retrieval.dense.query_cache import (
    CachedQueryEmbeddingProvider,
    QueryEmbeddingCache,
    QueryEmbeddingObservation,
)
from epsa_rag.retrieval.dense.retriever import DenseRetriever
from epsa_rag.retrieval.hybrid_retriever import HybridRetriever
from epsa_rag.retrieval.persistence import (
    index_directory,
    load_bm25_index,
    load_dense_index,
    load_index_manifest,
)


class _ObservedPipeline:
    """Associate embedding observations with exactly one serial pipeline execution."""

    def __init__(
        self,
        pipeline: FixedBaselinePipeline | AdaptiveBaselinePipeline | EPSAPipeline,
        observations: list[QueryEmbeddingObservation],
    ) -> None:
        self._pipeline = pipeline
        self._observations = observations
        self._last: tuple[QueryEmbeddingObservation, ...] = ()

    def run(self, *, question_id: str, question: str) -> PipelineTrace:
        start = len(self._observations)
        try:
            return self._pipeline.run(question_id=question_id, question=question)
        finally:
            self._last = tuple(self._observations[start:])
            del self._observations[start:]

    def take_query_embedding_observations(self) -> tuple[QueryEmbeddingObservation, ...]:
        result, self._last = self._last, ()
        return result


def run_system_evaluation(
    *,
    run_id: str,
    dataset_directory: Path,
    corpus_directory: Path,
    index_root: Path,
    export_root: Path,
    repository_root: Path,
    config: SystemEvaluationConfig,
    query_cache_root: Path = DEFAULT_QUERY_CACHE_ROOT,
    allow_dirty: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> SystemRunSummary:
    """Validate artifacts, reserve/resume a run, and execute serial paired conditions."""

    commit, dirty, hashes = code_provenance(repository_root, allow_dirty=allow_dirty)
    corpus = FrozenCorpus.load(corpus_directory)
    benchmark = FrozenBenchmark.load(dataset_directory, corpus)
    if config.question_limit is not None and config.question_limit > len(benchmark.examples):
        raise ConfigurationError("question_limit exceeds the frozen benchmark size")
    examples = benchmark.examples[: config.question_limit]
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
    bm25 = load_bm25_index(bm25_directory, corpus=corpus)
    dense_index = load_dense_index(dense_directory, corpus=corpus)
    if bm25.config != config.retriever.bm25 or dense_index.config != config.retriever.dense:
        raise ConfigurationError("system run configuration differs from frozen indexes")
    all_queries = tuple(item.inference for item in benchmark.examples)
    cache = QueryEmbeddingCache(
        query_cache_root,
        config=config.retriever.dense,
        cache_version=config.query_embedding_cache_version,
    )
    retrieval_queries = tuple(
        RetrievalQuery(text=item.text, question_id=item.question_id) for item in all_queries
    )
    cache.load_collection(
        queries=retrieval_queries,
        dataset_version=benchmark.manifest.version,
        dataset_manifest_sha256=benchmark.manifest_sha256,
        dataset_file_sha256=benchmark.manifest.files[0].sha256,
    )
    manifests = (
        load_index_manifest(bm25_directory),
        load_index_manifest(dense_directory),
    )
    metadata = SystemRunMetadata(
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
        index_manifests=manifests,
        question_ids=tuple(item.inference.question_id for item in examples),
        full_dataset_question_count=len(benchmark.examples),
        configuration=config,
        configuration_fingerprint=config.fingerprint(),
    )
    store = ResumableSystemRunStore(export_root, metadata)
    observations: list[QueryEmbeddingObservation] = []
    client = OpenAI(
        timeout=config.openai_timeout_seconds,
        max_retries=config.openai_max_retries,
        base_url="https://api.openai.com/v1",
    )
    previous_threads = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(config.faiss_threads)
        live_embeddings = OpenAIEmbeddingProvider(config.retriever.dense, client=client)
        cached_embeddings = CachedQueryEmbeddingProvider(
            config=config.retriever.dense,
            mode="read-write",
            cache=cache,
            delegate=live_embeddings,
            observer=observations.append,
        )
        dense = DenseRetriever(dense_index, cached_embeddings)
        retriever = HybridRetriever(
            corpus=corpus,
            bm25=bm25,
            dense=dense,
            config=config.retriever,
        )
        answer_generator = OpenAIFinalAnswerGenerator(client=client)
        adaptive_controller = OpenAIAdaptiveRetrievalController(client=client)
        epsa_controller = EPSAController.research_v1()

        def factory(condition: SystemCondition) -> EvaluationPipeline:
            pipeline: FixedBaselinePipeline | AdaptiveBaselinePipeline | EPSAPipeline
            if condition.system is SystemVariant.FIXED:
                pipeline = FixedBaselinePipeline(
                    retriever=retriever,
                    answer_generator=answer_generator,
                    config=FixedBaselineConfig(top_k=condition.top_k),
                )
            elif condition.system is SystemVariant.ADAPTIVE:
                pipeline = AdaptiveBaselinePipeline(
                    retriever=retriever,
                    controller=adaptive_controller,
                    answer_generator=answer_generator,
                    config=AdaptiveBaselineConfig(top_k=condition.top_k),
                )
            else:
                pipeline = EPSAPipeline(
                    retriever=retriever,
                    controller=epsa_controller,
                    answer_generator=answer_generator,
                    config=EPSAPipelineConfig(top_k=condition.top_k),
                )
            return _ObservedPipeline(pipeline, observations)

        counter = TikTokenCounter(
            model=config.tokenizer_model,
            expected_encoding=config.tokenizer_encoding,
        )
        summary = evaluate_systems(
            examples=examples,
            metadata=metadata,
            store=store,
            pipeline_factory=factory,
            token_counter=counter,
            progress=progress,
        )
        load_system_export(store.directory)
        return summary
    finally:
        faiss.omp_set_num_threads(previous_threads)
        client.close()


def default_system_config(
    *,
    corpus_directory: Path,
    index_root: Path,
    top_ks: tuple[int, ...] = (5, 10, 20, 30),
    question_limit: int | None = None,
    bm25_version: str = "bm25-v1",
    dense_version: str = "dense-openai-small-faiss-flatip-v1",
) -> SystemEvaluationConfig:
    if not top_ks:
        raise ValueError("top_ks must be nonempty")
    retriever = configuration_from_indexes(
        corpus_directory=corpus_directory,
        index_root=index_root,
        bm25_version=bm25_version,
        dense_version=dense_version,
        mode="hybrid",
        retriever_version=DEFAULT_HYBRID_RETRIEVER_VERSION,
        fusion=RRFConfig(
            result_k=max(top_ks),
            bm25_weight=HYBRID_V2_BM25_WEIGHT,
            dense_weight=HYBRID_V2_DENSE_WEIGHT,
        ),
    )
    return SystemEvaluationConfig(
        top_ks=top_ks,
        question_limit=question_limit,
        retriever=retriever,
    )
