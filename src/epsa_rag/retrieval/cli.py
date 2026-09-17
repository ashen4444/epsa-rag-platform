"""Small inspectable command-line interface for the Phase 3 retriever."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NoReturn

from pydantic import ValidationError

from epsa_rag.core.exceptions import EpsaRagError
from epsa_rag.core.models import RetrievalQuery
from epsa_rag.retrieval.config import (
    DEFAULT_HYBRID_RETRIEVER_VERSION,
    HYBRID_V2_BM25_WEIGHT,
    HYBRID_V2_DENSE_WEIGHT,
    HybridRetrieverConfig,
    RRFConfig,
)
from epsa_rag.retrieval.corpus import FrozenCorpus
from epsa_rag.retrieval.dense.embeddings import OpenAIEmbeddingProvider
from epsa_rag.retrieval.dense.retriever import DenseRetriever
from epsa_rag.retrieval.hybrid_retriever import HybridRetriever
from epsa_rag.retrieval.persistence import (
    index_directory,
    load_bm25_index,
    load_dense_index,
    load_index_manifest,
)
from epsa_rag.retrieval.pipeline import DEFAULT_CORPUS_DIRECTORY, DEFAULT_INDEX_ROOT


def build_parser() -> argparse.ArgumentParser:
    """Create the one-query inspection command line."""

    parser = argparse.ArgumentParser(
        description="Run one query through the shared BM25 + dense + RRF retriever."
    )
    parser.add_argument("query")
    parser.add_argument("--question-id")
    parser.add_argument("--corpus-directory", type=Path, default=DEFAULT_CORPUS_DIRECTORY)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--bm25-index-version", default="bm25-v1")
    parser.add_argument(
        "--dense-index-version", default="dense-openai-small-faiss-flatip-v1"
    )
    parser.add_argument(
        "--retriever-version", default=DEFAULT_HYBRID_RETRIEVER_VERSION
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rrf-rank-constant", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=HYBRID_V2_BM25_WEIGHT)
    parser.add_argument("--dense-weight", type=float, default=HYBRID_V2_DENSE_WEIGHT)
    return parser


def _parser_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


def main() -> int:
    """Load verified indexes, retrieve once, and print canonical JSON."""

    parser = build_parser()
    arguments = parser.parse_args()
    try:
        corpus = FrozenCorpus.load(arguments.corpus_directory)
        bm25_directory = index_directory(
            arguments.index_root,
            kind="bm25",
            corpus_version=corpus.manifest.version,
            index_version=arguments.bm25_index_version,
        )
        dense_directory = index_directory(
            arguments.index_root,
            kind="dense",
            corpus_version=corpus.manifest.version,
            index_version=arguments.dense_index_version,
        )
        bm25_manifest = load_index_manifest(bm25_directory)
        dense_manifest = load_index_manifest(dense_directory)
        config = HybridRetrieverConfig(
            retriever_version=arguments.retriever_version,
            bm25=bm25_manifest.configuration,
            dense=dense_manifest.configuration,
            fusion=RRFConfig(
                rank_constant=arguments.rrf_rank_constant,
                bm25_weight=arguments.bm25_weight,
                dense_weight=arguments.dense_weight,
                result_k=arguments.top_k,
            ),
        )
        bm25 = load_bm25_index(bm25_directory, corpus=corpus)
        dense_index = load_dense_index(dense_directory, corpus=corpus)
        dense = DenseRetriever(dense_index, OpenAIEmbeddingProvider(config.dense))
        retriever = HybridRetriever(
            corpus=corpus,
            bm25=bm25,
            dense=dense,
            config=config,
        )
        result = retriever.retrieve(
            RetrievalQuery(text=arguments.query, question_id=arguments.question_id)
        )
    except (EpsaRagError, ValidationError, ValueError) as error:
        _parser_error(parser, str(error))
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - installed entry point.
    raise SystemExit(main())
