"""CLI orchestration for immutable Phase 3 retrieval index construction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, NoReturn

from pydantic import ValidationError

from epsa_rag.core.exceptions import EpsaRagError
from epsa_rag.retrieval.config import BM25Config, DenseConfig
from epsa_rag.retrieval.corpus import FrozenCorpus
from epsa_rag.retrieval.dense.embeddings import OpenAIEmbeddingProvider
from epsa_rag.retrieval.persistence import BuiltIndex, build_bm25_index, build_dense_index

DEFAULT_CORPUS_DIRECTORY = Path("data/corpus/hotpotqa_10000_v1")
DEFAULT_INDEX_ROOT = Path("data/indexes")


def build_indexes(
    *,
    corpus_directory: Path,
    index_root: Path,
    kind: Literal["all", "bm25", "dense"],
    bm25_config: BM25Config,
    dense_config: DenseConfig,
) -> tuple[BuiltIndex, ...]:
    """Build the requested index kinds from one verified frozen corpus."""

    corpus = FrozenCorpus.load(corpus_directory)
    built: list[BuiltIndex] = []
    if kind in {"all", "bm25"}:
        built.append(build_bm25_index(corpus=corpus, index_root=index_root, config=bm25_config))
    if kind in {"all", "dense"}:
        provider = OpenAIEmbeddingProvider(
            dense_config,
            progress_callback=lambda completed, total: print(
                f"Embedded {completed}/{total} corpus chunks", file=sys.stderr
            ),
        )
        built.append(
            build_dense_index(
                corpus=corpus,
                index_root=index_root,
                config=dense_config,
                embedding_provider=provider,
            )
        )
    return tuple(built)


def build_parser() -> argparse.ArgumentParser:
    """Create the Phase 3 index-building command line."""

    parser = argparse.ArgumentParser(
        description="Build immutable BM25 and exact FAISS retrieval indexes."
    )
    parser.add_argument("--corpus-directory", type=Path, default=DEFAULT_CORPUS_DIRECTORY)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--kind", choices=("all", "bm25", "dense"), default="all")
    parser.add_argument("--bm25-index-version", default="bm25-v1")
    parser.add_argument(
        "--dense-index-version", default="dense-openai-small-faiss-flatip-v1"
    )
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    return parser


def _parser_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


def main() -> int:
    """Build requested indexes and print their stable locations and fingerprints."""

    parser = build_parser()
    arguments = parser.parse_args()
    try:
        bm25_config = BM25Config(
            index_version=arguments.bm25_index_version,
            candidate_k=arguments.candidate_k,
        )
        dense_config = DenseConfig(
            index_version=arguments.dense_index_version,
            candidate_k=arguments.candidate_k,
            batch_size=arguments.embedding_batch_size,
        )
        built = build_indexes(
            corpus_directory=arguments.corpus_directory,
            index_root=arguments.index_root,
            kind=arguments.kind,
            bm25_config=bm25_config,
            dense_config=dense_config,
        )
    except (EpsaRagError, ValidationError) as error:
        _parser_error(parser, str(error))

    summary = [
        {
            "configuration_fingerprint": item.manifest.configuration_fingerprint,
            "directory": str(item.directory),
            "index_kind": item.manifest.index_kind,
            "index_version": item.manifest.index_version,
        }
        for item in built
    ]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - installed entry point.
    raise SystemExit(main())
