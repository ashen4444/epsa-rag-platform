"""Run, inspect, and compare retrieval diagnostic exports from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAIError

from epsa_rag.core.exceptions import EpsaRagError
from epsa_rag.evaluation.retrieval.exports import compare_exports, load_export
from epsa_rag.evaluation.retrieval.models import EvaluationConfig
from epsa_rag.evaluation.retrieval.pipeline import (
    DEFAULT_QUERY_CACHE_ROOT,
    build_benchmark_query_cache,
    configuration_from_indexes,
    run_benchmark,
)
from epsa_rag.retrieval.config import RRFConfig
from epsa_rag.retrieval.pipeline import DEFAULT_CORPUS_DIRECTORY, DEFAULT_INDEX_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the frozen retrieval benchmark.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run an immutable diagnostic benchmark export")
    run.add_argument("--run-id", required=True)
    run.add_argument("--mode", choices=("hybrid", "bm25", "dense"), default="hybrid")
    run.add_argument(
        "--dataset-directory", type=Path, default=Path("data/datasets/hotpotqa_1000_v1")
    )
    run.add_argument("--corpus-directory", type=Path, default=DEFAULT_CORPUS_DIRECTORY)
    run.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    run.add_argument("--export-root", type=Path, default=Path("data/exports/retrieval"))
    run.add_argument("--repository-root", type=Path, default=Path.cwd())
    run.add_argument("--bm25-index-version", default="bm25-v1")
    run.add_argument("--dense-index-version", default="dense-openai-small-faiss-flatip-v1")
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--cutoffs", type=int, nargs="+", default=[1, 5, 10])
    run.add_argument("--rrf-rank-constant", type=int, default=60)
    run.add_argument("--bm25-weight", type=float, default=1.0)
    run.add_argument("--dense-weight", type=float, default=1.0)
    run.add_argument("--warmup-questions", type=int, default=0)
    run.add_argument("--question-limit", type=int)
    run.add_argument("--openai-timeout-seconds", type=float, default=60)
    run.add_argument("--openai-max-retries", type=int, default=2)
    run.add_argument("--faiss-threads", type=int, default=1)
    run.add_argument(
        "--query-embedding-cache",
        choices=("disabled", "read-only", "read-write"),
        default="disabled",
    )
    run.add_argument(
        "--query-embedding-cache-version",
        default="query-embeddings-openai-small-v1",
    )
    run.add_argument("--query-cache-root", type=Path, default=DEFAULT_QUERY_CACHE_ROOT)
    run.add_argument("--allow-dirty-dev-run", action="store_true")
    cache = commands.add_parser(
        "cache-queries", help="Build the immutable embedding cache for the frozen benchmark"
    )
    cache.add_argument(
        "--dataset-directory", type=Path, default=Path("data/datasets/hotpotqa_1000_v1")
    )
    cache.add_argument("--corpus-directory", type=Path, default=DEFAULT_CORPUS_DIRECTORY)
    cache.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    cache.add_argument("--repository-root", type=Path, default=Path.cwd())
    cache.add_argument("--query-cache-root", type=Path, default=DEFAULT_QUERY_CACHE_ROOT)
    cache.add_argument(
        "--query-embedding-cache-version",
        default="query-embeddings-openai-small-v1",
    )
    cache.add_argument("--dense-index-version", default="dense-openai-small-faiss-flatip-v1")
    cache.add_argument("--openai-timeout-seconds", type=float, default=60)
    cache.add_argument("--openai-max-retries", type=int, default=2)
    inspect = commands.add_parser("inspect", help="Inspect a run, failures, or one question")
    inspect.add_argument("directory", type=Path)
    selection = inspect.add_mutually_exclusive_group()
    selection.add_argument("--question-id")
    selection.add_argument("--failures", action="store_true")
    compare = commands.add_parser("compare", help="List paired improvements and regressions")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.add_argument("--metric", default="recall@10")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "cache-queries":
            manifest = build_benchmark_query_cache(
                dataset_directory=args.dataset_directory,
                corpus_directory=args.corpus_directory,
                index_root=args.index_root,
                repository_root=args.repository_root,
                query_cache_root=args.query_cache_root,
                cache_version=args.query_embedding_cache_version,
                dense_index_version=args.dense_index_version,
                openai_timeout_seconds=args.openai_timeout_seconds,
                openai_max_retries=args.openai_max_retries,
            )
            print(
                json.dumps(
                    {
                        "cache_version": manifest.cache_version,
                        "dataset_version": manifest.dataset_version,
                        "question_count": manifest.question_count,
                        "model": manifest.model,
                        "dimensions": manifest.dimensions,
                        "collection_manifest": str(
                            args.query_cache_root
                            / manifest.cache_version
                            / "collections"
                            / manifest.dataset_version
                            / "manifest.json"
                        ),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "compare":
            print(
                json.dumps(compare_exports(args.before, args.after, metric=args.metric), indent=2)
            )
            return 0
        if args.command == "inspect":
            summary, traces = load_export(args.directory)
            if args.question_id:
                trace = next(
                    (t for t in traces if t.question.question_id == args.question_id), None
                )
                if trace is None:
                    raise ValueError("question ID is not present in this export")
                print(trace.model_dump_json(indent=2))
            elif args.failures:
                k = max(summary.metadata.configuration.cutoffs)
                print(
                    json.dumps(
                        [
                            {
                                "question_id": t.question.question_id,
                                "question": t.question.text,
                                "status": t.status,
                                "error_type": t.error_type,
                                "recall": t.metrics[f"recall@{k}"],
                                "missing_gold_chunk_ids": t.missing_gold_identities[str(k)],
                            }
                            for t in traces
                            if t.status == "failed" or t.metrics[f"recall@{k}"] < 1
                        ],
                        indent=2,
                    )
                )
            else:
                print(summary.model_dump_json(indent=2))
            return 0
        retriever_config = configuration_from_indexes(
            corpus_directory=args.corpus_directory,
            index_root=args.index_root,
            bm25_version=args.bm25_index_version,
            dense_version=args.dense_index_version,
            mode=args.mode,
            fusion=RRFConfig(
                result_k=args.top_k,
                rank_constant=args.rrf_rank_constant,
                bm25_weight=args.bm25_weight,
                dense_weight=args.dense_weight,
            ),
        )
        config = EvaluationConfig(
            mode=args.mode,
            cutoffs=tuple(args.cutoffs),
            retriever=retriever_config,
            warmup_questions=args.warmup_questions,
            question_limit=args.question_limit,
            openai_timeout_seconds=args.openai_timeout_seconds,
            openai_max_retries=args.openai_max_retries,
            faiss_threads=args.faiss_threads,
            query_embedding_cache=args.query_embedding_cache,
            query_embedding_cache_version=args.query_embedding_cache_version,
        )
        summary = run_benchmark(
            run_id=args.run_id,
            dataset_directory=args.dataset_directory,
            corpus_directory=args.corpus_directory,
            index_root=args.index_root,
            export_root=args.export_root,
            repository_root=args.repository_root,
            config=config,
            allow_dirty=args.allow_dirty_dev_run,
            query_cache_root=args.query_cache_root,
            progress=lambda done, total: (
                print(f"Evaluated {done}/{total}", file=sys.stderr)
                if done % 25 == 0 or done == total
                else None
            ),
        )
        print(
            json.dumps(
                {
                    "run_id": summary.metadata.run_id,
                    "status": summary.status,
                    "completed_questions": summary.completed_questions,
                    "full_benchmark": summary.full_benchmark,
                    "git_dirty": summary.metadata.git_dirty,
                    "metrics": summary.metrics,
                    "latency_p50_ms": summary.latency_p50_ms,
                    "latency_p95_ms": summary.latency_p95_ms,
                    "retrieval_core_latency_p50_ms": summary.retrieval_core_latency_p50_ms,
                    "retrieval_core_latency_p95_ms": summary.retrieval_core_latency_p95_ms,
                    "throughput_questions_per_second": summary.throughput_questions_per_second,
                    "retrieval_core_throughput_questions_per_second": (
                        summary.retrieval_core_throughput_questions_per_second
                    ),
                    "export_directory": str(args.export_root / args.run_id),
                },
                indent=2,
            )
        )
        return 0 if summary.status == "completed" else 1
    except OpenAIError:
        parser.error("OpenAI client unavailable; set OPENAI_API_KEY in the process environment")
    except (EpsaRagError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
