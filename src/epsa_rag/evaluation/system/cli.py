"""Run or inspect resumable paired system evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAIError

from epsa_rag.core.exceptions import EpsaRagError
from epsa_rag.evaluation.retrieval.pipeline import DEFAULT_QUERY_CACHE_ROOT
from epsa_rag.evaluation.system.models import SystemEvaluationConfig
from epsa_rag.evaluation.system.pipeline import default_system_config, run_system_evaluation
from epsa_rag.evaluation.system.storage import load_system_export
from epsa_rag.retrieval.pipeline import DEFAULT_CORPUS_DIRECTORY, DEFAULT_INDEX_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate fixed, adaptive, and EPSA RAG systems.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run or resume a paired system evaluation")
    run.add_argument("--run-id", required=True)
    run.add_argument(
        "--dataset-directory", type=Path, default=Path("data/datasets/hotpotqa_1000_v1")
    )
    run.add_argument("--corpus-directory", type=Path, default=DEFAULT_CORPUS_DIRECTORY)
    run.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    run.add_argument("--export-root", type=Path, default=Path("data/exports/system"))
    run.add_argument("--repository-root", type=Path, default=Path.cwd())
    run.add_argument("--query-cache-root", type=Path, default=DEFAULT_QUERY_CACHE_ROOT)
    run.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 20, 30])
    run.add_argument("--question-limit", type=int)
    run.add_argument("--bm25-index-version", default="bm25-v1")
    run.add_argument("--dense-index-version", default="dense-openai-small-faiss-flatip-v1")
    run.add_argument(
        "--query-embedding-cache-version", default="query-embeddings-openai-small-v1"
    )
    run.add_argument("--openai-timeout-seconds", type=float, default=60)
    run.add_argument("--openai-max-retries", type=int, default=2)
    run.add_argument("--faiss-threads", type=int, default=1)
    run.add_argument("--allow-dirty-dev-run", action="store_true")
    inspect = commands.add_parser("inspect", help="Inspect a completed system export")
    inspect.add_argument("directory", type=Path)
    inspect.add_argument("--question-id")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            summary, traces = load_system_export(args.directory)
            if args.question_id:
                selected = tuple(
                    trace
                    for trace in traces
                    if trace.question.question_id == args.question_id
                )
                if not selected:
                    raise ValueError("question ID is not present in this export")
                print(json.dumps([item.model_dump(mode="json") for item in selected], indent=2))
            else:
                print(summary.model_dump_json(indent=2))
            return 0
        base_config = default_system_config(
            corpus_directory=args.corpus_directory,
            index_root=args.index_root,
            top_ks=tuple(sorted(set(args.top_k))),
            question_limit=args.question_limit,
            bm25_version=args.bm25_index_version,
            dense_version=args.dense_index_version,
        )
        config_data = base_config.model_dump()
        config_data.update(
            {
                "query_embedding_cache_version": args.query_embedding_cache_version,
                "openai_timeout_seconds": args.openai_timeout_seconds,
                "openai_max_retries": args.openai_max_retries,
                "faiss_threads": args.faiss_threads,
            }
        )
        config = SystemEvaluationConfig.model_validate(config_data)
        summary = run_system_evaluation(
            run_id=args.run_id,
            dataset_directory=args.dataset_directory,
            corpus_directory=args.corpus_directory,
            index_root=args.index_root,
            export_root=args.export_root,
            repository_root=args.repository_root,
            config=config,
            query_cache_root=args.query_cache_root,
            allow_dirty=args.allow_dirty_dev_run,
            progress=lambda done, total: (
                print(f"Evaluated {done}/{total} question-conditions", file=sys.stderr)
                if done % 25 == 0 or done == total
                else None
            ),
        )
        print(
            json.dumps(
                {
                    "run_id": summary.metadata.run_id,
                    "status": summary.status,
                    "completed_traces": summary.completed_traces,
                    "failed_traces": summary.failed_traces,
                    "full_benchmark": summary.full_benchmark,
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
