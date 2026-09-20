"""Run and inspect fixed development Component 03 diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from epsa_rag.core.exceptions import EpsaRagError
from epsa_rag.data.io import sha256_file
from epsa_rag.evaluation.components.evidence_units import (
    EvidenceUnitEvaluationConfig,
    EvidenceUnitRunSummary,
    evaluate_evidence_units,
    gold_coverage_diagnostics,
    inspect_trace,
    load_development_inputs,
    load_export,
    write_export,
)
from epsa_rag.evaluation.retrieval.provenance import code_provenance, runtime_provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate EPSA Component 03 on fixed development retrieval"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Create an inference-only development export")
    run.add_argument("--run-id", required=True)
    run.add_argument("--dataset-directory", type=Path, required=True)
    run.add_argument("--retrieval-export-directory", type=Path, required=True)
    run.add_argument("--export-root", type=Path, default=Path("data/exports/evidence-units"))
    run.add_argument("--retrieval-depth", type=int, default=10)
    run.add_argument(
        "--chunk-mode", choices=("rule_based_v1", "rule_based_v2"), default="rule_based_v2"
    )
    run.add_argument(
        "--unit-mode", choices=("rule_based_v1", "rule_based_v2"), default="rule_based_v2"
    )
    inspect = commands.add_parser("inspect", help="Inspect a run or question trace")
    inspect.add_argument("directory", type=Path)
    inspect.add_argument("--question-id")
    gold = commands.add_parser("gold-coverage", help="Join development gold labels after inference")
    gold.add_argument("directory", type=Path)
    gold.add_argument("--dataset-directory", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            if args.question_id:
                print(inspect_trace(args.directory, args.question_id).model_dump_json(indent=2))
            else:
                print(
                    EvidenceUnitRunSummary.model_validate_json(
                        (args.directory / "run.json").read_text(encoding="utf-8")
                    ).model_dump_json(indent=2)
                )
            return 0
        if args.command == "gold-coverage":
            _, traces = load_export(args.directory)
            diagnostics = gold_coverage_diagnostics(traces, args.dataset_directory)
            print(
                json.dumps(
                    {
                        "gold_sentence_count": sum(
                            item.gold_sentence_count for item in diagnostics
                        ),
                        "retrieved_gold_sentence_count": sum(
                            item.retrieved_gold_sentence_count for item in diagnostics
                        ),
                        "questions_with_all_gold_sentences": sum(
                            not item.missing_gold_evidence_unit_ids for item in diagnostics
                        ),
                        "note": (
                            "Evaluation-only retrieved gold sentence coverage; "
                            "not extractor accuracy."
                        ),
                    },
                    indent=2,
                )
            )
            return 0
        config = EvidenceUnitEvaluationConfig(
            chunk_mode=args.chunk_mode,
            unit_mode=args.unit_mode,
            retrieval_depth=args.retrieval_depth,
        )
        manifest, corpus_version, retrieval_run_id, _, inputs = load_development_inputs(
            args.dataset_directory,
            args.retrieval_export_directory,
            depth=config.retrieval_depth,
        )
        git_sha, git_dirty, source_hashes = code_provenance(
            Path(__file__).resolve().parents[4], allow_dirty=True
        )
        summary, traces, events = evaluate_evidence_units(
            inputs,
            run_id=args.run_id,
            dataset_version=manifest.version,
            dataset_manifest_sha256=sha256_file(args.dataset_directory / "manifest.json"),
            corpus_version=corpus_version,
            retrieval_run_id=retrieval_run_id,
            git_commit_sha=git_sha,
            git_dirty=git_dirty,
            source_sha256=source_hashes,
            runtime=runtime_provenance(),
            config=config,
        )
        directory = write_export(args.export_root, summary=summary, traces=traces, events=events)
        print(
            json.dumps(
                {
                    "run_id": summary.run_id,
                    "completed_questions": summary.completed_questions,
                    "failed_questions": summary.failed_questions,
                    "diagnostics": summary.diagnostics.model_dump(),
                    "export_directory": str(directory),
                    "note": "Structural diagnostics are not Component 03 accuracy metrics.",
                },
                indent=2,
            )
        )
        return 0
    except (EpsaRagError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
