"""CLI for reproducible Component 01 development diagnostics and trace inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from epsa_rag.core.exceptions import EpsaRagError
from epsa_rag.data.io import read_jsonl, sha256_file
from epsa_rag.evaluation.components.question_analyzer import (
    QuestionAnalysisTrace,
    QuestionAnalyzerRunSummary,
    evaluate_questions,
    load_inference_questions,
    render_development_report,
    write_export,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the small, non-tuning Component 01 command-line interface."""

    parser = argparse.ArgumentParser(
        description="Evaluate the deterministic EPSA Question Analyzer."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the fixed development benchmark diagnostic")
    run.add_argument("--run-id", required=True)
    run.add_argument(
        "--dataset-directory", type=Path, default=Path("data/datasets/hotpotqa_1000_v1")
    )
    run.add_argument("--export-root", type=Path, default=Path("data/exports/question-analysis"))
    run.add_argument(
        "--markdown-output",
        type=Path,
        help="Optional new Markdown file for manual per-question trace review.",
    )
    inspect = commands.add_parser("inspect", help="Inspect a summary or one stored inference trace")
    inspect.add_argument("directory", type=Path)
    inspect.add_argument("--question-id")
    return parser


def main() -> int:
    """Run a diagnostic export or inspect a persisted structured trace."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            if args.question_id:
                trace = next(
                    (
                        QuestionAnalysisTrace.model_validate(record)
                        for record in read_jsonl(args.directory / "traces.jsonl")
                        if record.get("question", {}).get("question_id") == args.question_id
                    ),
                    None,
                )
                if trace is None:
                    raise ValueError("question ID is not present in this export")
                print(trace.model_dump_json(indent=2))
            else:
                print(
                    QuestionAnalyzerRunSummary.model_validate_json(
                        (args.directory / "run.json").read_text(encoding="utf-8")
                    ).model_dump_json(indent=2)
                )
            return 0
        manifest, questions = load_inference_questions(args.dataset_directory)
        summary, traces, events = evaluate_questions(
            questions,
            run_id=args.run_id,
            dataset_version=manifest.version,
            dataset_manifest_sha256=sha256_file(args.dataset_directory / "manifest.json"),
        )
        directory = write_export(args.export_root, summary=summary, traces=traces, events=events)
        if args.markdown_output is not None:
            with args.markdown_output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    render_development_report(
                        traces, run_id=summary.run_id, dataset_version=summary.dataset_version
                    )
                )
        print(
            json.dumps(
                {
                    "run_id": summary.run_id,
                    "completed_questions": summary.completed_questions,
                    "failed_questions": summary.failed_questions,
                    "diagnostics": summary.diagnostics.model_dump(),
                    "export_directory": str(directory),
                    "note": "Diagnostics are not accuracy metrics; benchmark labels are not used.",
                },
                indent=2,
            )
        )
        return 0
    except (EpsaRagError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
