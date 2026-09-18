"""Small storage CLI; no API server and no research execution commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from epsa_observability.connection import check_schema, create_engine
from epsa_observability.errors import StorageError
from epsa_observability.importer import import_export
from epsa_observability.migrate import upgrade
from epsa_observability.repository import ExperimentRepository


def _serialize(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError("Unsupported output value")


def main(argv: list[str] | None = None) -> int:
    # Experiment traces preserve native Unicode text. Windows consoles may otherwise default to a
    # legacy encoding that cannot render valid HotPotQA source text.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description="EPSA PostgreSQL experiment storage")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate", help="Explicitly upgrade the database schema")
    commands.add_parser("check", help="Check connectivity and schema compatibility")
    importer = commands.add_parser("import", help="Import a finalized Phase 4 export atomically")
    importer.add_argument("directory", type=Path)
    importer.add_argument(
        "--benchmark-role", choices=["development", "test", "diagnostic"], required=True
    )
    listing = commands.add_parser("list", help="List runs, including incomplete runs")
    listing.add_argument("--limit", type=int, default=50)
    listing.add_argument("--benchmark-role", choices=["development", "test", "diagnostic"])
    listing.add_argument("--status", choices=["registered", "running", "completed", "failed"])
    inspect = commands.add_parser("inspect", help="Inspect one run or one question's exact events")
    inspect.add_argument("run_id")
    inspect.add_argument("--question-id")
    args = parser.parse_args(argv)
    engine = None
    try:
        database_url = os.environ.get("EPSA_DATABASE_URL")
        if not database_url:
            raise StorageError("Set EPSA_DATABASE_URL in the process environment first.")
        engine = create_engine(database_url)
        output: Any
        if args.command == "migrate":
            upgrade(engine)
            output = {"schema": "ready"}
        elif args.command == "check":
            check_schema(engine)
            output = {"schema": "compatible"}
        else:
            repository = ExperimentRepository(engine)
            if args.command == "import":
                output = import_export(
                    repository, args.directory, benchmark_role=args.benchmark_role
                )
            elif args.command == "list":
                output = repository.list_runs(
                    limit=args.limit, benchmark_role=args.benchmark_role, status=args.status
                )
            elif args.question_id:
                output = repository.get_question_trace(args.run_id, args.question_id)
            else:
                output = repository.get_run(args.run_id)
            if output is None:
                raise StorageError("Requested run or question was not found.")
        print(json.dumps(output, ensure_ascii=False, indent=2, default=_serialize, allow_nan=False))
        return 0
    except StorageError as error:
        print(f"Experiment storage: {error}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
