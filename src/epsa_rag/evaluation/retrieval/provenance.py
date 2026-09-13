"""Git and runtime identity without collecting environment values or credentials."""

from __future__ import annotations

import platform
import subprocess
from importlib.metadata import distributions
from pathlib import Path

from epsa_rag.core.exceptions import ConfigurationError
from epsa_rag.data.io import sha256_file


def code_provenance(root: Path, *, allow_dirty: bool) -> tuple[str, bool, dict[str, str]]:
    """Require a real commit, detect tracked/untracked edits, and hash implemented sources."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    except (OSError, subprocess.CalledProcessError) as error:
        raise ConfigurationError("evaluation requires a Git repository with a commit") from error
    if dirty and not allow_dirty:
        raise ConfigurationError(
            "commit the implementation before a research run; "
            "--allow-dirty-dev-run permits an explicitly marked development diagnostic"
        )
    paths = [*sorted((root / "src").rglob("*.py")), root / "pyproject.toml"]
    hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}
    return commit, dirty, hashes


def runtime_provenance() -> dict[str, str]:
    """Record installed distributions and hardware/OS descriptors for timing interpretation."""

    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    for distribution in distributions():
        name = distribution.metadata.get("Name")
        if name:
            result[f"package:{name.lower()}"] = distribution.version
    return dict(sorted(result.items()))
