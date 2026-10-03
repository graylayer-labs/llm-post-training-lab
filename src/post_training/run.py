"""Provenance recorded in every run's ``summary.json``."""

from __future__ import annotations

import platform
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

LIBRARIES = ("torch", "transformers", "trl", "peft", "datasets")


def _git(repo: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return r.stdout.strip()


def _version(lib: str) -> str | None:
    try:
        return metadata.version(lib)
    except metadata.PackageNotFoundError:
        return None


def run_provenance(repo: Path | None = None) -> dict[str, Any]:
    """Describe the code and machine a run came from.

    ``repo`` defaults to the checkout holding this package, not the CWD. A
    dirty tree (tracked changes or untracked files) marks the run
    ``scratch``. If git is unavailable or this is not a repo, the commit is
    None and the run is ``scratch`` (fail closed).
    """
    repo = repo or Path(__file__).resolve().parent
    commit = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    dirty = True if commit is None or status is None else bool(status)
    return {
        "commit": commit,
        "dirty": dirty,
        "scratch": dirty,
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "versions": {lib: _version(lib) for lib in LIBRARIES},
    }
