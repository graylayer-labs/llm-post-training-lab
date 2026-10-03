"""Provenance recorded in every run's ``summary.json``."""

from __future__ import annotations

import json
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


def install_problem(
    module_file: Path, toplevel: Path, direct_url: str | None
) -> str | None:
    """Why the imported code may not be the checked-out code, or None if it is.

    The git commit only describes the code that ran if the package is an
    editable install and its source sits in the checkout's ``src/``.
    """
    try:
        info = json.loads(direct_url or "{}").get("dir_info", {})
    except ValueError:
        info = {}
    if not info.get("editable"):
        return "package is not an editable install"
    if not module_file.resolve().is_relative_to((toplevel / "src").resolve()):
        return "package source is not inside the repository's src/"
    return None


def _default_install_problem(repo: Path) -> str | None:
    top = _git(repo, "rev-parse", "--show-toplevel")
    if top is None:
        return None
    try:
        dist = metadata.distribution("llm-post-training-lab")
        url = dist.read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        url = None
    return install_problem(Path(__file__), Path(top), url)


def run_provenance(repo: Path | None = None) -> dict[str, Any]:
    """Describe the code and machine a run came from.

    ``repo`` defaults to the checkout holding this package, not the CWD. A
    dirty tree (tracked changes or untracked files) marks the run
    ``scratch``. If git is unavailable or this is not a repo, the commit is
    None and the run is ``scratch`` (fail closed). When ``repo`` is not given,
    a non-editable install, or a package living outside the checkout's
    ``src/``, also marks the run ``scratch``. ``scratch_reason`` says why.
    """
    default = repo is None
    repo = repo or Path(__file__).resolve().parent
    commit = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    dirty = True if commit is None or status is None else bool(status)
    reason = None
    if commit is None or status is None:
        reason = "git unavailable or not a repository"
    elif default:
        reason = _default_install_problem(repo)
    if reason is None and dirty:
        reason = "working tree has uncommitted or untracked changes"
    return {
        "commit": commit,
        "dirty": dirty,
        "scratch": reason is not None,
        "scratch_reason": reason,
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "versions": {lib: _version(lib) for lib in LIBRARIES},
    }
