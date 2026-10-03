"""Crash-safe, keyed JSONL files for long generation jobs.

The first line of a cache file is a header ``{"cache_key": ..., ...}``. Each
later line is one record, appended and fsynced as soon as its batch finishes,
so a crash loses at most one batch. On a re-run the records are reused only
when the header's key matches the new key exactly. A last line without its
newline, or any line that is not valid JSON, marks where a crash cut the file:
it and everything after it are dropped.

The same pattern is used by the eval harness (``eval/harness.py`` on
``feat/3-eval-harness``); once both are merged one should import the other.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def adapter_hash(adapter: Path) -> str:
    """sha256 over every file in the adapter directory, by sorted name."""
    h = hashlib.sha256()
    for f in sorted(p for p in adapter.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(adapter)).encode() + b"\0")
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to a temp file beside ``path``, then ``os.replace`` it."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_header(path: Path) -> dict[str, Any] | None:
    """The header of a cache file, or None if absent or unreadable."""
    if not path.exists():
        return None
    with path.open() as f:
        first = f.readline()
    if not first.endswith("\n"):
        return None
    try:
        header = json.loads(first)
    except ValueError:
        return None
    return header if isinstance(header, dict) else None


def read_cache(path: Path, key: dict[str, Any]) -> list[dict[str, Any]]:
    """Records cached under ``key``, or [] if the file is absent or keyed otherwise."""
    header = read_header(path)
    if header is None or header.get("cache_key") != key:
        return []
    # A complete file ends with "\n", so the final element is "". Anything
    # else there is a partial line from a crash.
    lines = path.read_text().split("\n")[1:-1]
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            break
    return rows


def append_records(f: Any, records: list[dict[str, Any]]) -> None:
    """Append one JSON line per record to an open file, then fsync it."""
    for rec in records:
        f.write(json.dumps(rec) + "\n")
    f.flush()
    os.fsync(f.fileno())
