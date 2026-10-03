import subprocess
from pathlib import Path

from post_training.run import run_provenance


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _make_repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-b", "main")
    (path / "a.txt").write_text("one")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-m", "init")
    return path


def test_clean_tree_is_not_scratch(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "r")
    p = run_provenance(repo)
    assert p["dirty"] is False
    assert p["scratch"] is False
    assert p["branch"] == "main"
    assert isinstance(p["commit"], str) and len(p["commit"]) == 40


def test_modified_file_is_dirty(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "r")
    (repo / "a.txt").write_text("two")
    p = run_provenance(repo)
    assert p["dirty"] is True
    assert p["scratch"] is True


def test_untracked_file_is_dirty(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "r")
    (repo / "new.txt").write_text("x")
    assert run_provenance(repo)["dirty"] is True


def test_not_a_repo_fails_closed(tmp_path: Path) -> None:
    p = run_provenance(tmp_path)
    assert p["commit"] is None
    assert p["scratch"] is True


def test_records_versions_and_machine() -> None:
    p = run_provenance()
    for lib in ("torch", "transformers", "trl", "peft", "datasets"):
        assert p["versions"][lib]
    assert p["python"]
    assert p["hostname"]
    assert p["platform"]
