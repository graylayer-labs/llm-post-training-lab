"""Blind correctness grading of base, SFT and DPO answers.

Two steps. ``sheet`` picks prompts and writes what a grader sees, with the
three systems' answers shuffled under per-prompt labels A/B/C, and a separate
key. ``unblind`` reads the graders' grades, checks them strictly, maps them
back to systems and reports rates, agreement and a split by rubric result.

    uv run python tools/blind_grading.py sheet \\
        --gens-dir outputs/eval/generations --rows outputs/sft/eval_rows.json \\
        --n 50 --seed 0 --out outputs/grading/v1
    uv run python tools/blind_grading.py unblind --dir outputs/grading/v1 \\
        --grades outputs/grading/v1/grades_a.jsonl --grades ...

A grades file has one JSON line per prompt and label:
``{"index": 3, "label": "A", "grade": "correct|partly|wrong", "reason": "..."}``.

The rubric result per answer is recomputed with ``rubric.score`` at sheet
time (the generation files hold only text, tokens and ``stopped``) and saved
in ``key.json``, never in ``sheet.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from post_training.eval.grading import (
    SYSTEMS,
    build_sheet,
    check_generations,
    parse_grade_lines,
    pick_indices,
    render_markdown,
    rubric_overall,
    summarise,
    unblind,
    validate_grades,
)
from post_training.eval.rubric import RUBRIC_VERSION
from post_training.run import run_provenance

RESAMPLES = 1000
BOOTSTRAP_SEED = 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_gens(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return json.loads(lines[0])["cache_key"], [json.loads(ln) for ln in lines[1:]]


def cmd_sheet(args: argparse.Namespace) -> None:
    rows = json.loads(args.rows.read_text())
    gens: dict[str, list[dict[str, Any]]] = {}
    inputs = {"rows_sha256": _sha256(args.rows), "generations_sha256": {}}
    max_new: set[int] = set()
    for s in SYSTEMS:
        path = args.gens_dir / f"{s}.jsonl"
        header, gens[s] = _read_gens(path)
        inputs["generations_sha256"][s] = _sha256(path)
        max_new.add(header["generation"]["max_new_tokens"])
        try:
            check_generations(gens[s], rows, str(path))
        except ValueError as e:
            raise SystemExit(str(e)) from e
    if len(max_new) != 1:
        raise SystemExit(f"systems disagree on max_new_tokens: {max_new}")
    indices = pick_indices(len(rows), args.n, args.seed)
    sheet, key = build_sheet(rows, gens, indices, args.seed)
    rubric = rubric_overall(rows, gens, indices, max_new.pop())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sheet.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in sheet)
    )
    (args.out / "key.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n": args.n,
                "indices": indices,
                "labels": key,
                "rubric_overall": rubric,
                "rubric_version": RUBRIC_VERSION,
                "inputs": inputs,
                "provenance": run_provenance(),
            },
            indent=1,
        )
    )
    print(f"wrote {args.out}/sheet.jsonl and key.json ({len(sheet)} prompts)")


def cmd_unblind(args: argparse.Namespace) -> None:
    key_file = json.loads((args.dir / "key.json").read_text())
    sheet_indices = [
        json.loads(ln)["index"]
        for ln in (args.dir / "sheet.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    if sheet_indices != key_file["indices"]:
        raise SystemExit("sheet.jsonl indices do not match key.json")
    by_grader = {}
    for path in args.grades:
        try:
            grades = parse_grade_lines(path.read_text(), str(path))
            validate_grades(grades, key_file["indices"])
        except ValueError as e:
            raise SystemExit(f"{path}: {e}") from e
        name = path.stem.removeprefix("grades_")
        if name in by_grader:
            raise SystemExit(f"two grade files for grader {name!r}")
        by_grader[name] = unblind(grades, key_file["labels"])
    results = {
        "n_prompts": len(key_file["indices"]),
        "resamples": RESAMPLES,
        "sheet_seed": key_file["seed"],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "summary": summarise(
            by_grader,
            key_file["rubric_overall"],
            resamples=RESAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
        "grade_files_sha256": {p.name: _sha256(p) for p in args.grades},
        "key_sha256": _sha256(args.dir / "key.json"),
        "sheet_provenance": key_file["provenance"],
        "rubric_version": key_file["rubric_version"],
        "provenance": run_provenance(),
    }
    (args.dir / "results.json").write_text(json.dumps(results, indent=1))
    md = render_markdown(results)
    (args.dir / "results.md").write_text(md)
    print(md)


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--gens-dir", type=Path, required=True)
    s.add_argument("--rows", type=Path, required=True)
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(fn=cmd_sheet)
    u = sub.add_parser("unblind")
    u.add_argument("--dir", type=Path, required=True)
    u.add_argument("--grades", type=Path, action="append", required=True)
    u.set_defaults(fn=cmd_unblind)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
