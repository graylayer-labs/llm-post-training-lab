"""Score the dataset's own reference answers with the rubric.

The rubric decides what DPO pushes the model away from, so a rule that fails
good answers is harmful. Reference answers are the nearest thing to good
answers we have: this prints, per rule, how many references it applies to and
how many pass, plus a few failing examples to read, and saves the counts to
``outputs/rubric_references/v<RUBRIC_VERSION>/summary.json``.

    uv run python tools/rubric_on_references.py
    uv run python tools/rubric_on_references.py --show 5

References are scored as if generation stopped cleanly (``stopped=True``, one
new token), so ``clean_stop`` always passes here. ``ungrounded_numbers``
also passes by construction, because the reference is one of its sources. To
show how strict that rule is, the summary also scores each finance reference
with the reference removed as a number source, which is the position a good
model answer that cites its own figures is in.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from post_training.data.finance import load_splits
from post_training.eval.rubric import (
    RUBRIC_VERSION,
    check_ungrounded_numbers,
    is_finance_row,
    score,
)


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _provenance() -> dict[str, Any]:
    # TODO(#10): use post_training.run.run_provenance() once #10 merges.
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--dataset", default="gbharti/finance-alpaca")
    p.add_argument("--train-size", type=int, default=2000)
    p.add_argument("--eval-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show", type=int, default=3, help="failing examples per rule")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/rubric_references") / f"v{RUBRIC_VERSION}",
    )
    args = p.parse_args()

    splits = load_splits(
        args.dataset,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    rows = splits.train + splits.eval
    applies: dict[str, int] = defaultdict(int)
    passed: dict[str, int] = defaultdict(int)
    failures: dict[str, list[str]] = defaultdict(list)
    overall = 0
    for row in rows:
        res = score(row, row["output"], stopped=True, new_tokens=1, max_new_tokens=2)
        overall += res.overall
        for r in res.rules:
            if not r.applies:
                continue
            applies[r.name] += 1
            passed[r.name] += r.passed
            if not r.passed:
                failures[r.name].append(f"{row['instruction'][:70]!r}: {r.reason}")

    finance = [r for r in rows if is_finance_row(r)]
    ungrounded_without_ref = sum(
        not check_ungrounded_numbers(dict(r, output=""), r["output"]).passed
        for r in finance
    )

    summary = {
        "rubric_version": RUBRIC_VERSION,
        "data": {
            "dataset": args.dataset,
            "train_size": args.train_size,
            "eval_size": args.eval_size,
            "seed": args.seed,
            "rows_scored": len(rows),
        },
        "scored_as": {"stopped": True, "new_tokens": 1, "max_new_tokens": 2},
        "finance_rows": len(finance),
        "finance_share": _rate(len(finance), len(rows)),
        "overall": {"passed": overall, "rate": _rate(overall, len(rows))},
        "rules": {
            name: {
                "applies": applies[name],
                "passed": passed[name],
                "rate": _rate(passed[name], applies[name]),
            }
            for name in applies
        },
        "ungrounded_numbers_without_reference": {
            "finance_rows": len(finance),
            "failed": ungrounded_without_ref,
            "fail_rate": _rate(ungrounded_without_ref, len(finance)),
        },
        "provenance": _provenance(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"rubric v{RUBRIC_VERSION}, {len(rows)} reference answers")
    print(f"finance rows: {len(finance)} ({summary['finance_share']:.1%})")
    print(f"overall pass: {overall} ({summary['overall']['rate']:.1%})")
    for name, rule in summary["rules"].items():
        print(
            f"{name:22s} applies {rule['applies']:5d}  "
            f"pass {rule['passed']:5d}  ({rule['rate']:.1%})"
        )
    u = summary["ungrounded_numbers_without_reference"]
    print(
        f"ungrounded_numbers with the reference removed as a source: "
        f"{u['failed']} of {u['finance_rows']} fail ({u['fail_rate']:.1%})"
    )
    for name, items in failures.items():
        print(f"\n{name}: first {min(args.show, len(items))} of {len(items)} failures")
        for line in items[: args.show]:
            print(f"  {line}")
    print(f"\nwrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
