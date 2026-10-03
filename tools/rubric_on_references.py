"""Score the dataset's own reference answers with the rubric.

The rubric decides what DPO pushes the model away from, so a rule that fails
good answers is harmful. Reference answers are the nearest thing to good
answers we have: this prints, per rule, how many references it applies to and
how many pass, plus a few failing examples to read.

    uv run python tools/rubric_on_references.py
    uv run python tools/rubric_on_references.py --show 5

References are scored as if generation stopped cleanly (``stopped=True``, one
new token), so ``clean_stop`` always passes here. ``hallucinated_numbers``
also passes by construction, because the reference is one of its sources.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from post_training.data.finance import load_splits
from post_training.eval.rubric import RUBRIC_VERSION, is_finance_row, score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="gbharti/finance-alpaca")
    p.add_argument("--train-size", type=int, default=2000)
    p.add_argument("--eval-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show", type=int, default=3, help="failing examples per rule")
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

    finance = sum(is_finance_row(r) for r in rows)
    print(f"rubric v{RUBRIC_VERSION}, {len(rows)} reference answers")
    print(f"finance rows: {finance} ({finance / len(rows):.1%})")
    print(f"overall pass: {overall} ({overall / len(rows):.1%})")
    for name in applies:
        n, k = applies[name], passed[name]
        print(f"{name:22s} applies {n:5d}  pass {k:5d}  ({k / n:.1%})")
    for name, items in failures.items():
        print(f"\n{name}: first {min(args.show, len(items))} of {len(items)} failures")
        for line in items[: args.show]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
