"""Compare saved eval generations between systems, on CPU, without a model.

Reads the cached generations the eval harness wrote (one ``<system>.jsonl``
per system) and the held-out rows they answer, re-scores each answer with the
rubric, and saves length distributions, per-system rubric rates, and
like-for-like comparisons between pairs of systems to one JSON file. Its
first use is to test whether DPO's gains come from shorter answers: rejected
pair answers were about 2.9 times longer than chosen (docs/dpo-results.md).

    uv run python tools/compare_systems.py <generations dir> \\
        --eval-rows outputs/sft/eval_rows.json --out <file>

``<generations dir>`` is the harness's ``generations/`` directory, given as an
argument so the same tool works on any eval run.

For each ordered pair ``a->b`` the output has:

- ``both_stopped``: the prompts where both systems stopped cleanly (stop token
  before the limit), with mean new tokens, ROUGE-L and rubric pass rate for
  each. This removes never-stopping loops from both sides.
- ``b_on_prompts_where_a_stopped``: system b scored on only the prompts where
  a stopped cleanly, to set beside a's ``rubric_overall_when_stopped``.
- ``fixed`` / ``regressed``: prompts where a fails the rubric overall and b
  passes, and the reverse.
- ``b_shorter``: prompts where b used fewer new tokens than a.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rouge_score import rouge_scorer

from post_training.data.finance import Row, prompt_key
from post_training.eval.metrics import ROUGE_TYPE, ROUGE_USE_STEMMER, percentile
from post_training.eval.rubric import RUBRIC_VERSION, RubricResult, score
from post_training.run import run_provenance

Gen = dict[str, Any]


def read_generations(path: Path, rows: Sequence[Row]) -> list[Gen]:
    """The generation records in ``path``, checked against ``rows`` by order.

    The first line is the harness's cache-key header and is skipped. Raises
    ValueError if the count, an index or a prompt key does not match.
    """
    lines = path.read_text().splitlines()
    gens = [json.loads(line) for line in lines[1:] if line.strip()]
    if len(gens) != len(rows):
        raise ValueError(f"{path}: {len(gens)} generations for {len(rows)} rows")
    for i, (g, r) in enumerate(zip(gens, rows, strict=True)):
        if g.get("index") != i or g.get("prompt_key") != prompt_key(r):
            raise ValueError(f"{path}: row {i} does not match the eval rows")
    return gens


def _bins(max_new_tokens: int, short_below: int) -> list[tuple[str, int, int]]:
    edges = [0, short_below, 50, 100, 200, max_new_tokens]
    bins = [
        (f"<{hi}" if lo == 0 else f"{lo}-{hi - 1}", lo, hi)
        for lo, hi in zip(edges, edges[1:], strict=False)
        if hi > lo
    ]
    return [*bins, (f">={max_new_tokens}", max_new_tokens, 10**9)]


def length_stats(
    tokens: Sequence[int], *, max_new_tokens: int, short_below: int
) -> dict[str, Any]:
    """Distribution of new-token counts, with a histogram and short count."""
    xs = [float(t) for t in tokens]
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p10": percentile(xs, 10),
        "p25": percentile(xs, 25),
        "p75": percentile(xs, 75),
        "p90": percentile(xs, 90),
        "min": min(tokens),
        "max": max(tokens),
        "at_limit": sum(t >= max_new_tokens for t in tokens),
        "short": sum(t < short_below for t in tokens),
        "histogram": {
            label: sum(lo <= t < hi for t in tokens)
            for label, lo, hi in _bins(max_new_tokens, short_below)
        },
    }


def _kn(flags: Sequence[bool]) -> dict[str, Any]:
    n = len(flags)
    return {"k": sum(flags), "n": n, "rate": sum(flags) / n if n else None}


def _mean(xs: Sequence[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _clean(g: Gen, max_new_tokens: int) -> bool:
    return bool(g["stopped"]) and g["new_tokens"] < max_new_tokens


def analyse(
    rows: Sequence[Row],
    gens: dict[str, list[Gen]],
    *,
    max_new_tokens: int,
    short_below: int,
    n_examples: int,
    pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Per-system statistics and pairwise comparisons; see the module doc."""
    scorer = rouge_scorer.RougeScorer([ROUGE_TYPE], use_stemmer=ROUGE_USE_STEMMER)
    rouge: dict[str, list[float]] = {}
    rubric: dict[str, list[RubricResult]] = {}
    systems: dict[str, Any] = {}
    for name, gs in gens.items():
        rouge[name] = [
            scorer.score(r["output"], g["text"])[ROUGE_TYPE].fmeasure
            for r, g in zip(rows, gs, strict=True)
        ]
        rubric[name] = [
            score(
                r,
                g["text"],
                stopped=g["stopped"],
                new_tokens=g["new_tokens"],
                max_new_tokens=max_new_tokens,
            )
            for r, g in zip(rows, gs, strict=True)
        ]
        clean = [_clean(g, max_new_tokens) for g in gs]
        overall = [res.overall for res in rubric[name]]
        rule_names = [ru.name for ru in rubric[name][0].rules] if gs else []
        short = [i for i, g in enumerate(gs) if g["new_tokens"] < short_below]
        systems[name] = {
            "lengths": length_stats(
                [g["new_tokens"] for g in gs],
                max_new_tokens=max_new_tokens,
                short_below=short_below,
            ),
            "words_mean": _mean([len(g["text"].split()) for g in gs]),
            "stopped": {"k": sum(bool(g["stopped"]) for g in gs), "n": len(gs)},
            "stopped_cleanly": {"k": sum(clean), "n": len(gs)},
            "rouge_l_mean": _mean(rouge[name]),
            "rubric_overall": _kn(overall),
            "rubric_overall_when_stopped": _kn(
                [o for o, c in zip(overall, clean, strict=True) if c]
            ),
            "rubric_rules": {
                rn: _kn(
                    [
                        res.rule(rn).passed
                        for res in rubric[name]
                        if res.rule(rn).applies
                    ]
                )
                for rn in rule_names
            },
            "short_examples": [
                {
                    "index": i,
                    "prompt": rows[i]["instruction"],
                    "text": gs[i]["text"],
                    "new_tokens": gs[i]["new_tokens"],
                    "rouge_l": rouge[name][i],
                    "rubric_failed": list(rubric[name][i].failed),
                    "reference": rows[i]["output"],
                }
                for i in short[:n_examples]
            ],
        }

    out_pairs: dict[str, Any] = {}
    for a, b in pairs:
        ga, gb = gens[a], gens[b]
        ca = [_clean(g, max_new_tokens) for g in ga]
        cb = [_clean(g, max_new_tokens) for g in gb]
        oa = [res.overall for res in rubric[a]]
        ob = [res.overall for res in rubric[b]]
        both = [i for i in range(len(rows)) if ca[i] and cb[i]]
        a_stopped = [i for i in range(len(rows)) if ca[i]]
        fixed = [i for i in range(len(rows)) if not oa[i] and ob[i]]
        regressed = [i for i in range(len(rows)) if oa[i] and not ob[i]]
        ratios = [gb[i]["new_tokens"] / ga[i]["new_tokens"] for i in both]
        out_pairs[f"{a}->{b}"] = {
            "both_stopped": {
                "n": len(both),
                "mean_new_tokens": {
                    s: _mean([gens[s][i]["new_tokens"] for i in both]) for s in (a, b)
                },
                "median_length_ratio_b_over_a": (
                    statistics.median(ratios) if ratios else None
                ),
                "rouge_l": {s: _mean([rouge[s][i] for i in both]) for s in (a, b)},
                "rubric_overall": {
                    s: _kn([rubric[s][i].overall for i in both]) for s in (a, b)
                },
            },
            "b_on_prompts_where_a_stopped": {
                "n": len(a_stopped),
                "rubric_overall": _kn([ob[i] for i in a_stopped]),
                "mean_new_tokens": _mean([gb[i]["new_tokens"] for i in a_stopped]),
                "rouge_l": _mean([rouge[b][i] for i in a_stopped]),
            },
            "b_shorter": sum(
                gb[i]["new_tokens"] < ga[i]["new_tokens"] for i in range(len(rows))
            ),
            "fixed": {"k": len(fixed), "indices": fixed},
            "regressed": {
                "k": len(regressed),
                "indices": regressed,
                "rules_failed_by_b": {
                    str(i): list(rubric[b][i].failed) for i in regressed
                },
            },
        }
    return {"systems": systems, "pairs": out_pairs}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("generations", type=Path, help="directory of <system>.jsonl")
    p.add_argument("--eval-rows", type=Path, required=True)
    p.add_argument("--systems", nargs="+", default=["base", "sft", "dpo"])
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--short-below", type=int, default=20)
    p.add_argument("--examples", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    rows: list[Row] = json.loads(args.eval_rows.read_text())
    files = {s: args.generations / f"{s}.jsonl" for s in args.systems}
    gens = {s: read_generations(f, rows) for s, f in files.items()}
    pairs = list(zip(args.systems, args.systems[1:], strict=False))
    result = analyse(
        rows,
        gens,
        max_new_tokens=args.max_new_tokens,
        short_below=args.short_below,
        n_examples=args.examples,
        pairs=pairs,
    )
    out = {
        "inputs": {
            "generations_dir": str(args.generations),
            "systems": args.systems,
            "generations_sha256": {s: _sha256(f) for s, f in files.items()},
            "eval_rows": str(args.eval_rows),
            "eval_rows_sha256": _sha256(args.eval_rows),
            "n_rows": len(rows),
            "max_new_tokens": args.max_new_tokens,
            "short_below": args.short_below,
            "rubric_version": RUBRIC_VERSION,
            "rouge": {"type": ROUGE_TYPE, "use_stemmer": ROUGE_USE_STEMMER},
            "reference_words_mean": _mean([len(r["output"].split()) for r in rows]),
        },
        **result,
        "provenance": run_provenance(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    for name, s in result["systems"].items():
        ln = s["lengths"]
        print(
            f"{name:5s} tokens mean {ln['mean']:.1f} median {ln['median']:.1f}  "
            f"short {ln['short']}  rubric {s['rubric_overall']['k']}/"
            f"{s['rubric_overall']['n']}  when stopped "
            f"{s['rubric_overall_when_stopped']['k']}/"
            f"{s['rubric_overall_when_stopped']['n']}  ROUGE-L {s['rouge_l_mean']:.3f}"
        )
    for key, pr in result["pairs"].items():
        both = pr["both_stopped"]
        print(
            f"{key}: both stopped {both['n']}, tokens {both['mean_new_tokens']}, "
            f"ROUGE-L {both['rouge_l']}, fixed {pr['fixed']['k']}, "
            f"regressed {pr['regressed']['k']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
