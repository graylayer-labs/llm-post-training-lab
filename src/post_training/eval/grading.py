"""Blind correctness grading of base, SFT and DPO answers: the pure logic.

``tools/blind_grading.py`` is the command line around this. A sheet shows a
grader each prompt with the reference and three answers under labels A/B/C,
shuffled per prompt so the label says nothing about the system. The label to
system mapping is kept in a separate key. Grades come back per prompt and
label, are mapped to systems here, and are summarised per grader with
bootstrap intervals and inter-grader agreement.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

from post_training.data.finance import Row, prompt_key
from post_training.eval.metrics import bootstrap_ci, percentile
from post_training.eval.rubric import score

SYSTEMS = ("base", "sft", "dpo")
LABELS = ("A", "B", "C")
CLASSES = ("correct", "partly", "wrong")

Gen = dict[str, Any]
Key = dict[str, dict[str, str]]  # str(index) -> {label: system}
Grade = dict[str, Any]


def prompt_text(row: Row) -> str:
    """The prompt as the model saw it: instruction, then any input."""
    text = row["instruction"].strip()
    if row.get("input"):
        text = f"{text}\n\nContext: {row['input'].strip()}"
    return text


def pick_indices(total: int, n: int, seed: int) -> list[int]:
    """``n`` distinct prompt indices out of ``total``, sorted, seeded."""
    if not 0 < n <= total:
        raise ValueError(f"cannot pick {n} of {total} prompts")
    return sorted(random.Random(seed).sample(range(total), n))


def assign_labels(index: int, seed: int) -> dict[str, str]:
    """Label -> system for one prompt, a seeded shuffle that depends on the index."""
    order = list(SYSTEMS)
    random.Random(f"blind-grading:{seed}:{index}").shuffle(order)
    return dict(zip(LABELS, order, strict=True))


def build_sheet(
    rows: Sequence[Row],
    gens: Mapping[str, Sequence[Gen]],
    indices: Sequence[int],
    seed: int,
) -> tuple[list[dict[str, Any]], Key]:
    """The grader's sheet lines and the separate key.

    A sheet line holds only the index, prompt, reference and the three answer
    texts under their labels.
    """
    sheet: list[dict[str, Any]] = []
    key: Key = {}
    for i in indices:
        labels = assign_labels(i, seed)
        key[str(i)] = labels
        sheet.append(
            {
                "index": i,
                "prompt": prompt_text(rows[i]),
                "reference": rows[i]["output"],
                "answers": {lab: gens[sys][i]["text"] for lab, sys in labels.items()},
            }
        )
    return sheet, key


def rubric_overall(
    rows: Sequence[Row],
    gens: Mapping[str, Sequence[Gen]],
    indices: Sequence[int],
    max_new_tokens: int,
) -> dict[str, dict[str, bool]]:
    """str(index) -> {system: rubric overall passed}, recomputed with ``score``."""
    return {
        str(i): {
            s: score(
                rows[i],
                gens[s][i]["text"],
                stopped=gens[s][i]["stopped"],
                new_tokens=gens[s][i]["new_tokens"],
                max_new_tokens=max_new_tokens,
            ).overall
            for s in SYSTEMS
        }
        for i in indices
    }


def check_generations(gens: Sequence[Gen], rows: Sequence[Row], name: str) -> None:
    """Refuse unless ``gens`` answers ``rows`` one for one, in order."""
    if len(gens) != len(rows):
        raise ValueError(f"{name}: {len(gens)} generations for {len(rows)} rows")
    for i, (g, r) in enumerate(zip(gens, rows, strict=True)):
        if g.get("index") != i or g.get("prompt_key") != prompt_key(r):
            raise ValueError(f"{name}: row {i} does not match the eval rows")


def parse_grade_lines(text: str, name: str) -> list[Grade]:
    """Grade records from JSONL text; errors name the file and line number."""
    out: list[Grade] = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{name} line {n}: not JSON ({e.msg})") from e
        if not isinstance(rec, dict):
            raise ValueError(f"{name} line {n}: expected a JSON object")
        out.append(rec)
    return out


def validate_grades(grades: Sequence[Grade], indices: Sequence[int]) -> None:
    """Refuse unless every prompt x label is graded exactly once, validly."""
    expected = {(i, lab) for i in indices for lab in LABELS}
    seen: set[tuple[int, str]] = set()
    for g in grades:
        if not isinstance(g.get("index"), int) or isinstance(g.get("index"), bool):
            raise ValueError(f"bad index in {g!r}")
        if g.get("label") not in LABELS:
            raise ValueError(f"invalid label in {g!r}; want one of {LABELS}")
        if g.get("grade") not in CLASSES:
            raise ValueError(f"invalid grade in {g!r}; want one of {CLASSES}")
        reason = g.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"empty reason in {g!r}")
        pair = (g["index"], g["label"])
        if pair not in expected:
            raise ValueError(f"unexpected grade for {pair}")
        if pair in seen:
            raise ValueError(f"duplicate grade for {pair}")
        seen.add(pair)
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"{len(missing)} grades missing, first: {missing[:3]}")


def unblind(grades: Sequence[Grade], key: Key) -> dict[tuple[int, str], str]:
    """(index, system) -> grade."""
    return {(g["index"], key[str(g["index"])][g["label"]]): g["grade"] for g in grades}


def agreement(a: Sequence[str], b: Sequence[str]) -> float:
    """Fraction of items two graders gave the same class."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal, non-empty lists")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohens_kappa(
    a: Sequence[str], b: Sequence[str], classes: Sequence[str] = CLASSES
) -> float | None:
    """Cohen's kappa, (p_o - p_e) / (1 - p_e); None when p_e is 1."""
    n = len(a)
    p_o = agreement(a, b)
    p_e = sum((a.count(c) / n) * (b.count(c) / n) for c in classes)
    if p_e == 1:
        return None
    return (p_o - p_e) / (1 - p_e)


def _rate(flags: Sequence[bool], resamples: int, seed: int) -> dict[str, Any]:
    k, n = sum(flags), len(flags)
    ci = bootstrap_ci(flags, resamples=resamples, seed=seed)
    return {
        "k": k,
        "n": n,
        "rate": k / n if n else None,
        "ci95": list(ci) if ci else None,
    }


def paired_diff(
    a: Sequence[bool], b: Sequence[bool], *, resamples: int, seed: int
) -> dict[str, Any]:
    """Paired bootstrap of mean(b) - mean(a): resample prompts, keep pairs."""
    n = len(a)
    if n != len(b) or not n:
        raise ValueError("need two equal, non-empty lists")
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        idx = rng.choices(range(n), k=n)
        diffs.append(sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n)
    return {
        "diff": sum(b) / n - sum(a) / n,
        "ci95": [percentile(diffs, 2.5), percentile(diffs, 97.5)],
        "n": n,
    }


DIFFS = (("sft - base", "base", "sft"), ("dpo - sft", "sft", "dpo"))


def summarise(
    by_grader: Mapping[str, Mapping[tuple[int, str], str]],
    rubric: Mapping[str, Mapping[str, bool]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Per-grader rates, rubric split and inter-grader agreement."""
    graders: dict[str, Any] = {}
    for name, unblinded in by_grader.items():
        systems: dict[str, Any] = {}
        for s in SYSTEMS:
            items = sorted((i, g) for (i, sys), g in unblinded.items() if sys == s)
            grades = [g for _, g in items]
            split: dict[str, Any] = {}
            for label, want in (("rubric_pass", True), ("rubric_fail", False)):
                sub = [g == "correct" for i, g in items if rubric[str(i)][s] is want]
                split[label] = _rate(sub, resamples, seed)
            systems[s] = {
                "counts": {c: grades.count(c) for c in CLASSES},
                "correct": _rate([g == "correct" for g in grades], resamples, seed),
                "correct_or_partly": _rate(
                    [g != "wrong" for g in grades], resamples, seed
                ),
                "correct_by_rubric": split,
            }
        flags = {
            s: [
                g == "correct"
                for _, g in sorted(
                    (i, g) for (i, sys), g in unblinded.items() if sys == s
                )
            ]
            for s in SYSTEMS
        }
        graders[name] = {
            "systems": systems,
            "paired_correct_diff": {
                label: paired_diff(flags[a], flags[b], resamples=resamples, seed=seed)
                for label, a, b in DIFFS
            },
        }
    pairs: dict[str, Any] = {}
    for g1, g2 in combinations(by_grader, 2):
        keys = sorted(by_grader[g1])
        a = [by_grader[g1][k] for k in keys]
        b = [by_grader[g2][k] for k in keys]
        pairs[f"{g1} vs {g2}"] = {
            "n": len(keys),
            "exact_match": agreement(a, b),
            "cohens_kappa": cohens_kappa(a, b),
        }
    return {"graders": graders, "agreement": pairs}


def both_pass_diffs(
    by_grader: Mapping[str, Mapping[tuple[int, str], str]],
    rubric: Mapping[str, Mapping[str, bool]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired correct-rate differences on prompts where both systems pass.

    For each grader and each pair in ``DIFFS`` (label "b - a"), keep only the
    prompts where the rubric's overall rule passes both ``a`` and ``b``, then
    run ``paired_diff`` on their "correct" flags. This separates "correct"
    from "does not loop": an answer that fails the rubric is mostly a loop or
    a truncation, so it drops out of both sides.
    """
    out: dict[str, Any] = {}
    for name, unblinded in by_grader.items():
        per: dict[str, Any] = {}
        for label, a, b in DIFFS:
            idx = sorted(int(i) for i, r in rubric.items() if r[a] and r[b])
            fa = [unblinded[(i, a)] == "correct" for i in idx]
            fb = [unblinded[(i, b)] == "correct" for i in idx]
            d: dict[str, Any] = (
                paired_diff(fa, fb, resamples=resamples, seed=seed)
                if idx
                else {"diff": None, "ci95": None, "n": 0}
            )
            d.update(a_correct=sum(fa), b_correct=sum(fb), indices=idx)
            per[label] = d
        out[name] = per
    return out


def _pct(r: Mapping[str, Any]) -> str:
    if not r["n"]:
        return "n/a (0/0)"
    lo, hi = r["ci95"]
    return (
        f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']}) [{100 * lo:.1f}, {100 * hi:.1f}]"
    )


def render_markdown(results: Mapping[str, Any]) -> str:
    """results.md from the ``results`` dict that tools/blind_grading.py saves."""
    s = results["summary"]
    prov = results["provenance"]
    out = [
        "# Blind grading results",
        "",
        f"{results['n_prompts']} prompts, graders: {', '.join(s['graders'])}. "
        f"Commit `{(prov['commit'] or 'none')[:7]}`, scratch {prov['scratch']}. "
        f"Intervals are 95% percentile bootstrap, {results['resamples']} "
        f"resamples, bootstrap seed {results['bootstrap_seed']} "
        f"(sheet seed {results['sheet_seed']}).",
        "",
    ]
    for name, g in s["graders"].items():
        systems = g["systems"]
        out += [
            f"## Grader {name}",
            "",
            "| System | correct | correct or partly | correct, rubric pass | "
            "correct, rubric fail |",
            "|---|---|---|---|---|",
        ]
        for sys, v in systems.items():
            by = v["correct_by_rubric"]
            out.append(
                f"| {sys} | {_pct(v['correct'])} | {_pct(v['correct_or_partly'])} | "
                f"{_pct(by['rubric_pass'])} | {_pct(by['rubric_fail'])} |"
            )
        out += [
            "",
            "Paired difference in correct rate (same prompts, resampled "
            "together), 95% interval:",
            "",
        ]
        for label, d in g["paired_correct_diff"].items():
            lo, hi = d["ci95"]
            out.append(
                f"- {label}: {100 * d['diff']:+.1f} points "
                f"[{100 * lo:+.1f}, {100 * hi:+.1f}], n = {d['n']}"
            )
        out.append("")
    out += [
        "Overlapping per-system intervals do not mean there is no difference: "
        "the systems answer the same prompts, so the paired intervals above "
        "are the tighter comparison.",
        "",
    ]
    if s["agreement"]:
        out += [
            "## Agreement between graders",
            "",
            "| Pair | n | exact match | Cohen's kappa |",
            "|---|---|---|---|",
        ]
        for pair, v in s["agreement"].items():
            kappa = (
                "undefined" if v["cohens_kappa"] is None else f"{v['cohens_kappa']:.3f}"
            )
            out.append(
                f"| {pair} | {v['n']} | {100 * v['exact_match']:.1f}% | {kappa} |"
            )
        out.append("")
    return "\n".join(out)
