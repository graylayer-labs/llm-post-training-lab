import json
import random

import numpy as np
import pytest

from post_training.eval.grading import (
    SYSTEMS,
    agreement,
    assign_labels,
    build_sheet,
    check_generations,
    cohens_kappa,
    paired_diff,
    parse_grade_lines,
    pick_indices,
    unblind,
    validate_grades,
)

SHEET_KEYS = {"index", "prompt", "reference", "answers"}


def _rows(n: int) -> list[dict]:
    return [
        {"instruction": f"q{i}", "input": "", "output": f"ref{i}", "text": ""}
        for i in range(n)
    ]


def _gens(n: int) -> dict[str, list[dict]]:
    return {
        s: [
            {
                "index": i,
                "prompt_key": f"q{i}",
                "text": f"{s} answer {i}",
                "new_tokens": 7 + k,
                "stopped": bool(k % 2),
            }
            for i in range(n)
        ]
        for k, s in enumerate(SYSTEMS)
    }


def test_pick_indices_is_seeded_sorted_and_unique() -> None:
    a = pick_indices(200, 50, seed=0)
    assert a == pick_indices(200, 50, seed=0)
    assert a != pick_indices(200, 50, seed=1)
    assert a == sorted(set(a)) and len(a) == 50
    assert all(0 <= i < 200 for i in a)
    with pytest.raises(ValueError):
        pick_indices(10, 11, seed=0)


def test_assign_labels_deterministic_and_varies_across_prompts() -> None:
    m = assign_labels(3, seed=0)
    assert m == assign_labels(3, seed=0)
    assert sorted(m) == ["A", "B", "C"] and sorted(m.values()) == sorted(SYSTEMS)
    orders = {tuple(assign_labels(i, seed=0).values()) for i in range(40)}
    assert len(orders) > 1
    assert any(assign_labels(i, seed=0) != assign_labels(i, seed=1) for i in range(40))


def test_sheet_leaks_nothing_about_the_system() -> None:
    sheet, key = build_sheet(_rows(8), _gens(8), [1, 4, 6], seed=0)
    assert [r["index"] for r in sheet] == [1, 4, 6]
    for line in sheet:
        assert set(line) == SHEET_KEYS
        assert set(line["answers"]) == {"A", "B", "C"}
        assert all(isinstance(t, str) for t in line["answers"].values())
    # Only the answer text is carried: no flags, lengths or system names.
    # (The test answers embed the system name, so strip answer text first.)
    stripped = [{**line, "answers": sorted(line["answers"])} for line in sheet]
    blob = json.dumps(stripped)
    for bad in ("stopped", "new_tokens", "system", "length", *SYSTEMS):
        assert bad not in blob
    # The key (kept apart) maps label -> system and agrees with the sheet.
    for line in sheet:
        for label, system in key[str(line["index"])].items():
            assert line["answers"][label] == f"{system} answer {line['index']}"


def test_each_system_is_balanced_across_labels() -> None:
    _, key = build_sheet(_rows(200), _gens(200), pick_indices(200, 50, 0), seed=0)
    for label in "ABC":
        for system in SYSTEMS:
            assert sum(m[label] == system for m in key.values()) >= 10


def test_check_generations_refuses_misordered_rows() -> None:
    rows = _rows(4)
    good = _gens(4)["sft"]
    check_generations(good, rows, "sft")
    swapped = [good[1], good[0], *good[2:]]
    with pytest.raises(ValueError, match="row 0"):
        check_generations(swapped, rows, "sft")
    with pytest.raises(ValueError, match="3 generations"):
        check_generations(good[:3], rows, "sft")
    wrong_key = [{**good[0], "prompt_key": "other"}, *good[1:]]
    with pytest.raises(ValueError, match="row 0"):
        check_generations(wrong_key, rows, "sft")


def test_paired_diff_ci_matches_independent_percentiles() -> None:
    a = [True, False, False, True, False, False]
    b = [True, True, False, True, True, False]
    n = len(a)
    rng = random.Random(5)
    diffs = []
    for _ in range(200):
        idx = rng.choices(range(n), k=n)
        diffs.append(sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n)
    out = paired_diff(a, b, resamples=200, seed=5)
    assert out["diff"] == pytest.approx(2 / 6)
    assert out["ci95"] == pytest.approx(
        [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    )


def test_parse_grade_lines_reports_line_numbers() -> None:
    ok = '{"index": 0, "label": "A", "grade": "wrong", "reason": "r"}'
    assert len(parse_grade_lines(f"{ok}\n\n{ok}\n", "g.jsonl")) == 2
    with pytest.raises(ValueError, match="g.jsonl line 2"):
        parse_grade_lines(f"{ok}\nnot json\n", "g.jsonl")
    with pytest.raises(ValueError, match="line 2"):
        parse_grade_lines(f"{ok}\n[1, 2]\n", "g.jsonl")


def test_sheet_is_deterministic() -> None:
    a = build_sheet(_rows(8), _gens(8), [0, 2, 5], seed=3)
    assert a == build_sheet(_rows(8), _gens(8), [0, 2, 5], seed=3)


def test_unblind_maps_grades_back_to_systems() -> None:
    key = {
        "0": {"A": "dpo", "B": "base", "C": "sft"},
        "5": {"A": "sft", "B": "dpo", "C": "base"},
    }
    grades = [
        {"index": 0, "label": "A", "grade": "correct", "reason": "r"},
        {"index": 0, "label": "B", "grade": "wrong", "reason": "r"},
        {"index": 0, "label": "C", "grade": "partly", "reason": "r"},
        {"index": 5, "label": "A", "grade": "wrong", "reason": "r"},
        {"index": 5, "label": "B", "grade": "correct", "reason": "r"},
        {"index": 5, "label": "C", "grade": "wrong", "reason": "r"},
    ]
    assert unblind(grades, key) == {
        (0, "dpo"): "correct",
        (0, "base"): "wrong",
        (0, "sft"): "partly",
        (5, "sft"): "wrong",
        (5, "dpo"): "correct",
        (5, "base"): "wrong",
    }


def test_agreement_and_kappa_match_hand_computation() -> None:
    a = ["correct", "correct", "correct", "partly", "wrong", "wrong"]
    b = ["correct", "correct", "partly", "partly", "wrong", "correct"]
    # p_o = 4/6; marginals a (3,1,2), b (3,2,1); p_e = (9+2+2)/36 = 13/36.
    assert agreement(a, b) == pytest.approx(4 / 6)
    assert cohens_kappa(a, b) == pytest.approx(11 / 23)
    assert cohens_kappa(a, a) == pytest.approx(1.0)
    # One class only: p_e = 1, kappa undefined.
    assert cohens_kappa(["wrong"] * 3, ["wrong"] * 3) is None


def _full(indices: list[int]) -> list[dict]:
    return [
        {"index": i, "label": lab, "grade": "correct", "reason": "ok"}
        for i in indices
        for lab in "ABC"
    ]


def test_validate_accepts_complete_grades() -> None:
    validate_grades(_full([0, 3]), [0, 3])


def test_validate_refuses_missing_duplicate_and_bad_grades() -> None:
    good = _full([0, 3])
    with pytest.raises(ValueError, match="missing"):
        validate_grades(good[:-1], [0, 3])
    with pytest.raises(ValueError, match="duplicate"):
        validate_grades([*good, good[0]], [0, 3])
    with pytest.raises(ValueError, match="unexpected"):
        validate_grades([*good, {**good[0], "index": 9}], [0, 3])
    with pytest.raises(ValueError, match="grade"):
        validate_grades([{**good[0], "grade": "great"}, *good[1:]], [0, 3])
    with pytest.raises(ValueError, match="reason"):
        validate_grades([{**good[0], "reason": ""}, *good[1:]], [0, 3])
    with pytest.raises(ValueError, match="label"):
        validate_grades([{**good[0], "label": "D"}, *good[1:]], [0, 3])
