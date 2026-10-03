"""Tests for tools/compare_systems.py, loaded from its file path."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from post_training.data.finance import prompt_key

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "compare_systems.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_systems", _TOOL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_systems"] = mod
    spec.loader.exec_module(mod)
    return mod


cs = _load()

LOOP = "It is a good idea. " * 6

ROWS: list[dict[str, Any]] = [
    {"instruction": "Name a colour.", "input": "", "output": "Blue is a colour."},
    {"instruction": "Name a fruit.", "input": "", "output": "An apple is a fruit."},
    {"instruction": "Name a tree.", "input": "", "output": "An oak is a tree."},
    {"instruction": "Name a bird.", "input": "", "output": "A robin is a bird."},
]


def _gen(i: int, text: str, new_tokens: int, stopped: bool) -> dict[str, Any]:
    return {
        "index": i,
        "prompt_key": prompt_key(ROWS[i]),
        "text": text,
        "new_tokens": new_tokens,
        "stopped": stopped,
    }


# "a" loops on row 1 and never stops on row 3; "b" stops everywhere, shorter.
GENS: dict[str, list[dict[str, Any]]] = {
    "a": [
        _gen(0, "Blue is a colour.", 30, True),
        _gen(1, LOOP, 384, False),
        _gen(2, "An oak is a tree that grows tall.", 60, True),
        _gen(3, "A robin is a bird.", 384, False),
    ],
    "b": [
        _gen(0, "Blue.", 5, True),
        _gen(1, "An apple is a fruit.", 10, True),
        _gen(2, "An oak is a tree.", 20, True),
        _gen(3, "A robin is a bird.", 8, True),
    ],
}


def _write_jsonl(path: Path, gens: list[dict[str, Any]]) -> None:
    lines = [json.dumps({"cache_key": {"system": path.stem}})]
    lines += [json.dumps(g) for g in gens]
    path.write_text("\n".join(lines) + "\n")


def test_read_generations_skips_header_and_checks_alignment(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    _write_jsonl(path, GENS["a"])
    assert cs.read_generations(path, ROWS) == GENS["a"]
    with pytest.raises(ValueError, match="row 0"):
        cs.read_generations(path, list(reversed(ROWS)))
    with pytest.raises(ValueError, match="4 generations for 3 rows"):
        cs.read_generations(path, ROWS[:3])


def test_length_stats() -> None:
    s = cs.length_stats([5, 10, 20, 8], max_new_tokens=384, short_below=20)
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(10.75)
    assert s["median"] == pytest.approx(9.0)
    assert s["min"] == 5 and s["max"] == 20
    assert s["at_limit"] == 0
    assert s["short"] == 3  # 5, 10, 8 are below 20
    assert sum(s["histogram"].values()) == 4
    assert s["histogram"]["<20"] == 3


def test_analyse_counts_and_both_stopped_subset() -> None:
    out = cs.analyse(
        ROWS, GENS, max_new_tokens=384, short_below=20, n_examples=2, pairs=[("a", "b")]
    )
    a, b = out["systems"]["a"], out["systems"]["b"]
    assert a["stopped"] == {"k": 2, "n": 4}
    assert b["stopped"] == {"k": 4, "n": 4}
    # a: row 1 loops (repetition fails) and rows 1, 3 fail clean_stop.
    assert a["rubric_overall"]["k"] == 2
    assert a["rubric_overall_when_stopped"] == {"k": 2, "n": 2, "rate": 1.0}
    assert a["rubric_rules"]["repetition"]["k"] == 3
    assert b["rubric_overall"]["k"] == 4
    assert b["lengths"]["short"] == 3

    pair = out["pairs"]["a->b"]
    both = pair["both_stopped"]
    assert both["n"] == 2  # rows 0 and 2
    assert both["mean_new_tokens"] == {"a": 45.0, "b": 12.5}
    # Row 0: a exact (1.0), b "Blue." (P 1, R 1/4, F 0.4). Row 2: a has
    # 5 of its 8 tokens in the 5-token reference (F 10/13), b exact (1.0).
    assert both["rouge_l"]["a"] == pytest.approx((1.0 + 10 / 13) / 2)
    assert both["rouge_l"]["b"] == pytest.approx((0.4 + 1.0) / 2)
    assert pair["b_shorter"] == 4
    assert pair["fixed"]["k"] == 2  # rows 1 and 3: a fails overall, b passes
    assert pair["fixed"]["indices"] == [1, 3]
    assert pair["regressed"]["k"] == 0
    assert pair["b_on_prompts_where_a_stopped"]["rubric_overall"]["k"] == 2


def test_analyse_short_examples_and_rouge_per_row() -> None:
    out = cs.analyse(
        ROWS, GENS, max_new_tokens=384, short_below=20, n_examples=2, pairs=[]
    )
    ex = out["systems"]["b"]["short_examples"]
    assert [e["index"] for e in ex] == [0, 1]
    assert ex[0]["text"] == "Blue."
    assert ex[0]["reference"] == "Blue is a colour."
    # Identical answer and reference score ROUGE-L 1.0.
    assert ex[1]["rouge_l"] == pytest.approx(1.0)


def test_main_writes_provenance(tmp_path: Path) -> None:
    gen_dir = tmp_path / "generations"
    gen_dir.mkdir()
    for name, gens in GENS.items():
        _write_jsonl(gen_dir / f"{name}.jsonl", gens)
    rows_file = tmp_path / "rows.json"
    rows_file.write_text(json.dumps(ROWS))
    out = tmp_path / "analysis.json"
    cs.main(
        [
            str(gen_dir),
            "--eval-rows",
            str(rows_file),
            "--systems",
            "a",
            "b",
            "--out",
            str(out),
        ]
    )
    data = json.loads(out.read_text())
    assert "provenance" in data and "commit" in data["provenance"]
    assert data["inputs"]["systems"] == ["a", "b"]
    assert set(data["inputs"]["generations_sha256"]) == {"a", "b"}
    assert "a->b" in data["pairs"]
