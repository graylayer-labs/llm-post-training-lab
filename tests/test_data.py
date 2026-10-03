from post_training.data.finance import (
    SYSTEM_PROMPT,
    FinanceSplits,
    make_splits,
    prompt_key,
    to_messages,
)


def test_to_messages_without_input() -> None:
    row = {"instruction": "What is an ETF?", "input": "", "output": "A fund."}
    msgs = to_messages(row)
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1] == {"role": "user", "content": "What is an ETF?"}
    assert msgs[2] == {"role": "assistant", "content": "A fund."}


def test_to_messages_appends_input_as_context() -> None:
    row = {"instruction": "Summarise.", "input": "Rates rose.", "output": "Up."}
    user = to_messages(row)[1]["content"]
    assert user.startswith("Summarise.")
    assert "Rates rose." in user


def test_make_splits_is_deterministic_and_disjoint() -> None:
    rows = [
        {"instruction": f"q{i}", "input": "", "output": "a" * 50} for i in range(50)
    ]
    a = make_splits(rows, train_size=20, eval_size=5, seed=0)
    b = make_splits(rows, train_size=20, eval_size=5, seed=0)
    assert isinstance(a, FinanceSplits)
    assert [r["instruction"] for r in a.train] == [r["instruction"] for r in b.train]
    assert len(a.train) == 20 and len(a.eval) == 5
    train_qs = {r["instruction"] for r in a.train}
    assert train_qs.isdisjoint(r["instruction"] for r in a.eval)


def test_make_splits_filters_short_and_long_outputs() -> None:
    rows = [
        {"instruction": "short", "input": "", "output": "x"},
        {"instruction": "long", "input": "", "output": "y" * 10_000},
        {"instruction": "ok", "input": "", "output": "z" * 100},
    ]
    s = make_splits(rows, train_size=1, eval_size=0, seed=0, min_chars=20)
    assert [r["instruction"] for r in s.train] == ["ok"]


def test_prompt_key_collapses_whitespace_and_keeps_case() -> None:
    a = {"instruction": "  What is  an ETF? ", "input": "", "output": "x"}
    b = {"instruction": "What is an ETF?", "input": None, "output": "y"}
    c = {"instruction": "what is an etf?", "input": "", "output": "x"}
    d = {"instruction": "What is an ETF?", "input": " Context\n here ", "output": "x"}
    assert prompt_key(a) == prompt_key(b)
    assert prompt_key(a) != prompt_key(c)
    assert prompt_key(a) != prompt_key(d)


def test_make_splits_drops_duplicate_prompts_and_counts_them() -> None:
    def row(ins: str, inp: str = "", out: str = "a") -> dict[str, str]:
        return {"instruction": ins, "input": inp, "output": out * 50}

    rows = [
        row("q1"),
        row("q1", out="b"),  # exact prompt duplicate
        row("q2"),
        row("  q2 "),  # whitespace-only difference
        row("q3", "ctx here"),
        row("q3", "ctx   here\n"),  # whitespace in input
        row("q3"),  # same instruction, no input: a different prompt
        row("Q1"),  # case differs: kept
        row("q4"),
    ]
    s = make_splits(rows, train_size=4, eval_size=2, seed=0)
    assert s.duplicates_dropped == 3
    keys = [prompt_key(r) for r in s.train + s.eval]
    assert len(keys) == 6 and len(set(keys)) == 6
    assert {prompt_key(r) for r in s.train}.isdisjoint(prompt_key(r) for r in s.eval)


def test_make_splits_keeps_first_occurrence_in_dataset_order() -> None:
    rows = [
        {"instruction": "q", "input": "", "output": "first" + "a" * 50},
        {"instruction": " q", "input": "", "output": "second" + "a" * 50},
    ]
    s = make_splits(rows, train_size=1, eval_size=0, seed=0)
    assert s.train[0]["output"].startswith("first")
    assert s.duplicates_dropped == 1


def test_splits_stats_reports_sizes_and_duplicates_dropped() -> None:
    s = FinanceSplits(train=[{}, {}, {}], eval=[{}], duplicates_dropped=2)
    assert s.stats() == {"train_rows": 3, "eval_rows": 1, "duplicates_dropped": 2}
