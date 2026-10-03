from post_training.data.finance import (
    SYSTEM_PROMPT,
    FinanceSplits,
    make_splits,
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
