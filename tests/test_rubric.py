"""Rubric rules on hand-written answers. Expected values are written by hand."""

from decimal import Decimal

import pytest

from post_training.eval.rubric import (
    RUBRIC_VERSION,
    RubricResult,
    check_clean_stop,
    check_domain_terms,
    check_format,
    check_repetition,
    check_ungrounded_numbers,
    is_finance_row,
    number_values,
    score,
)

FINANCE_ROW = {
    "instruction": "Should I pay off my mortgage early or invest in an index fund?",
    "input": "",
    "output": (
        "It depends on your mortgage rate. If the rate is 3% and you expect "
        "the fund to return 7% a year, investing usually wins, but paying "
        "off the loan is a guaranteed return."
    ),
}
GENERAL_ROW = {
    "instruction": "Offer three tips for getting a good night's sleep.",
    "input": "",
    "output": "1. Keep a schedule. 2. Avoid screens. 3. Relax before bed.",
}


# --- format ---------------------------------------------------------------


def test_format_passes_plain_answer() -> None:
    r = check_format("Index funds spread your money across many stocks.")
    assert r.applies and r.passed


@pytest.mark.parametrize("answer", ["", "   \n\t "])
def test_format_fails_empty_answer(answer: str) -> None:
    r = check_format(answer)
    assert not r.passed
    assert "empty" in r.reason


@pytest.mark.parametrize(
    "answer",
    [
        "A bond pays interest.<|im_end|>",
        "A bond pays interest.\n<|im_start|>user\nAnd stocks?",
        "A bond pays interest.<|endoftext|>",
    ],
)
def test_format_fails_leaked_chat_marker(answer: str) -> None:
    r = check_format(answer)
    assert not r.passed
    assert "marker" in r.reason


@pytest.mark.parametrize(
    "answer",
    [
        "A bond pays interest.\nuser: what about stocks?",
        "A bond pays interest.\n  Assistant: Stocks are shares.",
        "Human: tell me more",
        "A bond pays interest.\nSYSTEM: be concise",
        "A bond pays interest.\n### Instruction:\nWhat is a stock?",
        "A bond pays interest.\nQuestion: what about stocks?",
        "Instruction: explain bonds.",
        "A bond pays interest.\n  Response: Stocks are shares.",
        # A Qwen turn decoded with skip_special_tokens leaves the bare role.
        "A bond pays interest.\nuser\nAnd stocks?",
        "A bond pays interest.\n  Assistant  \nStocks are shares.",
        "A bond pays interest.\nsystem\nYou are concise.",
    ],
)
def test_format_fails_role_line(answer: str) -> None:
    r = check_format(answer)
    assert not r.passed
    assert "role" in r.reason


def test_format_allows_role_word_mid_line() -> None:
    r = check_format("The user: the person who holds the account, pays the fee.")
    assert r.passed


def test_format_allows_role_word_starting_a_longer_line() -> None:
    r = check_format("A fee is charged.\nUser accounts are free for a year.")
    assert r.passed


# --- clean stop -----------------------------------------------------------


def test_clean_stop_passes_when_stopped_under_limit() -> None:
    r = check_clean_stop(stopped=True, new_tokens=80, max_new_tokens=256)
    assert r.applies and r.passed


def test_clean_stop_fails_when_not_stopped() -> None:
    r = check_clean_stop(stopped=False, new_tokens=80, max_new_tokens=256)
    assert not r.passed
    assert "stop token" in r.reason


def test_clean_stop_fails_at_token_limit() -> None:
    r = check_clean_stop(stopped=True, new_tokens=256, max_new_tokens=256)
    assert not r.passed
    assert "limit" in r.reason


# --- repetition -----------------------------------------------------------


def test_repetition_passes_varied_answer() -> None:
    r = check_repetition(FINANCE_ROW["output"])
    assert r.applies and r.passed


def test_repetition_fails_ngram_loop() -> None:
    answer = (
        "You should save money. You should save money every month. "
        "Then you should save money for retirement, and you should save money "
        "for a house."
    )
    r = check_repetition(answer)
    assert not r.passed
    assert "'you should save money' occurs 4 times" in r.reason


def test_repetition_allows_four_gram_three_times() -> None:
    answer = (
        "You should save money now. Later you should save money again, "
        "and you should save money when you retire."
    )
    assert check_repetition(answer).passed


def test_repetition_fails_duplicated_sentence() -> None:
    answer = (
        "An ETF trades on an exchange like a stock. It has low fees. "
        "An ETF trades on an exchange like a stock!"
    )
    r = check_repetition(answer)
    assert not r.passed
    assert "sentence" in r.reason


def test_repetition_ignores_short_repeated_sentences() -> None:
    answer = "Yes. It depends on fees and taxes. Yes. It also depends on time."
    assert check_repetition(answer).passed


# --- finance classification and domain terms ------------------------------


@pytest.mark.parametrize(
    "instruction",
    [
        "Wash sale rule with dividend reinvestment",
        "Can one get a house mortgage without buying a house?",
        "Is a Roth IRA better than a 401(k)?",
        "How are capital gains taxed?",
    ],
)
def test_is_finance_row_true_for_finance_prompts(instruction: str) -> None:
    assert is_finance_row({"instruction": instruction, "input": "", "output": "x"})


@pytest.mark.parametrize(
    "instruction",
    [
        "Offer three tips for getting a good night's sleep.",
        "Write a jQuery script to hide all elements with the class hide-this.",
        "Share a story about the river bank in spring.",
        "What is your heart rate after running?",
    ],
)
def test_is_finance_row_false_for_general_prompts(instruction: str) -> None:
    assert not is_finance_row({"instruction": instruction, "input": "", "output": "x"})


@pytest.mark.parametrize(
    "instruction",
    [
        "Investigate the world's smallest mountain.",
        "Call a taxi to the airport.",
        "Describe a wealth of ideas for a party.",
    ],
)
def test_is_finance_row_false_for_lookalike_words(instruction: str) -> None:
    assert not is_finance_row({"instruction": instruction, "input": "", "output": "x"})


def test_is_finance_row_ignores_finance_words_in_reference_only() -> None:
    row = {
        "instruction": "List five jobs that require analytical thinking.",
        "input": "",
        "output": "1. Data scientist 2. Financial analyst 3. Investment banker",
    }
    assert not is_finance_row(row)


def test_is_finance_row_reads_the_input_field() -> None:
    row = {"instruction": "Summarise.", "input": "The bond yield rose.", "output": "x"}
    assert is_finance_row(row)


def test_domain_terms_passes_finance_answer() -> None:
    r = check_domain_terms(FINANCE_ROW, "Paying the loan early saves interest.")
    assert r.applies and r.passed


@pytest.mark.parametrize(
    "answer",
    [
        "I refused to pay the bill.",
        "Checks don't expire in the US.",
        "Sell a fixed number of shares each month.",
        "Put $200 aside today.",
    ],
)
def test_domain_terms_accepts_broader_words_in_the_answer(answer: str) -> None:
    # Words too ambiguous to class a prompt as finance still show that an
    # answer to a finance prompt stayed on topic.
    assert check_domain_terms(FINANCE_ROW, answer).passed


def test_domain_terms_fails_finance_row_without_terms() -> None:
    r = check_domain_terms(FINANCE_ROW, "It depends on what makes you happy.")
    assert r.applies and not r.passed


def test_domain_terms_does_not_apply_to_general_row() -> None:
    r = check_domain_terms(GENERAL_ROW, "Go to bed at the same time.")
    assert not r.applies


# --- number normalisation -------------------------------------------------


def D(s: str) -> Decimal:
    return Decimal(s)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$1,000", [{D("1000")}]),
        ("1000", [{D("1000")}]),
        ("5%", [{D("5"), D("0.05")}]),
        ("5 percent", [{D("5"), D("0.05")}]),
        ("0.05", [{D("0.05")}]),
        ("3.50", [{D("3.5")}]),
        ("$1.5 million", [{D("1.5"), D("1500000")}]),
        ("$50k", [{D("50"), D("50000")}]),
        ("-2.5%", [{D("2.5"), D("0.025")}]),
        (".5%", [{D("0.5"), D("0.005")}]),
        ("rates of 3% and 7%", [{D("3"), D("0.03")}, {D("7"), D("0.07")}]),
    ],
)
def test_number_values(text: str, expected: list[set[Decimal]]) -> None:
    assert [set(v) for v in number_values(text)] == expected


@pytest.mark.parametrize(
    "text",
    [
        "Put it in a 401(k) or a 403(b).",
        "The S&P 500 is an index.",
        "File a W-2 and a 1099.",
        "Q1 results and the 2nd payment.",
    ],
)
def test_number_values_skips_names_and_codes(text: str) -> None:
    assert number_values(text) == []


# --- ungrounded numbers ---------------------------------------------------


def test_numbers_pass_when_found_in_reference() -> None:
    r = check_ungrounded_numbers(FINANCE_ROW, "At 3% the fund's 7% return wins.")
    assert r.applies and r.passed


def test_numbers_fail_when_invented() -> None:
    r = check_ungrounded_numbers(FINANCE_ROW, "Funds return 12% a year.")
    assert not r.passed
    assert "12%" in r.reason


def test_numbers_match_across_percent_and_decimal() -> None:
    assert check_ungrounded_numbers(FINANCE_ROW, "A rate of 0.03 is low.").passed
    row = dict(FINANCE_ROW, output="A rate of 0.04 beats inflation.")
    assert check_ungrounded_numbers(row, "4% beats inflation.").passed


def test_numbers_match_across_currency_and_commas() -> None:
    row = dict(FINANCE_ROW, input="I owe 25000 on the loan.")
    assert check_ungrounded_numbers(row, "Pay the $25,000 loan off.").passed


def test_numbers_exempt_years_and_small_integers() -> None:
    answer = "Since 2008, take 3 steps: budget, save, invest. In 1999 it was 10."
    assert check_ungrounded_numbers(FINANCE_ROW, answer).passed


def test_small_integer_with_percent_is_not_exempt() -> None:
    r = check_ungrounded_numbers(FINANCE_ROW, "Expect 5% from bonds.")
    assert not r.passed


def test_small_integer_with_currency_is_not_exempt() -> None:
    r = check_ungrounded_numbers(FINANCE_ROW, "It costs $8 a month.")
    assert not r.passed


def test_numbers_fail_correct_but_unstated_arithmetic() -> None:
    # The rule checks grounding, not truth: $50 at 5% is $52.50 after a year, but
    # the figure appears in neither the prompt nor the reference.
    row = dict(FINANCE_ROW, input="I invest $50 at 5% a year.", output="It grows.")
    r = check_ungrounded_numbers(row, "That is $52.50 after a year.")
    assert not r.passed
    assert "$52.50" in r.reason


def test_numbers_pass_answer_without_numbers() -> None:
    assert check_ungrounded_numbers(FINANCE_ROW, "Pay the loan off.").passed


def test_numbers_do_not_apply_to_general_row() -> None:
    r = check_ungrounded_numbers(GENERAL_ROW, "Sleep 8 hours, 42 nights.")
    assert not r.applies


# --- overall --------------------------------------------------------------


def test_score_passes_good_finance_answer() -> None:
    res = score(
        FINANCE_ROW,
        "If your mortgage rate is 3%, investing at 7% usually wins.",
        stopped=True,
        new_tokens=20,
        max_new_tokens=256,
    )
    assert isinstance(res, RubricResult)
    assert res.version == RUBRIC_VERSION == "1"
    assert res.overall
    assert res.failed == ()
    assert [r.name for r in res.rules] == [
        "format",
        "clean_stop",
        "repetition",
        "domain_terms",
        "ungrounded_numbers",
    ]


def test_score_fails_overall_when_one_rule_fails() -> None:
    res = score(
        FINANCE_ROW,
        "Invest in the fund.",
        stopped=False,
        new_tokens=256,
        max_new_tokens=256,
    )
    assert not res.overall
    assert res.failed == ("clean_stop",)
    assert not res.rule("clean_stop").passed


def test_score_ignores_rules_that_do_not_apply() -> None:
    res = score(
        GENERAL_ROW,
        "Keep a fixed bedtime and dim the lights.",
        stopped=True,
        new_tokens=12,
        max_new_tokens=256,
    )
    assert res.overall
    assert not res.rule("domain_terms").applies
    assert not res.rule("ungrounded_numbers").applies


def test_score_to_dict_is_json_ready() -> None:
    res = score(GENERAL_ROW, "", stopped=True, new_tokens=1, max_new_tokens=256)
    d = res.to_dict()
    assert d["version"] == "1"
    assert d["overall"] is False
    assert d["failed"] == ["format"]
    fmt = d["rules"]["format"]
    assert set(fmt) == {"applies", "passed", "reason"}
    assert fmt["applies"] is True and fmt["passed"] is False


def test_every_rule_is_documented_in_the_module_docstring() -> None:
    import post_training.eval.rubric as rubric

    res = score(GENERAL_ROW, "x", stopped=True, new_tokens=1, max_new_tokens=2)
    for r in res.rules:
        assert f"``{r.name}``" in (rubric.__doc__ or ""), r.name
