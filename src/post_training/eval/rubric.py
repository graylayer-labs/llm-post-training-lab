"""Rule-based rubric for generated answers.

Each rule is a pure function returning a ``RuleResult``: whether the rule
applies to the row, whether the answer passed, and a short reason. ``score``
runs all of them; the answer passes overall when every rule that applies
passes. DPO pair building (#2) uses a failing answer as the rejected side, and
the eval harness (#3) reports per-rule pass rates, so the rules are tuned to
be conservative: a rule that fires on good answers would teach DPO to avoid
them. Bump ``RUBRIC_VERSION`` whenever a rule or threshold changes, because
saved pairs and eval tables record the version they were scored with.

Rules:

- ``format``: the answer is non-empty and has no leaked chat markers
  (``<|im_end|>`` and the like) or lines that start a new role turn.
- ``clean_stop``: generation ended on a stop token, under the token limit.
- ``repetition``: no word 4-gram occurs 4 or more times, and no sentence of 4
  or more words occurs twice.
- ``domain_terms`` (finance rows only): the answer uses at least one term
  from ``ANSWER_FINANCE_TERMS`` (broader than the list that classes rows) or
  a currency amount.
- ``ungrounded_numbers`` (finance rows only): every figure in the answer
  also appears in the prompt or the reference answer, after normalisation.
  This checks grounding, not truth: correct arithmetic on the prompt's
  figures fails it, and a false figure copied from the reference passes.

A row is a finance row when its prompt (instruction and input, not the
reference answer) contains a finance term. The reference is left out because
general Alpaca answers mention finance in passing (a list of jobs that names
"financial analyst"), and classing those rows as finance would require
finance words of a correct answer that does not need them.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

RUBRIC_VERSION = "1"

Row = dict[str, Any]

# --- format ---------------------------------------------------------------

# Any special token in Qwen's <|...|> form, e.g. <|im_start|>, <|im_end|>,
# <|endoftext|>. The decoded answer should never contain one.
_CHAT_MARKER = re.compile(r"<\|[a-z_]+\|>")
# A line that opens a new turn: "user:", "Assistant:", "Human:", "system:",
# "Question:", "Instruction:", "Response:", an Alpaca-style "### Instruction"
# header, or a line holding only a role name. Only at the start of a line, so a
# role word inside a sentence is fine.
_ROLE_LINE = re.compile(
    r"^[ \t]*(?:"
    r"(?:user|assistant|system|human|question|instruction|response)[ \t]*:"
    r"|###[ \t]*(?:instruction|input|response)"
    # A bare role line is what a Qwen turn leaves when decoded with
    # skip_special_tokens: "<|im_start|>user" becomes "user".
    r"|(?:user|assistant|system)[ \t]*$"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# --- repetition -----------------------------------------------------------

# A 4-gram seen 4 times is a loop. Real SFT loops repeat a clause 10 or more
# times, while good prose repeats a 4-gram up to 3 times ("can be used to",
# a poem's refrain): at 3, 2.3% of the 2,200 reference answers failed, at
# 4, 0.8% (tools/rubric_on_references.py).
# Sentences under 4 words ("Yes.", "1. Stocks") repeat legitimately in lists,
# so only longer ones count as duplicates.
REPEAT_NGRAM = 4
REPEAT_MIN_COUNT = 4
DUP_SENTENCE_MIN_WORDS = 4

_WORD = re.compile(r"[a-z0-9']+")
_SENTENCE_SPLIT = re.compile(r"[.!?\n]+")

# --- finance terms ----------------------------------------------------------

# Matched case-insensitively at a word start; entries ending in "*" are stems.
# Words with a common non-finance sense are left out or kept only in
# compounds: "bank" (river bank), "rate" (heart rate), "share" (share a
# story), "interest" (of interest), "credit" (give credit), "market".
FINANCE_TERMS: tuple[str, ...] = (
    "invest",
    "invests",
    "invested",
    "investing",
    "investment*",
    "investor*",
    "financ*",
    "dividend*",
    "mortgag*",
    "loan*",
    "debt*",
    "tax",
    "taxes",
    "taxed",
    "taxable",
    "taxation",
    "retire*",
    "pension*",
    "broker*",
    "portfolio*",
    "stock*",
    "bond",
    "bonds",
    "shareholder*",
    "equity",
    "equities",
    "inflation",
    "interest rate*",
    "compound interest",
    "credit card*",
    "credit score*",
    "credit report*",
    "credit union*",
    "bank account*",
    "banking",
    "savings",
    "budget*",
    "insur*",
    "annuit*",
    "salary",
    "salaries",
    "income*",
    "expense*",
    "deduct*",
    "refinanc*",
    "money",
    "currenc*",
    "forex",
    "crypto*",
    "bitcoin*",
    "trading",
    "trader*",
    "asset*",
    "liabilit*",
    "revenue*",
    "profit*",
    "accounting",
    "accountant*",
    "earnings",
    "valuation*",
    "capital gain*",
    "fund",
    "funds",
    "etf*",
    "ira",
    "iras",
    "401(k)",
    "401k",
    "roth",
    "apr",
    "paycheck*",
    "wage*",
    "econom*",
    "wealth management",
    "wealthy",
    "cash",
    "debit*",
    "lender*",
    "lending",
    "brokerage*",
    "securities",
    "payment*",
    "net worth",
)


def _term_pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    parts = []
    for t in terms:
        if t.endswith("*"):
            parts.append(re.escape(t[:-1]) + r"\w*")
        else:
            parts.append(re.escape(t) + r"(?!\w)")
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + ")", re.IGNORECASE)


_FINANCE = _term_pattern(FINANCE_TERMS)

# The answer side is broader. Once the prompt is known to be about finance,
# words too ambiguous to class a prompt ("pay", "shares", "price") still show
# the answer stayed on topic, and a stricter list would fail good references
# ("I refused to pay the bill", "Checks don't expire").
ANSWER_FINANCE_TERMS: tuple[str, ...] = (
    *FINANCE_TERMS,
    "wealth*",
    "pay",
    "pays",
    "paid",
    "paying",
    "bank*",
    "interest",
    "credit*",
    "account*",
    "deposit*",
    "check",
    "checks",
    "checking",
    "cheque*",
    "share",
    "shares",
    "price*",
    "cost*",
    "fee",
    "fees",
    "earn*",
    "market*",
    "dollar*",
    "cents",
    "afford*",
    "sell*",
    "sold",
    "buy*",
    "bought",
    "rent*",
    "bill",
    "bills",
    "claim*",
    "spend*",
    "spent",
    "owe*",
    "business*",
    "contract*",
    "order*",
)
# A currency amount ("$200", "£5") also counts as staying on topic.
_ANSWER_FINANCE = re.compile(
    _term_pattern(ANSWER_FINANCE_TERMS).pattern + r"|[$£€]\s?\d",
    re.IGNORECASE,
)

# --- numbers ----------------------------------------------------------------

# Bare integers in these ranges are exempt: list ordinals and counts ("3
# steps") and years. A percentage, currency amount or decimal is never exempt.
EXEMPT_SMALL_INT_MAX = 10
EXEMPT_YEAR_RANGE = (1900, 2099)

# Names that contain digits but are not figures. Removed before extraction.
_NAMED_NUMBERS = re.compile(
    r"\b\d{3}\s?\(\s?[a-z]\s?\)"  # 401(k), 403(b), 457(b)
    r"|\b40[13]k\b"
    r"|S&P\s?\d+"
    r"|\b1099(?:-[A-Za-z]+)?\b"
    r"|\b(?:form|schedule)\s+\d+\w*",
    re.IGNORECASE,
)
_MULTIPLIERS = {
    "thousand": 10**3,
    "k": 10**3,
    "million": 10**6,
    "m": 10**6,
    "mm": 10**6,
    "billion": 10**9,
    "bn": 10**9,
    "b": 10**9,
    "trillion": 10**12,
}
# Not preceded by a word character or "." (Q1, v2.0) or by a letter and a
# hyphen (W-2); not followed by a word character (2nd, 10x). A leading minus
# sign is not captured, so -2.5% and 2.5% are the same figure.
_NUMBER = re.compile(
    r"(?<![\w.])(?<![A-Za-z]-)"
    r"(?P<cur>[$£€]\s?)?"
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:\s?(?P<mult>thousand|million|billion|trillion|mm|bn|k|m|b)(?!\w))?"
    r"(?P<pct>\s?%|\s?percent(?!\w))?"
    r"(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Number:
    text: str
    values: frozenset[Decimal]
    exempt: bool


def _numbers(text: str) -> list[_Number]:
    text = _NAMED_NUMBERS.sub(" ", text)
    found = []
    for m in _NUMBER.finditer(text):
        num = m.group("num")
        base = Decimal(num.replace(",", ""))
        values = {base}
        if m.group("mult"):
            values.add(base * _MULTIPLIERS[m.group("mult").lower()])
        if m.group("pct"):
            values.add(base / 100)
        plain = not (m.group("cur") or m.group("mult") or m.group("pct"))
        exempt = (
            plain
            and "." not in num
            and "," not in num
            and (
                base <= EXEMPT_SMALL_INT_MAX
                or EXEMPT_YEAR_RANGE[0] <= base <= EXEMPT_YEAR_RANGE[1]
            )
        )
        found.append(_Number(m.group(0).strip(), frozenset(values), exempt))
    return found


def number_values(text: str) -> list[frozenset[Decimal]]:
    """The normalised values of each figure in ``text``, in order.

    Each figure maps to a set of equivalent values: ``$1,000`` to {1000};
    ``5%`` to {5, 0.05}, so it matches a source that says ``5`` or ``0.05``;
    ``$1.5 million`` and ``$50k`` to both the written and the multiplied value.
    """
    return [n.values for n in _numbers(text)]


# --- results ----------------------------------------------------------------


@dataclass(frozen=True)
class RuleResult:
    name: str
    applies: bool
    passed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"applies": self.applies, "passed": self.passed, "reason": self.reason}


@dataclass(frozen=True)
class RubricResult:
    version: str
    rules: tuple[RuleResult, ...]

    @property
    def failed(self) -> tuple[str, ...]:
        """Names of the rules that apply and failed."""
        return tuple(r.name for r in self.rules if r.applies and not r.passed)

    @property
    def overall(self) -> bool:
        return not self.failed

    def rule(self, name: str) -> RuleResult:
        for r in self.rules:
            if r.name == name:
                return r
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "overall": self.overall,
            "failed": list(self.failed),
            "rules": {r.name: r.to_dict() for r in self.rules},
        }


def _ok(name: str, reason: str = "ok") -> RuleResult:
    return RuleResult(name, applies=True, passed=True, reason=reason)


def _fail(name: str, reason: str) -> RuleResult:
    return RuleResult(name, applies=True, passed=False, reason=reason)


def _skip(name: str, reason: str) -> RuleResult:
    return RuleResult(name, applies=False, passed=True, reason=reason)


# --- rules ------------------------------------------------------------------


def check_format(answer: str) -> RuleResult:
    if not answer.strip():
        return _fail("format", "empty answer")
    if m := _CHAT_MARKER.search(answer):
        return _fail("format", f"leaked chat marker {m.group(0)}")
    if m := _ROLE_LINE.search(answer):
        return _fail("format", f"role line {m.group(0).strip()!r}")
    return _ok("format")


def check_clean_stop(
    *, stopped: bool, new_tokens: int, max_new_tokens: int
) -> RuleResult:
    if not stopped:
        return _fail("clean_stop", f"no stop token after {new_tokens} tokens")
    if new_tokens >= max_new_tokens:
        return _fail("clean_stop", f"hit the {max_new_tokens}-token limit")
    return _ok("clean_stop")


def check_repetition(answer: str) -> RuleResult:
    words = _WORD.findall(answer.lower())
    grams = Counter(
        tuple(words[i : i + REPEAT_NGRAM]) for i in range(len(words) - REPEAT_NGRAM + 1)
    )
    if grams:
        gram, count = grams.most_common(1)[0]
        if count >= REPEAT_MIN_COUNT:
            return _fail("repetition", f"{' '.join(gram)!r} occurs {count} times")
    sentences = Counter(
        " ".join(ws)
        for s in _SENTENCE_SPLIT.split(answer.lower())
        if len(ws := _WORD.findall(s)) >= DUP_SENTENCE_MIN_WORDS
    )
    if sentences:
        sentence, count = sentences.most_common(1)[0]
        if count >= 2:
            return _fail("repetition", f"sentence {sentence!r} occurs {count} times")
    return _ok("repetition")


def _prompt_text(row: Row) -> str:
    return f"{row['instruction']}\n{row.get('input') or ''}"


def is_finance_row(row: Row) -> bool:
    """True when the prompt (instruction and input) uses a finance term."""
    return _FINANCE.search(_prompt_text(row)) is not None


def check_domain_terms(row: Row, answer: str) -> RuleResult:
    if not is_finance_row(row):
        return _skip("domain_terms", "not a finance row")
    if m := _ANSWER_FINANCE.search(answer):
        return _ok("domain_terms", f"uses {m.group(0)!r}")
    return _fail("domain_terms", "no finance term in a finance answer")


def check_ungrounded_numbers(row: Row, answer: str) -> RuleResult:
    """Fail when a figure in the answer is in neither the prompt nor the reference.

    This checks grounding, not truth. It cannot tell whether a figure is false:
    correct arithmetic on the prompt's figures ("$50 at 5% is $52.50 after a
    year") fails, and a wrong figure that happens to appear in the reference
    passes.
    """
    if not is_finance_row(row):
        return _skip("ungrounded_numbers", "not a finance row")
    source = set().union(
        *(n.values for n in _numbers(f"{_prompt_text(row)}\n{row['output']}"))
    )
    invented = [
        n.text for n in _numbers(answer) if not n.exempt and not (n.values & source)
    ]
    if invented:
        return _fail(
            "ungrounded_numbers",
            "not in prompt or reference: " + ", ".join(invented),
        )
    return _ok("ungrounded_numbers")


def score(
    row: Row,
    answer: str,
    *,
    stopped: bool,
    new_tokens: int,
    max_new_tokens: int,
) -> RubricResult:
    """Score one generated answer against its dataset row."""
    return RubricResult(
        version=RUBRIC_VERSION,
        rules=(
            check_format(answer),
            check_clean_stop(
                stopped=stopped, new_tokens=new_tokens, max_new_tokens=max_new_tokens
            ),
            check_repetition(answer),
            check_domain_terms(row, answer),
            check_ungrounded_numbers(row, answer),
        ),
    )
