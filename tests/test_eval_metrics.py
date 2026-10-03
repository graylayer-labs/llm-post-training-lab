import math
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from post_training.eval.metrics import (
    answer_nll,
    bootstrap_ci,
    percentile,
    perplexity,
    reference_nll,
    rouge_l,
    rubric_rates,
)
from post_training.eval.rubric import score


def _uniform(n_pos: int, vocab: int) -> torch.Tensor:
    return torch.zeros(n_pos, vocab)


def test_answer_nll_counts_only_answer_tokens() -> None:
    # 5 tokens, the first 2 are prompt: 3 answer tokens, each at p = 1/4.
    nll, n = answer_nll(_uniform(5, 4), torch.tensor([0, 1, 2, 3, 0]), n_prompt=2)
    assert n == 3
    assert nll == pytest.approx(3 * math.log(4))


def test_answer_nll_uses_the_previous_position_to_predict_each_token() -> None:
    # Position t's logits predict token t + 1. Put all mass on token 2 at
    # position 1, so the single answer token (index 2, value 2) costs 0.
    logits = torch.full((3, 4), -1e9)
    logits[1, 2] = 0.0
    nll, n = answer_nll(logits, torch.tensor([0, 1, 2]), n_prompt=2)
    assert n == 1
    assert nll == pytest.approx(0.0)


def test_perplexity_is_token_weighted_not_row_mean() -> None:
    # Row A: 3 tokens at p = 1/4. Row B: 1 token at p = 1/2.
    # Token-weighted: exp((3 ln 4 + ln 2) / 4) = 2 ** 1.75 ~ 3.364.
    # The mean of per-row perplexities would be (4 + 2) / 2 = 3.
    parts = [(3 * math.log(4), 3), (math.log(2), 1)]
    assert perplexity(parts) == pytest.approx(2**1.75)


def test_perplexity_of_nothing_is_none() -> None:
    assert perplexity([]) is None


class TemplateTok:
    """Each message is [10] + chars + [11]; the generation prompt adds [10]."""

    def apply_chat_template(
        self,
        messages: Any,
        tokenize: bool = True,
        return_dict: bool = True,
        add_generation_prompt: bool = False,
    ) -> dict[str, list[int]]:
        ids: list[int] = []
        for m in messages:
            ids += [10, *(12 + ord(c) % 20 for c in m["content"]), 11]
        if add_generation_prompt:
            ids.append(10)
        return {"input_ids": ids}


class UniformModel(torch.nn.Module):
    """Returns zero logits over ``vocab`` tokens: every token has p = 1/vocab."""

    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.seen: list[int] = []

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(self, input_ids: torch.Tensor, **kw: Any) -> Any:
        self.seen.append(input_ids.shape[1])
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, self.vocab))


ROW = {"instruction": "Why?", "input": "", "output": "Because."}


def test_reference_nll_masks_the_prompt_and_keeps_the_end_marker() -> None:
    nll, n, truncated = reference_nll(
        UniformModel(32), TemplateTok(), ROW, max_length=512
    )
    # Answer tokens: the 8 characters of "Because." and the closing [11].
    assert n == len("Because.") + 1
    assert nll == pytest.approx(n * math.log(32))
    assert truncated is False


def test_reference_nll_truncates_like_sft() -> None:
    model, tok = UniformModel(32), TemplateTok()
    _, n_full, _ = reference_nll(model, tok, ROW, max_length=512)
    total = model.seen[-1]
    nll, n, truncated = reference_nll(model, tok, ROW, max_length=total - 3)
    assert truncated is True
    assert n == n_full - 3
    assert model.seen[-1] == total - 3


def test_rouge_l_hand_computed() -> None:
    # LCS 3 words: precision 3/3, recall 3/6, F1 = 2 * 1 * 0.5 / 1.5 = 2/3.
    assert rouge_l(["the cat sat"], ["the cat sat on the mat"]) == pytest.approx(2 / 3)
    assert rouge_l(["a b", "x"], ["a b", "y"]) == pytest.approx(0.5)


def test_percentile_interpolates_linearly() -> None:
    values = [float(i) for i in range(1001)]
    assert percentile(values, 2.5) == pytest.approx(25.0)
    assert percentile(values, 97.5) == pytest.approx(975.0)
    assert percentile([0.0, 1.0], 50) == pytest.approx(0.5)


def test_bootstrap_ci_degenerate_inputs() -> None:
    assert bootstrap_ci([True, True, True], resamples=1000, seed=0) == (1.0, 1.0)
    assert bootstrap_ci([False, False], resamples=1000, seed=0) == (0.0, 0.0)
    assert bootstrap_ci([], resamples=1000, seed=0) is None


def test_bootstrap_ci_two_values_spans_zero_to_one() -> None:
    # Resampling [1, 0] twice gives a mean of 0 with p = 1/4, 0.5 with p = 1/2
    # and 1 with p = 1/4. Out of 1,000 draws, far more than 25 land on each
    # end, so the 2.5th and 97.5th percentiles are exactly 0 and 1.
    assert bootstrap_ci([True, False], resamples=1000, seed=0) == (0.0, 1.0)


def test_bootstrap_ci_matches_an_independent_numpy_computation() -> None:
    # random.Random.choices without weights draws index floor(random() * n)
    # for each pick. Redraw those indices here and take numpy's percentiles.
    xs = [True] * 6 + [False] * 4 + [True] * 2
    n, resamples, seed = len(xs), 1000, 11
    rng = random.Random(seed)
    values = np.array([1.0 if x else 0.0 for x in xs])
    means = [
        values[[int(rng.random() * n) for _ in range(n)]].mean()
        for _ in range(resamples)
    ]
    lo, hi = np.percentile(means, [2.5, 97.5])
    got = bootstrap_ci(xs, resamples=resamples, seed=seed)
    assert got is not None
    assert got == pytest.approx((lo, hi))
    assert got[0] < 8 / 12 < got[1]


def test_bootstrap_ci_is_seeded() -> None:
    xs = [True] * 7 + [False] * 3
    a = bootstrap_ci(xs, resamples=1000, seed=3)
    assert a == bootstrap_ci(xs, resamples=1000, seed=3)
    assert a is not None
    lo, hi = a
    assert 0.0 <= lo <= 0.7 <= hi <= 1.0


def test_rubric_rates_count_only_rows_a_rule_applies_to() -> None:
    finance = {"instruction": "How do dividends work?", "input": "", "output": "x"}
    general = {"instruction": "Name a colour.", "input": "", "output": "Blue."}
    results = [
        score(
            finance, "Dividends are paid.", stopped=True, new_tokens=5, max_new_tokens=9
        ),
        score(general, "Blue.", stopped=False, new_tokens=9, max_new_tokens=9),
    ]
    rates = rubric_rates(results, resamples=200, seed=0)
    assert rates["domain_terms"]["n"] == 1 and rates["domain_terms"]["k"] == 1
    assert rates["clean_stop"]["n"] == 2 and rates["clean_stop"]["k"] == 1
    assert rates["clean_stop"]["rate"] == pytest.approx(0.5)
    assert rates["overall"]["n"] == 2 and rates["overall"]["k"] == 1
    assert set(rates["format"]) == {"k", "n", "rate", "ci95"}


def test_rubric_rate_with_no_applicable_rows_is_none() -> None:
    general = {"instruction": "Name a colour.", "input": "", "output": "Blue."}
    res = [score(general, "Blue.", stopped=True, new_tokens=2, max_new_tokens=9)]
    rates = rubric_rates(res, resamples=100, seed=0)
    assert rates["domain_terms"] == {"k": 0, "n": 0, "rate": None, "ci95": None}
