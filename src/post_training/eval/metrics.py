"""Metrics for the eval harness: perplexity, ROUGE-L and rubric pass rates.

Perplexity is of the reference answers, answer tokens only. Each row is
rendered with the same chat template and prompt/completion split that TRL
used in SFT: the prompt is the system and user turns plus the generation
prompt, and the answer is everything after it, ``<|im_end|>`` included. The
corpus perplexity is ``exp`` of the token-weighted mean NLL (total NLL over
total answer tokens), not the mean of per-row perplexities.

Rubric scoring itself lives in ``post_training.eval.rubric``; this module
only counts its results.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from post_training.data.finance import Row, prompt_messages, to_messages
from post_training.eval.rubric import RubricResult

ROUGE_TYPE = "rougeL"
ROUGE_USE_STEMMER = True


def answer_nll(
    logits: torch.Tensor, input_ids: torch.Tensor, n_prompt: int
) -> tuple[float, int]:
    """Summed NLL of tokens ``n_prompt:`` and their count.

    ``logits`` is ``[seq, vocab]`` for one sequence; position ``t`` predicts
    token ``t + 1``.
    """
    targets = input_ids[n_prompt:]
    if len(targets) == 0:
        return 0.0, 0
    pred = logits[n_prompt - 1 : len(input_ids) - 1].float()
    nll = F.cross_entropy(pred, targets.to(pred.device), reduction="sum")
    return float(nll), int(len(targets))


def perplexity(parts: Sequence[tuple[float, int]]) -> float | None:
    """``exp`` of the token-weighted mean NLL over ``(nll_sum, n_tokens)``."""
    total = sum(n for _, n in parts)
    if total == 0:
        return None
    return math.exp(sum(nll for nll, _ in parts) / total)


def chat_ids(tok: Any, messages: Any, add_generation_prompt: bool) -> list[int]:
    """Token ids of ``messages`` under the tokenizer's chat template, as TRL
    tokenizes them for SFT."""
    return list(
        tok.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=add_generation_prompt,
        )["input_ids"]
    )


@torch.no_grad()
def reference_nll(
    model: Any, tok: Any, row: Row, max_length: int
) -> tuple[float, int, bool]:
    """NLL of the reference answer given the prompt, as SFT scored it.

    Returns ``(nll_sum, n_answer_tokens, truncated)``. The sequence is cut at
    ``max_length`` tokens, as TRL truncated it in training.
    """
    prompt = chat_ids(tok, prompt_messages(row), add_generation_prompt=True)
    full = chat_ids(tok, to_messages(row), add_generation_prompt=False)
    if full[: len(prompt)] != prompt:
        raise ValueError("chat template: prompt is not a prefix of prompt+answer")
    truncated = len(full) > max_length
    full = full[:max_length]
    ids = torch.tensor([full], device=model.device)
    logits = model(input_ids=ids).logits[0]
    nll, n = answer_nll(logits, ids[0], len(prompt))
    return nll, n, truncated


def rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float | None:
    """Mean ROUGE-L F1 of each prediction against its reference."""
    from rouge_score import rouge_scorer

    if not predictions:
        return None
    scorer = rouge_scorer.RougeScorer([ROUGE_TYPE], use_stemmer=ROUGE_USE_STEMMER)
    scores = [
        scorer.score(ref, pred)[ROUGE_TYPE].fmeasure
        for pred, ref in zip(predictions, references, strict=True)
    ]
    return sum(scores) / len(scores)


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile with linear interpolation (numpy's default)."""
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def bootstrap_ci(
    passed: Sequence[bool], *, resamples: int, seed: int, level: float = 0.95
) -> tuple[float, float] | None:
    """Percentile bootstrap interval for a pass rate, or None with no rows."""
    n = len(passed)
    if n == 0:
        return None
    rng = random.Random(seed)
    xs = [1.0 if p else 0.0 for p in passed]
    means = [sum(rng.choices(xs, k=n)) / n for _ in range(resamples)]
    tail = 100 * (1 - level) / 2
    return percentile(means, tail), percentile(means, 100 - tail)


def _rate(passed: list[bool], resamples: int, seed: int) -> dict[str, Any]:
    n = len(passed)
    return {
        "k": sum(passed),
        "n": n,
        "rate": sum(passed) / n if n else None,
        "ci95": bootstrap_ci(passed, resamples=resamples, seed=seed),
    }


def rubric_rates(
    results: Sequence[RubricResult], *, resamples: int, seed: int
) -> dict[str, dict[str, Any]]:
    """Per-rule pass rates over the rows each rule applies to, plus overall.

    Each entry is ``{k, n, rate, ci95}``: k passes out of n applicable rows
    and a percentile bootstrap interval. ``overall`` counts every row.
    """
    names = [r.name for r in results[0].rules] if results else []
    out = {
        name: _rate(
            [r.rule(name).passed for r in results if r.rule(name).applies],
            resamples,
            seed,
        )
        for name in names
    }
    out["overall"] = _rate([r.overall for r in results], resamples, seed)
    return out
