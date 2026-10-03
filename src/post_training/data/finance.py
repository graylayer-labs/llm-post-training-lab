"""Finance Q&A data: load, filter, split, and render to chat messages.

The dataset is Alpaca-style (instruction / input / output). We convert each row
to the chat-messages format so the model's own chat template, not ad hoc string
glue, decides the prompt layout. The same renderer is used for SFT, DPO and
evaluation so the three stages are never comparing different prompts.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

SYSTEM_PROMPT = (
    "You are a concise personal-finance assistant. Answer the question "
    "directly in one or two short paragraphs."
)

Row = dict[str, Any]
Message = dict[str, str]


def to_messages(row: Row) -> list[Message]:
    user = row["instruction"].strip()
    if row.get("input"):
        user = f"{user}\n\nContext: {row['input'].strip()}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["output"].strip()},
    ]


def prompt_messages(row: Row) -> list[Message]:
    """Messages up to and excluding the assistant turn (for generation)."""
    return to_messages(row)[:-1]


@dataclass(frozen=True)
class FinanceSplits:
    train: list[Row]
    eval: list[Row]


def make_splits(
    rows: Iterable[Row],
    *,
    train_size: int,
    eval_size: int,
    seed: int,
    min_chars: int = 40,
    max_chars: int = 1500,
) -> FinanceSplits:
    """Deterministic, disjoint train/eval split after length filtering.

    Filtering by output length removes one-word answers (nothing to learn
    from) and forum-length essays (would be truncated anyway and skew loss).
    """
    kept = [
        r
        for r in rows
        if min_chars <= len(r["output"]) <= max_chars and r["instruction"].strip()
    ]
    rng = random.Random(seed)
    rng.shuffle(kept)
    if len(kept) < train_size + eval_size:
        raise ValueError(f"only {len(kept)} rows after filtering")
    return FinanceSplits(
        train=kept[:train_size], eval=kept[train_size : train_size + eval_size]
    )


def load_splits(dataset: str, **kw: Any) -> FinanceSplits:
    from datasets import load_dataset

    ds = load_dataset(dataset, split="train")
    return make_splits(list(ds), **kw)
