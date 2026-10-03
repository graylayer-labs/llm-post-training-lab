"""Probability and rank of one token in a next-token distribution."""

from __future__ import annotations

import torch


def token_prob_rank(
    logits: torch.Tensor, position: int, token_id: int
) -> tuple[float, int]:
    """Probability and rank of ``token_id`` in the distribution at ``position``.

    ``logits`` is [seq, vocab]. Rank 1 is the top token. Ties share the best
    rank: rank = 1 + the number of tokens with a strictly greater logit.
    """
    row = logits[position].float()
    prob = torch.softmax(row, dim=-1)[token_id].item()
    rank = 1 + int((row > row[token_id]).sum().item())
    return prob, rank
