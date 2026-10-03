import math

import torch

from post_training.eval.stop_token import token_prob_rank


def test_prob_and_rank_hand_computed() -> None:
    # Position 1 logits [0, ln2, ln1, ln1] -> softmax denominators 1+2+1+1=5.
    ln2 = math.log(2)
    logits = torch.tensor([[9.0, 9.0, 9.0, 9.0], [0.0, ln2, 0.0, 0.0]])
    prob, rank = token_prob_rank(logits, position=1, token_id=1)
    assert math.isclose(prob, 2 / 5, rel_tol=1e-6)
    assert rank == 1
    prob, rank = token_prob_rank(logits, position=1, token_id=0)
    assert math.isclose(prob, 1 / 5, rel_tol=1e-6)
    assert rank == 2  # only token 1 is strictly greater


def test_ties_share_the_best_rank() -> None:
    logits = torch.tensor([[1.0, 3.0, 3.0, 0.0, 3.0]])
    assert token_prob_rank(logits, 0, 1)[1] == 1
    assert token_prob_rank(logits, 0, 4)[1] == 1
    assert token_prob_rank(logits, 0, 0)[1] == 4  # three strictly greater
    assert token_prob_rank(logits, 0, 3)[1] == 5


def test_stable_with_large_logits() -> None:
    logits = torch.tensor([[1000.0, 1000.0]])
    prob, _ = token_prob_rank(logits, 0, 0)
    assert math.isclose(prob, 0.5, rel_tol=1e-6)
