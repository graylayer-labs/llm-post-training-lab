"""Batched, left-padded generation with the project's shared stop-token set.

Stop tokens are ``<|im_end|>`` and ``<|endoftext|>``. The base checkpoint's
generation config lists only ``<|endoftext|>``, but the chat template closes
the assistant turn with ``<|im_end|>``, and that is the token SFT teaches the
model to emit. Without it in the stop set the SFT model would end its answer
and then keep going into a new turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


def stop_token_ids(tok: Any) -> list[int]:
    """Ids of the stop tokens, in order, without duplicates or missing ones."""
    ids: list[int] = []
    for t in (*STOP_TOKENS,):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != getattr(tok, "unk_token_id", None):
            ids.append(int(i))
    if tok.eos_token_id is not None:
        ids.append(int(tok.eos_token_id))
    return list(dict.fromkeys(ids))


def trim_at_stop(
    new_ids: torch.Tensor, stop: Sequence[int]
) -> tuple[torch.Tensor, bool]:
    """Cut generated ids after the first stop token (inclusive).

    Returns the kept ids and whether a stop token was found. Anything after the
    first stop token is padding and is dropped.
    """
    for pos, i in enumerate(new_ids.tolist()):
        if i in stop:
            return new_ids[: pos + 1], True
    return new_ids, False


@torch.no_grad()
def generate(
    model: Any,
    tok: Any,
    messages: Sequence[Sequence[dict[str, str]]],
    max_new_tokens: int,
    batch_size: int = 8,
    *,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 0,
    settings_out: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate one reply per prompt, in input order.

    Each result is ``{text, new_tokens, stopped}``. ``new_tokens`` counts the
    generated tokens up to and including the first stop token, never padding.
    Sampling is seeded per batch (``seed + batch index``), so a run is
    reproducible for a fixed batch size. The seed goes through the global
    ``torch.manual_seed``: this transformers version's ``generate`` takes no
    ``generator`` argument. ``top_k=0`` turns top-k off. If ``settings_out`` is
    given it is filled with every setting used, for the caller to record.
    """
    stop = stop_token_ids(tok)
    texts = [
        tok.apply_chat_template(list(m), tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    sampling: dict[str, Any] = (
        {
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": 1.0,
        }
        if do_sample
        else {"do_sample": False}
    )
    if settings_out is not None:
        settings_out.update(
            sampling,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            seed=seed,
            stop_token_ids=stop,
        )
    results: list[dict[str, Any]] = []
    old_side = tok.padding_side
    tok.padding_side = "left"
    try:
        for b, start in enumerate(range(0, len(texts), batch_size)):
            enc = tok(
                texts[start : start + batch_size], return_tensors="pt", padding=True
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            torch.manual_seed(seed + b)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                eos_token_id=stop,
                pad_token_id=tok.pad_token_id,
                **sampling,
            )
            for row in out[:, enc["input_ids"].shape[1] :]:
                kept, stopped = trim_at_stop(row, stop)
                results.append(
                    {
                        "text": tok.decode(kept, skip_special_tokens=True).strip(),
                        "new_tokens": len(kept),
                        "stopped": stopped,
                    }
                )
    finally:
        tok.padding_side = old_side
    return results
