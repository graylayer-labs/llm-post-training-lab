"""Greedy answers from the base model and the SFT adapter on held-out prompts.

Reads the eval rows a run saved (``<run>/eval_rows.json``), takes the first N,
renders each with the same chat template used in training, and decodes
greedily from the base model and then from base + adapter. Prints both
answers, the number of new tokens, and whether generation ended on a stop
token or ran into ``--max-new-tokens``.

    uv run python tools/compare_generations.py outputs/sft -n 5

Stop tokens are ``<|im_end|>`` and ``<|endoftext|>``. The base checkpoint's
generation config lists only ``<|endoftext|>``, but the chat template closes
the assistant turn with ``<|im_end|>``, and that is the token SFT teaches the
model to emit. Without it in the stop set the SFT model would end its answer
and then keep going into a new turn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from post_training.data.finance import prompt_messages
from post_training.train.common import load_model_and_tokenizer


@torch.no_grad()
def answer(model: Any, tok: Any, row: dict, max_new_tokens: int) -> dict:
    text = tok.apply_chat_template(
        prompt_messages(row), tokenize=False, add_generation_prompt=True
    )
    enc = tok(text, return_tensors="pt").to(model.device)
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=stop,
        pad_token_id=tok.pad_token_id,
    )
    new = out[0, enc["input_ids"].shape[1] :]
    return {
        "text": tok.decode(new, skip_special_tokens=True).strip(),
        "new_tokens": len(new),
        "stopped": int(new[-1]) in stop,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--max-new-tokens", type=int, default=256)
    a = p.parse_args()

    rows = json.loads((a.run_dir / "eval_rows.json").read_text())[: a.n]
    results: list[dict[str, Any]] = [{"instruction": r["instruction"]} for r in rows]
    for name, adapter in (("base", None), ("sft", str(a.run_dir / "adapter"))):
        model, tok = load_model_and_tokenizer(a.model, adapter=adapter)
        model.eval()
        for res, row in zip(results, rows, strict=True):
            res[name] = answer(model, tok, row, a.max_new_tokens)
        del model

    for i, res in enumerate(results):
        print(f"\n### {i + 1}. {res['instruction']}")
        for name in ("base", "sft"):
            g = res[name]
            end = "stop token" if g["stopped"] else "hit max_new_tokens"
            print(f"\n[{name}] {g['new_tokens']} tokens, {end}\n{g['text']}")
    (a.run_dir / "generations.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
