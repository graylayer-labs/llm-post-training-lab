"""Greedy answers from the base model and the SFT adapter on held-out prompts.

Reads the eval rows a run saved (``<run>/eval_rows.json``), takes the first N,
renders each with the same chat template used in training, and decodes
greedily from the base model and then from base + adapter. Prints both
answers, the number of new tokens, and whether generation ended on a stop
token or ran into ``--max-new-tokens``.

    uv run python tools/compare_generations.py outputs/sft -n 5

Generation (stop tokens, left padding) is in ``post_training.generate``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from post_training.data.finance import prompt_messages
from post_training.generate import generate
from post_training.train.common import load_model_and_tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    a = p.parse_args()

    rows = json.loads((a.run_dir / "eval_rows.json").read_text())[: a.n]
    results: list[dict[str, Any]] = [{"instruction": r["instruction"]} for r in rows]
    settings: dict[str, Any] = {}
    for name, adapter in (("base", None), ("sft", str(a.run_dir / "adapter"))):
        model, tok = load_model_and_tokenizer(a.model, adapter=adapter)
        model.eval()
        outs = generate(
            model,
            tok,
            [prompt_messages(r) for r in rows],
            a.max_new_tokens,
            a.batch_size,
            settings_out=settings,
        )
        for res, out in zip(results, outs, strict=True):
            res[name] = out
        del model

    print(f"generation settings: {json.dumps(settings)}")
    for i, res in enumerate(results):
        print(f"\n### {i + 1}. {res['instruction']}")
        for name in ("base", "sft"):
            g = res[name]
            end = "stop token" if g["stopped"] else "hit max_new_tokens"
            print(f"\n[{name}] {g['new_tokens']} tokens, {end}\n{g['text']}")
    (a.run_dir / "generations.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
