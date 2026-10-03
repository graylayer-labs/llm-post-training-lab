"""Probe how strongly a model predicts ``<|im_end|>`` after a reference answer.

For each held-out row of a run, build the full chat (system, user, assistant
reference) as SFT trains on it, then teacher-force it through the model. At
the position of the last reference-answer token, record the probability and
rank of ``<|im_end|>`` as the next token. Runs the base model and base plus
``RUN_DIR/adapter``.

    uv run python tools/stop_token_probe.py outputs/sft -n 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

from post_training.data.finance import to_messages
from post_training.eval.stop_token import token_prob_rank
from post_training.run import run_provenance
from post_training.train.common import load_model_and_tokenizer

END = "<|im_end|>"


def probe_rows(
    model: Any, tok: Any, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    end_id = tok.convert_tokens_to_ids(END)
    out = []
    for row in rows:
        text = tok.apply_chat_template(to_messages(row), tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        # The template ends the assistant turn with <|im_end|>, then a newline.
        end_pos = int((ids[0] == end_id).nonzero()[-1].item())
        with torch.no_grad():
            logits = model(ids.to(model.device)).logits[0]
        prob, rank = token_prob_rank(logits, end_pos - 1, end_id)
        out.append({"prob": prob, "rank": rank})
    return out


def summarise(per_row: list[dict[str, Any]]) -> dict[str, Any]:
    probs = [r["prob"] for r in per_row]
    ranks = [r["rank"] for r in per_row]
    return {
        "median_prob": statistics.median(probs),
        "mean_prob": statistics.fmean(probs),
        "median_rank": statistics.median(ranks),
        "share_rank_1": sum(r == 1 for r in ranks) / len(ranks),
        "per_row": per_row,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("-n", type=int, default=None, help="first N rows (default: all)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--scratch", action="store_true", help="allow a scratch tree")
    a = p.parse_args()

    prov = run_provenance()
    if prov["scratch"] and not a.scratch:
        sys.exit(f"scratch tree ({prov['scratch_reason']}); pass --scratch to proceed")
    out = a.out or a.run_dir / "stop_token_probe.json"
    if prov["scratch"] and not out.name.startswith("scratch-"):
        out = out.with_name(f"scratch-{out.name}")

    rows = json.loads((a.run_dir / "eval_rows.json").read_text())[: a.n]
    adapter = a.run_dir / "adapter"
    result: dict[str, Any] = {
        "provenance": prov,
        "run_dir": str(a.run_dir),
        "adapter": str(adapter),
        "n": len(rows),
        "model_name": a.model,
        "token": END,
        "models": {},
    }
    for label, ad in (("base", None), ("sft", str(adapter))):
        model, tok = load_model_and_tokenizer(a.model, ad)
        model.eval()
        result["models"][label] = summarise(probe_rows(model, tok, rows))
        del model
    out.write_text(json.dumps(result, indent=2) + "\n")
    for label, m in result["models"].items():
        print(
            f"{label}: median_prob={m['median_prob']:.4g} "
            f"mean_prob={m['mean_prob']:.4g} median_rank={m['median_rank']} "
            f"share_rank_1={m['share_rank_1']:.2f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
