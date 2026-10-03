"""Peak accelerator memory for a few training steps: LoRA vs full fine-tune.

Runs the same SFTTrainer setup as ``lab sft`` for ``--steps`` optimiser steps
(no gradient accumulation, no eval) in one of three modes and prints peak
memory, trainable parameters and seconds per step. Run each mode in its own
process so one run's cached allocations do not leak into the next.

    uv run python tools/memory_lora_vs_full.py --mode lora --rank 16
    uv run python tools/memory_lora_vs_full.py --mode lora --rank 64
    uv run python tools/memory_lora_vs_full.py --mode full
    uv run python tools/memory_lora_vs_full.py --mode lora --loss chunked_nll

Peak memory is the highest MPS driver allocation seen after any step (MPS has
no true peak counter). It includes weights, gradients, AdamW state, cached
activations and allocator slack. ``--loss nll`` (what ``lab sft`` uses)
materialises the full batch x sequence x 152k-vocab logits; TRL's default
``chunked_nll`` computes the loss in chunks and never holds them all at once.
LoRA runs also train the chat-token embedding rows, as ``lab sft`` does.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from typing import Any

from post_training.config import LoraSettings
from post_training.data.finance import load_splits
from post_training.train.common import (
    PeakMemoryCallback,
    ReleaseCacheCallback,
    load_model_and_tokenizer,
)
from post_training.train.sft import build_lora_config, to_chat_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["lora", "full"], required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--loss", choices=["nll", "chunked_nll"], default="nll")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    a = p.parse_args()

    from trl import SFTConfig, SFTTrainer

    splits = load_splits(
        "gbharti/finance-alpaca", train_size=a.steps * a.batch_size, eval_size=1, seed=0
    )
    model, tok = load_model_and_tokenizer(a.model)
    lora = dataclasses.replace(LoraSettings(), r=a.rank, alpha=2 * a.rank)
    # chunked_nll bypasses PEFT's trainable-token wrapper, so drop it there.
    tokens = list(lora.trainable_tokens) if a.loss == "nll" else []
    token_ids = tok.convert_tokens_to_ids(tokens) if tokens else None
    peak = PeakMemoryCallback()
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir="outputs/_memory_probe",
            max_steps=a.steps,
            per_device_train_batch_size=a.batch_size,
            learning_rate=2e-4 if a.mode == "lora" else 1e-5,
            max_length=512,
            bf16=True,
            completion_only_loss=True,
            save_strategy="no",
            eval_strategy="no",
            loss_type=a.loss,
            report_to=[],
        ),
        train_dataset=to_chat_dataset(splits.train),
        processing_class=tok,
        peft_config=build_lora_config(lora, token_ids) if a.mode == "lora" else None,
        callbacks=[peak, ReleaseCacheCallback()],
    )
    m: Any = trainer.model
    t0 = time.time()
    trainer.train()
    print(
        json.dumps(
            {
                "mode": a.mode,
                "loss": a.loss,
                "rank": a.rank if a.mode == "lora" else None,
                "trainable_params": sum(
                    p.numel() for p in m.parameters() if p.requires_grad
                ),
                "peak_memory_gb": round(peak.peak_gb, 2),
                "sec_per_step": round((time.time() - t0) / a.steps, 2),
            }
        )
    )


if __name__ == "__main__":
    main()
