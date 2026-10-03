"""Supervised fine-tuning with LoRA adapters.

Loss is next-token cross-entropy on the assistant turn only. TRL's
SFTTrainer handles the chat template, prompt masking and packing; this module
only turns a config into the objects it needs and records what happened.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig

from post_training.config import LoraSettings, SftConfig, to_dict
from post_training.data.finance import Row, load_splits, to_messages
from post_training.train.common import (
    PeakMemoryCallback,
    ReleaseCacheCallback,
    device_report,
    load_model_and_tokenizer,
)


def build_lora_config(
    lora: LoraSettings, token_ids: list[int] | None = None
) -> LoraConfig:
    """LoRA on the projections, plus full updates to selected embedding rows.

    ``token_ids`` rows of ``embed_tokens`` are trained densely; with tied
    weights that also updates the matching output-logit rows.
    """
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        trainable_token_indices={"embed_tokens": token_ids} if token_ids else None,
        task_type="CAUSAL_LM",
    )


def to_chat_dataset(rows: list[Row]) -> Dataset:
    """Prompt/completion format so TRL masks prompt tokens out of the loss.

    With the plain ``messages`` format TRL would train on the system and user
    turns too, unless the chat template carries ``{% generation %}`` markers,
    which Qwen2.5's does not.
    """
    msgs = [to_messages(r) for r in rows]
    return Dataset.from_dict(
        {"prompt": [m[:-1] for m in msgs], "completion": [m[-1:] for m in msgs]}
    )


def build_sft_args(cfg: SftConfig, n_train: int) -> Any:
    """TRL SFTConfig for a run with ``n_train`` training rows."""
    from trl import SFTConfig

    steps_per_epoch = max(1, n_train // (cfg.train.batch_size * cfg.train.grad_accum))
    warmup_steps = int(cfg.train.warmup_ratio * steps_per_epoch * cfg.train.epochs)
    return SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.train.epochs,
        learning_rate=cfg.train.learning_rate,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        max_length=cfg.train.max_length,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=cfg.train.logging_steps,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=True,
        completion_only_loss=True,
        # TRL's default "chunked_nll" multiplies hidden states by lm_head.weight
        # directly, bypassing PEFT's trainable-token wrapper on the tied
        # embedding, so the <|im_end|> row would get no gradient.
        loss_type="nll",
        report_to=[],
        seed=cfg.train.seed,
    )


def run_sft(cfg: SftConfig) -> dict[str, Any]:
    from trl import SFTTrainer

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = load_splits(
        cfg.data.dataset,
        train_size=cfg.data.train_size,
        eval_size=cfg.data.eval_size,
        seed=cfg.data.seed,
        min_chars=cfg.data.min_chars,
        max_chars=cfg.data.max_chars,
    )
    (out / "eval_rows.json").write_text(json.dumps(splits.eval, indent=1))

    model, tok = load_model_and_tokenizer(cfg.model_name)
    trainable_before = sum(p.numel() for p in model.parameters())

    args = build_sft_args(cfg, len(splits.train))
    peak = PeakMemoryCallback()
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=to_chat_dataset(splits.train),
        eval_dataset=to_chat_dataset(splits.eval),
        processing_class=tok,
        peft_config=build_lora_config(
            cfg.lora, tok.convert_tokens_to_ids(list(cfg.lora.trainable_tokens))
        ),
        # Order matters: sample the peak before the cache is released.
        callbacks=[peak, ReleaseCacheCallback()],
    )
    peft_model: Any = trainer.model
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

    t0 = time.time()
    eval_before = trainer.evaluate()
    train_result = trainer.train()
    eval_after = trainer.evaluate()
    peft_model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")

    summary = {
        "config": to_dict(cfg),
        "device": device_report(),
        "peak_memory_gb": round(peak.peak_gb, 2),
        "params_total": trainable_before,
        "params_trainable": trainable,
        "trainable_pct": round(100 * trainable / trainable_before, 3),
        "eval_loss_before": eval_before["eval_loss"],
        "eval_loss_after": eval_after["eval_loss"],
        "train_loss": train_result.training_loss,
        "wall_seconds": round(time.time() - t0, 1),
        "log_history": trainer.state.log_history,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
