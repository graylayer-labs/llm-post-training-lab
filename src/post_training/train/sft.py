"""Supervised fine-tuning with LoRA adapters.

Loss is next-token cross-entropy on the assistant turn only. TRL's
SFTTrainer handles the chat template, prompt masking and packing; this module
only turns a config into the objects it needs and records what happened.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig

from post_training.config import LoraSettings, SftConfig, to_dict
from post_training.data.finance import Row, load_splits, to_messages
from post_training.run import run_provenance
from post_training.train.common import (
    PeakMemoryCallback,
    ReleaseCacheCallback,
    bf16_supported,
    checkpoint_dirs,
    device_report,
    last_complete_checkpoint,
    load_model_and_tokenizer,
    train_loss_from_log,
)


class ResumeError(RuntimeError):
    """A start or resume that would lose, overwrite or mix up a run."""


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
        output_dir=str(Path(cfg.output_dir) / "checkpoints"),
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
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        bf16=bf16_supported(),
        completion_only_loss=True,
        # TRL's default "chunked_nll" multiplies hidden states by lm_head.weight
        # directly, bypassing PEFT's trainable-token wrapper on the tied
        # embedding, so the <|im_end|> row would get no gradient.
        loss_type="nll",
        report_to=[],
        seed=cfg.train.seed,
    )


RESUME_NOTE = (
    "The Trainer restores optimizer, scheduler and data order on resume, but "
    "not the MPS random-number state. With LoRA dropout a resumed MPS run is "
    "statistically equivalent to an uninterrupted one, not bit-identical "
    "(on CPU it matched exactly). train_loss is the mean of logged losses; "
    "wall_seconds, peak_memory_gb and trainer-side timings cover only the "
    "resumed part."
)


def _check_resume(
    out: Path, current: dict[str, Any], cfg: SftConfig, scratch: bool
) -> tuple[dict[str, Any], bool]:
    """Original provenance of the run in ``out`` and whether it was missing.

    Raises ResumeError if the code, tree or config differ from the original
    run, unless ``scratch``.
    """
    problems = []
    saved = out / "provenance.json"
    missing = not saved.exists()
    original = current if missing else json.loads(saved.read_text())
    if missing:
        problems.append(f"{saved} is missing; cannot tell which commit started it")
    elif current["commit"] != original["commit"]:
        problems.append(
            f"commit is {current['commit']}, run started at {original['commit']}"
        )
    if current["dirty"]:
        problems.append("working tree is dirty")
    elif current["scratch"]:
        problems.append(f"this checkout is scratch ({current['scratch_reason']})")
    saved_cfg = out / "config.json"
    if not saved_cfg.exists():
        problems.append(f"{saved_cfg} is missing; cannot check the config")
    elif json.loads(saved_cfg.read_text()) != json.loads(json.dumps(to_dict(cfg))):
        problems.append("config differs from the one the run started with")
    if problems and not scratch:
        raise ResumeError(
            "; ".join(problems) + ". A resumed run must be one commit's code "
            "and one config; fix the above or pass --scratch."
        )
    return original, missing


def run_sft(
    cfg: SftConfig, resume: bool = False, scratch: bool = False
) -> dict[str, Any]:
    from trl import SFTTrainer

    out = Path(cfg.output_dir)
    ckpt_dir = out / "checkpoints"
    if checkpoint_dirs(ckpt_dir) and not resume:
        raise ResumeError(
            f"{ckpt_dir} already holds checkpoints; pass --resume to continue "
            "that run or use a new output_dir. Nothing was changed."
        )
    last = last_complete_checkpoint(ckpt_dir)
    if resume and last is None:
        raise ResumeError(
            f"--resume given but no complete checkpoint (one with "
            f"trainer_state.json) under {ckpt_dir}"
        )

    # Capture first: the record must describe the code that ran, not the tree
    # as it stands after training.
    current = run_provenance()
    prov_missing = False
    provenance = current
    if resume:
        provenance, prov_missing = _check_resume(out, current, cfg, scratch)
    out.mkdir(parents=True, exist_ok=True)
    if not resume:
        (out / "provenance.json").write_text(json.dumps(provenance, indent=1))
        (out / "config.json").write_text(json.dumps(to_dict(cfg), indent=1))
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
    before_file = out / "eval_before.json"
    if resume and before_file.exists():
        eval_before = json.loads(before_file.read_text())
    else:
        eval_before = trainer.evaluate()
        before_file.write_text(json.dumps(eval_before, indent=1))
    resumed_at = datetime.now(UTC).isoformat(timespec="seconds")
    if resume:
        trainer.train(resume_from_checkpoint=str(last))
    else:
        trainer.train()
    eval_after = trainer.evaluate()
    peft_model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")

    summary = {
        "config": to_dict(cfg),
        "provenance": provenance,
        "device": device_report(),
        "peak_memory_gb": round(peak.peak_gb, 2),
        "data_stats": splits.stats(),
        "params_total": trainable_before,
        "params_trainable": trainable,
        "trainable_pct": round(100 * trainable / trainable_before, 3),
        "eval_loss_before": eval_before["eval_loss"],
        "eval_loss_after": eval_after["eval_loss"],
        "train_loss": train_loss_from_log(trainer.state.log_history),
        "wall_seconds": round(time.time() - t0, 1),
        "log_history": trainer.state.log_history,
    }
    if resume:
        summary["resumed_from"] = int(Path(str(last)).name.split("-")[-1])
        summary["resume_note"] = RESUME_NOTE
        if prov_missing:
            summary["original_provenance_missing"] = True
        summary["resumed_at"] = resumed_at
        summary["resume_provenance"] = current
        # Timing, peak memory and train_loss cover only the resumed part; the
        # time before the crash was not recorded and is not guessed.
        summary["resumed_segment_only"] = True
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if not cfg.train.keep_checkpoints:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return summary
