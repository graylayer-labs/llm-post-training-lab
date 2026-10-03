"""Direct preference optimisation on top of the SFT model.

**Reference model.** DPO needs a frozen reference, and here it must be the
SFT model, not the base. With a PEFT policy and no ``ref_model``, TRL takes
the reference log-probs with the adapters disabled, which is the *base*
model when the SFT adapter is the active one. TRL's other route, passing the
SFT ``PeftModel`` and letting it copy the adapter to a "ref" adapter, copies
only parameters with ``.default.`` in their name. That misses the SFT
adapter's ``trainable_tokens_delta.default`` (the trained ``<|im_start|>`` and
``<|im_end|>`` rows), so the reference would silently not be SFT either.

So the SFT adapter is merged into the base weights first, and DPO trains a
fresh LoRA on the merged model. The reference is that model with the LoRA
disabled: exactly the merged SFT model, in the same memory, and correct by
construction. The trained policy is ``merge(base, sft adapter)`` plus the
LoRA saved in ``<output_dir>/adapter``.

**Rejected samples that never stopped.** TRL renders each completion through
the chat template, which closes it with ``<|im_end|>``. For a rejected sample
that hit the token limit that token was never generated, and keeping it
would push down "stop here" after a ramble. It is removed from those
completions.

**Off-policy chosen answers.** The chosen answers are human references,
which the SFT model finds unlikely. DPO can raise the margin while pushing
both log-probs down, so the summary keeps chosen and rejected log-probs over
training, not only the margin.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from post_training.config import DpoConfig, to_dict
from post_training.eval.harness import adapter_hash
from post_training.run import run_provenance
from post_training.train.checkpoint import (
    RESUME_NOTE,
    check_resume,
    plan_start,
)
from post_training.train.common import (
    PeakMemoryCallback,
    ReleaseCacheCallback,
    bf16_supported,
    device_report,
    pick_device,
    train_loss_from_log,
)
from post_training.train.sft import ResumeError, build_lora_config

__all__ = ["ResumeError", "run_dpo"]

END_OF_TURN = "<|im_end|>"
REFERENCE_NOTE = (
    "SFT adapter merged into the base weights; DPO trains a fresh LoRA on the "
    "merged model; TRL computes reference log-probs with that LoRA disabled, "
    "i.e. the merged SFT model."
)
LOGPS_NOTE = (
    "Chosen answers are human references (off-policy). Watch logps/chosen: a "
    "growing margin with falling chosen log-probs means DPO is pushing both "
    "answers down."
)
TRAJECTORY_KEYS = (
    "loss",
    "rewards/chosen",
    "rewards/rejected",
    "rewards/margins",
    "rewards/accuracies",
    "logps/chosen",
    "logps/rejected",
)


def load_sft_merged(
    model_name: str,
    adapter: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | None = None,
) -> tuple[Any, Any]:
    """The SFT model as plain weights: base with the SFT adapter merged in."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok: Any = AutoTokenizer.from_pretrained(adapter)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model: Any = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    return model.to(device or pick_device()), tok


def split_pairs(
    pairs: Sequence[dict[str, Any]], fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Seeded (train, held-out) split; both keep the file order."""
    n = len(pairs)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_held = min(max(1, round(fraction * n)), n - 1) if n >= 2 else 0
    held = set(idx[:n_held])
    return (
        [p for i, p in enumerate(pairs) if i not in held],
        [p for i, p in enumerate(pairs) if i in held],
    )


def to_dpo_dataset(pairs: Sequence[dict[str, Any]]) -> Dataset:
    """Conversational prompt / chosen / rejected, plus whether rejected stopped."""
    return Dataset.from_dict(
        {
            "prompt": [p["prompt"] for p in pairs],
            "chosen": [[{"role": "assistant", "content": p["chosen"]}] for p in pairs],
            "rejected": [
                [{"role": "assistant", "content": p["rejected"]}] for p in pairs
            ],
            "rejected_stopped": [bool(p["rejected_stopped"]) for p in pairs],
        }
    )


def strip_unstopped_end(example: dict[str, Any], end_id: int) -> dict[str, Any]:
    """Cut a never-stopped rejected completion before its added end-of-turn."""
    ids = list(example["rejected_ids"])
    if not example["rejected_stopped"] and end_id in ids:
        ids = ids[: len(ids) - 1 - ids[::-1].index(end_id)]
    return {"rejected_ids": ids}


class SftReferenceDPOTrainer(DPOTrainer):
    """DPOTrainer that drops the end-of-turn token TRL adds to unstopped samples."""

    def _prepare_dataset(self, dataset: Any, *args: Any, **kw: Any) -> Any:
        ds = super()._prepare_dataset(dataset, *args, **kw)
        end = self._tokenizer.convert_tokens_to_ids(END_OF_TURN)
        return ds.map(strip_unstopped_end, fn_kwargs={"end_id": end})


def build_dpo_args(cfg: DpoConfig, n_train: int, **overrides: Any) -> Any:
    """TRL DPOConfig for a run with ``n_train`` training pairs."""
    t = cfg.train
    steps_per_epoch = max(1, n_train // (t.batch_size * t.grad_accum))
    kw: dict[str, Any] = {
        "output_dir": str(Path(cfg.output_dir) / "checkpoints"),
        "num_train_epochs": t.epochs,
        "learning_rate": t.learning_rate,
        "per_device_train_batch_size": t.batch_size,
        "per_device_eval_batch_size": t.batch_size,
        "gradient_accumulation_steps": t.grad_accum,
        "max_length": t.max_length,
        "warmup_steps": int(t.warmup_ratio * steps_per_epoch * t.epochs),
        "lr_scheduler_type": "cosine",
        "logging_steps": t.logging_steps,
        "eval_strategy": "no",
        "save_strategy": "steps",
        "save_steps": t.save_steps,
        "save_total_limit": t.save_total_limit,
        "bf16": bf16_supported(),
        "beta": t.beta,
        # The reference is the policy with its LoRA off, so it costs no
        # extra memory; precomputing would only add a pass over the data.
        "precompute_ref_log_probs": False,
        "report_to": [],
        "seed": t.seed,
        **overrides,
    }
    return DPOConfig(**kw)


def make_trainer(
    cfg: DpoConfig,
    model: Any,
    tok: Any,
    args: Any,
    train_pairs: Sequence[dict[str, Any]],
    eval_pairs: Sequence[dict[str, Any]],
    callbacks: Sequence[Any] = (),
) -> Any:
    """The trainer: fresh LoRA on the merged SFT model, no separate ref model."""
    tokens = list(cfg.lora.trainable_tokens)
    ids = tok.convert_tokens_to_ids(tokens) if tokens else None
    return SftReferenceDPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=to_dpo_dataset(train_pairs),
        eval_dataset=to_dpo_dataset(eval_pairs),
        processing_class=tok,
        peft_config=build_lora_config(cfg.lora, ids),
        callbacks=list(callbacks),
    )


def _trajectory(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"step": h["step"], **{k: h[k] for k in TRAJECTORY_KEYS if k in h}}
        for h in log
        if "loss" in h
    ]


def _change(traj: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    vals = [t[key] for t in traj if key in t]
    if not vals:
        return None
    return {"first": vals[0], "last": vals[-1], "delta": vals[-1] - vals[0]}


def _load_pairs(
    cfg: DpoConfig, a_hash: str
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    pdir = Path(cfg.pairs_dir)
    manifest = json.loads((pdir / "manifest.json").read_text())
    if manifest["adapter_sha256"] != a_hash:
        raise ValueError(
            f"pairs in {pdir} were sampled from another SFT adapter "
            f"({manifest['adapter_sha256'][:12]}, not {a_hash[:12]}); rebuild "
            "them with `lab pairs` against this SFT run"
        )
    raw = (pdir / "pairs.jsonl").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if manifest["pairs_sha256"] != sha:
        raise ValueError(f"{pdir / 'pairs.jsonl'} changed since its manifest")
    pairs = [json.loads(x) for x in raw.decode().splitlines() if x]
    return manifest, pairs, sha


def run_dpo(
    cfg: DpoConfig,
    resume: bool = False,
    scratch: bool = False,
    *,
    extra_callbacks: Sequence[Any] = (),
) -> dict[str, Any]:
    if cfg.reference != "sft":
        raise ValueError(
            f"reference={cfg.reference!r}: only 'sft' is supported; DPO here "
            "must stay close to the SFT model, not the base"
        )
    out = Path(cfg.output_dir)
    last = plan_start(out, resume)
    # Capture first: the record must describe the code that ran.
    current = run_provenance()
    provenance, prov_missing = current, False
    if resume:
        provenance, prov_missing = check_resume(out, current, to_dict(cfg), scratch)

    run = Path(cfg.sft_run_dir)
    adapter = run / "adapter"
    a_hash = adapter_hash(adapter)
    manifest, pairs, pairs_sha = _load_pairs(cfg, a_hash)
    t = cfg.train
    train_pairs, held = split_pairs(pairs, t.eval_fraction, t.seed)
    if not train_pairs or not held:
        raise ValueError(f"{len(pairs)} pairs: too few to hold any out")
    model_name = json.loads((run / "summary.json").read_text())["config"]["model_name"]

    out.mkdir(parents=True, exist_ok=True)
    if not resume:
        (out / "provenance.json").write_text(json.dumps(provenance, indent=1))
        (out / "config.json").write_text(json.dumps(to_dict(cfg), indent=1))

    model, tok = load_sft_merged(model_name, str(adapter))
    params_total = sum(p.numel() for p in model.parameters())
    args = build_dpo_args(cfg, len(train_pairs))
    peak = PeakMemoryCallback()
    trainer = make_trainer(
        cfg,
        model,
        tok,
        args,
        train_pairs,
        held,
        # Order matters: sample the peak before the cache is released.
        callbacks=[peak, ReleaseCacheCallback(), *extra_callbacks],
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

    log = trainer.state.log_history
    traj = _trajectory(log)
    summary: dict[str, Any] = {
        "config": to_dict(cfg),
        "reference": {
            "model": "sft",
            "construction": REFERENCE_NOTE,
            "base_model": model_name,
            "sft_run_dir": str(run),
            "sft_adapter": str(adapter),
            "sft_adapter_sha256": a_hash,
        },
        "policy": f"merge({model_name}, {adapter}) + LoRA in {out / 'adapter'}",
        "provenance": provenance,
        "device": device_report(),
        "peak_memory_gb": round(peak.peak_gb, 2),
        "pairs": {
            "dir": cfg.pairs_dir,
            "pairs_sha256": pairs_sha,
            "rubric_version": manifest.get("rubric_version"),
            "n_pairs": len(pairs),
            "n_train": len(train_pairs),
            "n_held_out": len(held),
            "held_out_indices": [p["index"] for p in held],
            "held_out_seed": t.seed,
        },
        "params_total": params_total,
        "params_trainable": trainable,
        "eval_before": eval_before,
        "eval_after": eval_after,
        "train_loss": train_loss_from_log(log),
        "global_step": trainer.state.global_step,
        "trajectory": traj,
        "logps_change": {
            "chosen": _change(traj, "logps/chosen"),
            "rejected": _change(traj, "logps/rejected"),
        },
        "logps_note": LOGPS_NOTE,
        "wall_seconds": round(time.time() - t0, 1),
        "log_history": log,
    }
    if resume:
        summary["resumed_from"] = int(Path(str(last)).name.split("-")[-1])
        summary["resume_note"] = RESUME_NOTE
        if prov_missing:
            summary["original_provenance_missing"] = True
        summary["resumed_at"] = resumed_at
        summary["resume_provenance"] = current
        summary["resumed_segment_only"] = True
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if not t.keep_checkpoints:
        shutil.rmtree(out / "checkpoints", ignore_errors=True)
    return summary
