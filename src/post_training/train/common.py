"""Shared model loading and device reporting."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback


def checkpoint_dirs(ckpt_dir: Path) -> list[Path]:
    """All ``checkpoint-N`` directories under ``ckpt_dir``, newest first."""
    if not ckpt_dir.is_dir():
        return []
    found = [
        (int(m.group(1)), d)
        for d in ckpt_dir.iterdir()
        if d.is_dir() and (m := re.fullmatch(r"checkpoint-(\d+)", d.name))
    ]
    return [d for _, d in sorted(found, key=lambda t: t[0], reverse=True)]


def last_complete_checkpoint(ckpt_dir: Path) -> Path | None:
    """Newest checkpoint that has ``trainer_state.json``.

    The Trainer writes that file last, so a directory without it is a save
    that was cut short by the crash and must not be resumed from.
    """
    for d in checkpoint_dirs(ckpt_dir):
        if (d / "trainer_state.json").exists():
            return d
    return None


def train_loss_from_log(log_history: list[dict[str, Any]]) -> float | None:
    """Mean of the logged step-wise training losses, or None if none logged.

    The Trainer's own ``training_loss`` is wrong after a resume (its running
    total restarts at 0 but is divided by the full step count). The log
    survives checkpoints. This equals what an uninterrupted run reports when
    logging covers every step; with ``logging_steps > 1`` each entry is the
    mean over its window, so a short final window makes the two differ
    slightly.
    """
    losses = [h["loss"] for h in log_history if "loss" in h]
    return sum(losses) / len(losses) if losses else None


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bf16_supported() -> bool:
    """Whether mixed-precision bf16 training works here: CUDA with bf16, or MPS.

    CPU-only machines such as CI runners fall back to fp32.
    """
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    return torch.backends.mps.is_available()


def load_model_and_tokenizer(
    model_name: str, adapter: str | None = None
) -> tuple[Any, Any]:
    tok: Any = AutoTokenizer.from_pretrained(adapter or model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model.to(pick_device()), tok


def accelerator_memory_gb() -> float:
    """Memory currently held by the accelerator's allocator, in GiB."""
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2**30
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 2**30
    return 0.0


class PeakMemoryCallback(TrainerCallback):
    """Track peak accelerator memory by sampling after every optimiser step.

    MPS has no ``max_memory_allocated``, so a single read at the end of a run
    only shows what the allocator still holds then, not the high-water mark.
    """

    def __init__(self, read_gb: Callable[[], float] = accelerator_memory_gb):
        self.read_gb = read_gb
        self.peak_gb = 0.0

    def _sample(self) -> None:
        self.peak_gb = max(self.peak_gb, self.read_gb())

    def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        self._sample()

    def on_evaluate(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        self._sample()


def release_accelerator_cache() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


class ReleaseCacheCallback(TrainerCallback):
    """Return cached allocator blocks to the driver after every step.

    Every batch has a different sequence length, so each step asks for new
    batch x seq x vocab logits buffers (152k vocab). On MPS the caching
    allocator keeps the old ones and driver memory grows step after step
    until the process runs out of memory.
    """

    def __init__(self, release: Callable[[], None] = release_accelerator_cache):
        self.release = release

    def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        self.release()


def device_report() -> dict[str, Any]:
    rep: dict[str, Any] = {"device": pick_device(), "torch": torch.__version__}
    if rep["device"] == "mps":
        rep["mps_alloc_gb_at_end"] = round(accelerator_memory_gb(), 2)
    elif rep["device"] == "cuda":
        rep["cuda_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return rep
