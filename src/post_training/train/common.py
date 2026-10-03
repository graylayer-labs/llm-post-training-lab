"""Shared model loading and device reporting."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
