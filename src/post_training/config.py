"""Typed, YAML-backed configuration for every stage.

Configs are plain dataclasses so that a run is fully described by one file
and the diff between two runs is a diff between two YAML files.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class LoraSettings:
    """LoRA hyper-parameters.

    r is the rank of the update; alpha scales it (the effective scale is
    alpha / r). Targeting attention *and* MLP projections is what the QLoRA
    paper found necessary to match full fine-tuning.

    trainable_tokens lists tokens whose embedding rows are trained in full.
    A base model has barely seen the chat-template tokens, and LoRA on the
    projections cannot change the (tied) embedding / output rows, so without
    this the model never learns to emit ``<|im_end|>`` and stop.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    trainable_tokens: tuple[str, ...] = ("<|im_start|>", "<|im_end|>")


@dataclass(frozen=True)
class DataSettings:
    dataset: str = "gbharti/finance-alpaca"
    train_size: int = 2000
    eval_size: int = 200
    seed: int = 0
    min_chars: int = 40
    max_chars: int = 1500


@dataclass(frozen=True)
class TrainSettings:
    epochs: float = 1.0
    learning_rate: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 4
    max_length: int = 512
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    seed: int = 0
    save_steps: int = 25
    save_total_limit: int = 2
    keep_checkpoints: bool = False


@dataclass(frozen=True)
class SftConfig:
    model_name: str
    output_dir: str
    lora: LoraSettings = field(default_factory=LoraSettings)
    data: DataSettings = field(default_factory=DataSettings)
    train: TrainSettings = field(default_factory=TrainSettings)


@dataclass(frozen=True)
class SystemSettings:
    """One system to evaluate: the base model, plus a LoRA adapter or none."""

    name: str
    adapter: str | None = None


def _default_systems() -> tuple[SystemSettings, ...]:
    return (
        SystemSettings("base", None),
        SystemSettings("sft", "outputs/sft/adapter"),
        SystemSettings("dpo", "outputs/dpo/adapter"),
    )


@dataclass(frozen=True)
class GenerationSettings:
    """Greedy decoding, the same for every system.

    batch_size is a setting, not a speed knob: batched bf16 on MPS changes
    greedy outputs (docs/decisions.md), so it stays fixed across systems.
    """

    max_new_tokens: int = 384
    batch_size: int = 1
    seed: int = 0


@dataclass(frozen=True)
class PerplexitySettings:
    """Truncate prompt + answer at SFT's max_length, as TRL did in training."""

    max_length: int = 512


@dataclass(frozen=True)
class BootstrapSettings:
    resamples: int = 1000
    seed: int = 0


@dataclass(frozen=True)
class EvalConfig:
    """Score each system on the held-out rows saved by the SFT run.

    eval_rows is the SFT run's ``eval_rows.json``. limit keeps the first N
    rows (for smoke runs); None keeps them all.
    """

    model_name: str
    output_dir: str
    eval_rows: str
    limit: int | None = None
    systems: tuple[SystemSettings, ...] = field(default_factory=_default_systems)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    perplexity: PerplexitySettings = field(default_factory=PerplexitySettings)
    bootstrap: BootstrapSettings = field(default_factory=BootstrapSettings)


def _build[T](cls: type[T], raw: Any) -> T:
    if not is_dataclass(cls):
        return raw
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        hint = hints[f.name]
        if is_dataclass(hint):
            value = _build(hint, value)
        elif get_origin(hint) is tuple:
            item = get_args(hint)[0]
            value = tuple(_build(item, v) if is_dataclass(item) else v for v in value)
        kwargs[f.name] = value
    return cls(**kwargs)


def load_config[T](path: str | Path, cls: type[T]) -> T:
    """Load a YAML file into a dataclass, nested sections included."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _build(cls, raw)


def to_dict(cfg: Any) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
