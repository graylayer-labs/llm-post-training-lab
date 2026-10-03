"""Checkpoint and resume rules shared by the training stages.

This mirrors the SFT checkpoint/resume work in #21 (branch
``feat/21-ckpt-resume``), which keeps the same helpers in ``train/common.py``
and ``train/sft.py``. Once both are merged, SFT and DPO should share one copy.

A run keeps ``provenance.json`` and ``config.json`` from its first start. A
resume must use the same commit, a clean tree and the same config, unless the
caller passes ``scratch``. A checkpoint directory without
``trainer_state.json`` is a save cut short by a crash and is never resumed
from.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

RESUME_NOTE = (
    "The Trainer restores optimizer, scheduler and data order on resume, but "
    "not the MPS random-number state. A resumed MPS run is statistically "
    "equivalent to an uninterrupted one, not bit-identical. train_loss is the "
    "mean of logged losses; wall_seconds, peak_memory_gb and trainer-side "
    "timings cover only the resumed part."
)


class ResumeError(RuntimeError):
    """A start or resume that would lose, overwrite or mix up a run."""


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
    """Newest checkpoint that has ``trainer_state.json`` (written last)."""
    for d in checkpoint_dirs(ckpt_dir):
        if (d / "trainer_state.json").exists():
            return d
    return None


def train_loss_from_log(log_history: list[dict[str, Any]]) -> float | None:
    """Mean of the logged step-wise training losses, or None if none logged.

    The Trainer's own ``training_loss`` is wrong after a resume; the log
    survives checkpoints.
    """
    losses = [h["loss"] for h in log_history if "loss" in h]
    return sum(losses) / len(losses) if losses else None


def plan_start(out: Path, resume: bool) -> Path | None:
    """The checkpoint to resume from, or None for a fresh start.

    Refuses a fresh start over existing checkpoints, and a resume with no
    complete checkpoint.
    """
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
    return last


def check_resume(
    out: Path, current: dict[str, Any], cfg: dict[str, Any], scratch: bool
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
    elif json.loads(saved_cfg.read_text()) != json.loads(json.dumps(cfg)):
        problems.append("config differs from the one the run started with")
    if problems and not scratch:
        raise ResumeError(
            "; ".join(problems) + ". A resumed run must be one commit's code "
            "and one config; fix the above or pass --scratch."
        )
    return original, missing
