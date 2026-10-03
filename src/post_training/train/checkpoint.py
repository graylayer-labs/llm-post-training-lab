"""Start and resume checks for a checkpointed training run.

The same rules as SFT's (#21, ``train/sft.py``), with the config passed as a
plain dict so any stage can use them. SFT still has its own copy, typed to
``SftConfig``; the two can merge in a later refactor.

A run keeps ``provenance.json`` and ``config.json`` from its first start. A
resume must use the same commit, a clean tree and the same config, unless the
caller passes ``scratch``. A checkpoint directory without
``trainer_state.json`` is a save cut short by a crash and is never resumed
from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from post_training.train.common import checkpoint_dirs, last_complete_checkpoint
from post_training.train.sft import ResumeError

RESUME_NOTE = (
    "The Trainer restores optimizer, scheduler and data order on resume, but "
    "not the MPS random-number state. A resumed MPS run is statistically "
    "equivalent to an uninterrupted one, not bit-identical. train_loss is the "
    "mean of logged losses; wall_seconds, peak_memory_gb and trainer-side "
    "timings cover only the resumed part."
)


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
