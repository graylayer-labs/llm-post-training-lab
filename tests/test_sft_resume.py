import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from post_training.config import SftConfig, TrainSettings
from post_training.data.finance import FinanceSplits
from post_training.train import sft

PROV = {"commit": "aaa", "dirty": False, "scratch": False, "scratch_reason": None}


class StubModel(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(1, 1)

    def save_pretrained(self, path: Any) -> None:
        pass


class StubTok:
    def convert_tokens_to_ids(self, tokens: Any) -> list[int]:
        return [0 for _ in tokens]

    def save_pretrained(self, path: Any) -> None:
        pass


class Calls:
    def __init__(self) -> None:
        self.evaluate = 0
        self.resume_from: list[Any] = []


def _setup(
    monkeypatch: pytest.MonkeyPatch, prov: dict[str, Any] | None = None
) -> Calls:
    calls = Calls()

    class StubTrainer:
        def __init__(self, **kw: Any) -> None:
            self.model = StubModel()
            self.state = SimpleNamespace(log_history=[], global_step=0)

        def evaluate(self) -> dict[str, float]:
            calls.evaluate += 1
            return {"eval_loss": 1.0 + calls.evaluate}

        def train(self, resume_from_checkpoint: Any = None) -> SimpleNamespace:
            calls.resume_from.append(resume_from_checkpoint)
            return SimpleNamespace(training_loss=0.5)

    monkeypatch.setattr("trl.SFTTrainer", StubTrainer)
    monkeypatch.setattr(
        sft, "load_splits", lambda *a, **k: FinanceSplits(train=[], eval=[])
    )
    monkeypatch.setattr(
        sft, "load_model_and_tokenizer", lambda name: (StubModel(), StubTok())
    )
    monkeypatch.setattr(sft, "build_sft_args", lambda cfg, n: None)
    monkeypatch.setattr(sft, "to_chat_dataset", lambda rows: None)
    monkeypatch.setattr(sft, "build_lora_config", lambda *a: None)
    monkeypatch.setattr(sft, "run_provenance", lambda: dict(prov or PROV))
    return calls


def _cfg(out: Path, **train: Any) -> SftConfig:
    return SftConfig(model_name="m", output_dir=str(out), train=TrainSettings(**train))


def _checkpoints(out: Path, *steps: int) -> None:
    for s in steps:
        d = out / "checkpoints" / f"checkpoint-{s}"
        d.mkdir(parents=True)
        (d / "trainer_state.json").write_text("{}")


def _crashed_run(out: Path, *steps: int) -> None:
    _checkpoints(out, *steps)
    (out / "provenance.json").write_text(json.dumps(PROV))
    (out / "eval_before.json").write_text(json.dumps({"eval_loss": 9.0}))


def test_train_settings_checkpoint_defaults() -> None:
    t = TrainSettings()
    assert (t.save_steps, t.save_total_limit, t.keep_checkpoints) == (25, 2, False)


def test_sft_args_save_checkpoints_under_output_dir(tmp_path: Path) -> None:
    args = sft.build_sft_args(_cfg(tmp_path, save_steps=5, save_total_limit=3), 100)
    assert args.save_strategy == "steps"
    assert args.save_steps == 5
    assert args.save_total_limit == 3
    assert Path(args.output_dir) == tmp_path / "checkpoints"


def test_fresh_run_writes_provenance_and_eval_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch)
    sft.run_sft(_cfg(tmp_path))
    assert json.loads((tmp_path / "provenance.json").read_text())["commit"] == "aaa"
    assert json.loads((tmp_path / "eval_before.json").read_text())["eval_loss"] == 2.0


def test_refuses_to_start_over_existing_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _setup(monkeypatch)
    _checkpoints(tmp_path, 25)
    with pytest.raises(sft.ResumeError, match="--resume"):
        sft.run_sft(_cfg(tmp_path))
    assert calls.resume_from == []
    assert (tmp_path / "checkpoints" / "checkpoint-25").exists()


def test_resume_without_checkpoints_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch)
    with pytest.raises(sft.ResumeError, match="no checkpoint"):
        sft.run_sft(_cfg(tmp_path), resume=True)


def test_resume_picks_last_checkpoint_and_reuses_eval_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _setup(monkeypatch)
    _crashed_run(tmp_path, 25, 50)
    summary = sft.run_sft(_cfg(tmp_path, keep_checkpoints=True), resume=True)
    assert calls.resume_from == [str(tmp_path / "checkpoints" / "checkpoint-50")]
    assert calls.evaluate == 1  # only the "after" eval
    assert summary["eval_loss_before"] == 9.0
    assert summary["resumed_from"] == 50
    assert "resumed_at" in summary
    assert summary["provenance"]["commit"] == "aaa"
    assert summary["resumed_segment_only"] is True


def test_resume_refuses_on_commit_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _setup(monkeypatch, {**PROV, "commit": "bbb"})
    _crashed_run(tmp_path, 25)
    with pytest.raises(sft.ResumeError, match="commit"):
        sft.run_sft(_cfg(tmp_path), resume=True)
    assert calls.resume_from == []


def test_resume_refuses_on_dirty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, {**PROV, "dirty": True})
    _crashed_run(tmp_path, 25)
    with pytest.raises(sft.ResumeError, match="dirty"):
        sft.run_sft(_cfg(tmp_path), resume=True)


def test_scratch_allows_mismatch_and_keeps_original_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, {**PROV, "commit": "bbb", "dirty": True})
    _crashed_run(tmp_path, 25)
    summary = sft.run_sft(_cfg(tmp_path), resume=True, scratch=True)
    assert summary["provenance"]["commit"] == "aaa"
    assert summary["resume_provenance"]["commit"] == "bbb"


def test_checkpoints_deleted_after_summary_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch)
    _crashed_run(tmp_path, 25)
    sft.run_sft(_cfg(tmp_path), resume=True)
    assert (tmp_path / "summary.json").exists()
    assert not (tmp_path / "checkpoints").exists()


def test_keep_checkpoints_true_keeps_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch)
    _crashed_run(tmp_path, 25)
    sft.run_sft(_cfg(tmp_path, keep_checkpoints=True), resume=True)
    assert (tmp_path / "checkpoints" / "checkpoint-25").exists()


def test_cli_passes_resume_and_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from post_training import cli

    got: dict[str, Any] = {}

    def fake(cfg: Any, resume: bool = False, scratch: bool = False) -> dict[str, Any]:
        got.update(resume=resume, scratch=scratch)
        return {"log_history": []}

    monkeypatch.setattr(sft, "run_sft", fake)
    p = tmp_path / "c.yaml"
    p.write_text("model_name: m\noutput_dir: o\n")
    monkeypatch.setattr(
        "sys.argv", ["lab", "sft", "--config", str(p), "--resume", "--scratch"]
    )
    cli.main()
    assert got == {"resume": True, "scratch": True}


def test_cli_turns_resume_error_into_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from post_training import cli

    def fake(cfg: Any, resume: bool = False, scratch: bool = False) -> dict[str, Any]:
        raise sft.ResumeError("pass --resume")

    monkeypatch.setattr(sft, "run_sft", fake)
    p = tmp_path / "c.yaml"
    p.write_text("model_name: m\noutput_dir: o\n")
    monkeypatch.setattr("sys.argv", ["lab", "sft", "--config", str(p)])
    with pytest.raises(SystemExit, match="pass --resume"):
        cli.main()
