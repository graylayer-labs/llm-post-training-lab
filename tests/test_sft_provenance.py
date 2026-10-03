import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from post_training.config import SftConfig
from post_training.run import run_provenance
from post_training.train import sft


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


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_summary_provenance_is_captured_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "f.txt").write_text("x")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-m", "init")

    class StubTrainer:
        def __init__(self, **kw: Any) -> None:
            self.model = StubModel()
            self.state = SimpleNamespace(log_history=[])

        def evaluate(self) -> dict[str, float]:
            return {"eval_loss": 1.0}

        def train(self) -> SimpleNamespace:
            (repo / "dirty.txt").write_text("changed during training")
            return SimpleNamespace(training_loss=0.5)

    monkeypatch.setattr("trl.SFTTrainer", StubTrainer)
    monkeypatch.setattr(
        sft, "load_splits", lambda *a, **k: SimpleNamespace(train=[], eval=[])
    )
    monkeypatch.setattr(
        sft, "load_model_and_tokenizer", lambda name: (StubModel(), StubTok())
    )
    monkeypatch.setattr(sft, "build_sft_args", lambda cfg, n: None)
    monkeypatch.setattr(sft, "to_chat_dataset", lambda rows: None)
    monkeypatch.setattr(sft, "build_lora_config", lambda *a: None)
    monkeypatch.setattr(sft, "run_provenance", lambda: run_provenance(repo))

    out = tmp_path / "out"
    sft.run_sft(SftConfig(model_name="m", output_dir=str(out)))

    prov = json.loads((out / "summary.json").read_text())["provenance"]
    assert prov["dirty"] is False
    assert prov["scratch"] is False
