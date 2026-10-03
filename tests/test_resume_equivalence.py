"""Resume with the real HF Trainer on a tiny CPU model (no stubs)."""

from pathlib import Path
from typing import Any

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from post_training.train.common import last_complete_checkpoint, train_loss_from_log


class _Stop(TrainerCallback):
    def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        if state.global_step == 6:
            raise KeyboardInterrupt


def _model() -> Any:
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=True,
    )
    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(Qwen2ForCausalLM(cfg), lora)


def _trainer(out: Path, callbacks: list[Any] | None = None) -> Trainer:
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 64, (32, 8), generator=g).tolist()
    args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        learning_rate=1e-2,
        save_strategy="steps",
        save_steps=3,
        save_total_limit=2,
        use_cpu=True,
        report_to=[],
        logging_steps=1,
        seed=0,
        lr_scheduler_type="cosine",
        warmup_steps=1,
    )
    ds = Dataset.from_dict({"input_ids": ids, "labels": ids})
    return Trainer(model=_model(), args=args, train_dataset=ds, callbacks=callbacks)


def test_train_loss_from_log_matches_after_resume(tmp_path: Path) -> None:
    full = _trainer(tmp_path / "full")
    full.train()

    broken = _trainer(tmp_path / "broken", callbacks=[_Stop()])
    with pytest.raises(KeyboardInterrupt):
        broken.train()
    last = last_complete_checkpoint(tmp_path / "broken")
    assert last is not None

    resumed = _trainer(tmp_path / "broken")
    resumed.train(resume_from_checkpoint=str(last))

    expect = train_loss_from_log(full.state.log_history)
    got = train_loss_from_log(resumed.state.log_history)
    assert got == pytest.approx(expect, rel=1e-4)
    assert resumed.state.global_step == full.state.global_step
