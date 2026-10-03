"""DPO: the reference is the SFT model, pairs load as built, runs resume.

The model tests use a tiny random Qwen2 (tied embeddings, like Qwen2.5-0.5B)
and a word-level tokenizer built in memory, so they run offline on CPU.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
    TrainerCallback,
)

from post_training.config import (
    DpoConfig,
    DpoTrainSettings,
    LoraSettings,
    load_config,
    to_dict,
)
from post_training.eval.harness import adapter_hash
from post_training.train import dpo

PROV = {"commit": "aaa", "dirty": False, "scratch": False, "scratch_reason": None}
WORDS = [
    "system",
    "user",
    "assistant",
    "you",
    "are",
    "a",
    "concise",
    "finance",
    "helper",
    "what",
    "is",
    "how",
    "do",
    "i",
    "the",
    "cat",
    "sat",
    "down",
    "stock",
    "bond",
    "rate",
    "pay",
    "tax",
    "save",
    "money",
    "because",
    "it",
    "yes",
    "no",
    "answer",
    "question",
    "good",
    "bad",
    "long",
    "short",
]
SPECIAL = ["<pad>", "<|im_start|>", "<|im_end|>", "<|endoftext|>", "[UNK]"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}"
    "<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _tokenizer() -> Any:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {w: i for i, w in enumerate([*SPECIAL, *WORDS])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="[UNK]",
        pad_token="<pad>",
        eos_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tok.chat_template = CHAT_TEMPLATE
    return tok


def _base(vocab: int) -> Any:
    cfg = Qwen2Config(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg)


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A saved tiny base model and a saved, clearly non-zero SFT adapter."""
    root = tmp_path_factory.mktemp("models")
    tok = _tokenizer()
    base = _base(len(tok))
    base.save_pretrained(root / "base")
    tok.save_pretrained(root / "base")
    ids = tok.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])
    sft = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=TARGETS,
            init_lora_weights=False,
            trainable_token_indices={"embed_tokens": ids},
            task_type="CAUSAL_LM",
        ),
    )
    torch.manual_seed(1)
    with torch.no_grad():
        for n, p in sft.named_parameters():
            if "trainable_tokens_delta" in n:
                p.normal_(0, 2.0)
    sft.save_pretrained(str(root / "sft"))
    tok.save_pretrained(root / "sft")
    return {"base": str(root / "base"), "sft": str(root / "sft"), "tok": tok}


_REAL_LOAD = dpo.load_sft_merged


def _load_cpu(model_name: str, adapter: str, **_: Any) -> tuple[Any, Any]:
    return _REAL_LOAD(model_name, adapter, dtype=torch.float32, device="cpu")


def _sft_peft(models: dict[str, Any]) -> Any:
    base = AutoModelForCausalLM.from_pretrained(models["base"], dtype=torch.float32)
    return PeftModel.from_pretrained(base, models["sft"]).eval()


def _plain_base(models: dict[str, Any]) -> Any:
    return AutoModelForCausalLM.from_pretrained(
        models["base"], dtype=torch.float32
    ).eval()


PROMPT = [
    {"role": "system", "content": "you are a concise finance helper"},
    {"role": "user", "content": "what is a bond"},
]


def _pair(i: int, stopped: bool = True) -> dict[str, Any]:
    return {
        "index": i,
        "prompt_index": i,
        "prompt_key": f"what is a bond {i}",
        "prompt": PROMPT,
        "chosen": "a bond is a loan because it pay rate",
        "rejected": "the cat sat down the cat sat down the cat sat down",
        "rejected_sample": 0,
        "rejected_stopped": stopped,
        "rejected_new_tokens": 12 if stopped else 384,
        "rejected_reasons": ["repetition"],
        "rejected_failed_rules": ["repetition"],
        "rejected_rubric": {"failed": ["repetition"]},
    }


# --- pure helpers -------------------------------------------------------------


def test_split_pairs_is_seeded_disjoint_and_complete() -> None:
    ps = [_pair(i) for i in range(20)]
    train, held = dpo.split_pairs(ps, fraction=0.1, seed=0)
    assert len(held) == 2
    assert len(train) == 18
    assert {p["index"] for p in train} | {p["index"] for p in held} == set(range(20))
    assert not {p["index"] for p in train} & {p["index"] for p in held}
    again, held2 = dpo.split_pairs(ps, fraction=0.1, seed=0)
    assert held2 == held and again == train
    _, other = dpo.split_pairs(ps, fraction=0.1, seed=1)
    assert other != held


def test_strip_end_of_turn_only_for_unstopped_rejected() -> None:
    end = 2
    ex = {"rejected_ids": [5, 6, end, 9], "rejected_stopped": False}
    assert dpo.strip_unstopped_end(ex, end)["rejected_ids"] == [5, 6]
    ex = {"rejected_ids": [5, 6, end, 9], "rejected_stopped": True}
    assert dpo.strip_unstopped_end(ex, end)["rejected_ids"] == [5, 6, end, 9]


def test_change_needs_two_points() -> None:
    assert dpo._change([{"logps/chosen": -1.0}], "logps/chosen") is None
    assert dpo._change([], "logps/chosen") is None
    two = [{"logps/chosen": -1.0}, {"logps/chosen": -3.0}]
    assert dpo._change(two, "logps/chosen") == {
        "first": -1.0,
        "last": -3.0,
        "delta": -2.0,
    }


def test_dpo_train_defaults() -> None:
    t = DpoTrainSettings()
    assert (t.epochs, t.logging_steps, t.save_steps) == (3, 1, None)


def _plan(n: int, **train: Any) -> dict[str, int]:
    base: dict[str, Any] = {"batch_size": 2, "grad_accum": 8, "epochs": 3}
    cfg = DpoConfig(output_dir="o", train=DpoTrainSettings(**{**base, **train}))
    return dpo.plan_steps(cfg, n)


def test_plan_counts_optimizer_steps_like_the_trainer() -> None:
    # 100 pairs / batch 2 = 50 micro-batches; ceil(50 / 8) = 7 per epoch.
    assert _plan(100)["total_steps"] == 21
    assert _plan(100, epochs=1)["total_steps"] == 7


def test_plan_saves_about_five_times_by_default() -> None:
    assert _plan(100)["save_steps"] == 4  # round(21 / 5)
    assert _plan(4, batch_size=1, grad_accum=1, epochs=1)["save_steps"] == 1


def test_plan_caps_save_steps_so_two_checkpoints_happen() -> None:
    assert _plan(100, save_steps=50)["save_steps"] == 10  # 21 // 2
    assert _plan(100, save_steps=3)["save_steps"] == 3


def test_plan_warms_up_at_least_one_step_on_ten_or_more() -> None:
    assert _plan(100, warmup_ratio=0.01)["warmup_steps"] == 1
    assert _plan(100, warmup_ratio=0.1)["warmup_steps"] == 2
    small = _plan(8, batch_size=1, grad_accum=1, epochs=1, warmup_ratio=0.01)
    assert small["warmup_steps"] == 0


def test_dpo_args_use_the_plan(tmp_path: Path) -> None:
    cfg = DpoConfig(output_dir=str(tmp_path))
    args = dpo.build_dpo_args(cfg, 100)
    plan = dpo.plan_steps(cfg, 100)
    assert args.save_steps == plan["save_steps"]
    assert args.warmup_steps == plan["warmup_steps"]
    assert args.logging_steps == 1


def test_truncation_counts() -> None:
    rows = [
        {"prompt_ids": [1] * 5, "chosen_ids": [2] * 5, "rejected_ids": [3] * 2},
        {"prompt_ids": [1] * 5, "chosen_ids": [2] * 2, "rejected_ids": [3] * 9},
        {"prompt_ids": [1] * 3, "chosen_ids": [2] * 2, "rejected_ids": [3] * 2},
    ]
    assert dpo.truncation_counts(rows, max_length=8, n_input=4) == {
        "pairs": 3,
        "chosen_truncated": 1,
        "rejected_truncated": 1,
        "dropped_prompt_too_long": 1,
    }


def test_reference_other_than_sft_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sft"):
        dpo.run_dpo(DpoConfig(output_dir=str(tmp_path), reference="base"))


@pytest.mark.parametrize("name", ["dpo.yaml", "dpo_smoke.yaml"])
def test_shipped_dpo_configs_load(name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / name, DpoConfig)
    assert cfg.reference == "sft"
    assert cfg.lora.trainable_tokens == ()
    assert 0 < cfg.train.eval_fraction < 1
    assert cfg.train.logging_steps == 1
    if name == "dpo.yaml":
        assert cfg.train.epochs == 3
        assert cfg.train.save_steps is None  # worked out from the pair count


def test_dpo_args_checkpoint_under_output_dir(tmp_path: Path) -> None:
    cfg = DpoConfig(
        output_dir=str(tmp_path),
        train=DpoTrainSettings(beta=0.2, save_steps=7, save_total_limit=3),
    )
    args = dpo.build_dpo_args(cfg, 100)
    assert Path(args.output_dir) == tmp_path / "checkpoints"
    assert args.save_strategy == "steps"
    assert (args.save_steps, args.save_total_limit, args.beta) == (7, 3, 0.2)
    assert args.precompute_ref_log_probs is False


# --- the reference model ------------------------------------------------------


def test_merged_model_is_the_sft_model(models: dict[str, Any]) -> None:
    merged, _ = _load_cpu(models["base"], models["sft"])
    ids = torch.tensor([[1, 7, 8, 2, 1, 9, 10, 2]])
    with torch.no_grad():
        m = merged.eval()(input_ids=ids).logits
        s = _sft_peft(models)(input_ids=ids).logits
        b = _plain_base(models)(input_ids=ids).logits
    assert torch.allclose(m, s, atol=1e-4)
    assert not torch.allclose(m, b, atol=1e-2)


def _seq_logps(model: Any, batch: dict[str, Any]) -> torch.Tensor:
    with torch.no_grad():
        logits = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits[:, :-1]
    lp = logits.log_softmax(-1).gather(-1, batch["input_ids"][:, 1:, None])
    return (lp.squeeze(-1) * batch["completion_mask"][:, 1:]).sum(-1)


def _trainer(models: dict[str, Any], tmp_path: Path, ps: list[dict[str, Any]]) -> Any:
    model, tok = _load_cpu(models["base"], models["sft"])
    cfg = DpoConfig(
        output_dir=str(tmp_path),
        lora=LoraSettings(r=4, alpha=8, trainable_tokens=()),
        train=DpoTrainSettings(batch_size=4, grad_accum=1, max_length=64),
    )
    args = dpo.build_dpo_args(cfg, len(ps), use_cpu=True, bf16=False)
    return dpo.make_trainer(cfg, model, tok, args, ps, ps[:1])


def test_trainer_reference_is_sft_not_base(
    models: dict[str, Any], tmp_path: Path
) -> None:
    ps = [_pair(0), _pair(1, stopped=False)]
    trainer = _trainer(models, tmp_path, ps)
    assert trainer.ref_model is None  # reference = the policy with LoRA off
    assert "ref" not in trainer.model.peft_config
    batch = next(iter(trainer.get_train_dataloader()))
    ref_c, ref_r = trainer.compute_ref_log_probs(trainer.model, batch)
    ref = torch.cat([ref_c, ref_r])
    sft = _seq_logps(_sft_peft(models), batch)
    base = _seq_logps(_plain_base(models), batch)
    assert torch.allclose(ref, sft, atol=1e-3), (ref, sft)
    assert not torch.allclose(ref, base, atol=1e-1), (ref, base)

    # Training moves the policy, never the reference.
    with torch.no_grad():
        for n, p in trainer.model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.5)
    ref2 = torch.cat(trainer.compute_ref_log_probs(trainer.model, batch))
    policy = _seq_logps(trainer.model, batch)
    assert torch.allclose(ref2, ref, atol=1e-5)
    assert not torch.allclose(policy, ref, atol=1e-2)


def test_unstopped_rejected_gets_no_end_of_turn_token(
    models: dict[str, Any], tmp_path: Path
) -> None:
    ps = [_pair(0), _pair(1, stopped=False)]
    trainer = _trainer(models, tmp_path, ps)
    end = models["tok"].convert_tokens_to_ids("<|im_end|>")
    ds = trainer.train_dataset
    assert end in ds[0]["chosen_ids"] and end in ds[1]["chosen_ids"]
    assert end in ds[0]["rejected_ids"]
    assert end not in ds[1]["rejected_ids"]
    assert ds[1]["rejected_ids"]  # the sampled text itself is kept


# --- end to end on the tiny model ---------------------------------------------


def _write_inputs(
    root: Path, models: dict[str, Any], n: int = 8, a_hash: str | None = None
) -> tuple[Path, Path]:
    run = root / "sft_run"
    run.mkdir()
    (run / "adapter").symlink_to(models["sft"])
    (run / "summary.json").write_text(
        json.dumps({"config": {"model_name": models["base"]}})
    )
    pdir = root / "pairs"
    pdir.mkdir()
    ps = [_pair(i, stopped=i % 2 == 0) for i in range(n)]
    text = "".join(json.dumps(p) + "\n" for p in ps)
    (pdir / "pairs.jsonl").write_text(text)
    manifest = {
        "sft_run_dir": str(run),
        "adapter_sha256": a_hash or adapter_hash(run / "adapter"),
        "pairs_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "counts": {"pairs_kept": n},
    }
    (pdir / "manifest.json").write_text(json.dumps(manifest))
    return run, pdir


def _cfg(root: Path, run: Path, pdir: Path, **train: Any) -> DpoConfig:
    t: dict[str, Any] = {
        "epochs": 1,
        "batch_size": 1,
        "grad_accum": 1,
        "max_length": 64,
        "logging_steps": 1,
        "save_steps": 2,
        "eval_fraction": 0.25,
        "learning_rate": 1e-3,
        **train,
    }
    return DpoConfig(
        output_dir=str(root / "out"),
        sft_run_dir=str(run),
        pairs_dir=str(pdir),
        lora=LoraSettings(r=4, alpha=8, trainable_tokens=()),
        train=DpoTrainSettings(**t),
    )


@pytest.fixture
def cpu_run(monkeypatch: pytest.MonkeyPatch) -> None:
    real_args = dpo.build_dpo_args
    monkeypatch.setattr(
        dpo,
        "build_dpo_args",
        lambda cfg, n: real_args(cfg, n, use_cpu=True, bf16=False),
    )
    monkeypatch.setattr(dpo, "load_sft_merged", _load_cpu)
    monkeypatch.setattr(dpo, "run_provenance", lambda: dict(PROV))


class Crash(TrainerCallback):
    def __init__(self, at: int) -> None:
        self.at = at

    def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        if state.global_step == self.at:
            raise RuntimeError("simulated crash")


def test_tiny_run_records_reference_logps_and_held_out(
    models: dict[str, Any], tmp_path: Path, cpu_run: None
) -> None:
    run, pdir = _write_inputs(tmp_path, models)
    s = dpo.run_dpo(_cfg(tmp_path, run, pdir))
    out = tmp_path / "out"
    assert s["reference"]["model"] == "sft"
    assert s["reference"]["sft_adapter_sha256"] == adapter_hash(run / "adapter")
    assert s["config"]["reference"] == "sft"
    assert s["pairs"]["n_train"] == 6 and s["pairs"]["n_held_out"] == 2
    for k in ("eval_rewards/accuracies", "eval_rewards/margins", "eval_logps/chosen"):
        assert k in s["eval_after"] and k in s["eval_before"]
    # Policy starts as the reference, so the margin starts at zero.
    assert s["eval_before"]["eval_rewards/margins"] == pytest.approx(0, abs=1e-4)
    traj = s["trajectory"]
    assert len(traj) == 6
    assert all("logps/chosen" in t and "logps/rejected" in t for t in traj)
    assert set(s["logps_change"]) == {"chosen", "rejected"}
    assert s["peak_memory_gb"] >= 0 and s["wall_seconds"] > 0
    assert (out / "adapter" / "adapter_config.json").exists()
    assert not (out / "checkpoints").exists()
    saved = json.loads((out / "summary.json").read_text())
    assert saved["reference"]["model"] == "sft"
    assert s["total_steps"] == 6
    assert s["steps_plan"]["save_steps"] == 2
    assert set(s["truncation"]) == {"max_length", "train", "held_out"}
    assert s["truncation"]["train"]["pairs"] == 6


def test_held_out_lengths_and_per_token_logps(
    models: dict[str, Any], tmp_path: Path, cpu_run: None
) -> None:
    run, pdir = _write_inputs(tmp_path, models)
    s = dpo.run_dpo(_cfg(tmp_path, run, pdir))
    for when in ("before", "after"):
        h = s["held_out_lengths"][when]
        for side in ("chosen", "rejected"):
            assert h[f"{side}_tokens_mean"] > 0
            assert h[f"{side}_logp_per_token_mean"] < 0
    before = s["held_out_lengths"]["before"]
    # Same pairs, same model: the summed log-prob matches TRL's eval metric.
    assert before["chosen_logp_sum_mean"] == pytest.approx(
        s["eval_before"]["eval_logps/chosen"], abs=1e-2
    )
    # Rejected samples in _pair are longer than chosen ones.
    assert before["rejected_tokens_mean"] > before["chosen_tokens_mean"]


PROBE = torch.tensor([[1, 7, 8, 2, 1, 9, 10, 23, 24, 2]])


class CaptureLogits(TrainerCallback):
    """Logits of the in-memory policy at the end of training."""

    logits: torch.Tensor | None = None

    def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        model = kw["model"]
        model.eval()
        with torch.no_grad():
            self.logits = model(input_ids=PROBE.to(model.device)).logits.cpu()


def test_run_records_which_base_adapter_the_lora_sits_on(
    models: dict[str, Any], tmp_path: Path, cpu_run: None
) -> None:
    run, pdir = _write_inputs(tmp_path, models)
    dpo.run_dpo(_cfg(tmp_path, run, pdir))
    meta = json.loads((tmp_path / "out" / "adapter" / "base_adapter.json").read_text())
    assert meta == {
        "path": str(run / "adapter"),
        "sha256": adapter_hash(run / "adapter"),
    }


def test_harness_loads_the_policy_that_was_trained(
    models: dict[str, Any], tmp_path: Path, cpu_run: None
) -> None:
    from post_training.eval import harness

    run, pdir = _write_inputs(tmp_path, models)
    cap = CaptureLogits()
    dpo.run_dpo(_cfg(tmp_path, run, pdir), extra_callbacks=[cap])
    assert cap.logits is not None
    out = str(tmp_path / "out" / "adapter")
    model, _ = harness.load_stacked(
        models["base"], str(run / "adapter"), out, dtype=torch.float32, device="cpu"
    )
    with torch.no_grad():
        got = model.eval()(input_ids=PROBE).logits
        plain = PeftModel.from_pretrained(_plain_base(models), out).eval()
        on_base = plain(input_ids=PROBE).logits
    assert torch.allclose(got, cap.logits, atol=1e-4)
    assert not torch.allclose(on_base, cap.logits, atol=1e-2)


def test_edited_pairs_file_is_refused(
    models: dict[str, Any], tmp_path: Path, cpu_run: None
) -> None:
    run, pdir = _write_inputs(tmp_path, models)
    with (pdir / "pairs.jsonl").open("a") as f:
        f.write(json.dumps(_pair(99)) + "\n")
    with pytest.raises(ValueError, match="changed"):
        dpo.run_dpo(_cfg(tmp_path, run, pdir))


def test_pairs_from_another_adapter_are_refused(
    models: dict[str, Any], tmp_path: Path, cpu_run: None
) -> None:
    run, pdir = _write_inputs(tmp_path, models, a_hash="0" * 64)
    with pytest.raises(ValueError, match="adapter"):
        dpo.run_dpo(_cfg(tmp_path, run, pdir))


def test_crash_then_resume_finishes_the_run(
    models: dict[str, Any], tmp_path: Path, cpu_run: None
) -> None:
    run, pdir = _write_inputs(tmp_path, models)
    cfg = _cfg(tmp_path, run, pdir)
    with pytest.raises(RuntimeError, match="simulated crash"):
        dpo.run_dpo(cfg, extra_callbacks=[Crash(at=3)])
    out = tmp_path / "out"
    assert (out / "checkpoints" / "checkpoint-2").exists()
    assert (out / "eval_before.json").exists()

    with pytest.raises(dpo.ResumeError, match="--resume"):
        dpo.run_dpo(cfg)
    s = dpo.run_dpo(cfg, resume=True)
    assert s["resumed_from"] == 2
    assert s["global_step"] == 6
    assert s["resumed_segment_only"] is True
    assert "MPS" in s["resume_note"]


def _fake_checkpoint(out: Path, step: int, cfg: DpoConfig) -> None:
    d = out / "checkpoints" / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "trainer_state.json").write_text("{}")
    (out / "provenance.json").write_text(json.dumps(PROV))
    (out / "config.json").write_text(json.dumps(to_dict(cfg)))


@pytest.mark.parametrize(
    ("prov", "match"),
    [
        ({**PROV, "commit": "bbb"}, "commit"),
        ({**PROV, "dirty": True}, "dirty"),
        ({**PROV, "scratch": True, "scratch_reason": "not editable"}, "not editable"),
    ],
)
def test_resume_refuses_other_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prov: dict[str, Any], match: str
) -> None:
    cfg = DpoConfig(output_dir=str(tmp_path / "out"))
    _fake_checkpoint(tmp_path / "out", 4, cfg)
    monkeypatch.setattr(dpo, "run_provenance", lambda: dict(prov))
    with pytest.raises(dpo.ResumeError, match=match):
        dpo.run_dpo(cfg, resume=True)


def test_resume_refuses_another_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = DpoConfig(output_dir=str(tmp_path / "out"))
    _fake_checkpoint(tmp_path / "out", 4, cfg)
    monkeypatch.setattr(dpo, "run_provenance", lambda: dict(PROV))
    other = DpoConfig(
        output_dir=str(tmp_path / "out"), train=DpoTrainSettings(beta=0.5)
    )
    with pytest.raises(dpo.ResumeError, match="config"):
        dpo.run_dpo(other, resume=True)


def test_resume_without_complete_checkpoint_is_an_error(tmp_path: Path) -> None:
    cfg = DpoConfig(output_dir=str(tmp_path / "out"))
    (tmp_path / "out" / "checkpoints" / "checkpoint-3").mkdir(parents=True)
    with pytest.raises(dpo.ResumeError, match="no complete checkpoint"):
        dpo.run_dpo(cfg, resume=True)


def test_check_resume_with_scratch_marks_the_run_scratch(tmp_path: Path) -> None:
    from post_training.train.checkpoint import check_resume

    cfg = DpoConfig(output_dir=str(tmp_path))
    _fake_checkpoint(tmp_path, 4, cfg)
    current = {**PROV, "commit": "bbb"}
    original, missing = check_resume(tmp_path, current, to_dict(cfg), scratch=True)
    assert missing is False
    assert original["commit"] == "aaa"
    assert original["scratch"] is True
    assert "commit is bbb" in original["scratch_reason"]


def test_scratch_resume_of_a_dpo_run_is_marked_scratch(
    models: dict[str, Any],
    tmp_path: Path,
    cpu_run: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, pdir = _write_inputs(tmp_path, models)
    cfg = _cfg(tmp_path, run, pdir)
    with pytest.raises(RuntimeError, match="simulated crash"):
        dpo.run_dpo(cfg, extra_callbacks=[Crash(at=3)])
    monkeypatch.setattr(dpo, "run_provenance", lambda: {**PROV, "commit": "bbb"})
    s = dpo.run_dpo(cfg, resume=True, scratch=True)
    assert s["provenance"]["scratch"] is True
    assert "commit is bbb" in s["provenance"]["scratch_reason"]
    assert s["resume_provenance"]["commit"] == "bbb"


def test_sft_and_dpo_share_one_resume_implementation() -> None:
    from post_training.train import checkpoint, sft

    assert sft.ResumeError is checkpoint.ResumeError
    assert not hasattr(sft, "_check_resume")


# --- CLI ----------------------------------------------------------------------


def test_cli_passes_resume_and_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from post_training import cli

    got: dict[str, Any] = {}

    def fake(cfg: Any, resume: bool = False, scratch: bool = False) -> dict[str, Any]:
        got.update(cfg=cfg, resume=resume, scratch=scratch)
        return {"log_history": []}

    monkeypatch.setattr(dpo, "run_dpo", fake)
    p = tmp_path / "c.yaml"
    p.write_text("output_dir: o\n")
    monkeypatch.setattr(
        "sys.argv", ["lab", "dpo", "--config", str(p), "--resume", "--scratch"]
    )
    cli.main()
    assert got == {"cfg": DpoConfig(output_dir="o"), "resume": True, "scratch": True}


def test_cli_turns_resume_error_into_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from post_training import cli

    def fake(cfg: Any, resume: bool = False, scratch: bool = False) -> dict[str, Any]:
        raise dpo.ResumeError("pass --resume")

    monkeypatch.setattr(dpo, "run_dpo", fake)
    p = tmp_path / "c.yaml"
    p.write_text("output_dir: o\n")
    monkeypatch.setattr("sys.argv", ["lab", "dpo", "--config", str(p)])
    with pytest.raises(SystemExit, match="pass --resume"):
        cli.main()
