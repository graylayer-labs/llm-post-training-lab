from post_training.config import LoraSettings
from post_training.train.sft import build_lora_config, to_chat_dataset


def test_build_lora_config_uses_settings() -> None:
    cfg = build_lora_config(LoraSettings(r=4, alpha=8, dropout=0.1))
    assert cfg.r == 4
    assert cfg.lora_alpha == 8
    assert cfg.lora_dropout == 0.1
    assert cfg.task_type == "CAUSAL_LM"
    assert "q_proj" in list(cfg.target_modules or [])


def test_to_chat_dataset_splits_prompt_and_completion() -> None:
    rows = [{"instruction": "q", "input": "", "output": "a"}]
    ds = to_chat_dataset(rows)
    assert set(ds.column_names) == {"prompt", "completion"}
    assert [m["role"] for m in ds[0]["prompt"]] == ["system", "user"]
    assert ds[0]["completion"] == [{"role": "assistant", "content": "a"}]


def test_peak_memory_callback_keeps_the_maximum_sample() -> None:
    from post_training.train.common import PeakMemoryCallback

    samples = iter([1.0, 3.0, 2.0])
    cb = PeakMemoryCallback(read_gb=lambda: next(samples))
    for _ in range(3):
        cb.on_step_end(None, None, None)
    assert cb.peak_gb == 3.0


def test_build_lora_config_trains_only_the_given_embedding_rows() -> None:
    cfg = build_lora_config(LoraSettings(), token_ids=[151644, 151645])
    assert cfg.trainable_token_indices == {"embed_tokens": [151644, 151645]}


def test_build_lora_config_without_token_ids_trains_no_embeddings() -> None:
    assert build_lora_config(LoraSettings()).trainable_token_indices is None


def test_sft_args_use_unchunked_loss_and_completion_masking(tmp_path) -> None:
    from post_training.config import SftConfig
    from post_training.train.sft import build_sft_args

    args = build_sft_args(SftConfig(model_name="m", output_dir=str(tmp_path)), 2000)
    # chunked_nll reads lm_head.weight directly and so skips the trainable
    # token rows that PEFT wraps around the tied embedding.
    assert args.loss_type == "nll"
    assert args.completion_only_loss is True
    assert args.warmup_steps == 3  # 3% of 125 optimiser steps


def test_release_cache_callback_frees_the_cache_after_every_step() -> None:
    from post_training.train.common import ReleaseCacheCallback

    calls: list[int] = []
    cb = ReleaseCacheCallback(release=lambda: calls.append(1))
    cb.on_step_end(None, None, None)
    cb.on_step_end(None, None, None)
    assert len(calls) == 2


def _hide_accelerators(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def test_bf16_supported_is_false_on_cpu(monkeypatch) -> None:
    from post_training.train.common import bf16_supported

    _hide_accelerators(monkeypatch)
    assert bf16_supported() is False


def test_bf16_supported_is_true_on_mps(monkeypatch) -> None:
    import torch

    from post_training.train.common import bf16_supported

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert bf16_supported() is True


def test_sft_args_build_without_bf16_on_cpu(monkeypatch, tmp_path) -> None:
    from post_training.config import SftConfig
    from post_training.train.sft import build_sft_args

    _hide_accelerators(monkeypatch)
    args = build_sft_args(SftConfig(model_name="m", output_dir=str(tmp_path)), 2000)
    assert args.bf16 is False
