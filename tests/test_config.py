from pathlib import Path

from post_training.config import LoraSettings, SftConfig, load_config


def test_load_sft_config_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "sft.yaml"
    p.write_text(
        "model_name: Qwen/Qwen2.5-0.5B\n"
        "output_dir: outputs/sft\n"
        "lora: {r: 8, alpha: 16}\n"
        "data: {train_size: 10, eval_size: 2}\n"
        "train: {epochs: 1, learning_rate: 0.0001}\n"
    )
    cfg = load_config(p, SftConfig)
    assert cfg.model_name == "Qwen/Qwen2.5-0.5B"
    assert cfg.lora == LoraSettings(r=8, alpha=16)
    assert cfg.data.train_size == 10
    assert cfg.train.learning_rate == 1e-4


def test_lora_defaults_target_attention_and_mlp() -> None:
    lora = LoraSettings()
    assert lora.r == 16
    assert lora.alpha == 32
    assert {"q_proj", "v_proj", "down_proj"} <= set(lora.target_modules)


def test_lora_trains_chat_token_embeddings_by_default() -> None:
    assert LoraSettings().trainable_tokens == ("<|im_start|>", "<|im_end|>")


def test_trainable_tokens_from_yaml_become_a_tuple(tmp_path: Path) -> None:
    p = tmp_path / "sft.yaml"
    p.write_text(
        "model_name: m\noutput_dir: o\nlora: {trainable_tokens: ['<|im_end|>']}\n"
    )
    assert load_config(p, SftConfig).lora.trainable_tokens == ("<|im_end|>",)
