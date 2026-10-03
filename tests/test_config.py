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


def test_load_eval_config_builds_systems(tmp_path: Path) -> None:
    from post_training.config import EvalConfig, GenerationSettings, SystemSettings

    p = tmp_path / "eval.yaml"
    p.write_text(
        "model_name: m\n"
        "output_dir: o\n"
        "eval_rows: outputs/sft/eval_rows.json\n"
        "limit: 10\n"
        "systems:\n"
        "  - {name: base, adapter: null}\n"
        "  - {name: sft, adapter: outputs/sft/adapter}\n"
        "generation: {max_new_tokens: 64}\n"
    )
    cfg = load_config(p, EvalConfig)
    assert cfg.systems == (
        SystemSettings("base", None),
        SystemSettings("sft", "outputs/sft/adapter"),
    )
    assert cfg.limit == 10
    assert cfg.generation == GenerationSettings(max_new_tokens=64)


def test_eval_defaults_are_greedy_unbatched_and_cover_three_systems() -> None:
    from post_training.config import EvalConfig

    cfg = EvalConfig(model_name="m", output_dir="o", eval_rows="r.json")
    assert [s.name for s in cfg.systems] == ["base", "sft", "dpo"]
    assert cfg.systems[0].adapter is None
    assert cfg.generation.max_new_tokens == 384
    assert cfg.generation.batch_size == 1
    assert cfg.bootstrap.resamples == 1000
    assert cfg.perplexity.max_length == 512


def test_shipped_eval_configs_load() -> None:
    from post_training.config import EvalConfig

    root = Path(__file__).resolve().parents[1] / "configs"
    full = load_config(root / "eval.yaml", EvalConfig)
    smoke = load_config(root / "eval_smoke.yaml", EvalConfig)
    assert [s.name for s in full.systems] == ["base", "sft", "dpo"]
    assert full.limit is None and full.generation.max_new_tokens == 384
    assert smoke.output_dir.startswith("outputs/scratch-")
    assert all("outputs/sft/" not in (s.adapter or "") for s in smoke.systems)
