import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from post_training.config import EvalConfig, GenerationSettings, SystemSettings
from post_training.data.finance import prompt_key
from post_training.eval import harness
from post_training.eval.harness import (
    prompt_set_hash,
    read_cache,
    run_eval,
    write_atomic,
)

PAD, IM_END, EOT = 0, 1, 2


class HarnessTok:
    """Char-level stub tokenizer that also renders a chat template.

    Each message is [10] + chars + [11]; the generation prompt adds [10].
    """

    padding_side = "right"
    pad_token_id = PAD
    eos_token_id = EOT
    chat_template = "stub-template-v1"
    all_special_tokens = ["<|im_end|>", "<|endoftext|>"]

    def convert_tokens_to_ids(self, t: str) -> int:
        return {"<|im_end|>": IM_END, "<|endoftext|>": EOT}[t]

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": PAD, "<|im_end|>": IM_END, "<|endoftext|>": EOT}

    def _ids(self, messages: Any, add_generation_prompt: bool) -> list[int]:
        ids: list[int] = []
        for m in messages:
            ids += [10, *(12 + ord(c) % 40 for c in m["content"]), 11]
        if add_generation_prompt:
            ids.append(10)
        return ids

    def apply_chat_template(
        self,
        messages: Any,
        tokenize: bool = False,
        return_dict: bool = False,
        add_generation_prompt: bool = False,
    ) -> Any:
        ids = self._ids(messages, add_generation_prompt)
        if tokenize:
            return {"input_ids": ids}
        return ",".join(map(str, ids))

    def __call__(
        self, texts: list[str], return_tensors: str = "pt", padding: bool = True
    ) -> dict[str, torch.Tensor]:
        ids = [[int(x) for x in t.split(",")] for t in texts]
        n = max(map(len, ids))
        rows, masks = [], []
        for x in ids:
            pad = [PAD] * (n - len(x))
            rows.append(pad + x)
            masks.append([0] * len(pad) + [1] * len(x))
        return {"input_ids": torch.tensor(rows), "attention_mask": torch.tensor(masks)}

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        kept = [int(i) for i in ids if not (skip_special_tokens and int(i) < 3)]
        return "".join(chr(97 + i % 26) for i in kept)


def _model() -> Any:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        pad_token_id=PAD,
    )
    return LlamaForCausalLM(cfg).eval()


ROWS = [
    {"instruction": "How do dividends work?", "input": "", "output": "Paid."},
    {"instruction": "Name a colour.", "input": "", "output": "Blue."},
    {"instruction": "What is a bond?", "input": "", "output": "A loan."},
]


class OtherTemplateTok(HarnessTok):
    """Same prompt ids as HarnessTok, but a different chat-template string."""

    chat_template = "stub-template-v2"


class ShiftedTok(HarnessTok):
    """A template that renders different prompt ids."""

    chat_template = "stub-template-shifted"

    def _ids(self, messages: Any, add_generation_prompt: bool) -> list[int]:
        return [3, *super()._ids(messages, add_generation_prompt)]


class Loader:
    def __init__(self, adapter_tok: type[HarnessTok] = HarnessTok) -> None:
        self.calls: list[str | None] = []
        self.adapter_tok = adapter_tok

    def __call__(self, model_name: str, adapter: str | None = None) -> Any:
        self.calls.append(adapter)
        return _model(), (self.adapter_tok() if adapter else HarnessTok())


def _cfg(tmp_path: Path, **kw: Any) -> EvalConfig:
    rows_path = tmp_path / "eval_rows.json"
    rows_path.write_text(json.dumps(ROWS))
    adapter = tmp_path / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-v1")
    cfg = EvalConfig(
        model_name="stub",
        output_dir=str(tmp_path / "out"),
        eval_rows=str(rows_path),
        systems=(
            SystemSettings("base", None),
            SystemSettings("sft", str(adapter)),
            SystemSettings("dpo", str(tmp_path / "missing")),
        ),
        generation=GenerationSettings(max_new_tokens=4, batch_size=1),
    )
    return dataclasses.replace(cfg, **kw)


def _system(results: dict[str, Any], name: str) -> dict[str, Any]:
    return next(s for s in results["systems"] if s["name"] == name)


def test_prompt_set_hash_is_sha256_of_sorted_keys() -> None:
    keys = sorted(prompt_key(r) for r in ROWS)
    expected = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    assert prompt_set_hash(ROWS) == expected
    assert prompt_set_hash(list(reversed(ROWS))) == expected


def test_missing_adapter_is_skipped_not_crashed(tmp_path: Path) -> None:
    loader = Loader()
    results = run_eval(_cfg(tmp_path), load=loader)
    dpo = _system(results, "dpo")
    assert dpo["status"] == "skipped"
    assert "missing" in dpo["reason"]
    assert str(tmp_path / "missing") not in loader.calls
    assert _system(results, "base")["status"] == "scored"
    assert _system(results, "sft")["status"] == "scored"


def test_outputs_and_table_shape(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    results = run_eval(cfg, load=Loader())
    out = Path(cfg.output_dir)
    assert json.loads((out / "results.json").read_text()) == results
    assert results["prompt_set"]["n"] == 3
    assert results["prompt_set"]["sha256"] == prompt_set_hash(ROWS)
    assert "commit" in results["provenance"]
    base = _system(results, "base")
    for k in ("perplexity", "rouge_l", "rubric", "mean_new_tokens", "stopped_share"):
        assert k in base
    assert base["rubric"]["overall"]["n"] == 3
    md = (out / "results.md").read_text()
    assert md.count("| base |") == 1 and "skipped" in md
    lines = (out / "generations" / "base.jsonl").read_text().splitlines()
    assert "cache_key" in json.loads(lines[0])
    assert len(lines) == 1 + len(ROWS)


def test_cache_reused_only_when_every_key_matches(tmp_path: Path) -> None:
    first = run_eval(_cfg(tmp_path), load=Loader())
    assert _system(first, "base")["rows_generated"] == 3

    again = run_eval(_cfg(tmp_path), load=Loader())
    assert _system(again, "base")["rows_from_cache"] == 3
    assert _system(again, "base")["rows_generated"] == 0
    assert _system(again, "base")["rouge_l"] == _system(first, "base")["rouge_l"]

    changed = run_eval(
        _cfg(tmp_path, generation=GenerationSettings(max_new_tokens=3)), load=Loader()
    )
    assert _system(changed, "base")["rows_from_cache"] == 0
    assert _system(changed, "base")["rows_generated"] == 3


def test_changed_adapter_weights_invalidate_the_cache(tmp_path: Path) -> None:
    run_eval(_cfg(tmp_path), load=Loader())
    cfg = _cfg(tmp_path)
    (tmp_path / "adapter" / "adapter_model.safetensors").write_bytes(b"weights-v2")
    res = run_eval(cfg, load=Loader())
    assert _system(res, "sft")["rows_from_cache"] == 0
    assert _system(res, "base")["rows_from_cache"] == 3


def test_key_records_tokenizer_template_revision_and_dtype(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    res = run_eval(cfg, load=Loader())
    base = _system(res, "base")
    path = Path(cfg.output_dir) / "generations" / "base.jsonl"
    key = json.loads(path.read_text().splitlines()[0])["cache_key"]
    expected_template = hashlib.sha256(b"stub-template-v1").hexdigest()
    for record in (key, base):
        assert record["chat_template_sha256"] == expected_template
        assert len(record["tokenizer_sha256"]) == 64
        assert record["dtype"] == "torch.float32"
        assert "model_revision" in record
    assert "first_prompt_ids_sha256" in base


def test_changed_chat_template_invalidates_the_cache(tmp_path: Path) -> None:
    run_eval(_cfg(tmp_path), load=Loader())
    res = run_eval(_cfg(tmp_path), load=Loader(adapter_tok=OtherTemplateTok))
    assert _system(res, "sft")["rows_from_cache"] == 0
    assert _system(res, "base")["rows_from_cache"] == 3


def test_prompt_ids_that_differ_across_systems_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prompt ids"):
        run_eval(_cfg(tmp_path), load=Loader(adapter_tok=ShiftedTok))


def test_results_md_carries_the_metric_notes(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_eval(cfg, load=Loader())
    md = (Path(cfg.output_dir) / "results.md").read_text()
    assert "lowercases" in md and "1 200 50" in md
    assert "not a paired test" in md
    assert "new_tokens < max_new_tokens" in md


def test_resume_after_a_partial_file(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    full = run_eval(cfg, load=Loader())
    path = Path(cfg.output_dir) / "generations" / "base.jsonl"
    complete = path.read_text().splitlines()
    # Simulate a crash after the first row: header + 1 row survive.
    path.write_text("\n".join(complete[:2]) + "\n")

    res = run_eval(cfg, load=Loader())
    base = _system(res, "base")
    assert base["rows_from_cache"] == 1 and base["rows_generated"] == 2
    assert path.read_text().splitlines() == complete
    assert base["rouge_l"] == _system(full, "base")["rouge_l"]


def test_truncated_last_line_is_dropped(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_eval(cfg, load=Loader())
    path = Path(cfg.output_dir) / "generations" / "base.jsonl"
    complete = path.read_text().splitlines()
    header = json.loads(complete[0])["cache_key"]
    # Header, one full row, then half of the next row with no newline.
    path.write_text(complete[0] + "\n" + complete[1] + "\n" + complete[2][:10])

    assert len(read_cache(path, header)) == 1
    res = run_eval(cfg, load=Loader())
    assert _system(res, "base")["rows_from_cache"] == 1
    assert path.read_text().splitlines() == complete


def test_read_cache_with_another_key_returns_nothing(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_eval(cfg, load=Loader())
    path = Path(cfg.output_dir) / "generations" / "base.jsonl"
    assert read_cache(path, {"different": True}) == []
    assert read_cache(tmp_path / "absent.jsonl", {}) == []


def test_resume_with_batches_restarts_at_a_batch_boundary(tmp_path: Path) -> None:
    gen = GenerationSettings(max_new_tokens=4, batch_size=2)
    cfg = _cfg(tmp_path, generation=gen)
    run_eval(cfg, load=Loader())
    path = Path(cfg.output_dir) / "generations" / "base.jsonl"
    complete = path.read_text().splitlines()
    # Only the first row of the first batch of 2 survived.
    path.write_text("\n".join(complete[:2]) + "\n")
    res = run_eval(cfg, load=Loader())
    assert _system(res, "base")["rows_from_cache"] == 0
    assert path.read_text().splitlines() == complete


def test_write_atomic_leaves_old_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "results.json"
    write_atomic(target, "old")

    def boom(*a: Any, **k: Any) -> None:
        raise OSError("crash")

    monkeypatch.setattr(harness.os, "replace", boom)
    with pytest.raises(OSError):
        write_atomic(target, "new")
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["results.json"]


def test_cli_eval_runs_the_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from post_training import cli

    p = tmp_path / "eval.yaml"
    p.write_text("model_name: m\noutput_dir: o\neval_rows: r.json\n")
    seen: list[EvalConfig] = []

    def fake(cfg: EvalConfig) -> dict[str, Any]:
        seen.append(cfg)
        return {"systems": []}

    monkeypatch.setattr(harness, "run_eval", fake)
    monkeypatch.setattr(harness, "render_markdown", lambda r: "TABLE\n")
    monkeypatch.setattr("sys.argv", ["lab", "eval", "--config", str(p)])
    cli.main()
    assert seen[0].eval_rows == "r.json"
    assert "TABLE" in capsys.readouterr().out


def test_model_revision_is_resolved_from_the_hub_cache_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    from post_training.eval import harness

    fake = "/c/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db64/config.json"
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda repo, name: fake
    )
    assert harness.model_revision("Qwen/Qwen2.5-0.5B") == "060db64"


def test_model_revision_is_none_when_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    from post_training.eval import harness

    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda repo, name: None
    )
    assert harness.model_revision("Qwen/Qwen2.5-0.5B") is None
