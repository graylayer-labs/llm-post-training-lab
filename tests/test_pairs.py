import json
from pathlib import Path
from typing import Any

import pytest

from post_training.config import PairsConfig, load_config
from post_training.data.finance import FinanceSplits, prompt_key, to_messages
from post_training.eval.rubric import RubricResult, RuleResult
from post_training.train import pairs

PROV = {"commit": "aaa", "dirty": False, "scratch": False, "scratch_reason": None}

# --- the rejection rule -------------------------------------------------------


def _rubric(**failed: bool) -> RubricResult:
    names = ("format", "clean_stop", "repetition", "domain_terms", "ungrounded_numbers")
    return RubricResult(
        version="1",
        rules=tuple(
            RuleResult(n, applies=True, passed=not failed.get(n, False), reason="x")
            for n in names
        ),
    )


def test_format_failure_rejects() -> None:
    assert pairs.classify(_rubric(format=True), stopped=True) == (
        "rejected",
        ["format"],
    )


def test_repetition_failure_rejects() -> None:
    assert pairs.classify(_rubric(repetition=True), stopped=True) == (
        "rejected",
        ["repetition"],
    )


def test_no_stop_with_repetition_names_both_reasons() -> None:
    verdict, reasons = pairs.classify(
        _rubric(repetition=True, clean_stop=True), stopped=False
    )
    assert verdict == "rejected"
    assert reasons == ["repetition", "no_stop_and_repetition"]


@pytest.mark.parametrize(
    "failed", [{"ungrounded_numbers": True}, {"domain_terms": True}]
)
def test_number_and_domain_rules_never_reject(failed: dict[str, bool]) -> None:
    assert pairs.classify(_rubric(**failed), stopped=True) == ("pass", [])


def test_only_hitting_the_limit_is_ambiguous() -> None:
    assert pairs.classify(_rubric(clean_stop=True), stopped=False) == (
        "ambiguous",
        [],
    )


def test_limit_plus_number_failure_is_still_ambiguous() -> None:
    r = _rubric(clean_stop=True, ungrounded_numbers=True, domain_terms=True)
    assert pairs.classify(r, stopped=False) == ("ambiguous", [])


# --- config -------------------------------------------------------------------


def test_pairs_config_defaults_and_yaml(tmp_path: Path) -> None:
    p = tmp_path / "pairs.yaml"
    p.write_text("output_dir: o\nn_prompts: 3\nk: 2\n")
    cfg = load_config(p, PairsConfig)
    assert cfg.sft_run_dir == "outputs/sft"
    assert (cfg.n_prompts, cfg.k, cfg.max_new_tokens) == (3, 2, 384)


def test_short_token_budget_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="384"):
        pairs.run_pairs(PairsConfig(output_dir=str(tmp_path), max_new_tokens=64))


def test_short_token_budget_is_refused_when_the_config_loads(tmp_path: Path) -> None:
    p = tmp_path / "pairs.yaml"
    p.write_text("output_dir: o\nmax_new_tokens: 64\n")
    with pytest.raises(ValueError, match="384"):
        load_config(p, PairsConfig)


# --- end to end with stubs ----------------------------------------------------


def _row(i: int, finance: bool = False) -> dict[str, Any]:
    q = f"What is a dividend number {i}?" if finance else f"Name colour {i}."
    return {"instruction": q, "input": "", "output": f" Reference answer {i}. "}


TRAIN = [_row(i) for i in range(6)]
EVAL = [_row(100 + i) for i in range(2)]

# Per (prompt index, sample index): text, stopped, new_tokens.
GOOD = ("A clean short answer.", True, 5)
REPEAT = ("the cat sat down " * 6, True, 24)
NOSTOP_REPEAT = ("the cat sat down " * 6, False, 384)
LIMIT_ONLY = ("A long but varied answer that ran out of room.", False, 384)
LEAK = ("Answer <|im_start|>user hi", True, 9)
OUTCOMES = {
    0: [GOOD, REPEAT],  # pair from sample 1
    1: [LEAK, NOSTOP_REPEAT],  # two failing: pick sample 0 (first)
    2: [GOOD, GOOD],  # no failing sample
    3: [LIMIT_ONLY, GOOD],  # ambiguous skipped, no pair
    4: [NOSTOP_REPEAT, LIMIT_ONLY],  # pair from sample 0
    5: [GOOD, LEAK],  # pair from sample 1
}


class Gen:
    """Stub for post_training.generate.generate, keyed on the user message."""

    def __init__(self, crash_after: int | None = None) -> None:
        self.calls = 0
        self.crash_after = crash_after
        self.prompts_seen: list[str] = []

    def __call__(
        self,
        model: Any,
        tok: Any,
        messages: Any,
        max_new_tokens: int,
        batch_size: int = 8,
        **kw: Any,
    ) -> list[dict[str, Any]]:
        if self.crash_after is not None and self.calls >= self.crash_after:
            raise RuntimeError("simulated crash")
        self.calls += 1
        settings = kw.get("settings_out")
        if settings is not None:
            settings.update(
                do_sample=kw["do_sample"],
                temperature=kw["temperature"],
                top_p=kw["top_p"],
                top_k=kw["top_k"],
                repetition_penalty=1.0,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                seed=kw["seed"],
                stop_token_ids=[1, 2],
            )
        out = []
        seen: dict[str, int] = {}
        for m in messages:
            user = m[-1]["content"]
            self.prompts_seen.append(user)
            i = int(user.rstrip(".?").split()[-1])
            j = seen.get(user, 0)
            seen[user] = j + 1
            text, stopped, n = OUTCOMES[i][j]
            out.append({"text": text, "stopped": stopped, "new_tokens": n})
        return out


class Tok:
    vocab = {"<pad>": 0, "<|im_end|>": 1, "<|endoftext|>": 2}

    def convert_tokens_to_ids(self, t: str) -> int:
        return {"<|im_end|>": 1, "<|endoftext|>": 2}[t]

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    all_special_tokens = ["<|im_end|>", "<|endoftext|>"]
    eos_token_id = 2
    unk_token_id = None


class OtherTok(Tok):
    vocab = {**Tok.vocab, "extra": 3}


def _sft_run(root: Path, eval_rows: list[dict[str, Any]] = EVAL) -> Path:
    run = root / "sft"
    (run / "adapter").mkdir(parents=True)
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
    (run / "eval_rows.json").write_text(json.dumps(eval_rows))
    data = {
        "dataset": "d",
        "train_size": 6,
        "eval_size": 2,
        "seed": 0,
        "min_chars": 40,
        "max_chars": 1500,
    }
    (run / "summary.json").write_text(
        json.dumps({"config": {"model_name": "m", "data": data}})
    )
    return run


def _setup(
    monkeypatch: pytest.MonkeyPatch,
    gen: Gen,
    train: list[dict[str, Any]] = TRAIN,
    tok: type[Tok] = Tok,
    revision: str = "rev1",
) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_splits(dataset: str, **kw: Any) -> FinanceSplits:
        seen["splits"] = (dataset, kw)
        return FinanceSplits(train=list(train), eval=list(EVAL))

    def fake_load(model_name: str, adapter: str | None = None) -> tuple[Any, Any]:
        seen["load"] = (model_name, adapter)
        return object(), tok()

    monkeypatch.setattr(pairs, "load_splits", fake_splits)
    monkeypatch.setattr(pairs, "load_model_and_tokenizer", fake_load)
    monkeypatch.setattr(pairs, "generate", gen)
    monkeypatch.setattr(pairs, "run_provenance", lambda: dict(PROV))
    monkeypatch.setattr(pairs, "release_model", lambda model: None)
    monkeypatch.setattr(pairs, "model_revision", lambda name: revision)
    return seen


def _cfg(tmp_path: Path, run: Path, **kw: Any) -> PairsConfig:
    base: dict[str, Any] = {
        "output_dir": str(tmp_path / "pairs"),
        "sft_run_dir": str(run),
        "n_prompts": 6,
        "k": 2,
        "batch_size": 4,
    }
    return PairsConfig(**{**base, **kw})


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x]


def test_rebuilds_the_sft_runs_split_and_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _setup(monkeypatch, Gen())
    run = _sft_run(tmp_path)
    pairs.run_pairs(_cfg(tmp_path, run))
    dataset, kw = seen["splits"]
    assert dataset == "d"
    assert kw == {
        "train_size": 6,
        "eval_size": 2,
        "seed": 0,
        "min_chars": 40,
        "max_chars": 1500,
    }
    assert seen["load"] == ("m", str(run / "adapter"))


def test_split_mismatch_with_saved_eval_rows_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gen = Gen()
    _setup(monkeypatch, gen)
    run = _sft_run(tmp_path, eval_rows=[_row(100), _row(999)])
    with pytest.raises(pairs.SplitMismatchError, match="eval_rows"):
        pairs.run_pairs(_cfg(tmp_path, run))
    assert gen.calls == 0


def test_pairs_follow_the_rejection_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, Gen())
    run = _sft_run(tmp_path)
    pairs.run_pairs(_cfg(tmp_path, run))
    got = _read_jsonl(tmp_path / "pairs" / "pairs.jsonl")
    assert [p["prompt_index"] for p in got] == [0, 1, 4, 5]
    assert [p["rejected_sample"] for p in got] == [1, 0, 0, 1]
    for p in got:
        fails = set(p["rejected_rubric"]["failed"])
        assert fails & {"format", "repetition"}, p
        assert p["rejected_reasons"]
        assert set(p["rejected_reasons"]) <= {
            "format",
            "repetition",
            "no_stop_and_repetition",
        }


def test_every_rejected_sample_fails_format_or_repetition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, Gen())
    run = _sft_run(tmp_path)
    pairs.run_pairs(_cfg(tmp_path, run))
    samples = _read_jsonl(tmp_path / "pairs" / "samples.jsonl")[1:]
    for s in samples:
        failed = set(s["rubric"]["failed"])
        rejected = s["verdict"] == "rejected"
        rule = bool(failed & {"format", "repetition"}) or (
            not s["stopped"] and "repetition" in failed
        )
        assert rejected == rule, s


def test_chosen_is_the_reference_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, Gen())
    run = _sft_run(tmp_path)
    pairs.run_pairs(_cfg(tmp_path, run))
    by_key = {prompt_key(r): r for r in TRAIN}
    for p in _read_jsonl(tmp_path / "pairs" / "pairs.jsonl"):
        row = by_key[p["prompt_key"]]
        assert p["chosen"] == to_messages(row)[-1]["content"]
        assert p["prompt"] == to_messages(row)[:-1]


def test_no_pair_prompt_is_a_held_out_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, Gen())
    run = _sft_run(tmp_path)
    pairs.run_pairs(_cfg(tmp_path, run))
    held_out = {prompt_key(r) for r in json.loads((run / "eval_rows.json").read_text())}
    got = _read_jsonl(tmp_path / "pairs" / "pairs.jsonl")
    assert got
    assert not {p["prompt_key"] for p in got} & held_out


def test_overlap_between_train_and_eval_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gen = Gen()
    _setup(monkeypatch, gen, train=[*TRAIN[:5], EVAL[0]])
    run = _sft_run(tmp_path)
    with pytest.raises(pairs.SplitMismatchError, match="held-out"):
        pairs.run_pairs(_cfg(tmp_path, run))
    assert gen.calls == 0


def test_manifest_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(monkeypatch, Gen())
    run = _sft_run(tmp_path)
    m = pairs.run_pairs(_cfg(tmp_path, run))
    c = m["counts"]
    assert c["prompts"] == 6
    assert c["samples"] == 12
    assert c["samples_rejected"] == 5
    assert c["rejections_per_rule"] == {
        "format": 2,
        "repetition": 3,
        "no_stop_and_repetition": 2,
    }
    assert c["ambiguous_skipped"] == 2
    assert c["prompts_without_rejected_sample"] == 2
    assert c["prompts_all_samples_pass"] == 1
    assert c["pairs_kept"] == 4
    assert m == json.loads((tmp_path / "pairs" / "manifest.json").read_text())


def test_manifest_records_how_to_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch, Gen())
    run = _sft_run(tmp_path)
    m = pairs.run_pairs(_cfg(tmp_path, run, temperature=0.7, seed=3))
    assert m["adapter"] == str(run / "adapter")
    assert len(m["adapter_sha256"]) == 64
    assert m["rubric_version"] == "1"
    assert m["generation"]["temperature"] == 0.7
    assert m["generation"]["seed"] == 3
    assert m["generation"]["do_sample"] is True
    assert m["generation"]["k"] == 2
    assert m["provenance"]["commit"] == "aaa"
    assert len(m["pairs_sha256"]) == 64
    assert "samples_per_minute" in m["throughput"]


def test_number_and_domain_failures_are_logged_per_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = [_row(i, finance=True) for i in range(6)]
    _setup(monkeypatch, Gen(), train=train)
    run = _sft_run(tmp_path)
    m = pairs.run_pairs(_cfg(tmp_path, run))
    samples = _read_jsonl(tmp_path / "pairs" / "samples.jsonl")[1:]
    assert all("domain_terms" in s["rubric"]["rules"] for s in samples)
    # Finance prompts, answers without finance terms: domain_terms fails often,
    # but it never makes a sample rejected on its own.
    fails = m["counts"]["rule_failures"]
    assert fails["domain_terms"] > 0
    assert m["counts"]["samples_rejected"] == 5


def test_crash_then_rerun_resumes_from_the_saved_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _sft_run(tmp_path)
    _setup(monkeypatch, Gen(crash_after=2))
    with pytest.raises(RuntimeError, match="simulated crash"):
        pairs.run_pairs(_cfg(tmp_path, run))
    samples = tmp_path / "pairs" / "samples.jsonl"
    assert len(_read_jsonl(samples)) == 1 + 8  # header + 2 batches of 4

    gen = Gen()
    _setup(monkeypatch, gen)
    m = pairs.run_pairs(_cfg(tmp_path, run))
    assert gen.calls == 1  # only the last batch
    assert m["counts"]["samples"] == 12
    assert m["throughput"]["samples_reused"] == 8
    rows = _read_jsonl(samples)[1:]
    assert [r["index"] for r in rows] == list(range(12))


def test_partial_last_line_is_dropped_and_regenerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _sft_run(tmp_path)
    _setup(monkeypatch, Gen(crash_after=2))
    with pytest.raises(RuntimeError):
        pairs.run_pairs(_cfg(tmp_path, run))
    samples = tmp_path / "pairs" / "samples.jsonl"
    with samples.open("a") as f:
        f.write('{"index": 8, "prompt_key": "cut sho')
    gen = Gen()
    _setup(monkeypatch, gen)
    pairs.run_pairs(_cfg(tmp_path, run))
    rows = _read_jsonl(samples)[1:]
    assert [r["index"] for r in rows] == list(range(12))


def test_half_batch_is_cut_back_to_a_batch_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _sft_run(tmp_path)
    _setup(monkeypatch, Gen(crash_after=2))
    with pytest.raises(RuntimeError):
        pairs.run_pairs(_cfg(tmp_path, run))
    samples = tmp_path / "pairs" / "samples.jsonl"
    lines = samples.read_text().splitlines(keepends=True)
    samples.write_text("".join(lines[:-2]))  # 6 of 8 rows kept
    gen = Gen()
    _setup(monkeypatch, gen)
    m = pairs.run_pairs(_cfg(tmp_path, run))
    assert gen.calls == 2
    assert m["throughput"]["samples_reused"] == 4


def test_raising_n_prompts_extends_without_resampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _sft_run(tmp_path)
    _setup(monkeypatch, Gen())
    pairs.run_pairs(_cfg(tmp_path, run, n_prompts=4))  # 8 samples, 2 batches
    samples = tmp_path / "pairs" / "samples.jsonl"
    first = samples.read_text().splitlines()

    gen = Gen()
    _setup(monkeypatch, gen)
    m = pairs.run_pairs(_cfg(tmp_path, run, n_prompts=6))
    assert gen.calls == 1  # only the new batch of prompts 4 and 5
    assert m["throughput"]["samples_reused"] == 8
    assert m["counts"]["prompts"] == 6
    again = samples.read_text().splitlines()
    assert again[1:9] == first[1:9]
    assert [json.loads(x)["index"] for x in again[1:]] == list(range(12))


def test_lowering_n_prompts_is_refused_and_keeps_the_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _sft_run(tmp_path)
    _setup(monkeypatch, Gen())
    pairs.run_pairs(_cfg(tmp_path, run, n_prompts=6))
    samples = tmp_path / "pairs" / "samples.jsonl"
    before = samples.read_text()
    with pytest.raises(pairs.CacheKeyError, match="n_prompts"):
        pairs.run_pairs(_cfg(tmp_path, run, n_prompts=4))
    assert samples.read_text() == before


def test_key_records_tokenizer_and_model_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _sft_run(tmp_path)
    _setup(monkeypatch, Gen())
    m = pairs.run_pairs(_cfg(tmp_path, run))
    header = json.loads(
        (tmp_path / "pairs" / "samples.jsonl").read_text().splitlines()[0]
    )
    key = header["cache_key"]
    assert len(key["tokenizer_sha256"]) == 64
    assert key["model_revision"] == "rev1"
    assert m["tokenizer_sha256"] == key["tokenizer_sha256"]
    assert m["model_revision"] == "rev1"


@pytest.mark.parametrize(
    "change", [{"tok": OtherTok}, {"revision": "rev2"}], ids=["tokenizer", "rev"]
)
def test_other_tokenizer_or_revision_does_not_reuse_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    run = _sft_run(tmp_path)
    _setup(monkeypatch, Gen())
    pairs.run_pairs(_cfg(tmp_path, run))
    _setup(monkeypatch, Gen(), **change)
    with pytest.raises(pairs.CacheKeyError):
        pairs.run_pairs(_cfg(tmp_path, run))


def test_existing_samples_from_other_settings_are_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _sft_run(tmp_path)
    _setup(monkeypatch, Gen())
    pairs.run_pairs(_cfg(tmp_path, run))
    samples = tmp_path / "pairs" / "samples.jsonl"
    before = samples.read_text()
    with pytest.raises(pairs.CacheKeyError, match="output_dir"):
        pairs.run_pairs(_cfg(tmp_path, run, temperature=0.5))
    assert samples.read_text() == before


def test_seeds_advance_per_batch_and_are_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeds: list[int] = []
    gen = Gen()

    def spy(*a: Any, **kw: Any) -> list[dict[str, Any]]:
        seeds.append(kw["seed"])
        return gen(*a, **kw)

    _setup(monkeypatch, gen)
    monkeypatch.setattr(pairs, "generate", spy)
    run = _sft_run(tmp_path)
    pairs.run_pairs(_cfg(tmp_path, run, seed=10))
    assert seeds == [10, 11, 12]


def test_each_prompt_is_sampled_k_times_in_one_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gen = Gen()
    _setup(monkeypatch, gen)
    run = _sft_run(tmp_path)
    pairs.run_pairs(_cfg(tmp_path, run, n_prompts=3))
    assert gen.prompts_seen == [
        TRAIN[0]["instruction"],
        TRAIN[0]["instruction"],
        TRAIN[1]["instruction"],
        TRAIN[1]["instruction"],
        TRAIN[2]["instruction"],
        TRAIN[2]["instruction"],
    ]


def test_cli_runs_pairs_from_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from post_training import cli

    got: list[PairsConfig] = []
    monkeypatch.setattr(pairs, "run_pairs", lambda cfg: got.append(cfg) or {})
    p = tmp_path / "c.yaml"
    p.write_text("output_dir: o\nk: 3\n")
    monkeypatch.setattr("sys.argv", ["lab", "pairs", "--config", str(p)])
    cli.main()
    assert got == [PairsConfig(output_dir="o", k=3)]


@pytest.mark.parametrize("name", ["pairs.yaml", "pairs_smoke.yaml"])
def test_shipped_pairs_configs_load(name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / name, PairsConfig)
    assert cfg.max_new_tokens >= 384
    assert cfg.batch_size % cfg.k == 0
