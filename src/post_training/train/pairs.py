"""Build DPO preference pairs from the SFT model's own samples.

For each training prompt of an SFT run, the SFT model answers ``k`` times
with sampling. Each sample is scored with the rule-based rubric:

- **rejected**: it fails ``format``, or ``repetition``, or it did not stop
  and fails ``repetition``. ``ungrounded_numbers`` and ``domain_terms`` never
  make a sample rejected: the number rule fails 37.5% of finance references
  and fails correct arithmetic, so rejecting on it would teach DPO to avoid
  concrete figures. They are logged per sample and counted.
- **ambiguous**: not rejected, but it failed ``clean_stop`` (it hit the
  token limit). A long, good answer and one that never stops look the same
  here, so the sample is skipped and counted.
- **pass**: everything else.

The pair for a prompt is (reference answer, first rejected sample by sample
index). Taking the first, not the "worst", keeps the choice independent of
how badly a sample fails, so the set is not biased towards the most
degenerate, easiest-to-separate answers. Prompts without a rejected sample
give no pair.

The prompts are the SFT run's own training prompts. The split is rebuilt
from the run's ``summary.json`` and must match its saved ``eval_rows.json``
exactly by prompt key, which proves it is the same split. Samples are
appended to ``samples.jsonl`` batch by batch and a crashed run resumes from
the saved batches, as the eval harness does (``eval/harness.py``).
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from post_training.config import PairsConfig, to_dict
from post_training.data.finance import (
    FinanceSplits,
    Row,
    load_splits,
    prompt_key,
    prompt_messages,
    to_messages,
)
from post_training.eval.harness import (
    adapter_hash,
    model_revision,
    read_cache,
    tokenizer_hash,
    write_atomic,
)
from post_training.eval.rubric import RUBRIC_VERSION, RubricResult, score
from post_training.generate import generate, stop_token_ids
from post_training.run import run_provenance
from post_training.train.common import (
    load_model_and_tokenizer,
    release_accelerator_cache,
)

PAIRS_FORMAT_VERSION = "1"
MIN_NEW_TOKENS = 384
REJECT_REASONS = ("format", "repetition", "no_stop_and_repetition")
RULES = ("format", "clean_stop", "repetition", "domain_terms", "ungrounded_numbers")
REJECTION_RULE = (
    "rejected = fails format, OR fails repetition, OR (did not stop AND fails "
    "repetition); ungrounded_numbers and domain_terms never reject. ambiguous "
    "= not rejected but failed clean_stop (hit the token limit): skipped. The "
    "rejected sample of a pair is the first rejected one by sample index."
)


class SplitMismatchError(RuntimeError):
    """The rebuilt split is not the one the SFT run trained on."""


class CacheKeyError(RuntimeError):
    """samples.jsonl holds samples made with other settings."""


def classify(result: RubricResult, *, stopped: bool) -> tuple[str, list[str]]:
    """Verdict for one sample (rejected / ambiguous / pass) and why rejected."""
    failed = set(result.failed)
    reasons = [r for r in ("format", "repetition") if r in failed]
    if not stopped and "repetition" in failed:
        reasons.append("no_stop_and_repetition")
    if reasons:
        return "rejected", reasons
    if "clean_stop" in failed:
        return "ambiguous", []
    return "pass", []


def prompt_set_hash(rows: Sequence[Row]) -> str:
    """sha256 over the prompt keys in order, one per line."""
    return hashlib.sha256("\n".join(prompt_key(r) for r in rows).encode()).hexdigest()


def check_split(splits: FinanceSplits, eval_rows: Sequence[Row], source: Path) -> None:
    """Raise unless ``splits`` is the split whose eval rows were saved."""
    saved = {prompt_key(r) for r in eval_rows}
    rebuilt = {prompt_key(r) for r in splits.eval}
    if saved != rebuilt:
        raise SplitMismatchError(
            f"rebuilt eval split differs from {source / 'eval_rows.json'}: "
            f"{len(saved - rebuilt)} saved prompts missing, "
            f"{len(rebuilt - saved)} unexpected. The data config or the split "
            "code has changed since that SFT run."
        )
    overlap = saved & {prompt_key(r) for r in splits.train}
    if overlap:
        raise SplitMismatchError(
            f"{len(overlap)} training prompts are held-out prompts too"
        )


def read_header(path: Path) -> dict[str, Any] | None:
    """The header line of a samples file, or None if absent or unreadable."""
    if not path.exists():
        return None
    with path.open() as f:
        first = f.readline()
    if not first.endswith("\n"):
        return None
    try:
        header = json.loads(first)
    except ValueError:
        return None
    return header if isinstance(header, dict) else None


def append_records(f: Any, records: list[dict[str, Any]]) -> None:
    """Append one JSON line per record to an open file, then fsync it."""
    for rec in records:
        f.write(json.dumps(rec) + "\n")
    f.flush()
    os.fsync(f.fileno())


def release_model(model: Any) -> None:
    del model
    gc.collect()
    release_accelerator_cache()


def _resume_point(
    cached: list[dict[str, Any]], expected: Sequence[tuple[int, str]], bs: int
) -> int:
    """How many cached samples to keep: a valid prefix, cut to a batch boundary."""
    n = 0
    for c, (idx, key) in zip(cached, expected, strict=False):
        if c.get("index") != idx or c.get("prompt_key") != key:
            break
        n += 1
    return n if n == len(expected) else n - n % bs


def _score_record(row: Row, out: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    result = score(
        row,
        out["text"],
        stopped=out["stopped"],
        new_tokens=out["new_tokens"],
        max_new_tokens=max_new_tokens,
    )
    verdict, reasons = classify(result, stopped=out["stopped"])
    return {"rubric": result.to_dict(), "verdict": verdict, "reject_reasons": reasons}


def _sample(
    model: Any,
    tok: Any,
    rows: Sequence[Row],
    cfg: PairsConfig,
    path: Path,
    key: dict[str, Any],
    started: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, float]:
    """All samples, reusing saved batches; also the count reused and gen time."""
    gen = key["generation"]
    bs = cfg.batch_size
    plan = [(i, j) for i in range(len(rows)) for j in range(cfg.k)]
    expected = [(s, prompt_key(rows[i])) for s, (i, _) in enumerate(plan)]
    cached = read_cache(path, key)
    keep = _resume_point(cached, expected, bs)
    done = cached[:keep]
    header = {"cache_key": key, "started_provenance": started}
    write_atomic(path, "".join(json.dumps(x) + "\n" for x in [header, *done]))
    seconds = 0.0
    with path.open("a") as f:
        for start in range(keep, len(plan), bs):
            batch = plan[start : start + bs]
            seed = gen["seed"] + start // bs
            settings: dict[str, Any] = {}
            t0 = time.time()
            sampling: dict[str, Any] = (
                {
                    "do_sample": True,
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "top_k": cfg.top_k,
                }
                if cfg.decoding == "sample"
                else {"do_sample": False}
            )
            outs = generate(
                model,
                tok,
                [prompt_messages(rows[i]) for i, _ in batch],
                max_new_tokens=cfg.max_new_tokens,
                batch_size=bs,
                seed=seed,
                settings_out=settings,
                **sampling,
            )
            seconds += time.time() - t0
            if settings != {**gen, "seed": seed}:
                raise RuntimeError(f"generation settings drifted: {settings}")
            records = [
                {
                    "index": start + n,
                    "prompt_index": i,
                    "sample": j,
                    "prompt_key": prompt_key(rows[i]),
                    **o,
                    **_score_record(rows[i], o, cfg.max_new_tokens),
                }
                for n, ((i, j), o) in enumerate(zip(batch, outs, strict=True))
            ]
            append_records(f, records)
            done += records
            print(f"[pairs] sampled {len(done)}/{len(plan)}", flush=True)
    return done, keep, seconds


def build_pairs(
    rows: Sequence[Row], samples: Sequence[dict[str, Any]], k: int
) -> tuple[list[dict[str, Any]], int]:
    """One pair per prompt with a rejected sample; also how many were dropped.

    A pair is dropped only if its rejected text equals the reference.
    """
    out: list[dict[str, Any]] = []
    dropped = 0
    for i, row in enumerate(rows):
        group = samples[i * k : (i + 1) * k]
        rejected = [s for s in group if s["verdict"] == "rejected"]
        if not rejected:
            continue
        r = rejected[0]
        msgs = to_messages(row)
        chosen = msgs[-1]["content"]
        if r["text"].strip() == chosen:
            dropped += 1
            continue
        out.append(
            {
                "index": len(out),
                "prompt_index": i,
                "prompt_key": prompt_key(row),
                "prompt": msgs[:-1],
                "chosen": chosen,
                "rejected": r["text"],
                "rejected_sample": r["sample"],
                "rejected_stopped": r["stopped"],
                "rejected_new_tokens": r["new_tokens"],
                "rejected_reasons": r["reject_reasons"],
                "rejected_failed_rules": r["rubric"]["failed"],
                "rejected_rubric": r["rubric"],
            }
        )
    return out, dropped


def _counts(
    rows: Sequence[Row],
    samples: Sequence[dict[str, Any]],
    k: int,
    pairs: Sequence[dict[str, Any]],
    dropped: int,
) -> dict[str, Any]:
    verdicts = Counter(s["verdict"] for s in samples)
    reasons = Counter(r for s in samples for r in s["reject_reasons"])
    fails = Counter(f for s in samples for f in s["rubric"]["failed"])
    info = Counter(
        f
        for s in samples
        if s["verdict"] != "rejected"
        for f in s["rubric"]["failed"]
        if f in ("domain_terms", "ungrounded_numbers")
    )
    groups = [samples[i * k : (i + 1) * k] for i in range(len(rows))]
    with_rejected = sum(any(s["verdict"] == "rejected" for s in g) for g in groups)
    n = len(samples)
    return {
        "prompts": len(rows),
        "samples": n,
        "samples_rejected": verdicts["rejected"],
        "rejections_per_rule": {r: reasons[r] for r in REJECT_REASONS},
        "ambiguous_skipped": verdicts["ambiguous"],
        "samples_pass": verdicts["pass"],
        "rule_failures": {r: fails[r] for r in RULES},
        "number_and_domain_failures_not_rejected": {
            r: info[r] for r in ("domain_terms", "ungrounded_numbers")
        },
        "prompts_with_rejected_sample": with_rejected,
        "prompts_without_rejected_sample": len(rows) - with_rejected,
        "prompts_all_samples_pass": sum(
            all(s["verdict"] == "pass" for s in g) for g in groups
        ),
        "pairs_kept": len(pairs),
        "yield": len(pairs) / len(rows) if rows else None,
        "rejected_share_by_rule": {
            r: reasons[r] / n if n else None for r in REJECT_REASONS
        },
        "no_stop_share": sum(not s["stopped"] for s in samples) / n if n else None,
        "pairs_dropped_rejected_equals_chosen": dropped,
        "stopped_share": sum(s["stopped"] for s in samples) / n if n else None,
        "mean_new_tokens": sum(s["new_tokens"] for s in samples) / n if n else None,
    }


def run_pairs(cfg: PairsConfig) -> dict[str, Any]:
    """Sample, score and pair; write samples.jsonl, pairs.jsonl, manifest.json."""
    if cfg.max_new_tokens < MIN_NEW_TOKENS:
        raise ValueError(
            f"max_new_tokens={cfg.max_new_tokens}: need at least {MIN_NEW_TOKENS} "
            "so that an answer that never stops is told apart from a long one"
        )
    provenance = run_provenance()
    run = Path(cfg.sft_run_dir)
    sft_cfg = json.loads((run / "summary.json").read_text())["config"]
    model_name = sft_cfg["model_name"]
    data = dict(sft_cfg["data"])
    dataset = data.pop("dataset")
    splits = load_splits(dataset, **data)
    eval_rows = json.loads((run / "eval_rows.json").read_text())
    check_split(splits, eval_rows, run)
    rows = splits.train[: cfg.n_prompts]
    if len(rows) < cfg.n_prompts:
        raise ValueError(f"only {len(rows)} training prompts, asked {cfg.n_prompts}")

    adapter = run / "adapter"
    a_hash = adapter_hash(adapter)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples_path = out / "samples.jsonl"

    model, tok = load_model_and_tokenizer(model_name, str(adapter))
    try:
        if hasattr(model, "eval"):
            model.eval()
        # Exactly what generate() reports back, so drift is caught.
        generation: dict[str, Any] = (
            {
                "do_sample": True,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "repetition_penalty": 1.0,
            }
            if cfg.decoding == "sample"
            else {"do_sample": False}
        )
        generation |= {
            "max_new_tokens": cfg.max_new_tokens,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
            "stop_token_ids": stop_token_ids(tok),
        }
        # The key binds the whole ordered training split, not the first
        # n_prompts of it: prompts are taken in split order, so raising
        # n_prompts keeps every saved sample and samples only the new prompts.
        key = {
            "format_version": PAIRS_FORMAT_VERSION,
            "sft_run_dir": str(run),
            "model_name": model_name,
            "model_revision": model_revision(model_name),
            "adapter": str(adapter),
            "adapter_sha256": a_hash,
            "tokenizer_sha256": tokenizer_hash(tok),
            "train_prompts_sha256": prompt_set_hash(splits.train),
            "k": cfg.k,
            "rubric_version": RUBRIC_VERSION,
            "generation": generation,
        }
        header = read_header(samples_path)
        if samples_path.exists() and (header is None or header.get("cache_key") != key):
            raise CacheKeyError(
                f"{samples_path} holds samples made with other settings or is "
                "unreadable; use a new output_dir or move that file away. "
                "Nothing was changed."
            )
        saved = len(read_cache(samples_path, key))
        if saved > cfg.n_prompts * cfg.k:
            raise CacheKeyError(
                f"{samples_path} holds {saved} samples, more than n_prompts * k "
                f"= {cfg.n_prompts * cfg.k}; lowering n_prompts would drop them. "
                "Use a new output_dir. Nothing was changed."
            )
        started = (header or {}).get("started_provenance") or provenance
        samples, reused, seconds = _sample(
            model, tok, rows, cfg, samples_path, key, started
        )
    finally:
        release_model(model)

    pairs, dropped = build_pairs(rows, samples, cfg.k)
    held_out = {prompt_key(r) for r in eval_rows}
    leaked = [p["prompt_key"] for p in pairs if p["prompt_key"] in held_out]
    if leaked:
        raise SplitMismatchError(f"{len(leaked)} pair prompts are held-out prompts")
    pairs_text = "".join(json.dumps(p) + "\n" for p in pairs)
    write_atomic(out / "pairs.jsonl", pairs_text)

    generated = len(samples) - reused
    new_tokens = sum(s["new_tokens"] for s in samples[reused:])
    manifest = {
        "format_version": PAIRS_FORMAT_VERSION,
        "config": to_dict(cfg),
        "sft_run_dir": str(run),
        "model_name": model_name,
        "adapter": str(adapter),
        "adapter_sha256": a_hash,
        "data": sft_cfg["data"],
        "split_check": {
            "eval_rows_match": True,
            "eval_prompts": len(held_out),
            "pair_prompts_in_eval": 0,
        },
        "prompt_set_sha256": prompt_set_hash(rows),
        "train_prompts_sha256": key["train_prompts_sha256"],
        "model_revision": key["model_revision"],
        "tokenizer_sha256": key["tokenizer_sha256"],
        "rubric_version": RUBRIC_VERSION,
        "rejection_rule": REJECTION_RULE,
        "generation": {
            "decoding": cfg.decoding,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            **generation,
            "k": cfg.k,
        },
        "counts": _counts(rows, samples, cfg.k, pairs, dropped),
        "throughput": {
            "samples_generated": generated,
            "samples_reused": reused,
            "generation_seconds": round(seconds, 1),
            "samples_per_minute": (
                round(60 * generated / seconds, 2) if seconds > 0 else None
            ),
            "new_tokens_generated": new_tokens,
            "tokens_per_second": (
                round(new_tokens / seconds, 1) if seconds > 0 else None
            ),
        },
        "files": {"pairs": "pairs.jsonl", "samples": "samples.jsonl"},
        "pairs_sha256": hashlib.sha256(pairs_text.encode()).hexdigest(),
        "provenance": provenance,
        "sampling_started_provenance": started,
    }
    manifest = json.loads(json.dumps(manifest))
    write_atomic(out / "manifest.json", json.dumps(manifest, indent=1))
    return manifest
