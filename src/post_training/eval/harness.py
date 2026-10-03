"""Eval harness: score each system on the same held-out prompts.

For every system (the base model, or the base model plus a LoRA adapter) it
computes the perplexity of the reference answers, generates one greedy answer
per prompt, and scores the answers with ROUGE-L and the rubric. Results go to
``results.json`` and ``results.md`` in ``output_dir``.

Generations are cached in ``generations/<system>.jsonl``. The first line is a
header holding the cache key: system, adapter path, a hash of the adapter's
files, model, prompt-set hash and generation settings. Each later line is one
generated row, appended and flushed as soon as its batch finishes, so a crash
loses at most one batch. On a re-run the rows are reused only when the header
matches the new key exactly; a partial last line is dropped, and generation
resumes at the next batch boundary so batches keep their composition.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from post_training.config import EvalConfig, SystemSettings, to_dict
from post_training.data.finance import Row, prompt_key, prompt_messages
from post_training.eval.metrics import (
    ROUGE_TYPE,
    ROUGE_USE_STEMMER,
    chat_ids,
    perplexity,
    reference_nll,
    rouge_l,
    rubric_rates,
)
from post_training.eval.rubric import RUBRIC_VERSION, score
from post_training.generate import generate, stop_token_ids
from post_training.run import run_provenance

Loader = Callable[[str, str | None], tuple[Any, Any]]

RULES = ("format", "clean_stop", "repetition", "domain_terms", "ungrounded_numbers")
PERPLEXITY_NOTE = (
    "Perplexity of the reference answers: answer tokens only (prompt masked), "
    "SFT's chat template and prompt/completion split, <|im_end|> included, "
    "truncated at perplexity.max_length like SFT; exp of the token-weighted "
    "mean NLL over all answer tokens."
)
NOTES = (
    "ROUGE-L uses rouge-score's tokenizer, which lowercases and replaces "
    "punctuation with spaces (`$1,200.50` becomes `1 200 50`), with the "
    "Porter stemmer on.",
    "Bootstrap intervals use one shared seed across systems and are not a "
    "paired test: overlapping intervals do not mean there is no difference.",
    "An answer whose stop token lands exactly at max_new_tokens counts as "
    "stopped but fails clean_stop, whose rule is `new_tokens < max_new_tokens`.",
)


def prompt_set_hash(rows: Sequence[Row]) -> str:
    """sha256 over the sorted prompt keys, one per line."""
    keys = sorted(prompt_key(r) for r in rows)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def adapter_hash(adapter: Path) -> str:
    """sha256 over every file in the adapter directory, by sorted name."""
    h = hashlib.sha256()
    for f in sorted(p for p in adapter.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(adapter)).encode() + b"\0")
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tokenizer_hash(tok: Any) -> str:
    """sha256 over the tokenizer's vocabulary and special tokens."""
    return _sha256(
        json.dumps(
            {
                "vocab": sorted(tok.get_vocab().items()),
                "special": sorted(tok.all_special_tokens),
            }
        )
    )


def model_revision(model_name: str) -> str | None:
    """Commit hash of the cached Hub snapshot the base weights load from.

    ``model.config._commit_hash`` is empty when loading from the local cache,
    so read it from the snapshot path instead. None if the model is not cached.
    """
    import huggingface_hub

    path = huggingface_hub.try_to_load_from_cache(model_name, "config.json")
    if not isinstance(path, str):
        return None
    return Path(path).parent.name


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to a temp file beside ``path``, then ``os.replace`` it."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_cache(path: Path, key: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows cached under ``key``, or [] if the file is absent or keyed otherwise.

    A last line without its newline, or any line that is not valid JSON, is
    taken as the point a crash cut the file: it and everything after it are
    dropped.
    """
    if not path.exists():
        return []
    lines = path.read_text().split("\n")
    # A complete file ends with "\n", so the final element is "". Anything
    # else there is a partial line from a crash.
    lines = lines[:-1]
    try:
        header = json.loads(lines[0]) if lines else None
    except ValueError:
        return []
    if not isinstance(header, dict) or header.get("cache_key") != key:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines[1:]:
        try:
            rows.append(json.loads(line))
        except ValueError:
            break
    return rows


def _resume_point(
    cached: list[dict[str, Any]], rows: Sequence[Row], batch_size: int
) -> int:
    """How many cached rows to keep: a valid prefix, cut to a batch boundary."""
    n = 0
    for i, (c, r) in enumerate(zip(cached, rows, strict=False)):
        if c.get("index") != i or c.get("prompt_key") != prompt_key(r):
            break
        n = i + 1
    return n if n == len(rows) else n - n % batch_size


def _append(f: Any, records: list[dict[str, Any]]) -> None:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
    f.flush()
    os.fsync(f.fileno())


def _generations(
    model: Any,
    tok: Any,
    rows: Sequence[Row],
    path: Path,
    key: dict[str, Any],
    name: str,
) -> tuple[list[dict[str, Any]], int]:
    """All generations for ``rows``, reusing the cache; also the count reused."""
    gen = key["generation"]
    bs = gen["batch_size"]
    cached = read_cache(path, key)
    keep = _resume_point(cached, rows, bs)
    done = cached[:keep]
    # Rewrite the file to a clean state: header plus the rows kept.
    write_atomic(
        path,
        "".join(json.dumps(x) + "\n" for x in [{"cache_key": key}, *done]),
    )
    with path.open("a") as f:
        for start in range(keep, len(rows), bs):
            batch = rows[start : start + bs]
            settings: dict[str, Any] = {}
            out = generate(
                model,
                tok,
                [prompt_messages(r) for r in batch],
                max_new_tokens=gen["max_new_tokens"],
                batch_size=bs,
                seed=gen["seed"] + start // bs,
                settings_out=settings,
            )
            if settings != {**gen, "seed": gen["seed"] + start // bs}:
                raise RuntimeError(f"generation settings drifted: {settings}")
            records = [
                {"index": start + j, "prompt_key": prompt_key(r), **o}
                for j, (r, o) in enumerate(zip(batch, out, strict=True))
            ]
            _append(f, records)
            done += records
            print(f"[{name}] generated {len(done)}/{len(rows)}", flush=True)
    return done, keep


def _skip_reason(system: SystemSettings) -> str | None:
    if system.adapter is None:
        return None
    if not (Path(system.adapter) / "adapter_config.json").is_file():
        return f"adapter missing at {system.adapter}"
    return None


def _default_loader(model_name: str, adapter: str | None) -> tuple[Any, Any]:
    from post_training.train.common import load_model_and_tokenizer

    return load_model_and_tokenizer(model_name, adapter)


def _release(model: Any) -> None:
    from post_training.train.common import release_accelerator_cache

    del model
    gc.collect()
    release_accelerator_cache()


def _score_system(
    cfg: EvalConfig,
    system: SystemSettings,
    rows: Sequence[Row],
    p_hash: str,
    gen_dir: Path,
    load: Loader,
    first_ids: dict[str, str],
) -> dict[str, Any]:
    """Score one system. ``first_ids`` holds the first scored system's hash
    of the first row's prompt ids; every later system must match it."""
    out: dict[str, Any] = {"name": system.name, "adapter": system.adapter}
    if reason := _skip_reason(system):
        return {**out, "status": "skipped", "reason": reason}
    t0 = time.time()
    a_hash = adapter_hash(Path(system.adapter)) if system.adapter else None
    model, tok = load(cfg.model_name, system.adapter)
    model.eval()
    try:
        ids = chat_ids(tok, prompt_messages(rows[0]), add_generation_prompt=True)
        ids_hash = _sha256(json.dumps(ids))
        if first_ids.setdefault("sha256", ids_hash) != ids_hash:
            raise ValueError(
                f"{system.name}: prompt ids for the first row differ from the "
                "first system's; the tokenizer or chat template changed"
            )
        identity = {
            "chat_template_sha256": _sha256(str(tok.chat_template)),
            "tokenizer_sha256": tokenizer_hash(tok),
            "model_revision": model_revision(cfg.model_name),
            "dtype": str(next(model.parameters()).dtype),
        }
        key = {
            "system": system.name,
            "adapter": system.adapter,
            "adapter_sha256": a_hash,
            "model_name": cfg.model_name,
            **identity,
            "prompt_set_sha256": p_hash,
            "generation": {
                "do_sample": False,
                "max_new_tokens": cfg.generation.max_new_tokens,
                "batch_size": cfg.generation.batch_size,
                "seed": cfg.generation.seed,
                "stop_token_ids": stop_token_ids(tok),
            },
        }
        nll = [reference_nll(model, tok, r, cfg.perplexity.max_length) for r in rows]
        gens, reused = _generations(
            model, tok, rows, gen_dir / f"{system.name}.jsonl", key, system.name
        )
    finally:
        _release(model)

    max_new = cfg.generation.max_new_tokens
    rubric = [
        score(
            r,
            g["text"],
            stopped=g["stopped"],
            new_tokens=g["new_tokens"],
            max_new_tokens=max_new,
        )
        for r, g in zip(rows, gens, strict=True)
    ]
    n = len(rows)
    return {
        **out,
        "status": "scored",
        "adapter_sha256": a_hash,
        **identity,
        "first_prompt_ids_sha256": ids_hash,
        "generation_settings": key["generation"],
        "generations_file": str(gen_dir / f"{system.name}.jsonl"),
        "rows_from_cache": reused,
        "rows_generated": n - reused,
        "perplexity": perplexity([(x, k) for x, k, _ in nll]),
        "perplexity_answer_tokens": sum(k for _, k, _ in nll),
        "perplexity_truncated_rows": sum(t for _, _, t in nll),
        "rouge_l": rouge_l([g["text"] for g in gens], [r["output"] for r in rows]),
        "rubric": rubric_rates(
            rubric, resamples=cfg.bootstrap.resamples, seed=cfg.bootstrap.seed
        ),
        "mean_new_tokens": sum(g["new_tokens"] for g in gens) / n if n else None,
        "stopped_share": sum(g["stopped"] for g in gens) / n if n else None,
        "wall_seconds": round(time.time() - t0, 1),
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def _rate_cell(r: dict[str, Any]) -> str:
    if not r["n"]:
        return "n/a (0/0)"
    lo, hi = r["ci95"]
    return f"{_pct(r['rate'])} ({r['k']}/{r['n']}) [{_pct(lo)}, {_pct(hi)}]"


def render_markdown(results: dict[str, Any]) -> str:
    cols = [
        "System",
        "Perplexity",
        "ROUGE-L F1",
        *RULES,
        "Overall",
        "Mean new tokens",
        "Stopped %",
    ]
    lines = [
        "# Eval results",
        "",
        f"- Prompts: {results['prompt_set']['n']} from "
        f"`{results['prompt_set']['eval_rows']}`, sha256 "
        f"`{results['prompt_set']['sha256']}`",
        f"- Generation: `{json.dumps(results['generation'])}`",
        f"- Commit: `{results['provenance']['commit']}`"
        + (" (scratch)" if results["provenance"]["scratch"] else ""),
        f"- Rubric version {results['rubric_version']}. Rubric cells: pass % "
        "(k/n over the rows the rule applies to) [95% percentile bootstrap, "
        f"{results['bootstrap']['resamples']} resamples, seed "
        f"{results['bootstrap']['seed']}].",
        f"- {PERPLEXITY_NOTE}",
        f"- ROUGE-L: mean F1 against the reference, rouge-score, stemmer "
        f"{'on' if ROUGE_USE_STEMMER else 'off'}.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for s in results["systems"]:
        if s["status"] != "scored":
            cells = [f"skipped: {s['reason']}"] + [""] * (len(cols) - 2)
        else:
            cells = [
                "n/a" if s["perplexity"] is None else f"{s['perplexity']:.3f}",
                "n/a" if s["rouge_l"] is None else f"{s['rouge_l']:.3f}",
                *(_rate_cell(s["rubric"][r]) for r in RULES),
                _rate_cell(s["rubric"]["overall"]),
                f"{s['mean_new_tokens']:.1f}",
                _pct(s["stopped_share"]),
            ]
        lines.append(f"| {s['name']} | " + " | ".join(cells) + " |")
    lines += ["", "## Notes", "", *(f"- {n}" for n in NOTES)]
    return "\n".join(lines) + "\n"


_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def run_eval(cfg: EvalConfig, *, load: Loader = _default_loader) -> dict[str, Any]:
    """Score every system in ``cfg`` and write results.json and results.md."""
    provenance = run_provenance()
    names = [s.name for s in cfg.systems]
    if len(set(names)) != len(names) or not all(_NAME.match(n) for n in names):
        raise ValueError(f"system names must be unique file-safe words: {names}")
    t0 = time.time()
    eval_rows = Path(cfg.eval_rows)
    raw = eval_rows.read_bytes()
    rows: list[Row] = json.loads(raw)[: cfg.limit]
    p_hash = prompt_set_hash(rows)
    out = Path(cfg.output_dir)
    gen_dir = out / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    first_ids: dict[str, str] = {}
    systems = [
        _score_system(cfg, s, rows, p_hash, gen_dir, load, first_ids)
        for s in cfg.systems
    ]
    results = {
        "config": to_dict(cfg),
        "provenance": provenance,
        "prompt_set": {
            "n": len(rows),
            "sha256": p_hash,
            "eval_rows": cfg.eval_rows,
            "eval_rows_file_sha256": hashlib.sha256(raw).hexdigest(),
        },
        "generation": to_dict(cfg.generation) | {"do_sample": False},
        "perplexity": {"definition": PERPLEXITY_NOTE} | to_dict(cfg.perplexity),
        "rouge": {"type": ROUGE_TYPE, "use_stemmer": ROUGE_USE_STEMMER},
        "bootstrap": to_dict(cfg.bootstrap) | {"interval": "95% percentile"},
        "rubric_version": RUBRIC_VERSION,
        "systems": systems,
        "wall_seconds": round(time.time() - t0, 1),
    }
    # Round-trip so the returned dict is exactly what results.json holds.
    results = json.loads(json.dumps(results))
    write_atomic(out / "results.json", json.dumps(results, indent=1))
    write_atomic(out / "results.md", render_markdown(results))
    return results
