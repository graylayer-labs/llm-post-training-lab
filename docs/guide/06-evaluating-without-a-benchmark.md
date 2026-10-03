# 6. Evaluating without a benchmark

*Last updated 2026-10-03 from issues #3, #23*

## What we set out to do

Answer "is it better?" for a task with no public test set. Score base, SFT
and DPO on the same 200 held-out prompts, in one table, and say how far the
table can be trusted. The harness came in #23; the full run in #3
([eval-results.md][res]).

## The harness (#23)

`lab eval` is config-driven (`configs/eval.yaml`). Its design choices, from
the [#23 handoff][i23]:

- **Same prompts for every system.** The 200 rows come from the SFT run's
  saved `eval_rows.json`, with a prompt-set hash recorded and identical
  prompt ids asserted across systems ([decisions.md][heldout]).
- **Identity hashes.** Each system records its adapter hash, chat template,
  tokenizer, base-model revision and dtype, so a mismatch would show
  ([eval-results.md][res-how]). The DPO system is loaded as the SFT adapter
  merged into base plus the DPO LoRA, and its saved `base_adapter.json` is
  checked, so a wrong base fails instead of scoring an untrained model (see
  chapter 5).
- **Crash-safe cache.** Generations are cached and keyed on those hashes, so a
  changed adapter or template cannot reuse stale answers.
- **Batch size is a setting.** Batched bf16 decoding on MPS flips near-tied
  tokens, so the batch size (16) is recorded and held fixed for every
  system ([decisions.md][batch]).
- **Metrics.** Perplexity of the reference answers (answer tokens only,
  prompt masked as in SFT), ROUGE-L, the rubric's per-rule pass rates with
  95% bootstrap intervals, mean new tokens and stop share.

## What we found

One run: commit `76c9086`, clean, greedy, 384 new tokens, batch 16, one
seed. Nothing came from the cache. Source for the table:
[eval-results.md][res-table] (`outputs/eval/results.md`).

| | base | SFT | DPO |
|---|---|---|---|
| Rubric overall | 35.0% | 73.0% | 89.0% |
| Stopped | 39.5% | 86.5% | 100% |
| Repetition pass | 40.5% | 75.0% | 92.5% |
| Perplexity of references | 8.910 | 6.245 | 6.555 |
| ROUGE-L | 0.153 | 0.297 | 0.296 |
| Mean new tokens | 270.0 | 111.6 | 51.9 |

SFT's main change is stopping, the stop-token fix from chapter 3. DPO
stops every answer and loops far less ([eval-results.md][res-changed]).
DPO fixes 36 prompts and breaks 4 relative to SFT. SFT also fixes 88 and
breaks 12 relative to base, so no change is one-way.

### The length check

Chapter 5 flagged that DPO may only have learned "shorter". The saved
generations can test part of it, with `tools/compare_systems.py` at commit
`b872427` ([eval-results.md][res-short]):

- Where SFT already stopped (173 prompts), DPO is still shorter: 50.4
  tokens against 69.1, and slightly further from the references (ROUGE-L
  0.314 against 0.330).
- 48 DPO answers are under 20 tokens, against 27 for SFT.
- DPO passes 20 of the 27 prompts where SFT never stopped.

So part of DPO's gain is "shorter", not only "no loops". The doc says this
cannot be separated with these pairs; that needs pairs of similar length.

### What the rubric missed

The rubric checks behaviour, not truth. For "simplify `3x + 4xy + y - 2x -
3y`" the reference ends `= 4xy + x - 2y`. DPO answered in 12 tokens, "The
simplified expression is 2x + 2y". That is wrong and passes every rule
(index 50). Another DPO answer to "Name and describe four types of
renewable energy" names four and stops, dropping the descriptions; the
rubric counts it as a fix (index 2). A short loop can also pass, because
`repetition` counts repeats and a short answer has fewer chances to
repeat (index 29) ([eval-results.md][res-short]).

## Decisions and why

- **No model judge.** The owner was offered about 1,200 Claude API calls
  ($1 to $5) and chose to drop the judge, keeping cost inside the
  subscription ([issue #3][i3]; [decisions.md][nojudge]). The `anthropic`
  dependency was removed. Cost: nothing scores correctness or completeness.
- **Batch size 16 for the full run.** At batch size 1 the base model would
  have held a shared GPU for about 85 minutes ([decisions.md][batch]).

## Limits

All from [eval-results.md][res-limits].

- One seed, one run. Intervals are several points wide even on 200 prompts:
  SFT overall [66.5, 79.0], DPO [84.5, 93.0].
- The finance rules apply to only 24 of the 200 prompts, so they say almost
  nothing.
- ROUGE-L is overlap with one reference, and many finance references are
  forum posts.
- Perplexity is of the reference answers, not of the model's own. DPO's
  rise from 6.245 to 6.555 does not show its answers got worse.
- Decoding at batch 16 is not bit-identical to one-at-a-time.
- Wall time (1,562.8 s) was contended by another project's GPU jobs.

## How to reproduce it

Not re-run for this chapter: the full eval took 1,562.8 s of wall time on
the GPU. The documented commands, from [eval-results.md][res-how]:

```bash
uv run lab eval --config configs/eval.yaml
uv run python tools/compare_systems.py outputs/eval/generations \
    --eval-rows outputs/sft/eval_rows.json --out outputs/eval/analysis.json \
    --show 185 2 12 50 29 193
```

Needs the SFT and DPO runs in `outputs/`. Files: `src/post_training/eval/harness.py`,
`metrics.py`, `tools/compare_systems.py`.

## What we would do differently

- Add a correctness check, even a small hand-graded set. This follows from
  the rubric passing a wrong answer (index 50) and having no judge
  ([eval-results.md][res-limits]; [decisions.md][nojudge]).
- Use length-matched pairs in the next DPO run. This follows from the length
  check, which could not separate "no loops" from "shorter"
  ([eval-results.md][res-short]).
- Draw more finance prompts. This follows from the n = 24 finance slice,
  where one row moves a rate by 4.2 points ([eval-results.md][res-limits]).

[i3]: https://github.com/graylayer-labs/llm-post-training-lab/issues/3
[i23]: https://github.com/graylayer-labs/llm-post-training-lab/issues/23
[res]: ../eval-results.md
[res-how]: ../eval-results.md#how-these-numbers-were-made
[res-table]: ../eval-results.md#the-table
[res-changed]: ../eval-results.md#what-each-stage-changed
[res-short]: ../eval-results.md#is-it-just-shorter
[res-limits]: ../eval-results.md#limits
[heldout]: ../decisions.md#read-the-held-out-rows-from-the-sft-runs-saved-file
[batch]: ../decisions.md#treat-the-generation-batch-size-as-a-setting
[nojudge]: ../decisions.md#evaluate-without-a-model-judge
