# 5. Preference optimisation (DPO)

*Last updated 2026-10-03 from issues #2, #26*

## What we set out to do

Run DPO on top of the SFT model, with pairs built from the project's own
data, since real users rarely have preference data ([INTENT.md](../../INTENT.md)).
A pair is a prompt with a chosen and a rejected answer. Chosen is the
dataset's reference answer. Rejected is the SFT model's own answer when the
rubric from chapter 4 says it fails. The code came in #26 (PR #27); the
runs in #2 ([dpo-results.md][dpo]).

## What we found

### Building the pairs

The owner made two decisions, recorded on [issue #2][i2] and in
[decisions.md][reject].

**1. Reject on behaviour only.** A sample is rejected if it fails `format`
or `repetition`. `ungrounded_numbers` and `domain_terms` are logged, never
used to reject. The reason: with the reference removed as a source of
figures, the number rule fails 37.1% of finance reference answers
([issue #2][i2], from `outputs/rubric_references/v1/summary.json`).
Rejecting on it would teach DPO to avoid concrete figures. A sample that
only hit the 384-token limit is ambiguous and skipped.

**2. Reject greedy answers, not sampled ones.** With the clean SFT adapter,
temperature-1.0 sampling gave 0 rejected of 96 samples, so there was nothing
to pair. That was a scratch measurement and was not saved ([issue
#2][i2]). SFT's remaining failure is greedy looping, which is also how the
eval decodes. So each rejected answer is one greedy SFT answer
([decisions.md][greedy]).

**Counts** (from `outputs/pairs/manifest.json`, via [dpo-results.md][dpo-counts]):

- 500 prompts, one greedy answer each: 105 rejected, 393 passed, 2 ambiguous.
- Of the 105: 104 for `repetition` (66 of them never stopped), 1 for `format`.
- 105 pairs kept, 0 of the 200 eval prompts among the pair prompts.

The pairs teach one thing: do not loop. The 66 rejected answers that never
stopped have their `<|im_end|>` removed, because the model never wrote it
and DPO would otherwise push down "stop here" ([decisions.md][dropend]).

### The reference-model trap

DPO scores the policy against a frozen reference, and the reference must be
the SFT model. Getting it wrong does not crash; it trains against the wrong
model. SFT is base, a LoRA adapter and two trained embedding rows
(`<|im_start|>`, `<|im_end|>`; see chapter 3). TRL's two routes both fail:

- **Keep training the SFT adapter.** TRL's reference is then "adapters
  off", which is the base model.
- **Copy the adapter to a "ref" adapter.** TRL copies only parameters whose
  names contain `.default.`. The trained token rows are named
  `trainable_tokens_delta.default`, so they are skipped, and the reference
  is neither SFT nor base. On a tiny CPU model its log-probs were up to
  10.7 nats off SFT ([PR #27 body][pr27]; a test, not a saved run).

**The fix.** Merge the SFT adapter, token rows included, into the base
weights, train a fresh LoRA on the merged model, and let TRL take the
reference with that LoRA off. On the same tiny model this matched SFT to
1e-3 ([PR #27 body][pr27]), and a unit test checks it. The cost is that the
merge is in bf16, so the reference differs from unmerged SFT by rounding
([decisions.md][ref]). The DPO model must be loaded the same way; the eval
harness does (chapter 6).

### Training results

Held-out pairs are 10 of the 105 (95 train), 18 steps, one seed ([dpo-results.md][dpo-train]).
All numbers below are from `outputs/dpo/summary.json` unless noted.

| Held-out pairs | Before | After |
|---|---|---|
| DPO loss | 0.693 | 0.058 |
| Log-prob, chosen | -281.671 | -286.481 |
| Log-prob, rejected | -64.718 | -123.960 |

Source: [dpo-results.md][dpo-held]. The margin of 5.443 comes from the
rejected side falling. Chosen barely moved. Under SFT a looping answer costs
-0.182 per token against -2.250 for a reference answer, which is why greedy
decoding loops: each repeat is close to certain. After DPO the rejected
token costs -0.341; chosen went -2.250 to -2.286 ([dpo-results.md][dpo-len]).

**A misreading, caught by checking the saved file.** The lead's brief for
the results doc said chosen log-probs "rose by 67". The summary's
`logps_change` does show chosen -279.884 to -212.707 and rejected -45.897
to -102.188. But steps 1 and 18 score different batches of pairs, so the
chosen figure moved with the batch. The agent writing the doc checked the
brief against `outputs/dpo/summary.json` and used the like-for-like
measure instead: the same 10 held-out pairs before and after, where chosen
is flat to slightly down. The correct reading: chosen held steady and
rejected fell ([dpo-results.md][dpo-misread]).

### Memory

Peak sampled MPS memory was 13.19 GiB for DPO against 8.10 GiB for the SFT
re-run ([dpo-results.md][dpo-mem]). The expected causes are two answers per
row, a 768-token `max_length` against 512, full-vocabulary logits on each
sequence, and a reference forward pass each step. That is the expected
direction, not a measured breakdown; the run records the peak only. The
project's "GB" figures are GiB.

## Decisions and why

- Reject on format and repetition only: avoids teaching DPO to dodge figures
  ([decisions.md][reject]).
- Greedy rejected answers: sampling found no failures ([decisions.md][greedy]).
- Merge SFT, train a fresh LoRA: the only route that makes the reference
  SFT ([decisions.md][ref]).
- Held-out pair accuracy is a fit check, not a result: those pairs use SFT
  training prompts ([decisions.md][fit]).

## Limits

- 10 held-out pairs from training prompts: loss 0.058 shows fit, not
  generalisation. One seed, 105 pairs, 18 steps.
- **Length confound.** Rejected answers average 356.4 tokens and chosen
  124.1 on the held-out pairs ([dpo-results.md][dpo-len]). DPO may learn
  "shorter is better" rather than "do not repeat", and these pairs cannot
  tell the two apart. Chapter 6 tests it.
- 104 of 105 rejections are repetition. Nothing here teaches correctness.
- Times were contended by another project's GPU jobs; do not read them as
  cost.

## How to reproduce it

Both runs were at commit `0263788`, clean. Not re-run for this chapter: the
pairs run alone took 1,885.7 s of generation ([dpo-results.md][dpo]), over
the one-hour limit the owner waived on 2026-10-03, and it needs the GPU.

```bash
uv run lab pairs --config configs/pairs.yaml   # outputs/pairs/manifest.json
uv run lab dpo --config configs/dpo.yaml       # outputs/dpo/summary.json
```

Both need the SFT run in `outputs/sft/`. Files: `src/post_training/train/pairs.py`,
`dpo.py`, `checkpoint.py`.

## What we would do differently

- Build pairs whose chosen and rejected answers have similar lengths. This
  follows from the length confound ([dpo-results.md][dpo-len], Limits;
  [eval-results.md][res-short]).
- Hold out more than 10 pairs, and from prompts SFT did not train on. This
  follows from the small held-out set, which only shows fit
  ([decisions.md][fit]).

[i2]: https://github.com/graylayer-labs/llm-post-training-lab/issues/2
[pr27]: https://github.com/graylayer-labs/llm-post-training-lab/pull/27
[dpo]: ../dpo-results.md
[res-short]: ../eval-results.md#is-it-just-shorter
[dpo-counts]: ../dpo-results.md#counts
[dpo-train]: ../dpo-results.md#training-results
[dpo-held]: ../dpo-results.md#held-out-pairs-before-and-after
[dpo-len]: ../dpo-results.md#lengths-and-per-token-log-probs
[dpo-misread]: ../dpo-results.md#did-dpo-push-both-answers-down
[dpo-mem]: ../dpo-results.md#memory
[reject]: ../decisions.md#reject-samples-on-format-and-repetition-only
[greedy]: ../decisions.md#draw-rejected-answers-from-greedy-decoding
[dropend]: ../decisions.md#drop-the-end-of-turn-token-from-rejected-samples-that-never-stopped
[ref]: ../decisions.md#use-the-sft-model-as-dpos-reference-by-merging-its-adapter
[fit]: ../decisions.md#treat-held-out-pair-accuracy-as-a-fit-check-not-a-result
