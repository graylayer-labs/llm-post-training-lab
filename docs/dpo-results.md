# Part 2: DPO results

What DPO did on top of the LoRA SFT model, how the preference pairs were
built from the project's own data, and why the reference model needed care.
Issue [#2](https://github.com/graylayer-labs/llm-post-training-lab/issues/2);
the code came in
[#26](https://github.com/graylayer-labs/llm-post-training-lab/issues/26)
(PR [#27](https://github.com/graylayer-labs/llm-post-training-lab/pull/27)).
The reasons behind each setting are in [decisions.md](decisions.md).

These are training-side numbers on held-out *pairs*. They are not the
evaluation. Base, SFT and DPO are scored on the 200 held-out prompts in
[#3](https://github.com/graylayer-labs/llm-post-training-lab/issues/3), and
those results will get their own doc.

## How these numbers were made

Two runs, both from commit `0263788` on `main`, clean tree, `scratch` false.

| | Pairs | DPO |
|---|---|---|
| Command | `uv run lab pairs --config configs/pairs.yaml` | `uv run lab dpo --config configs/dpo.yaml` |
| Run directory | `outputs/pairs/` | `outputs/dpo/` |
| Main file | `manifest.json` | `summary.json` |
| Provenance | `provenance` in `manifest.json` | `provenance` in `summary.json` |
| Time | 1,885.7 s of generation (`throughput.generation_seconds`) | 1,118.8 s wall (`wall_seconds`) |

- Both runs start from the SFT re-run in `outputs/sft/` (adapter sha256
  `ade0646e…`, recorded as `adapter_sha256` in the manifest and
  `reference.sft_adapter_sha256` in the summary). See
  [sft-results.md](sft-results.md).
- Apple M4, 24 GB, MPS. torch 2.14.1, transformers 5.18.0, trl 1.14.1,
  peft 0.21.2.
- One seed (0) for each run. Small differences between runs are not
  evidence.
- Both times were measured while another project's GPU jobs were running on
  the same machine, so they are contended. The lead saw those jobs in the
  process list; the run files do not record them. Do not read the times as
  the cost of either stage.
- The pairs run was expected to take longer than this repo's one-hour limit
  for laptop runs (see "Ask the owner first" in `CLAUDE.md`). The owner
  agreed to it in the lead's session on 2026-10-03. Its generation took
  1,885.7 s in the end.
- DPO checkpoints were saved every 4 steps and removed after the run
  completed (`steps_plan.save_steps`).
- `outputs/` is gitignored, so these files are local only.

## Building the pairs

A DPO pair is one prompt with a *chosen* and a *rejected* answer. Chosen is
the dataset's reference answer. Rejected is the SFT model's own answer, when
the rule-based rubric
([#11](https://github.com/graylayer-labs/llm-post-training-lab/issues/11))
says it fails. The prompts are the first 500 of the SFT run's 2,000
*training* prompts, so no eval prompt can leak in.

The owner made two decisions, recorded on #2 and in
[decisions.md](decisions.md):

1. **Reject on behaviour only.** A sample is rejected if it fails the
   `format` rule or the `repetition` rule. The `ungrounded_numbers` and
   `domain_terms` rules are logged but never reject, because the number rule
   fails 37.1% of finance reference answers once the reference is removed as
   a source of figures (`outputs/rubric_references/v1/summary.json`).
   Rejecting on it would teach DPO to avoid concrete figures. A sample that
   only hit the 384-token limit, without failing a behaviour rule, is
   ambiguous and skipped.
2. **Use greedy answers.** With the clean SFT adapter, temperature-1.0
   sampling gave 0 rejected of 96 samples, so there was nothing to pair.
   That was a scratch measurement and was not saved. SFT's remaining failure
   is greedy looping, which is also how the eval harness decodes, so the
   owner chose one greedy answer per prompt.

### Counts

From `outputs/pairs/manifest.json`, `counts` unless another key is named.

| | Value | Key |
|---|---|---|
| Training prompts used | 500 | `prompts` |
| Greedy answers (one per prompt, 384 new tokens at most, batch 16) | 500 | `samples` |
| Rejected | 105 | `samples_rejected` |
| ... for `format` | 1 | `rejections_per_rule.format` |
| ... for `repetition` | 104 | `rejections_per_rule.repetition` |
| ... of those, repetition and never stopped | 66 | `rejections_per_rule.no_stop_and_repetition` |
| Ambiguous (hit the limit, no rule failed), skipped | 2 | `ambiguous_skipped` |
| Passed | 393 | `samples_pass` |
| Pairs kept | 105 | `pairs_kept` |
| Yield (pairs per prompt) | 0.21 | `yield` |
| Share of answers that stopped | 0.864 | `stopped_share` |
| Mean new tokens per answer | 103.158 | `mean_new_tokens` |
| Answers failing `ungrounded_numbers` / `domain_terms` (logged only) | 9 / 6 | `rule_failures` |
| Held-out eval prompts among the pair prompts | 0 of 200 | `split_check.pair_prompts_in_eval` |

- Every rejection but one is a repetition loop. The pairs teach one thing:
  do not loop.
- 66 of the 105 rejected answers never stopped. Their `<|im_end|>` is
  removed before training, because the model never produced it (see
  decisions.md, "Drop the end-of-turn token...").
- The split was rebuilt from the SFT run's `summary.json` and checked against
  its `eval_rows.json` (`split_check.eval_rows_match` true).
- The pair file is hashed (`pairs_sha256` `cbcca269…`), and the DPO summary
  records the same hash.

### Two example pairs

From `outputs/pairs/pairs.jsonl` (pair `index` 0 and 3). Shortened.

**Prompt:** "Generate a list of books related to business and finance"

- **Chosen** (reference): "1. The Intelligent Investor by Benjamin Graham
  2. Think and Grow Rich by Napoleon Hill 3. The Essays of Warren Buffett
  ..."
- **Rejected** (greedy SFT, 131 tokens, stopped): "1. "The Art of Business"
  by Peter Drucker 2. "The Art of Business" by Peter Drucker 3. "The Art of
  Business" by Peter Drucker ..." The rubric's reason: "'the art of
  business' occurs 10 times".

**Prompt:** "How do brokerage firms make money?"

- **Chosen** (reference): "Regarding "Interest on idle cash", brokerage
  firms must maintain a segregated account on the brokerage firm's books
  ..."
- **Rejected** (greedy SFT, 384 tokens, never stopped): "The brokerage
  firms make money by selling securities to investors and then charging
  them a commission for each transaction. They also make money by selling
  securities to other firms and charging them a commission for each
  transaction. They also make money by selling securities to other firms
  ..." The rubric's reason: "'make money by selling' occurs 20 times".

The loop starts with a plausible sentence, then the model copies itself.
The chosen answers are written by people in a forum or instruction style,
so they are not the model's own text. That matters below.

## Choosing the reference model

DPO pulls the policy towards chosen answers and away from rejected ones,
*relative to a frozen reference*. The reference must be the SFT model.
Getting that wrong does not crash; it quietly trains against the wrong
model. This was the hardest part of the stage.

The SFT model is base + a LoRA adapter + two trained embedding rows
(`<|im_start|>` and `<|im_end|>`, see the stop-token bug in
[sft-results.md](sft-results.md)). PEFT stores those rows as
`trainable_tokens_delta` parameters. TRL offers two routes, and both give
the wrong reference here:

- **Train the SFT adapter further.** With a PEFT policy, TRL computes the
  reference with the adapters disabled. If the adapter being trained is the
  SFT one, "adapters off" is the *base* model.
- **Copy the SFT adapter to a "ref" adapter.** TRL's copy only takes
  parameters whose names contain `.default.`. The trained token rows are
  named `...trainable_tokens_delta.default`, with no dot after `default`, so
  they are skipped. The "reference" is then SFT's LoRA without SFT's chat
  token rows: neither SFT nor base. In a test on a tiny CPU model, that
  route's reference log-probs were up to 10.7 nats off SFT's (PR #27 body;
  a test, not a saved run).

**What we do.** Merge the SFT adapter, token rows included, into the base
weights. Train a fresh LoRA on the merged model. TRL then takes the
reference with that fresh LoRA off, which is exactly the merged SFT model.
On the same tiny model this matched SFT to 1e-3 (PR #27 body). A unit test
checks that the reference log-probs equal SFT's and differ from base's.

The run records this. `summary.json` has `config.reference` `sft` and a
`reference.construction` string, and its `policy` is
"merge(Qwen/Qwen2.5-0.5B, outputs/sft/adapter) + LoRA in outputs/dpo/adapter".
The DPO model must be loaded the same way; the eval harness does. Costs:
the merge is in bf16, so the reference differs from the unmerged SFT model
by rounding. Full reasoning: decisions.md, "Use the SFT model as DPO's
reference by merging its adapter".

## Training results

Settings from `outputs/dpo/summary.json`: beta 0.1, LoRA r=16 on all seven
projections (8,798,208 trainable parameters, `params_trainable`), learning
rate 5e-5, 3 epochs, effective batch 16 pairs (2 × 8), `max_length` 768.
105 pairs, split 95 for training and 10 held out (`pairs.n_train`,
`pairs.n_held_out`). 18 optimiser steps with 1 warmup step
(`steps_plan`). No chosen or rejected answer was truncated, in training or
held out (`truncation`). `train_loss` 0.246 is the mean of the logged step
losses.

### Held-out pairs, before and after

The same 10 held-out pairs, scored by TRL before step 1 and after step 18.
From `outputs/dpo/summary.json`, keys under `eval_before` and `eval_after`.
Rounded to three decimals. Log-probs are summed over the answer's tokens;
a reward is beta × (policy log-prob − reference log-prob).

| | Before | After | Key |
|---|---|---|---|
| DPO loss | 0.693 | 0.058 | `eval_loss` |
| Reward, chosen | 0.000 | -0.481 | `eval_rewards/chosen` |
| Reward, rejected | 0.000 | -5.924 | `eval_rewards/rejected` |
| Reward margin | 0.000 | 5.443 | `eval_rewards/margins` |
| Reward accuracy | 0.0 | 1.0 | `eval_rewards/accuracies` |
| Log-prob, chosen | -281.671 | -286.481 | `eval_logps/chosen` |
| Log-prob, rejected | -64.718 | -123.960 | `eval_logps/rejected` |

- Before training the fresh LoRA is zero, so policy = reference, every
  reward is 0 and the loss is ln 2 = 0.693. Accuracy counts only strict
  wins, so ties give 0.0.
- After 18 steps the model separates all 10 held-out pairs. On 10 pairs
  from the training prompts this is a fit check, not a result (see Limits).

### Lengths and per-token log-probs

The same 10 pairs, from `held_out_lengths` in `summary.json`. "Before" is
the reference, the SFT model. Rounded to three decimals.

| | Before | After | Key (`before.` / `after.`) |
|---|---|---|---|
| Chosen length, tokens | 124.1 | 124.1 | `chosen_tokens_mean` |
| Rejected length, tokens | 356.4 | 356.4 | `rejected_tokens_mean` |
| Chosen log-prob, summed | -281.671 | -286.503 | `chosen_logp_sum_mean` |
| Rejected log-prob, summed | -64.718 | -123.971 | `rejected_logp_sum_mean` |
| Chosen log-prob per token | -2.250 | -2.286 | `chosen_logp_per_token_mean` |
| Rejected log-prob per token | -0.182 | -0.341 | `rejected_logp_per_token_mean` |

Before training these match the `eval_logps` above. After training they
differ slightly (−286.503 against −286.481 for chosen), because they are
computed in a separate pass.

**Why greedy decoding loops.** Under the SFT model a looping rejected answer
costs about −0.18 per token, against about −2.25 for a reference answer.
Once a loop starts, each repeat is close to certain, so greedy decoding
keeps taking it. The whole 356-token loop is *more* likely in total
(−64.7) than the 124-token reference (−281.7). Before DPO, SFT prefers its
own loop to the human answer by a wide margin.

**What DPO changed.** After training, a rejected token costs −0.341, nearly
double. The chosen answers barely moved: −2.250 to −2.286 per token.

### Did DPO push both answers down?

A known DPO failure with off-policy chosen answers (text the model did not
write) is to widen the margin by lowering both answers, the chosen one
less. It did not happen here in any large way:

- On the held-out pairs the margin of 5.443 comes from the rejected reward
  falling to −5.924, while the chosen reward fell only to −0.481. In
  log-prob terms, chosen fell by 4.81 nats over 124 tokens (−281.671 to
  −286.481) and rejected by 59.24 nats (−64.718 to −123.960).
- During training, the per-batch chosen reward stayed near zero, between
  −0.290 (step 17) and 0.201 (step 14), while the rejected reward fell
  from −0.006 at step 1 to −5.760 at step 17 and −4.820 at step 18
  (`trajectory`).

Selected steps from `trajectory` in `summary.json`, rounded to three
decimals. Each step is a different batch of up to 16 training pairs, so the
log-prob columns change partly because the pairs change. The reward columns
are each batch measured against the reference, so they compare across
steps.

| Step | Loss | Reward, chosen | Reward, rejected | Margin | Accuracy | Log-prob, chosen | Log-prob, rejected |
|---|---|---|---|---|---|---|---|
| 1 | 0.683 | 0.014 | -0.006 | 0.020 | 0.688 | -279.884 | -45.897 |
| 6 | 0.326 | -0.015 | -0.997 | 0.981 | 1.0 | -312.214 | -75.353 |
| 12 | 0.055 | -0.015 | -3.875 | 3.860 | 1.0 | -172.212 | -91.505 |
| 18 | 0.085 | -0.014 | -4.820 | 4.805 | 1.0 | -212.707 | -102.188 |

`logps_change` in the summary compares step 1 with step 18: chosen
−279.884 to −212.707 (+67.178) and rejected −45.897 to −102.188 (−56.291).
The rejected fall is consistent with the held-out pairs. The chosen *rise*
is mostly a change of batch: steps 1 and 18 score different pairs, and the
like-for-like measures above show chosen roughly flat, slightly down on the
held-out pairs. So the honest reading is that chosen held steady while
rejected fell, not that chosen rose.

## Limits

- **Only 10 held-out pairs, from training prompts.** The held-out pairs use
  SFT *training* prompts, and their chosen answers are references SFT was
  trained on. Loss 0.058 and accuracy 1.0 on them show that DPO fits this
  pair distribution, not that it generalises. See decisions.md, "Treat
  held-out pair accuracy as a fit check, not a result".
- **Small set, one seed.** 105 pairs, 18 steps, one seed for both the pairs
  and DPO. No interval can be given.
- **Length confound.** Rejected answers average 356.4 tokens and chosen
  124.1, about 2.9 times longer (`held_out_lengths`). Nearly every rejected
  answer is a long loop. DPO may partly learn "shorter is better" or "stop
  sooner" rather than "do not repeat". These pairs cannot tell those apart.
- **Narrow signal.** 104 of 105 rejections are for repetition. DPO here
  teaches the model away from loops, nothing about correctness or figures.
- **Not the eval.** Held-out pair loss and accuracy say nothing about
  answers on the 200 held-out eval prompts. That three-way comparison of
  base, SFT and DPO is
  [#3](https://github.com/graylayer-labs/llm-post-training-lab/issues/3),
  and its results come in a separate doc. No eval result is claimed here.

## Memory

Peak sampled MPS memory was 13.19 GiB for DPO (`outputs/dpo/summary.json`,
`peak_memory_gb`), against 8.10 GiB for the SFT re-run
(`outputs/sft/summary.json`, `peak_memory_gb`).

Why DPO needs more, at the same batch size of 2:

- **Two answers per row.** Each DPO row is a pair, so a micro-batch of 2
  pairs runs 4 sequences (2 chosen, 2 rejected), against 2 for SFT.
- **Longer sequences.** DPO allows 768 tokens of prompt and answer
  (`max_length`), against 512 for SFT, so that 384-token rejected loops fit
  without truncation. The rejected answers are the long ones.
- **The logits.** Every sequence carries full 152k-vocabulary logits, which
  were already the largest activation in SFT (see
  [sft-results.md](sft-results.md)).
- **A reference pass.** Reference log-probs are computed each step, with the
  DPO LoRA off (`precompute_ref_log_probs` is false), so each step runs a
  second forward pass without gradients.

This is the expected direction, not a measured breakdown: the run records
the peak only. Like the SFT figure, it is sampled per optimiser step and at
evaluation, so a spike between samples is missed.

**Units.** The project's "GB" figures are GiB. `accelerator_memory_gb` in
`src/post_training/train/common.py` divides by 2^30. The SFT results doc
writes "GB" for the same unit.
