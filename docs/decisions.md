# Decisions

One section per decision, grouped by topic, newest first within each group.
Each section has a title that states the choice, a date, and three lines:
**Chose**, **Over** and **Why**.
New decisions get a new section in the right group.
Sections marked **Planned** describe the design for a part that is not built
yet.

## Model and data

### Start from the Qwen2.5-0.5B base model

*2026-10-03*

- **Chose:** `Qwen/Qwen2.5-0.5B` base.
- **Over:** Qwen2.5-0.5B-Instruct.
- **Why:** An Instruct model already follows the chat format and stops, which
  would hide what SFT changes. On the base model, SFT's effect on format and
  stopping is visible.

### Train on a filtered 2,000-row slice of finance-alpaca

*2026-10-03*

- **Chose:** `gbharti/finance-alpaca` with a seeded 2,000 / 200 train / eval
  split, keeping answers of 40 to 1,500 characters.
- **Over:** the full set.
- **Why:** Short answers teach nothing, and very long ones get truncated at
  512 tokens. 2,000 rows keep one epoch near 15 minutes on the laptop. The
  set mixes finance Q&A with general Alpaca instructions, which the sample
  answers show.

### De-duplicate prompts before splitting

*2026-10-03*

- **Chose:** drop rows whose prompt (instruction + input, whitespace
  collapsed) repeats an earlier row, before the seeded shuffle. The first
  occurrence is kept.
- **Over:** keeping the Part 1 split and removing only the 4 overlapping
  rows from training.
- **Why:** finance-alpaca repeats prompts. Of the 55,835 rows that pass the
  length filter, 7,587 (13.6%) are duplicates, and 4 of the 200 Part 1
  held-out prompts were also training rows. Fixing the split at its source
  guards every later stage, and the DPO and eval code should reuse the same
  `prompt_key`. The cost: the reshuffle changes which rows are drawn, so only
  2 of the 200 new held-out prompts were in the old set, and the Part 1
  numbers are not comparable with the re-run in #12. Measured on
  2026-10-03, not saved. See #15.

## Training

### Use the plain "nll" loss

*2026-10-03*

- **Chose:** `loss_type="nll"`.
- **Over:** TRL's default `"chunked_nll"`.
- **Why:** The chunked loss multiplies hidden states by `lm_head.weight`
  directly. That bypasses PEFT's trainable-token wrapper, so the
  `<|im_end|>` row got no gradient.
- **Cost:** The full batch × sequence × 152k-vocab logits sit in memory. That
  is about 0.8 to 0.95 GB more at batch 2 in the memory probe (4.56 and
  4.70 GB against 3.75 GB). Those logits change shape every batch, so the MPS
  cache is now emptied after every step. See `docs/sft-results.md`.

### Train the chat-token embedding rows in full

*2026-10-03*

- **Chose:** train the `<|im_start|>` and `<|im_end|>` embedding rows in
  full, via PEFT `trainable_token_indices`. The two rows add 1,792 trainable
  parameters.
- **Over:** LoRA on the projections alone.
- **Why:** The base model has barely seen the chat tokens. LoRA cannot reach
  the tied embedding and output rows. After one epoch with LoRA alone, the
  model still did not stop in 5 of 5 samples.

### Compute the loss on the answer only

*2026-10-03*

- **Chose:** a completion-only loss, with prompt tokens masked and
  `<|im_end|>` kept in the target.
- **Over:** training on the whole conversation.
- **Why:** The model should learn to answer and to stop, not to reproduce the
  system prompt and question. Qwen2.5's chat template has no
  `{% generation %}` markers, so rows go to TRL as prompt/completion pairs.

### Apply LoRA to all seven projections

*2026-10-03*

- **Chose:** LoRA r=16, alpha=32 on all seven projections (q, k, v, o, gate,
  up, down). This gives 8.8M trainable parameters, 1.78% of the model.
- **Over:** attention-only LoRA.
- **Why:** The QLoRA paper found that adapting the MLP layers too was needed
  to match full fine-tuning. alpha/r = 2 is a common default.

### Use bf16 LoRA on Apple MPS

*2026-10-03*

- **Chose:** bf16 LoRA on Apple MPS.
- **Over:** QLoRA.
- **Why:** bitsandbytes 4-bit quantisation is CUDA-only. A 0.5B model in bf16
  fits easily in 24 GB.

## Evaluation and preferences (planned)

### Evaluate without a model judge

*2026-10-03*

- **Chose:** perplexity of the references, ROUGE-L and the rule-based
  rubric, with counts and a bootstrap interval on rubric pass rates.
- **Over:** adding a pairwise Claude judge (about 1,200 API calls, $1 to $5).
- **Why:** the owner's choice, to keep cost inside the existing
  subscription. The judge is a stated gap: nothing scores whether an answer
  is correct or helpful beyond what the rubric's rules check. See #3.

### Read the held-out rows from the SFT run's saved file

*2026-10-03*

- **Chose:** the eval harness reads the SFT run's `eval_rows.json` and
  records a sha256 over the sorted `prompt_key`s in `results.json`.
- **Over:** rebuilding the split from the data settings in the SFT run's
  `summary.json`.
- **Why:** a rebuild depends on today's split code and the dataset on the
  Hub. The Part 1 summary records the same data settings as today's config,
  but its split came from before prompts were de-duplicated (#15), so a
  rebuild would not give back its 200 rows. The saved file is what the
  adapter was held out from. See #3.

### Class finance rows by the prompt, and match figures as value sets

*2026-10-03*

- **Chose:** a row counts as finance when its instruction or input uses a
  finance term, such as "dividend", "market order" or "bank" outside a
  non-money compound like "river bank". The `domain_terms` and
  `ungrounded_numbers` rules apply only to this finance slice, and report
  "does not apply" on every other row. The answer is checked against a
  broader list ("pay", "shares", "price"). Each figure maps to a set of
  equal values: `5%` to {5, 0.05}, `$1.5 million` to {1.5, 1,500,000}. A
  figure passes if any value appears in the prompt or reference. Bare
  integers 0 to 10 and years 1900 to 2099 are exempt.
- **Over:** classing rows by the reference answer too, one term list for
  both sides, and exact string matching of numbers.
- **Why:** The rubric picks DPO's rejected answers, so a rule that fails
  good answers teaches the wrong thing. General Alpaca answers mention
  finance in passing ("financial analyst" in a list of jobs), and recipes or
  code carry numbers no reference repeats. Exact matching would fail `0.05`
  against `5%`. See #11.
- **Result:** `tools/rubric_on_references.py` on `main` at commit `723c41a`
  (clean tree) scored the 2,200 training and eval references, saved in
  `outputs/rubric_references/v1/summary.json`. 283 (12.9%) are finance
  rows. The references pass 98.6% overall (2,170 of 2,200), `repetition`
  99.1%, `domain_terms` 96.8% (274 of 283) and `format` 99.95% (2,199 of
  2,200). An earlier scratch run before the split was de-duplicated (#15)
  found 23.4% finance rows. De-duplication cut the share to 12.9%:
  finance-alpaca's finance rows were heavily duplicated, so the training
  data is mostly general Alpaca instructions.
- **Limit:** `ungrounded_numbers` checks grounding, not truth. A correct
  figure that the reference does not state fails it, for example "$52.50"
  for $50 after a year at 5%. With the reference removed as a number source,
  105 of the 283 finance references (37.1%) fail it. An earlier, narrower
  classifier gave 38.2%, measured but not saved. A good answer that cites
  its own figures is therefore often marked ungrounded.

### Score answers with a rubric and a pairwise judge

*2026-10-03*, **Superseded** by "Evaluate without a model judge": the
judge was dropped on 2026-10-03. The rubric part stands.

- **Chose:** a rule-based rubric plus a pairwise Claude judge with position
  swap, alongside perplexity and ROUGE-L. The rubric checks format, clean
  stopping, domain terms and numbers not found in the prompt or reference.
- **Over:** perplexity and ROUGE-L alone, or a public benchmark.
- **Why:** The task has no benchmark, and loss metrics miss the failures that
  matter here: answers that never stop, repeat themselves or invent figures.
  Each judged pair is sent in both orders. An order-dependent verdict counts
  as inconsistent, not a win. See #3.

### Use the SFT model as DPO's reference by merging its adapter

*2026-10-03*

- **Chose:** merge the SFT adapter into the base weights, train a fresh
  LoRA on the merged model, and let TRL take reference log-probs with that
  LoRA disabled. The reference is then exactly the merged SFT model.
  `reference: sft` is in the config and `summary.json`.
- **Over:** passing the SFT `PeftModel` to `DPOTrainer`.
- **Why:** With a PEFT policy, TRL's reference is the model with adapters
  off, which is the base model if the SFT adapter is the one being trained.
  TRL's other route copies the SFT adapter to a "ref" adapter, but it
  matches parameter names containing `.default.`. That misses
  `trainable_tokens_delta.default`, the trained `<|im_start|>` and
  `<|im_end|>` rows, so the reference would silently not be SFT. A test on
  a tiny model checks the reference log-probs equal SFT's and differ from
  base's.
- **Cost:** the merge is done in bf16, so the reference differs from the
  unmerged SFT model by rounding. The trained policy is
  merge(base, SFT adapter) plus the DPO LoRA, and must be loaded that way.

### Draw rejected answers from greedy decoding

*2026-10-03*, owner decision

- **Chose:** the rejected candidate for each prompt is the SFT model's
  greedy answer (`decoding: greedy`, one answer per prompt, batch size 16,
  384 tokens). The rejection rule is unchanged: format or repetition.
- **Over:** temperature-1.0 samples, and skipping DPO.
- **Why:** with the post-dedup SFT adapter (`outputs/sft`), 0 of 96
  temperature-1.0 samples were rejected (9% and 2% ambiguous in two
  scratch runs at batch 8 and 16). SFT's remaining failure is greedy
  looping: 21 of 96 greedy answers on the first 96 training prompts fail
  `repetition`, and 14 of them never stop. The eval harness also decodes
  greedily, so DPO targets the failure the eval measures. Greedy bf16
  output depends on batch size, so batch size is part of the samples
  cache key.
- **Figures:** measured but not saved, from the lead's and the
  implementer's scratch runs on 2026-10-03. The full run's `manifest.json`
  will hold the quotable ones (yield, rejected share by rule, share not
  stopped).

### Treat held-out pair accuracy as a fit check, not a result

*2026-10-03*

- **Chose:** report DPO's reward accuracy and margins on the held-out
  pairs only as a check that training fits the preference data. Claims
  about what DPO changed come from the eval harness on the SFT run's
  held-out prompts.
- **Over:** quoting held-out pair accuracy as DPO's effect.
- **Why:** the held-out pairs come from the same SFT *training* prompts as
  the training pairs, and their chosen answers are references SFT trained
  on. Their accuracy measures fit to that pair distribution, not
  generalisation. The summary also records completion lengths and
  per-token log-probs on those pairs. Rejected samples are long rambles,
  so a "shorter wins" shortcut would show there.

### Grow the pair set by raising n_prompts

*2026-10-03*

- **Chose:** the samples cache is keyed on the whole ordered training split,
  so a re-run with a larger `n_prompts` keeps every saved sample and only
  samples the new prompts. A smaller `n_prompts` is refused.
- **Over:** keying on the first `n_prompts` prompts, which forced a full
  resample to grow the set.
- **Why:** the full run is sized from measured throughput. Starting small
  and extending wastes nothing. The saved samples are reused whole only
  when the earlier `n_prompts * k` is a multiple of `batch_size`.
  Otherwise the earlier run's last, partial batch is sampled again, so
  that batches keep a fixed composition and seed.

### Drop the end-of-turn token from rejected samples that never stopped

*2026-10-03*

- **Chose:** remove the `<|im_end|>` that TRL's chat template appends to a
  rejected completion when the sample hit the token limit.
- **Over:** TRL's default rendering.
- **Why:** The model never produced that token. Keeping it would make DPO
  push down "stop here" after a ramble, against what SFT taught.

### Reject samples on format and repetition only

*2026-10-03*, owner decision

- **Chose:** a sample is rejected if it fails the rubric's `format` rule, or
  its `repetition` rule, or did not stop and fails `repetition`. Chosen is
  the dataset's reference answer. Samples use at least 384 new tokens. A
  sample that only hit the limit is ambiguous: it is skipped and counted.
  The pair takes the first rejected sample by sample index.
- **Over:** rejecting on any rubric failure, including `ungrounded_numbers`
  and `domain_terms`, and picking the worst sample.
- **Why:** The number rule fails 37.5% of finance references and fails
  correct arithmetic, so rejecting on it would teach DPO to avoid concrete
  figures. Those two rules are logged per sample and counted, never used to
  reject. At 384 tokens, a sample that hit the limit may be a long good
  answer, so it is not a safe rejected example. Taking the first rejected
  sample, not the worst, keeps the set from leaning towards the most
  degenerate, easiest-to-separate answers.
- **Note:** "did not stop and fails repetition" is a subset of "fails
  repetition", so the rule is format OR repetition. It is still counted
  separately in `manifest.json`.

### Build the DPO preference pairs from the project's own data

*2026-10-03*

- **Chose:** a home-made preference set for DPO. Chosen is the reference
  answer, and rejected is an SFT sample that fails the rubric. The prompts
  are the SFT run's own training prompts: the split is rebuilt from its
  `summary.json` and must match its `eval_rows.json`, or `lab pairs` stops.
- **Over:** a public preference dataset.
- **Why:** Real users rarely have preference data, so building pairs from SFT
  data is the realistic case. It also ties the signal to the same rubric
  Part 3 scores. See #2.

## Project setup

### Record each run's commit, tree state and library versions

*2026-10-03*

- **Chose:** every `summary.json` carries a `provenance` block: the commit,
  whether the tree was dirty (untracked files count), the branch, the
  machine and the torch, transformers, trl, peft and datasets versions. A run
  from a dirty tree is marked `scratch`.
- **Over:** trusting the commit message or the write-up to say which code
  made a run.
- **Why:** The Part 1 SFT run came from a dirty tree and recorded no commit,
  so its numbers could not be traced. See #10.

### Treat the generation batch size as a setting

*2026-10-03*

- **Chose:** batched, left-padded generation, with the batch size recorded
  beside the other generation settings and held fixed when systems are
  compared.
- **Over:** assuming batched greedy decoding matches one prompt at a time.
- **Why:** In bf16 on MPS, padding changes the numerics enough to flip
  near-tied tokens. On the first 5 held-out prompts, batch size 1 reproduced
  all 10 saved greedy answers exactly. Batch size 5 matched the 5 short
  answers but diverged on 4 of the 5 that ran to the token limit, in one
  case after 22 characters. Measured on 2026-10-03, not saved.
- **Applied:** the full eval (`configs/eval.yaml`) decodes at batch size 16
  for every system, the same setting the DPO pairs were drawn with. At
  batch size 1 the base model, which often runs to the 384-token limit,
  would have kept a shared GPU busy for about 85 minutes. The answers are
  therefore not bit-identical to one-at-a-time decoding, but every system
  is decoded the same way, which is what the comparison needs.

### Checkpoint training runs and resume after a crash

*2026-10-03*

- **Chose:** SFT saves a checkpoint (adapter, optimizer, scheduler, RNG
  state) every `train.save_steps` steps under `<output_dir>/checkpoints`, and
  `lab sft --resume` continues from the newest complete one. Starting over
  existing checkpoints is refused. A resume must match the original commit
  (`provenance.json`) and config (`config.json`), on a clean tree, unless
  `--scratch`. The pre-training eval is saved and reused. `train_loss` is the
  mean of the logged step losses.
- **Over:** `save_strategy="no"`, where a crash at step 120 of 125 loses the
  whole run.
- **Why:** The real Part 1 SFT run was killed (see #21), and the GPU is shared
  with other jobs. A resumed run must still be one commit's code, or its
  numbers cannot be traced.
- **Cost:** The Trainer's own `training_loss` is wrong after a resume (its
  running total restarts at 0 but is divided by the full step count; 3.380
  against 4.160 on a tiny CPU run), so the log is used instead. Wall time,
  peak memory and trainer timings of a resumed run cover only the resumed
  part, flagged by `resumed_segment_only`. The Trainer does not restore MPS
  RNG state, so with LoRA dropout a resumed MPS run is statistically
  equivalent to an uninterrupted one, not bit-identical; on CPU it matched
  exactly.

### Describe each run in a config file

*2026-10-03*

- **Chose:** config-driven runs: one YAML file per run, typed dataclasses and
  a `lab` CLI.
- **Over:** notebooks.
- **Why:** A run is then fully described by a commit and a file, and two runs
  differ by a YAML diff. `summary.json` stores the resolved config next to
  the numbers.

### Use uv, ruff, ty and pytest

*2026-10-03*

- **Chose:** uv, ruff, ty and pytest, with the same checks in CI.
- **Over:** pip and mypy.
- **Why:** One lockfile and fast installs. One quality gate runs the same
  locally and in CI.
