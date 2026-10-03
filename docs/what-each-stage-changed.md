# What each post-training stage changed

The front door to the write-up for issue
[#4](https://github.com/graylayer-labs/llm-post-training-lab/issues/4).
It says what SFT and DPO did to a 0.5B base model, what broke on the way,
how the model was judged without a benchmark, and what would differ at a
scale far beyond one laptop. Every number names the doc and the saved file
it came from, with the commit of the run.

The results docs hold the detail:

- [sft-results.md](sft-results.md): Part 1, LoRA SFT, commit `76739e6`.
- [dpo-results.md](dpo-results.md): Part 2, pairs and DPO, commit `0263788`.
- [eval-results.md](eval-results.md): Part 3, the three-way eval, commit
  `76c9086`, and the length check at `b872427`.
- [memory-and-scale.md](memory-and-scale.md): where the memory goes, and
  the 70B reasoning.
- [decisions.md](decisions.md): why each choice was made.

`outputs/` is gitignored, so the files named below are local to the
laptop that ran them. Each results doc records the commit and the
provenance block of its run.

## The question

From [INTENT.md](../INTENT.md). Organisations want to adapt an open LLM to
their own data. They have no benchmark for their task, limited hardware,
and recipes whose stages are opaque. So this project ran every stage by
hand on `Qwen/Qwen2.5-0.5B` on one Apple M4 laptop, to see:

- what SFT changes, and what DPO changes on top of it;
- where the memory goes during training;
- how to evaluate when no off-the-shelf benchmark exists.

## Base, SFT and DPO in one table

Base, SFT and DPO decoded on the same 200 held-out prompts, greedy, at most
384 new tokens, batch size 16, one seed. From
[eval-results.md](eval-results.md), "The table", copied from
`outputs/eval/results.md` at commit `76c9086`.

| System | Perplexity of references | ROUGE-L F1 | Rubric pass, overall (95% interval) | Stopped % | Mean new tokens |
|---|---|---|---|---|---|
| base | 8.910 | 0.153 | 35.0 (70/200) [28.5, 41.5] | 39.5 | 270.0 |
| sft | 6.245 | 0.297 | 73.0 (146/200) [66.5, 79.0] | 86.5 | 111.6 |
| dpo | 6.555 | 0.296 | 89.0 (178/200) [84.5, 93.0] | 100.0 | 51.9 |

How to read it, in short. Perplexity is of the dataset's *reference*
answers under each model, not of the model's own answers. ROUGE-L is
overlap with one reference. The rubric is rule-based: format, clean stop,
repetition, and two finance-only rules on 24 rows. Nothing in the table
scores whether an answer is correct. The full column definitions and the
per-rule rates are in eval-results.md.

## What SFT changed

SFT was a LoRA (r=16, all seven projections) on 2,000 rows of
`gbharti/finance-alpaca` (despite the name, mostly general instructions
after de-duplication: 12.9% of references are finance; see Limits), with
the prompt masked from the loss and the two chat-token embedding rows
trained in full. One epoch, 125 steps. Eval loss
on the 200 held-out answers fell from 2.169 to 1.714
([sft-results.md](sft-results.md), Runs table, `outputs/sft/summary.json`,
commit `76739e6`).

**Format and stopping.** The base model already keeps the chat format
(`format` 99.5%, 199/200), but it does not know how to end a turn: it stops
on 39.5% of prompts and runs 121 answers to the 384-token limit. After SFT
it stops on 86.5% and 27 answers hit the limit
([eval-results.md](eval-results.md), "SFT against base";
`outputs/eval/analysis.json`, `systems.*.lengths.at_limit`, commit
`b872427`). Mean new tokens fell from 270.0 to 111.6, the median from 384
to 66.

**Register (illustrative Part 1 samples).** In the Part 1 samples, the
finance answers take on the first-person forum style of the training data
("I would not do that. I would put the money in a savings account...")
([sft-results.md](sft-results.md), "Base vs SFT answers"). Those samples
came from a dirty tree, so they illustrate and are not quoted as results.

**Closer to the references.** ROUGE-L 0.153 to 0.297, nearly double.
Perplexity of the references 8.910 to 6.245. On the 69 prompts where both
base and SFT stopped, SFT is shorter (63.6 against 96.3 mean tokens) and
closer to the reference (ROUGE-L 0.322 against 0.284)
(`analysis.json`, `pairs."base->sft".both_stopped`). SFT fixes 88 prompts
that base failed and breaks 12 that base passed; all 12 now fail
`repetition` (`pairs."base->sft".fixed` and `.regressed`).

**The stop-token story.** The first SFT run, LoRA alone, ended none of its 5
sample answers. The chat template closes a turn with `<|im_end|>`, which
the base model has barely seen, and LoRA on the projections cannot reach
the tied embedding and output rows that decide it. The fix trains the
`<|im_start|>` and `<|im_end|>` embedding rows in full, which needed a
second fix: TRL's default `chunked_nll` loss bypasses PEFT's wrapper, so
the rows got no gradient until the loss was switched to `nll`. The saved
probe (`tools/stop_token_probe.py` at commit `723c41a`,
`outputs/sft/stop_token_probe.json`, all 200 held-out rows) measures the
effect at the end of each reference answer
([sft-results.md](sft-results.md), "The stop-token bug"):

| `<|im_end|>` as the next token | Base | SFT |
|---|---|---|
| Median probability | 2.065e-09 | 0.7417 |
| Median rank, of 152k | 123,031 | 1 |
| Share of rows where it ranks first | 0.00 | 0.795 |

That is a next-token measure on reference text. The eval table above is the
count on generated answers: 39.5% to 86.5%.

**Not fixed by SFT.** Repetition loops. The `repetition` pass rate rose from
40.5% to 75.0%, but 27 SFT answers still loop to the limit, and SFT's 12
regressions against base are all repetition. The loop is cheap under the
SFT model: a looping answer costs about −0.18 per token against about
−2.25 for a human reference, so once a loop starts greedy decoding keeps
it ([dpo-results.md](dpo-results.md), "Lengths and per-token log-probs",
`outputs/dpo/summary.json`, `held_out_lengths.before`).

## What DPO changed

DPO trained a fresh LoRA on the merged SFT model, with the merged SFT model
as the reference, on 105 preference pairs built from the project's own
data: 95 for training, 10 held out, 18 steps, beta 0.1
([dpo-results.md](dpo-results.md), "Training results",
`outputs/dpo/summary.json`, commit `0263788`).

**Where the pairs came from.** The chosen answer is the dataset's
reference. The rejected answer is the SFT model's own greedy answer, when
the rubric's `format` or `repetition` rule fails it. The prompts are 500 of
SFT's *training* prompts, so no eval prompt can leak in
(`split_check.pair_prompts_in_eval` 0). Of 500 greedy answers, 105 were
rejected, 104 of them for repetition, and 66 of those never stopped
(`outputs/pairs/manifest.json`, `counts`). The pairs teach one thing: do
not loop. Greedy answers were a fallback. With the clean SFT adapter,
temperature-1.0 sampling gave 0 rejected of 96 samples, so there was
nothing to pair; that was a scratch measurement, not saved
([decisions.md](decisions.md), "Draw rejected answers from greedy
decoding").

**Looping cut, every answer stops.** Stopped 86.5% to 100.0%; none of the
200 DPO answers reaches the limit, the longest is 375 tokens. `repetition`
75.0% to 92.5%, so 15 of 200 DPO answers still repeat, and a short loop
can stay under the rule's threshold (index 29, under "What it missed"
below). Overall rubric 73.0% to 89.0%: DPO fixes 36 prompts and
breaks 4 ([eval-results.md](eval-results.md), "DPO against SFT";
`analysis.json`, `pairs."sft->dpo"`). On the held-out pairs, the rejected
answers' per-token log-prob fell from −0.182 to −0.341, nearly double the
cost, while the chosen answers barely moved, −2.250 to −2.286
([dpo-results.md](dpo-results.md), "Lengths and per-token log-probs").

**But shorter, everywhere.** Mean new tokens 111.6 to 51.9. DPO's answer is
shorter than SFT's on 144 of 200 prompts. On the 173 prompts where SFT
already stopped, DPO averages 50.4 tokens against 69.1, median per-prompt
ratio 0.79 (`analysis.json`, `pairs."sft->dpo".both_stopped`). If DPO had
only learned to break loops, answers that already stopped would keep their
length. They did not. The pairs explain why: rejected answers average 356.4
tokens and chosen 124.1 (`outputs/dpo/summary.json`, `held_out_lengths`),
so "shorter" and "do not loop" were the same signal
([eval-results.md](eval-results.md), "Is it just 'shorter'?").

**ROUGE-L flat, perplexity up.** ROUGE-L 0.297 to 0.296. That headline is
an average of two moves: on the 27 prompts where SFT looped, DPO scores
0.180 against 0.087; on the 173 where SFT stopped, DPO scores 0.314
against SFT's 0.330 (`pairs."sft->dpo".a_did_not_stop.rouge_l` and
`.both_stopped.rouge_l`). Perplexity of the references rose from 6.245 to
6.555: DPO moved slightly away from the reference distribution, as
expected when a model is pushed towards short answers and away from its
own loops. It does not show DPO's answers got worse, and it does not show
they got better.

**Very short answers.** 48 DPO answers are under 20 tokens, against 27 for
SFT (`systems.*.lengths.short`). Many fit their prompt. Some drop part of
the task: asked to "name and describe four types of renewable energy", SFT
named and described them in 87 tokens, DPO named them in 17 and stopped.
The rubric counts that as a DPO fix (`analysis.json`, `examples`, index 2).

**Reading.** DPO did what the pairs mostly taught. It stops every time and
loops far less, and that is why the rubric rate rose 16 points. It also
learned "be short", and this run cannot separate the two. Doing so would
need pairs whose chosen and rejected answers have similar lengths.

## What broke along the way

Each item links the section that holds the detail.

- **The stop token.** LoRA alone could not teach `<|im_end|>`, and TRL's
  `chunked_nll` loss silently skipped the fix. Found from 0 of 5 samples
  stopping; fixed by training the two token rows and using `nll`.
  [sft-results.md, "The stop-token bug"](sft-results.md#the-stop-token-bug).
- **MPS cache retention and out-of-memory.** The `nll` loss holds the full
  batch × sequence × 152k logits, and each batch has a new sequence length,
  so the MPS caching allocator kept old blocks. At micro-batch 2 the run
  died with 2.84 GiB of tensors and 27.34 GiB held by the driver, against a
  30.19 GiB limit. Halving the batch did not help. Fixed by
  `torch.mps.empty_cache()` after every optimiser step.
  [sft-results.md, "Out of memory, then cache retention"](sft-results.md#out-of-memory-then-cache-retention).
- **Duplicated prompts and train/eval overlap.** finance-alpaca repeats
  prompts: 7,587 of the 55,835 rows that pass the length filter are
  duplicates, and 4 of the Part 1 run's 200 held-out prompts were also
  training rows. The split now de-duplicates by prompt before shuffling,
  and SFT was re-run on the clean split; the Part 1 numbers are not
  comparable with it.
  [decisions.md, "De-duplicate prompts before splitting"](decisions.md#de-duplicate-prompts-before-splitting);
  [sft-results.md, "How these numbers were made"](sft-results.md#how-these-numbers-were-made).
- **The killed run, checkpointing and the resumed-loss bug.** The first SFT
  re-run was killed at step 18 of 125 by another agent's `pkill -f` (the
  Claude agents and how they worked are in the README,
  ["How this was built"](../README.md#how-this-was-built)). Nothing had
  been saved. That led to checkpoints every 25 steps and
  `lab sft --resume`, which refuses a resume from a different commit or
  config. The Trainer's own `training_loss` is wrong after a resume (its
  running total restarts at 0 but is divided by the full step count: 3.380
  against 4.160 on a tiny CPU run), so `train_loss` is now the mean of the
  logged step losses.
  [sft-results.md, "Failures on the way"](sft-results.md#failures-on-the-way);
  [decisions.md, "Checkpoint training runs and resume after a crash"](decisions.md#checkpoint-training-runs-and-resume-after-a-crash).
- **TRL's reference copy skipped the trained token rows.** DPO's reference
  must be the SFT model. TRL's route that copies the policy adapter to a
  "ref" adapter matches parameter names containing `.default.`, which
  misses `trainable_tokens_delta.default`, the trained `<|im_start|>` and
  `<|im_end|>` rows. The reference would have been neither SFT nor base,
  with no error: on a tiny CPU model its log-probs were up to 10.7 nats off
  SFT's (PR #27, a test, not a saved run). The fix merges the SFT adapter
  into the base weights, trains a fresh LoRA on top, and lets TRL take the
  reference with that LoRA off, which matched SFT to 1e-3 on the same
  model and is checked by a unit test.
  [dpo-results.md, "Choosing the reference model"](dpo-results.md#choosing-the-reference-model).
- **Batch-size-dependent greedy decoding.** In bf16 on MPS, left padding
  changes the numerics enough to flip near-tied tokens. Batch size 1
  reproduced all 10 saved greedy answers on the first 5 held-out prompts;
  batch size 5 matched the 5 short answers but diverged on 4 of the 5 that
  ran to the limit, once after 22 characters (measured, not saved). So the
  batch size is recorded as a generation setting and held at 16 for every
  system, and it is part of the samples cache key for the pairs.
  [decisions.md, "Treat the generation batch size as a setting"](decisions.md#treat-the-generation-batch-size-as-a-setting).
- **A misreading, caught by checking the saved file.** The DPO summary's
  `logps_change` compares step 1 with step 18 and shows the chosen log-prob
  rising by +67.178 (−279.884 to −212.707). The lead Claude Code session
  read that as DPO raising the chosen answers, and wrote it into the brief
  for the results doc; the agent writing the doc checked it against the summary and
  corrected it. It is mostly a change of batch: steps 1 and 18 score
  different pairs. The like-for-like held-out pairs show chosen roughly
  flat, slightly down (−281.671 to −286.481), while rejected fell by 59.24
  nats. The honest reading is that chosen held steady while rejected fell.
  [dpo-results.md, "Did DPO push both answers down?"](dpo-results.md#did-dpo-push-both-answers-down).

## Evaluating without a benchmark

The task has no public test set, so the harness scores every stage on the
same 200 held-out prompts with three measures that need no judge: the
perplexity of the reference answers, ROUGE-L against the reference, and a
rule-based rubric with bootstrap intervals
([eval-results.md](eval-results.md), "How these numbers were made";
[decisions.md](decisions.md), "Evaluate without a model judge").

**What the rubric caught.** The failures that matter for this task and
that loss metrics miss: answers that never stop, answers that loop, leaked
chat markers, and figures not found in the prompt or reference. Stopping
and repetition are where base, SFT and DPO differ most, and the rubric
measures both directly and deterministically. It also chose the DPO
rejected answers, so the training signal and the evaluation agree on what
"bad" means.

**What it missed.**

- *Correctness.* "Given a mathematical expression, simplify it": input
  `3x + 4xy + y - 2x - 3y`, reference ending `= 4xy + x - 2y`. DPO answered
  "The simplified expression is 2x + 2y" in 12 tokens. Wrong, and it passes
  every rubric rule (`outputs/eval/analysis.json`, `examples`, index 50).
- *Completeness.* The renewable-energy half-answer above (index 2) counts
  as a fix.
- *Short loops.* A 46-token burger-topping list repeats three items twice
  but stays under the `repetition` rule's threshold, so it passes (index
  29). The rule counts repeats, so shorter answers pass it more easily,
  which is part of DPO's gain.
- *The finance slice is tiny.* `domain_terms` and `ungrounded_numbers`
  apply to 24 of the 200 prompts. One row moves a rate by 4.2 points, and
  SFT's `ungrounded_numbers` interval is [54.2, 91.7]. DPO moved each
  finance rule by one row (22/24 to 21/24 and 18/24 to 19/24); no change
  is claimed ([eval-results.md](eval-results.md), Limits).
- *Grounding is not truth.* `ungrounded_numbers` fails a correct figure the
  reference does not state. With the reference removed as a source, 105 of
  the 283 finance references (37.1%) fail it
  (`outputs/rubric_references/v1/summary.json`, commit `723c41a`;
  [decisions.md](decisions.md), "Class finance rows by the prompt"). That
  is why it was logged but never used to reject a DPO sample.

**Why a model judge is the next step.** Every gap above is a judgement
about meaning: is the answer right, does it do all of what was asked, is a
shorter answer a better one. A pairwise judge with position swap was
designed for this (decisions.md, "Score answers with a rubric and a
pairwise judge") and dropped by the owner to keep cost inside the existing
subscription: about 1,200 API calls, $1 to $5 (decisions.md, "Evaluate
without a model judge"). Its absence is the largest stated gap in this
write-up. Without it, the claim is "DPO stops and loops less", not "DPO
answers better".

## At much larger scale

Measured on the laptop, SFT peaked at 8.10 GiB and DPO at 13.19 GiB of
driver memory on a 0.5B model (`outputs/sft/summary.json` and
`outputs/dpo/summary.json`, `peak_memory_gb`, commits `76739e6` and
`0263788`). The base weights are 0.92 GiB of that. The rest is the
activations, the 152k-wide logits, and the allocator's cache; DPO costs
more because it runs four sequences of up to 768 tokens per micro-step
instead of two of 512, not because of the reference model, which costs
nothing here since it is the policy with its adapter switched off
([memory-and-scale.md](memory-and-scale.md), Sections 1 to 3).

At 70B the picture inverts. Full fine-tuning with mixed-precision AdamW
holds 16 bytes per parameter, 1.12 TB, so the state has to be sharded
across every GPU (ZeRO-3 or FSDP full shard, 17.5 GB per GPU over 64) and
the weight all-gathers cross the slow inter-node links. With LoRA the
optimiser state and base gradients disappear, the base can be sharded
within one node, and the cross-node traffic shrinks to a 0.4 GB adapter
all-reduce per step. The page's choice for SFT at 70B is LoRA with FSDP
within a node and data parallelism across nodes, at about 36 GB per 80 GB
GPU; for DPO the same, with precomputed reference log-probs or the
adapter-off reference. All of that is arithmetic with stated assumptions,
not measurement ([memory-and-scale.md](memory-and-scale.md), Section 4).

## Limits

- **One seed, one run per stage.** Every number above is from a single run
  decoded once, greedily. Small differences are not evidence.
- **200 prompts.** The rubric intervals are several points wide: SFT's
  overall pass rate is [66.5, 79.0] and DPO's [84.5, 93.0]. They are not
  paired, so overlapping intervals do not mean no difference.
- **A small DPO set.** 105 pairs, 104 of them for repetition, 18 optimiser
  steps. The held-out pairs come from SFT training prompts, so their loss
  of 0.058 and accuracy of 1.0 are a fit check, not a result
  ([dpo-results.md](dpo-results.md), Limits).
- **Length is confounded with quality.** Rejected answers were 2.9 times
  longer than chosen. DPO's gains in `repetition` and `Overall` partly
  reflect length.
- **The data is mostly not finance.** After de-duplication, 283 of the
  2,200 training and eval references (12.9%) are finance rows
  (`outputs/rubric_references/v1/summary.json`; [decisions.md](decisions.md),
  "Class finance rows by the prompt"). finance-alpaca's finance rows were
  heavily duplicated, so the model was mostly trained and scored on general
  Alpaca instructions, and the finance rules rest on 24 eval rows.
- **No model judge.** Nothing scores correctness, completeness or
  helpfulness. See "Evaluating without a benchmark".
- **Batch-size-dependent decoding.** The answers are not bit-identical to
  one-at-a-time decoding; every system was decoded the same way.
- **Contended timings.** Wall times were measured while another project's
  GPU jobs ran on the same machine, so no time in the docs is the cost of a
  stage.
- **Approximate peaks.** Memory is sampled once per step, so a spike inside
  a step is missed. The true peaks may be higher than 8.10 and 13.19 GiB.
