# Part 3: evaluation results

Base, SFT and DPO scored on the same 200 held-out prompts: what each stage
changed, and how far the numbers can be trusted. Issue
[#3](https://github.com/graylayer-labs/llm-post-training-lab/issues/3); the
harness came in
[#23](https://github.com/graylayer-labs/llm-post-training-lab/issues/23).
The reasons behind each setting are in [decisions.md](decisions.md).

This is one run with one seed, scored by perplexity, ROUGE-L and a rule-based
rubric. None of those says whether an answer is correct. A blind check of
50 of the prompts, graded by two model graders, does; it is in "Blind
correctness check" below
([#35](https://github.com/graylayer-labs/llm-post-training-lab/issues/35)).
See Limits.

## How these numbers were made

From `outputs/eval/results.json` unless another file is named.

- Command `uv run lab eval --config configs/eval.yaml`. The config is copied
  into `results.json` under `config`.
- Commit `76c9086` on `main`, clean tree, `scratch` false (`provenance`).
- Apple M4, 24 GB, MPS. torch 2.14.1, transformers 5.18.0, trl 1.14.1,
  peft 0.21.2.
- Prompts: the 200 held-out rows in `outputs/sft/eval_rows.json`, the same
  rows the SFT run held out. Prompt-set sha256 `69001e40…` over the sorted
  prompt keys (`prompt_set.sha256`). File sha256 `197ddf38…`
  (`prompt_set.eval_rows_file_sha256`). See decisions.md, "Read the
  held-out rows from the SFT run's saved file".
- Generation, the same for every system: greedy (`do_sample` false), at most
  384 new tokens, batch size 16, seed 0, stopping on token ids 151645
  (`<|im_end|>`) and 151643 (`generation`, `systems[].generation_settings`).
- Every system has the same chat template (sha256 `44d5f08f…`), tokenizer
  (`237cc6d1…`), base-model revision (`060db649…`), dtype bf16 and
  first-prompt token ids (`35cdc268…`). These are recorded per system, so a
  mismatch would show.

| System | Adapter | Adapter sha256 | Loaded as |
|---|---|---|---|
| base | none | none | `Qwen/Qwen2.5-0.5B` |
| sft | `outputs/sft/adapter` | `ade0646e…` | base + SFT LoRA and chat-token rows |
| dpo | `outputs/dpo/adapter` | `ff85f0c7…` | merge(base, SFT adapter) + DPO LoRA |

- **DPO loading.** DPO's LoRA was trained on the merged SFT model (see
  [dpo-results.md](dpo-results.md), "Choosing the reference model"). The
  config therefore gives the DPO system `base_adapter:
  outputs/sft/adapter`. The harness merges that adapter into the base
  weights with the same function DPO training used, then loads the DPO LoRA
  on top. It checks the SFT adapter's sha256 against the `base_adapter.json`
  that DPO saved beside its adapter, so a wrong base would fail rather than
  score an untrained model. `base_adapter_sha256` for DPO is `ade0646e…`,
  the SFT adapter.
- No generation came from the cache: `rows_generated` is 200 and
  `rows_from_cache` is 0 for every system.
- Wall time 1,562.8 s in total: base 432.0 s, SFT 868.7 s, DPO 262.0 s
  (`wall_seconds`). Another project's GPU jobs were running at the same
  time. The lead saw them in the process list; the run files do not record
  them. Do not read the times as the cost of decoding.
- The DPO adapter came from the run at commit `0263788`
  (`outputs/dpo/summary.json`), trained on the 105 pairs in
  `outputs/pairs/manifest.json`. None of the 200 eval prompts is among the
  pair prompts (`split_check.pair_prompts_in_eval` 0).
- `outputs/` is gitignored, so these files are local only.

## The table

Copied from `outputs/eval/results.md`.

| System | Perplexity | ROUGE-L F1 | format | clean_stop | repetition | domain_terms | ungrounded_numbers | Overall | Mean new tokens | Stopped % |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 8.910 | 0.153 | 99.5 (199/200) [98.5, 100.0] | 39.5 (79/200) [33.0, 46.5] | 40.5 (81/200) [34.0, 47.5] | 95.8 (23/24) [87.5, 100.0] | 87.5 (21/24) [74.9, 100.0] | 35.0 (70/200) [28.5, 41.5] | 270.0 | 39.5 |
| sft | 6.245 | 0.297 | 100.0 (200/200) [100.0, 100.0] | 86.5 (173/200) [81.5, 91.0] | 75.0 (150/200) [69.0, 81.0] | 91.7 (22/24) [79.2, 100.0] | 75.0 (18/24) [54.2, 91.7] | 73.0 (146/200) [66.5, 79.0] | 111.6 | 86.5 |
| dpo | 6.555 | 0.296 | 100.0 (200/200) [100.0, 100.0] | 100.0 (200/200) [100.0, 100.0] | 92.5 (185/200) [88.5, 96.0] | 87.5 (21/24) [70.8, 100.0] | 79.2 (19/24) [62.5, 95.8] | 89.0 (178/200) [84.5, 93.0] | 51.9 | 100.0 |

### How to read it

- **Perplexity.** How surprised the model is by the dataset's *reference*
  answers. It is computed on the answer tokens only, with the prompt masked
  as in SFT, `<|im_end|>` included. It is exp of the mean NLL over all
  16,180 answer tokens (`perplexity_answer_tokens`, the same for every
  system, 0 rows truncated). Lower means the reference text is more likely
  under the model. It does not look at the model's own answers.
- **ROUGE-L F1.** Longest-common-subsequence overlap between the model's
  answer and the reference, averaged over the 200 prompts. rouge-score
  lowercases, drops punctuation and stems words.
- **Rubric columns.** The rule-based rubric from
  [#11](https://github.com/graylayer-labs/llm-post-training-lab/issues/11),
  version 1. Each cell is pass % (passes/rows the rule applies to) [95%
  interval].
  - `format`: non-empty, no leaked chat markers or role lines.
  - `clean_stop`: the answer ended on a stop token before the 384-token
    limit.
  - `repetition`: no 4-word phrase four or more times, and no sentence of
    four or more words twice.
  - `domain_terms`, `ungrounded_numbers`: finance rows only, which is why
    n is 24. The first checks the answer uses a finance term. The second
    checks every figure in the answer appears in the prompt or the
    reference; it checks grounding, not truth.
  - `Overall`: the answer passes every rule that applies to it.
- **Mean new tokens** and **Stopped %** are over all 200 answers.
- **Intervals** are a 95% percentile bootstrap, 1,000 resamples, seed 0,
  computed for each system on its own. They are not paired. Two systems
  answer the *same* prompts, so a paired comparison would be tighter;
  overlapping intervals here do not mean there is no difference.

## What each stage changed

The per-prompt comparisons below come from
`outputs/eval/analysis.json`, written by `tools/compare_systems.py` at
commit `b872427` on a clean tree (`provenance` in the file, `scratch`
false). The tool reads only the saved generations and the eval rows, on
CPU. It re-scores every answer with the same rubric. Its totals match
`results.json`: overall passes 70, 146 and 178; ROUGE-L 0.153, 0.297 and
0.296.

Command:

```bash
uv run python tools/compare_systems.py outputs/eval/generations \
    --eval-rows outputs/sft/eval_rows.json --out outputs/eval/analysis.json \
    --show 185 2 12 50 29 193
```

"Stopped" below means stopped cleanly: a stop token before the 384-token
limit. In this run every answer that produced a stop token did so before the
limit.

### SFT against base

From `results.md` unless `analysis.json` is named.

- **Format.** Already near-perfect in base, 99.5% (199/200), and 100% after
  SFT.
- **Stopping.** The main change. Base stops on 39.5% of prompts, SFT on
  86.5%. 121 base answers run to the 384-token limit, against 27 for SFT
  (`analysis.json`, `systems.*.lengths.at_limit`). This is the stop-token
  fix from Part 1 at work: see [sft-results.md](sft-results.md).
- **Repetition.** Pass rate 40.5% to 75.0%. Many base loops repeat the
  prompt with a junk token, for example "Generate a list of tips for
  creating an effective presentation.acco" over and over
  (`generations/base.jsonl`, index 12).
- **Overall rubric.** 35.0% to 73.0%. When base does stop, it passes 70 of
  79 (88.6%) (`analysis.json`, `systems.base.rubric_overall_when_stopped`).
  Base's failures are mostly failures to stop.
- **Perplexity.** 8.910 to 6.245. The references are more likely under
  SFT, which was trained on rows like them.
- **ROUGE-L.** 0.153 to 0.297, nearly double.
- **Length.** Mean new tokens 270.0 to 111.6; median 384 to 66
  (`analysis.json`, `lengths.median`).
- **Not all one way.** SFT fixes 88 prompts that base failed and breaks 12
  that base passed. All 12 now fail `repetition`, 8 of them also
  `clean_stop` (`analysis.json`, `pairs."base->sft".fixed` and
  `.regressed`). On the 69 prompts where both stopped, SFT is shorter (63.6
  against 96.3 mean tokens) and closer to the reference (ROUGE-L 0.322
  against 0.284) (`pairs."base->sft".both_stopped`).

### DPO against SFT

From `results.md` unless `analysis.json` is named.

- **Stopping.** 86.5% to 100%. Every DPO answer stops, the longest at 375
  tokens; none reaches the limit (`analysis.json`,
  `systems.dpo.lengths.max` and `.at_limit`).
- **Repetition.** Pass rate 75.0% to 92.5%. 15 DPO answers still repeat.
- **Overall rubric.** 73.0% (146/200) to 89.0% (178/200). DPO fixes 36
  prompts and breaks 4 (`analysis.json`, `pairs."sft->dpo".fixed` and
  `.regressed`). Three of the 4 fail `repetition`, one fails
  `ungrounded_numbers`.
- **ROUGE-L.** Flat: 0.297 to 0.296.
- **Perplexity.** Slightly up, 6.245 to 6.555. The references are a little
  less likely under DPO than under SFT. See Limits for what that means.
- **Length.** Mean new tokens 111.6 to 51.9, roughly halved.
- **Finance rules.** `domain_terms` 22/24 to 21/24 and
  `ungrounded_numbers` 18/24 to 19/24. Each is one row on n = 24, inside
  wide intervals. No change is claimed.

### Is it just "shorter"?

The pairs gave DPO a reason to learn length. Rejected answers averaged 356.4
tokens and chosen ones 124.1, about 2.9 times longer
([dpo-results.md](dpo-results.md), Limits). So DPO may have learned "shorter
is better" or "stop sooner" rather than "do not loop". The generations can
test part of this. All figures in this section are from
`outputs/eval/analysis.json`.

**Length distributions** (new tokens, all 200 answers, `systems.*.lengths`):

| System | Mean | Median | p10 | p90 | Max | At the limit | Under 20 |
|---|---|---|---|---|---|---|---|
| base | 270.0 | 384 | 30.8 | 384 | 384 | 121 | 8 |
| sft | 111.6 | 66 | 16 | 384 | 384 | 27 | 27 |
| dpo | 51.9 | 41 | 12 | 92.6 | 375 | 0 | 48 |

| New tokens | <20 | 20–49 | 50–99 | 100–199 | 200–383 | 384 |
|---|---|---|---|---|---|---|
| base | 8 | 18 | 20 | 25 | 8 | 121 |
| sft | 27 | 45 | 64 | 31 | 6 | 27 |
| dpo | 48 | 64 | 69 | 16 | 3 | 0 |

In words (whitespace-split, not tokens), the references average 63.3 words
(`inputs.reference_words_mean`) and DPO answers 41.7 (`systems.dpo.words_mean`).
SFT answers average 90.4 words, inflated by its 27 loops.

**DPO is shorter even where SFT already stopped.** DPO's answer is shorter
than SFT's on 144 of 200 prompts (`pairs."sft->dpo".b_shorter`). On the 173
prompts where both stopped, DPO averages 50.4 tokens against 69.1 for SFT,
with a median per-prompt ratio of 0.79 (`pairs."sft->dpo".both_stopped`). If
DPO had only learned to break loops, answers that already stopped would keep
their length. They did not. That fits the length confound.

**Rubric, stopped SFT answers against DPO.** SFT's answers that stopped pass
146 of 173 (84.4%) (`systems.sft.rubric_overall_when_stopped`). DPO passes
178 of 200 (89.0%), and 158 of the same 173 prompts (91.3%)
(`pairs."sft->dpo".both_stopped.rubric_overall`). DPO passes 32 more prompts
than SFT overall. 20 of them are among the 27 prompts where SFT never
stopped: DPO passes 20 of those 27 (`pairs."sft->dpo".a_did_not_stop`). The
other 12 are on prompts SFT already finished, 84.4% to 91.3%. A stopped SFT
answer can only fail `repetition` or the two finance rules (`format` is
100% for both), and the finance rules cover only 24 prompts. So this gain is
most likely in `repetition`; the saved analysis does not split it by rule.
That rule counts repeats, and a shorter answer has fewer chances to repeat,
so this gain cannot be told apart from length.

**ROUGE-L on like-for-like prompts.** On the 173 prompts where both stopped,
DPO scores 0.314 against SFT's 0.330
(`pairs."sft->dpo".both_stopped.rouge_l`). On the 27 where SFT never
stopped, DPO scores 0.180 against SFT's 0.087
(`pairs."sft->dpo".a_did_not_stop.rouge_l`). The flat headline ROUGE-L is
the weighted average of these two: DPO gains on the prompts where SFT looped and loses a
little on the prompts SFT already handled.

**Very short answers.** 48 DPO answers are under 20 tokens, against 27 for
SFT and 8 for base (`systems.*.lengths.short`). On those 48 prompts, SFT's
answers averaged 42.9 tokens; DPO's average 13.1. ROUGE-L is 0.377 for DPO
against 0.402 for SFT, and the rubric passes 47 of 48 for DPO against 42 of
48 for SFT (`pairs."sft->dpo".b_short`). Many short answers fit their
prompt. Two from `systems.dpo.short_examples`:

- "Edit this sentence to make it logically sound: The organization whose
  its aim was to eliminate poverty was successful." DPO, 11 tokens: "The
  organization that aimed to eliminate poverty was successful." ROUGE-L
  0.842.
- "List the three strands of sustainable development". DPO, 14 tokens: "The
  three strands of sustainable development are economic, social and
  environmental." ROUGE-L 0.720.

Others drop part of the request or are wrong, and the rubric does not see
it:

- "Given a mathematical expression, simplify it." Input `3x + 4xy + y - 2x
  - 3y`; the reference ends `= 4xy + x - 2y`. SFT looped for 384 tokens. DPO, 12
  tokens: "The simplified expression is 2x + 2y". Wrong, and it passes every
  rubric rule (`examples`, index 50).
- "Create a list of unique burger toppings." DPO, 46 tokens, repeats items
  ("Pickled onions", "Pickled jalapenos", "Pickled mushrooms" each twice)
  but stops before the repetition rule's threshold, so it passes. SFT's
  answer looped "BBQ sauce, Ketchup, Mustard, Sriracha" to the limit
  (`examples`, index 29).

**Reading.** DPO did what the pairs mostly taught: it stops every time and
loops far less. The evidence also fits "shorter" being part of what it
learned. Its answers are shorter on prompts SFT already finished, and
slightly further from the references there. This run cannot separate "do
not loop" from "be short": that would need pairs whose chosen and rejected
answers have similar lengths. Whether the answers also got more *correct*
is a separate question, taken up in "Blind correctness check".

### Three examples

From `outputs/eval/generations/{base,sft,dpo}.jsonl` with token counts,
ROUGE-L and rubric failures from `outputs/eval/analysis.json`, `examples`.
Answers are shortened with "…".

**1. DPO fixed a loop** (index 185). "Explain the concept of latent
variable."

- Base, 115 tokens, stopped, fails `repetition` (one sentence twice): "Latent
  variable is a variable that is not directly observable but is inferred
  from other variables. Latent variables are often used in psychological and
  social science research…"
- SFT, 384 tokens, never stopped, fails `clean_stop` and `repetition` ("is
  a variable that" 24 times), ROUGE-L 0.135: "Latent variable is a variable
  that is not directly observable but is inferred from other variables. It
  is a variable that is not directly measured but is inferred from other
  variables. It is a variable that is not directly observable…"
- DPO, 32 tokens, passes, ROUGE-L 0.354: "Latent variables are variables
  that are not directly observable, but are inferred from observable
  variables. They are used to explain the relationship between two or more
  variables."

The first sentence is the same in SFT and DPO. SFT then restates it until
the limit; DPO adds one sentence and stops.

**2. DPO got shorter and lost part of the task** (index 2). "Name and
describe four types of renewable energy."

- Base, 141 tokens, passes, ROUGE-L 0.195: a general paragraph on renewable
  energy that lists five sources.
- SFT, 87 tokens, stopped, fails `repetition` ("energy is generated by" 4
  times), ROUGE-L 0.457: "Four types of renewable energy are solar, wind,
  hydro, and geothermal. Solar energy is generated by the sun and is used to
  power homes and businesses. Wind energy is generated by the wind…"
- DPO, 17 tokens, passes, ROUGE-L 0.312: "Four types of renewable energy are
  solar, wind, hydro, and geothermal."

The rubric counts this as a DPO fix. It is worse: the prompt asks to
describe each type, SFT did, and DPO stops after naming them. SFT's
"repetition" is a sentence pattern, not a loop. Index 193 is the other
direction: DPO's 182-token answer to "How would you define the balance
between work and life?" drifts to a bank "balance sheet" and fails
`repetition`, where SFT's 64-token answer passed.

**3. Typical** (index 12). "Generate a list of tips for creating an
effective presentation."

- Base, 384 tokens, never stopped, ROUGE-L 0.041: the prompt repeated with
  a junk token, "Generate a list of tips for creating an effective
  presentation.acco / acco / …"
- SFT, 136 tokens, stopped, fails `repetition` ("use a clear and" 4 times),
  ROUGE-L 0.259: "1. Know your audience and tailor your presentation to
  their needs. 2. Use visuals to help illustrate your points. 3. Keep your
  presentation concise and to the point. …"
- DPO, 105 tokens, passes, ROUGE-L 0.296: "1. Know your audience and tailor
  your presentation to their needs. 2. Use visuals to help illustrate your
  points. 3. Keep your slides simple and easy to read. …"

The usual pattern: SFT made the answer usable, and DPO kept SFT's opening
and wrote a slightly shorter answer with less repeated wording.

## Blind correctness check

The rubric checks how an answer behaves, not whether it is right. Issue
[#35](https://github.com/graylayer-labs/llm-post-training-lab/issues/35)
grades 50 of the 200 held-out prompts for correctness, blind, with two
model graders. **The grading was done by models, not by a person.**

### Method

- **The sheet.** `tools/blind_grading.py sheet` (code from
  [#37](https://github.com/graylayer-labs/llm-post-training-lab/issues/37))
  picked 50 of the 200 prompts with seed 0 and wrote
  `outputs/grading/v1/sheet.jsonl` at commit `66b837b` on `main`, clean
  tree (`key.json`, `provenance`). Each line holds only the prompt index,
  the prompt, the reference answer and the three systems' answers from
  `outputs/eval/generations`, under labels A, B and C. The label order is
  shuffled per prompt by a seeded shuffle, so a label says nothing about the
  system. No token count, stop flag or system name is on the sheet. The
  label-to-system mapping and each answer's rubric result are kept apart,
  in `key.json`.
- **The graders.** Two Claude subagents on different models, one Opus and
  one Sonnet, each graded all 150 answers on its own, without the key or
  the other's grades. Their grades are `grades_opus.jsonl` and
  `grades_sonnet.jsonl`, one line per prompt and label, each with a reason.
- **The grades.** Each answer is graded against the reference as
  `correct`, `partly` or `wrong`. The graders' full instructions are in
  [grading-briefs.md](grading-briefs.md); the saved reasons show how the
  grades were applied. An answer whose right content
  sits inside a loop can be graded `partly`, so `partly` mixes "incomplete"
  with "right but looping" (see "Limits of the check").
- **Unblinding.** `tools/blind_grading.py unblind` validates every grade
  (each prompt and label exactly once, a known grade, a non-empty reason),
  maps labels back to systems with the key and writes `results.json` and
  `results.md`. It first ran at commit `66b837b` on `main` and was re-run
  at `9bc7756` on `main`, clean tree, with identical results apart from the
  provenance (`results.json`, `provenance`). Intervals are 95% percentile bootstrap,
  1,000 resamples, bootstrap seed 0. The paired differences resample
  prompts and keep each prompt's three answers together.

```bash
uv run python tools/blind_grading.py sheet \
    --gens-dir outputs/eval/generations --rows outputs/sft/eval_rows.json \
    --n 50 --seed 0 --out outputs/grading/v1
uv run python tools/blind_grading.py unblind --dir outputs/grading/v1 \
    --grades outputs/grading/v1/grades_opus.jsonl \
    --grades outputs/grading/v1/grades_sonnet.jsonl
```

### Results

From `outputs/grading/v1/results.md`. Each cell is rate (count/50) [95%
interval].

| System | Opus: correct | Opus: correct or partly | Sonnet: correct | Sonnet: correct or partly |
|---|---|---|---|---|
| base | 10.0% (5/50) [2.0, 20.0] | 30.0% (15/50) [16.0, 44.0] | 12.0% (6/50) [4.0, 22.0] | 28.0% (14/50) [16.0, 40.0] |
| sft | 30.0% (15/50) [18.0, 42.0] | 66.0% (33/50) [52.0, 78.0] | 30.0% (15/50) [18.0, 42.0] | 72.0% (36/50) [60.0, 84.0] |
| dpo | 44.0% (22/50) [30.0, 58.0] | 72.0% (36/50) [60.0, 84.0] | 48.0% (24/50) [34.0, 62.0] | 78.0% (39/50) [66.0, 88.0] |

Paired difference in the correct rate, same 50 prompts, 95% interval:

| Difference | Opus | Sonnet |
|---|---|---|
| sft − base | +20.0 points [+6.0, +34.0] | +18.0 points [+4.0, +32.0] |
| dpo − sft | +14.0 points [+0.0, +28.0] | +18.0 points [+6.0, +28.0] |

Both graders put the systems in the same order: base, then SFT, then DPO.
SFT's gain over base excludes zero for both. DPO's gain over SFT excludes
zero for Sonnet; for Opus the interval's lower end is exactly 0.0.

**Agreement.** Over all 150 graded answers the two graders gave the same
grade on 84.0%, Cohen's kappa 0.756 (`results.json`, `agreement`). They
agree well, but they are two models from one family, so agreement is not
evidence that either is right (see "Limits of the check").

### Passing the rubric is not being correct

Of the 50 prompts, the rubric passes 13 base answers, 35 SFT answers and 47
DPO answers (`results.md`, the n in each split). Among the answers that pass
the rubric, the share graded correct is low:

| System | Opus: correct among rubric passes | Sonnet: correct among rubric passes |
|---|---|---|
| base | 30.8% (4/13) [7.7, 61.5] | 38.5% (5/13) [15.4, 69.2] |
| sft | 34.3% (12/35) [20.0, 51.4] | 40.0% (14/35) [25.7, 57.1] |
| dpo | 46.8% (22/47) [31.9, 61.7] | 51.1% (24/47) [36.2, 66.0] |

So most SFT answers that pass every rubric rule are not graded correct. The
rubric measures stopping and looping; it is not a proxy for correctness.
Answers that fail the rubric are rarely correct: SFT 3/15 (Opus) and 1/15
(Sonnet), DPO 0/3 for both.

### Correct, or just not looping?

A loop is graded `wrong` or at best `partly`, and DPO loops far less than
SFT. So DPO's higher correct rate could come only from not looping. To
separate the two, the paired difference is repeated on only the prompts
where the rubric's overall rule passes *both* systems, so a looping or
truncated answer is out on both sides.

From `outputs/grading/v1/subset.json`, written by
`tools/blind_grading.py subset`. It was first computed at branch commit
`b3b5567` before merge, then re-run at `9bc7756` on `main`, clean tree,
with identical results apart from the provenance (`provenance` in the
file). Same
bootstrap settings as above.

| Both pass the rubric | n | Opus: correct, a vs b | Opus: b − a | Sonnet: correct, a vs b | Sonnet: b − a |
|---|---|---|---|---|---|
| dpo − sft | 35 | 12 vs 15 | +8.6 points [−8.6, +22.9] | 14 vs 18 | +11.4 points [+0.0, +25.7] |
| sft − base | 8 | 4 vs 3 | −12.5 points [−37.5, +0.0] | 5 vs 4 | −12.5 points [−37.5, +0.0] |

```bash
uv run python tools/blind_grading.py subset --dir outputs/grading/v1 \
    --grades outputs/grading/v1/grades_opus.jsonl \
    --grades outputs/grading/v1/grades_sonnet.jsonl \
    --discarded outputs/grading/v1/discarded/grades_sonnet_shortcut.jsonl
```

**Reading.** On the 35 prompts where neither SFT nor DPO fails the rubric
(every prompt SFT passes, DPO also passes), DPO is still ahead for both
graders, by 3 prompts (Opus) and 4 (Sonnet). The intervals reach zero for
both: −8.6 to +22.9 for Opus, and a lower end of exactly 0.0 for Sonnet.
So the point estimates say DPO's gain is not only "stopped looping", but 35
prompts are too few to show it. The rest of the gain is on the 15 prompts
where SFT's answer failed the rubric, mostly loops. For Opus, DPO has 7
more correct answers than SFT overall (22 against 15), 3 of them on the
both-pass prompts and so 4 on those 15; for Sonnet, 9 more (24 against
15), 4 and 5. That split is arithmetic on `results.json` and
`subset.json`. For SFT against base only 8 prompts pass the rubric for
both, too few to read; SFT's gain over base comes from prompts where base
failed the rubric, mostly by not stopping.

The rubric is not a clean "does not loop" filter: a short loop can pass it
(index 29 above), and it also checks format and, on finance rows, terms and
figures. The subset removes long loops and truncations, not every loop.

### A discarded grader attempt

The first Sonnet grading was thrown away. The grader reported that it had
judged mostly from the first ~300 characters of each answer and from the
answer's length, and had graded partly with a script rather than by
reading. That is a process failure: a grader that looks at length is
measuring the thing DPO changed most, not correctness. The lead session
discarded the file and ran a new Sonnet grader, told to read every answer
in full and grade by hand, in batches. The kept `grades_sonnet.jsonl` is
that rerun.

The discarded grades are kept at
`outputs/grading/v1/discarded/grades_sonnet_shortcut.jsonl` and feed no
result above. `subset.json` (`discarded`) only describes how they differ.
Its 150 reasons use 3 distinct strings, one per grade, such as "Wrong,
off-topic or degenerate.". It marks 6, 12 and 16 answers correct for base,
SFT and DPO, against 6, 15 and 24 in the rerun. It agrees with the Opus
grades on 78.7% (kappa 0.671) and with the rerun on 74.7% (kappa 0.613),
below the 84.0% (kappa 0.756) between the two kept graders.

### Limits of the check

- **50 prompts.** Intervals are wide, up to 28 points for one system's
  correct rate, and one prompt moves a rate by 2 points.
- **Model graders, not a person.** Both graders are Claude models. They
  may share biases with each other, so their agreement can overstate how
  reliable the grades are. Nobody checked the grades by hand.
- **Blind to the label, not to the style.** The sheet hides which system
  wrote an answer, but a 384-token loop or a 12-token reply is visible. A
  grader can see degeneracy and length, and DPO's answers are the shortest.
  The discarded attempt shows a grader can lean on exactly those cues.
- **One seed.** The answers come from one greedy decode of one run per
  stage.
- **`partly` mixes two things.** An answer with right content inside a loop
  can be graded `partly`, so "correct or partly" entangles correctness with
  looping. The `correct` rate is the cleaner measure, and the both-pass
  subset is the attempt to separate the two.
- **Unequal briefs.** The Sonnet rerun was given stricter process rules
  (read every answer in full, no length or scripted grading) than the Opus
  grader. Both briefs are in [grading-briefs.md](grading-briefs.md).

## Limits

- **One seed, one run.** Every system was decoded once, greedily. Small
  differences are not evidence.
- **200 prompts.** Even on all 200 prompts the intervals are several
  points wide: SFT's overall pass rate is [66.5, 79.0], DPO's [84.5,
  93.0].
- **Only 24 finance rows.** `domain_terms` and `ungrounded_numbers` apply
  to 24 of the 200 prompts. Their intervals are wide, for example
  [54.2, 91.7] for SFT on `ungrounded_numbers`, and a one-row change moves
  the rate by 4.2 points. The finance-specific rules say almost nothing
  here, and most prompts are general Alpaca instructions (see
  decisions.md on the 12.9% finance share).
- **ROUGE-L against forum-style references.** Many finance references are
  forum posts ("See Berkshire Hathaway Inc. (BRK-A)… welcome to SE"). A
  good answer in a different style scores low. ROUGE-L measures overlap
  with one reference, not quality.
- **No model judge on the 200.** The owner dropped the paid judge
  ([#3](https://github.com/graylayer-labs/llm-post-training-lab/issues/3),
  decisions.md, "Evaluate without a model judge"). Nothing in the table
  scores correctness or completeness. The rubric passed DPO's wrong algebra
  (index 50) and its half-answer on renewables (index 2). Correctness is
  checked only on 50 prompts, by model graders, in "Blind correctness
  check", with its own limits.
- **Batch size 16.** Answers were decoded in batches of 16 with left
  padding. In bf16 on MPS this flips near-tied tokens, so the answers are
  not bit-identical to one-at-a-time decoding. Every system was decoded the
  same way. See decisions.md, "Treat the generation batch size as a
  setting".
- **Perplexity is of the reference answers.** It measures how likely the
  dataset's answers are under each model, not how good the model's own
  answers are. DPO's rise from 6.245 to 6.555 means it moved slightly away
  from the reference distribution, as expected when a model is pushed
  towards shorter answers and away from its own loops. It does not show
  that DPO's answers got worse.
- **Rubric thresholds interact with length.** `repetition` counts repeats,
  so shorter answers pass it more easily, and a short loop can pass (index
  29). Gains in `repetition` and `Overall` partly reflect length.
- **Contended wall time.** See "How these numbers were made".
