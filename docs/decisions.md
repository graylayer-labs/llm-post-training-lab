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
- **Result:** `outputs/rubric_references/v1/summary.json` (commit
  `34f850f`, from `tools/rubric_on_references.py`) covers the 2,200 training
  and eval references. 514 (23.4%) are finance rows. The references pass
  98.4% overall, `domain_terms` 96.5% and `repetition` 99.2%.
- **Limit:** `ungrounded_numbers` checks grounding, not truth. A correct
  figure that the reference does not state fails it, for example "$52.50"
  for $50 after a year at 5%. With the reference removed as a number source,
  193 of the 514 finance references (37.5%) fail it. An earlier, narrower
  classifier gave 38.2%, measured but not saved. A good answer that cites
  its own figures is therefore often marked ungrounded.

### Score answers with a rubric and a pairwise judge

*2026-10-03*, **Planned**

- **Chose:** a rule-based rubric plus a pairwise Claude judge with position
  swap, alongside perplexity and ROUGE-L. The rubric checks format, clean
  stopping, domain terms and numbers not found in the prompt or reference.
- **Over:** perplexity and ROUGE-L alone, or a public benchmark.
- **Why:** The task has no benchmark, and loss metrics miss the failures that
  matter here: answers that never stop, repeat themselves or invent figures.
  Each judged pair is sent in both orders. An order-dependent verdict counts
  as inconsistent, not a win. See #3.

### Build the DPO preference pairs from the project's own data

*2026-10-03*, **Planned**

- **Chose:** a home-made preference set for DPO. Chosen is the reference
  answer, and rejected is an SFT sample that fails the rubric.
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
