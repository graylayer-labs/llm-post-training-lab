# Part 1: LoRA SFT results

What LoRA supervised fine-tuning did to `Qwen/Qwen2.5-0.5B` base on finance
Q&A, what broke on the way, and where the memory went. Issue
[#1](https://github.com/graylayer-labs/llm-post-training-lab/issues/1).
The reasons behind each setting are in [decisions.md](decisions.md).

## How these numbers were made

The quotable run is the re-run for
[#12](https://github.com/graylayer-labs/llm-post-training-lab/issues/12).

- Commit `76739e6` on `main`, clean tree, `scratch` false
  (`outputs/sft/summary.json`, `provenance`).
- Config `configs/sft.yaml`: 2,000 training rows, 200 held-out rows, seed 0,
  one epoch, 125 optimiser steps, effective batch 16 (2 × 8).
- Run directory `outputs/sft/`. The held-out rows are the 200
  de-duplicated prompts in `outputs/sft/eval_rows.json`. De-duplication
  dropped 7,587 rows (`data_stats.duplicates_dropped`).
- Apple M4, 24 GB, MPS. torch 2.14.1, transformers 5.18.0, trl 1.14.1,
  peft 0.21.2.
- One seed per run. Small differences between runs are not evidence.
- The wall time was measured while another project's GPU jobs were running,
  so it is contended and not comparable with Part 1's 854 s.
- Checkpoints were saved every 25 steps and removed after the run completed
  ([#21](https://github.com/graylayer-labs/llm-post-training-lab/issues/21)).
- `train_loss` is the mean of the logged step losses, not the Trainer's own
  figure. See the checkpointing section of [decisions.md](decisions.md).
- `outputs/` is gitignored, so these files are local only.

The Part 1 runs (v1 and v2 below) are history. v2 came from a dirty working
tree just before commit `4c35d55`, so its code cannot be traced. Its split
also overlapped training: 4 of its 200 held-out prompts appeared among the
2,000 training rows, because the dataset repeats prompts and the split did
not remove them. `make_splits` now de-duplicates by prompt before shuffling
([#15](https://github.com/graylayer-labs/llm-post-training-lab/issues/15)).
The re-run draws different rows, so its losses are not comparable with
Part 1's. The Part 1 run is kept at
`outputs/sft_part1_overlapping_split/` (`summary.json`, `generations.json`,
`run.log`); v1's logs are in `outputs/sft_v1_lora_only/`.

```bash
uv run lab sft --config configs/sft.yaml
uv run python tools/stop_token_probe.py outputs/sft
```

## Runs

| | v1: LoRA only (Part 1) | v2: LoRA + chat-token rows (Part 1) | Re-run (#12, quotable) |
|---|---|---|---|
| Split | overlapping | overlapping | de-duplicated, disjoint |
| Tree | dirty | dirty | clean, commit `76739e6` |
| Trainable parameters | 8,798,208 | 8,800,000 | 8,800,000 |
| Loss | `chunked_nll` (TRL default) | `nll` | `nll` |
| Micro-batch × accumulation | 4 × 4 | 2 × 8 | 2 × 8 |
| Eval loss before | 2.302 | 2.283 | 2.169 |
| Eval loss after | 2.024 | 1.831 | 1.714 |
| Train loss | 2.171 (epoch mean) | 2.075 (epoch mean) | 1.873 (mean of logged step losses) |
| Wall time | 857 s | 854 s | 1,282.7 s (contended) |
| MPS memory at end of run | 3.71 GB | 5.16 GB | 6.0 GB |
| Peak MPS memory, sampled per step | not measured | 8.53 GB | 8.10 GB |
| Stop token, 5 samples | 0 of 5 | 4 of 5 | not sampled; see the probe below |

Notes on reading the table:

- Compare each run's before and after, not losses across runs. Each run
  scores a different 200 rows, and v1 and v2 differ in how the eval batch
  size follows the micro-batch, which most likely changes how the
  per-batch losses are averaged.
- On the 200 de-duplicated held-out answers, eval loss fell from 2.169 to
  1.714. v2 fell from 2.283 to 1.831 on its overlapping rows. The drops are
  0.455 and 0.452, so the clean re-run reproduces the size of the Part 1
  effect. One seed each; no interval.
- The re-run's wall time is 1,282.7 s against 854 s, but the re-run shared
  the GPU with another project's jobs. Do not read it as the cost of
  the de-duplicated split or of checkpointing.
- v1 recorded memory once, at the end of the run. Its 3.71 GB is end-of-run
  memory, not a peak.
- The 1,792 extra parameters in v2 and the re-run are the two embedding rows
  of width 896.

## The stop-token bug

**Symptom.** After v1, the SFT model ended none of its 5 sample answers. Every
one ran to the 256-token limit, mostly by repeating a sentence.

**Cause.** The chat template closes each assistant turn with `<|im_end|>`.
The base model has barely seen that token. LoRA adapts the attention and MLP
projections only. Qwen2.5-0.5B ties its input embedding to its output layer,
and LoRA does not touch either, so the logit row for `<|im_end|>` could not
learn much.

**Measured and saved.** `tools/stop_token_probe.py` feeds each reference
answer to the model, up to but not including its final `<|im_end|>`, and
records the probability and rank of `<|im_end|>` as the next token, out of
152,000 tokens. Run at commit `723c41a`, clean, on the re-run adapter and
the base model, over all 200 held-out rows (200 of 200 probed). Saved in
`outputs/sft/stop_token_probe.json`.

| | Base | SFT (re-run) |
|---|---|---|
| Median probability | 2.065e-09 | 0.7417 |
| Mean probability | 9.548e-07 | 0.6335 |
| Median rank | 123,031 | 1 |
| Share of rows where `<|im_end|>` ranks first | 0.00 | 0.795 |

The Part 1 text quoted two figures from an unsaved check: a base rank of
"about 115,000th" and a probability of "0.70" after the v2 fix. They were
close. The saved base median rank is 123,031. The 0.70 was probably a mean
rather than a median (the saved SFT mean is 0.6335 and the median 0.7417),
but it was never saved and came from the Part 1 adapter on other rows, so
which statistic it was is not known. Use the saved figures only.

**Fix.** PEFT's `trainable_token_indices` trains the `<|im_start|>` and
`<|im_end|>` embedding rows in full. With tied weights that also updates
their output rows. This needed a second change. TRL's default
`chunked_nll` loss reads `lm_head.weight` directly and skips PEFT's wrapper,
so the rows got no gradient. The run now uses `loss_type="nll"`.

**Result.** In Part 1, 4 of 5 SFT answers ended on `<|im_end|>`, against 0 of
5 for v1. On the re-run adapter the probe puts `<|im_end|>` first in 159 of
200 held-out reference positions (share 0.795), against none for the base
model. That is a next-token measure on reference text, not a count of
generated answers that stop. The eval harness
([#3](https://github.com/graylayer-labs/llm-post-training-lab/issues/3))
will measure stopping on generated answers.

## Out of memory, then a cache leak

The plain `nll` loss holds the full batch × sequence × 152k-vocab logits. Two
attempts at the v2 setup crashed before the cache fix. Logs are in
`outputs/sft_v1_lora_only/`.

| Attempt | Died at step | Tensors allocated | Other driver allocations | Limit |
|---|---|---|---|---|
| Micro-batch 4 | 11 of 125 | 6.86 GiB | 22.80 GiB | 30.19 GiB |
| Micro-batch 2 | about 20 of 125 | 2.84 GiB | 27.34 GiB | 30.19 GiB |

Halving the batch did not fix it. At batch 2 the live tensors were under
3 GiB, while the driver held more than 27 GiB besides. That points to the
allocator cache, not the model.

**Cause.** Each batch has a different sequence length, so each step asks for
logit buffers of a new shape. The MPS caching allocator kept the old blocks
instead of reusing them, and driver memory grew until the process died.

**Fix.** `ReleaseCacheCallback` calls `torch.mps.empty_cache()` after every
optimiser step. `PeakMemoryCallback` samples driver memory just before the
release, because MPS has no `max_memory_allocated`. The final Part 1 run
completed with a sampled peak of 8.53 GB.

## Failures on the way

The first attempt at the re-run was killed at step 18 of 125 by another
agent's pattern kill (`pkill -f`), which matched this run's process. The log
is at `outputs/failed/sft-killed-step18/run.log`. Nothing had been saved, so
the run started again from step 0. That led to checkpointing and
`lab sft --resume`
([#21](https://github.com/graylayer-labs/llm-post-training-lab/issues/21)),
and to a machine-wide guard against pattern kills. The re-run above
completed with checkpoints on.

## Base vs SFT answers

These answers are from the Part 1 run (v2), on its overlapping split, not
from the re-run. Greedy decoding, 256 new tokens at most, first 5 held-out
prompts of that split. Answers are shortened here. Full text is in
`outputs/sft_part1_overlapping_split/generations.json`. The eval harness
([#3](https://github.com/graylayer-labs/llm-post-training-lab/issues/3))
will give quotable answers.

| Prompt | Base | SFT (v2) |
|---|---|---|
| Should I get cash from credit card at 0% for 8 months and put it on loans? | 37 tokens, stops. "Yes, you should... This will help you save money and pay off your debts faster." | 256 tokens, no stop. "I would not do that. I would put the money in a savings account..." then repeats one sentence. |
| Buying shares in employer's company during IPO | 256 tokens, no stop. Repeats "You can earn a bonus by buying shares in your own company during IPO." | 71 tokens, stops. Answers in a forum style but contradicts itself. |
| Create a computer program for comparing two strings. | 256 tokens, no stop. Echoes the prompt with stray Cyrillic tokens. | 52 tokens, stops. A correct Python function. |
| Create a character sketch... using 7-10 sentences. | 256 tokens, no stop. | 67 tokens, stops. Four sentences, short of the 7 asked for. |
| Compare and contrast emotional intelligence and cognitive intelligence. | 255 tokens, stops. Fluent but claims EI is measured by the Myers-Briggs test. | 66 tokens, stops. Short and accurate. |

What changed:

- **Stopping and length.** SFT answers end on the stop token and are much
  shorter. This is the clearest effect.
- **Register.** Finance answers take on the first-person forum style of the
  training data ("I would not do that").
- **Not fixed.** Repetition loops remain, and factual quality is mixed. Five
  samples cannot say whether SFT improved correctness. That is Part 3's job.

## LoRA vs full fine-tune memory

`tools/memory_lora_vs_full.py`, 8 optimiser steps at batch 2, no
accumulation, `nll` loss unless stated. Each mode ran in its own process.
Rerun on 2026-10-03 from commit `c81b784`.

| Mode | Trainable parameters | Peak MPS memory | Seconds per step |
|---|---|---|---|
| LoRA r=16, `chunked_nll`, no token rows | 8,798,208 | 3.75 GB | 0.81 |
| LoRA r=16 (two runs) | 8,800,000 | 4.56 / 4.70 GB | 0.89 / 0.88 |
| LoRA r=64 | 35,194,624 | 5.28 GB | 0.94 |
| Full fine-tune | 494,032,768 | 6.59 GB | 1.60 |

- Full fine-tuning trains 56 times the parameters of LoRA r=16 for about
  1.4 times the memory. At this model size the activations and the 152k-vocab
  logits are a large share of memory, so cutting trainable parameters saves
  less than the parameter count suggests.
- Moving from `chunked_nll` to `nll` with the two token rows cost about 0.8
  to 0.95 GB at batch 2. Nearly all of that is the full logits.
- Two runs of the same r=16 setting differ by 0.14 GB. Read differences of
  that size as noise.
- The model is loaded in bf16, so the full fine-tune keeps its weights,
  gradients and AdamW state in bf16. A standard mixed-precision full
  fine-tune with fp32 master weights would need more.
- The probe trains on 16 short rows. Its peaks are not comparable to the
  8.53 GB of the full run, which also covers evaluation and longer batches.

Raw output is in `outputs/memory_probe/results.jsonl`.

## Open items

- **Repetition loops.** One SFT answer of five still loops until the token
  limit. The rubric (#11) marks such answers as failing `repetition`, so DPO
  may reduce them. A repetition penalty at decode time is another option, not yet tried.
- **One unexplained kill.** One run ended with exit code 137 (killed by the
  system) and no Python error. It has not happened again and no log was
  kept, so the cause is unknown.
- **Approximate peak memory on MPS.** The peak is sampled once per optimiser
  step and at evaluation. A spike inside a step, between samples, is missed.
  The true peak may be higher than the 8.10 GB of the re-run.
- **No answers from the re-run yet.** The sample answers above are from
  Part 1. The re-run adapter has a saved loss and a saved probe, but no
  generated answers until the eval harness runs it.
