# Part 1: LoRA SFT results

What LoRA supervised fine-tuning did to `Qwen/Qwen2.5-0.5B` base on finance
Q&A, what broke on the way, and where the memory went. Issue
[#1](https://github.com/graylayer-labs/llm-post-training-lab/issues/1).
The reasons behind each setting are in [decisions.md](decisions.md).

## How these numbers were made

- Config `configs/sft.yaml`: 2,000 training rows, 200 held-out rows, seed 0,
  one epoch, 125 optimiser steps, effective batch 16.
- Apple M4, 24 GB, MPS. torch 2.14.1, transformers 5.18.0, trl 1.14.1,
  peft 0.21.2.
- One seed per run. Small differences between runs are not evidence.
- The final run was made on the working tree just before commit `4c35d55`,
  not from a clean commit.
- Source files: `outputs/sft/` and `outputs/sft_v1_lora_only/`
  (`summary.json`, `run.log`, `generations.json`). `outputs/` is gitignored,
  so these files are local only.

```bash
uv run lab sft --config configs/sft.yaml
uv run python tools/compare_generations.py outputs/sft -n 5
```

## Runs

| | v1: LoRA only | v2: LoRA + chat-token rows (final) |
|---|---|---|
| Trainable parameters | 8,798,208 | 8,800,000 |
| Loss | `chunked_nll` (TRL default) | `nll` |
| Micro-batch × accumulation | 4 × 4 | 2 × 8 |
| Eval loss before | 2.302 | 2.283 |
| Eval loss after | 2.024 | 1.831 |
| Train loss, epoch mean | 2.171 | 2.075 |
| Wall time | 857 s | 854 s |
| MPS memory at end of run | 3.71 GB | 5.16 GB |
| Peak MPS memory, sampled per step | not measured | 8.53 GB |
| SFT samples ending on a stop token | 0 of 5 | 4 of 5 |

Notes on reading the table:

- Both runs score the same 200 eval rows. The "before" losses differ because
  the eval batch size follows the micro-batch, which most likely changes how
  the per-batch losses are averaged. Compare each run's before and after, not
  losses across runs.
- v1 recorded memory once, at the end of the run. Its 3.71 GB is end-of-run
  memory, not a peak. The two memory rows therefore do not show that v2 needs
  more than twice the memory.
- The 1,792 extra parameters in v2 are the two embedding rows of width 896.

## The stop-token bug

**Symptom.** After v1, the SFT model ended none of its 5 sample answers. Every
one ran to the 256-token limit, mostly by repeating a sentence.

**Cause.** The chat template closes each assistant turn with `<|im_end|>`.
The base model has barely seen that token. LoRA adapts the attention and MLP
projections only. Qwen2.5-0.5B ties its input embedding to its output layer,
and LoRA does not touch either, so the logit row for `<|im_end|>` could not
learn much.

**Measured but not saved.** At the end of a reference answer, the base model
ranked `<|im_end|>` about 115,000th of 152,000 tokens. Its probability at
that point was 0.00 for the base model and 0.70 after the v2 fix. Neither number
was written to a file. Treat them as indicative until a script records them.

**Fix.** PEFT's `trainable_token_indices` trains the `<|im_start|>` and
`<|im_end|>` embedding rows in full. With tied weights that also updates
their output rows. This needed a second change. TRL's default
`chunked_nll` loss reads `lm_head.weight` directly and skips PEFT's wrapper,
so the rows got no gradient. The run now uses `loss_type="nll"`.

**Result.** 4 of 5 SFT answers end on `<|im_end|>`, against 0 of 5 for v1.

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
release, because MPS has no `max_memory_allocated`. The final run completed
with a sampled peak of 8.53 GB.

## Base vs SFT answers

Greedy decoding, 256 new tokens at most, first 5 held-out prompts. Answers
are shortened here. Full text is in `outputs/sft/generations.json`.

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
  limit. The planned Part 2 rubric will mark such answers as rejected, so
  DPO may reduce them. A repetition penalty at decode time is another option, not yet tried.
- **One unexplained kill.** One run ended with exit code 137 (killed by the
  system) and no Python error. It has not happened again and no log was
  kept, so the cause is unknown.
- **Approximate peak memory on MPS.** The peak is sampled once per optimiser
  step and at evaluation. A spike inside a step, between samples, is missed.
  The true peak may be higher than 8.53 GB.
- **Unsaved measurements.** The `<|im_end|>` rank and probability figures
  above need a script that writes them to a file.
