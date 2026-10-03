# 2. Supervised fine-tuning

*Last updated 2026-10-03 from issues #12, #1*

## What we set out to do

Teach `Qwen/Qwen2.5-0.5B` to answer in the chat format, with a LoRA adapter,
on one laptop. Then re-run it cleanly, because the first run could not be
traced (chapter 4).

## What we found

The quotable run is the re-run for [#12][i12]: commit `76739e6` on `main`,
clean tree, `scratch` false (`outputs/sft/summary.json`). Config
`configs/sft.yaml`: 2,000 training rows, 200 held-out rows, seed 0, one
epoch, 125 optimiser steps, effective batch 16 (2 x 8)
([sft-results.md][res]). One seed, so small differences are not evidence.

| | Part 1 v2 (history) | Re-run (#12) |
|---|---|---|
| Split | overlapping, dirty tree | de-duplicated, clean `76739e6` |
| Eval loss before | 2.283 | 2.169 |
| Eval loss after | 1.831 | 1.714 |
| Peak MPS memory, sampled per step | 8.53 GB | 8.10 GB |
| Wall time | 854 s | 1,282.7 s (contended) |

Source for every cell: [the Runs table in sft-results.md][res]. v1 (LoRA
only) is in the same table: eval loss 2.302 to 2.024.

How to read it:

- Compare each run's before and after, not losses across runs. They score
  different 200 rows. The doc gives the drops as 0.455 (re-run) and 0.452
  (v2), so the clean run reproduces the size of the Part 1 effect
  ([sft-results.md][res]).
- The re-run's wall time shared the GPU with another project's jobs. It is
  not the cost of the de-duplicated split or of checkpointing
  ([sft-results.md][res]).
- The Part 1 runs stay as history, labelled overlapping split and dirty
  tree. The Part 1 run is kept at `outputs/sft_part1_overlapping_split/`.

## Decisions and why

- **LoRA on all seven projections** (q, k, v, o, gate, up, down), r=16,
  alpha=32: 8.8M trainable parameters, 1.78% of the model. The QLoRA paper
  found that adapting the MLP layers too was needed to match full
  fine-tuning ([decisions.md][lora]).
- **bf16 LoRA, not QLoRA.** bitsandbytes 4-bit is CUDA-only, and a 0.5B
  model in bf16 fits in 24 GB ([decisions.md][bf16]).
- **Completion-only loss.** Prompt tokens are masked and `<|im_end|>` stays
  in the target, so the model learns to answer and to stop. Qwen2.5's chat
  template has no `{% generation %}` markers, so rows go to TRL as
  prompt/completion pairs ([decisions.md][mask]).
- **Chat-token rows trained in full.** The `<|im_start|>` and `<|im_end|>`
  embedding rows are trained via PEFT `trainable_token_indices`, adding
  1,792 parameters (8,798,208 to 8,800,000) ([decisions.md][rows]). Why this
  was needed is chapter 3.
- **The MPS cache is emptied after every step.** Each batch has a different
  sequence length, so logit buffers change shape, and the MPS allocator kept
  old blocks. Two v2 attempts died at steps 11 and about 20 of 125 before
  the fix; the final Part 1 run completed ([sft-results.md][res]).

## LoRA vs full fine-tune memory

`tools/memory_lora_vs_full.py`, 8 optimiser steps at batch 2, each mode in
its own process, rerun 2026-10-03 from commit `c81b784`
([sft-results.md][res]).

| Mode | Trainable parameters | Peak MPS memory |
|---|---|---|
| LoRA r=16, `chunked_nll`, no token rows | 8,798,208 | 3.75 GB |
| LoRA r=16 (two runs) | 8,800,000 | 4.56 / 4.70 GB |
| LoRA r=64 | 35,194,624 | 5.28 GB |
| Full fine-tune | 494,032,768 | 6.59 GB |

The results doc reads this as: full fine-tuning trains 56 times the
parameters of LoRA r=16 for about 1.4 times the memory. At this size,
activations and the 152k-vocabulary logits are a large share, so cutting
parameters saves less than the count suggests. Two runs of the same setting
differ by 0.14 GB, so treat gaps that small as noise. The model is loaded in
bf16, so the full fine-tune keeps weights, gradients and AdamW state in bf16;
a mixed-precision fine-tune with fp32 master weights would need more. The
probe's peaks are not comparable with the 8.10 GB of the full run, which
also covers evaluation and longer batches.

## How to reproduce it

```bash
uv run lab sft --config configs/sft.yaml
uv run python tools/stop_token_probe.py outputs/sft
```

These are the commands from [sft-results.md][res]. Not re-run for this
chapter: a training run was using the GPU. The re-run took 1,282.7 s while
the GPU was contended; there is no uncontended time for the re-run.

## What we would do differently

Run the clean re-run first, from a clean commit with provenance on. The
Part 1 numbers survive only as history. Also, the re-run has no generated
answers yet; the sample answers in [sft-results.md][res] are from Part 1.
The evaluation chapters will fix that.

[i12]: https://github.com/graylayer-labs/llm-post-training-lab/issues/12
[res]: ../sft-results.md
[lora]: ../decisions.md#apply-lora-to-all-seven-projections
[bf16]: ../decisions.md#use-bf16-lora-on-apple-mps
[mask]: ../decisions.md#compute-the-loss-on-the-answer-only
[rows]: ../decisions.md#train-the-chat-token-embedding-rows-in-full
