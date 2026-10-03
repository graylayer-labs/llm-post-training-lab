# 3. The stop-token bug

*Last updated 2026-10-03 from issues #19, #12, #1*

## What we set out to do

Get the SFT model to end its answers. This was the most instructive failure
in the project: two separate causes hid each other, and the first fix
did not work.

## What we found

**Symptom.** After v1 (LoRA on the projections only), the SFT model ended
none of its 5 sample answers. Every one ran to the 256-token limit, mostly
by repeating a sentence ([sft-results.md][res]).

**Cause 1: LoRA cannot reach the token.** The chat template closes each
assistant turn with `<|im_end|>`, which the base model has barely seen.
LoRA adapts the attention and MLP projections only. Qwen2.5-0.5B ties its
input embedding to its output layer, and LoRA touches neither, so the
logit row for `<|im_end|>` could not learn much ([sft-results.md][res]).

**Fix 1, and why it was not enough.** PEFT's `trainable_token_indices`
trains the `<|im_start|>` and `<|im_end|>` embedding rows in full. With tied
weights that also updates their output rows ([sft-results.md][res]).

**Cause 2: the loss skipped the wrapper.** TRL's default `chunked_nll` loss
reads `lm_head.weight` directly and bypasses PEFT's wrapper, so the rows got
no gradient. The run now uses `loss_type="nll"` ([decisions.md][nll]). The
comment in `src/post_training/train/sft.py` says the same.

**The price.** The plain loss holds the full batch x sequence x 152k-vocab
logits. In the memory probe that is about 0.8 to 0.95 GB more at batch 2
(4.56 and 4.70 GB against 3.75 GB) ([decisions.md][nll]). Those logits
change shape every batch, which led to the cache-emptying callback
(chapter 2).

**Result in Part 1.** 4 of 5 SFT answers ended on `<|im_end|>`, against 0 of
5 for v1 ([sft-results.md][res]). Five samples is a small count.

## The saved probe

Part 1 quoted two figures from a check that was never saved. Issue
[#19][i19] added `tools/stop_token_probe.py` to save them. It feeds each
reference answer to the model up to, but not including, its final
`<|im_end|>`, and records that token's probability and rank as the next
token, out of 152,000. Run at commit `723c41a`, clean, on the re-run adapter
and the base model, 200 of 200 held-out rows probed. Saved in
`outputs/sft/stop_token_probe.json` ([sft-results.md][res];
[handoff note for #19][i19]).

| | Base | SFT (re-run) |
|---|---|---|
| Median probability | 2.065e-09 | 0.7417 |
| Mean probability | 9.548e-07 | 0.6335 |
| Median rank | 123,031 | 1 |
| Share of rows where the stop token ranks first | 0.00 | 0.795 |

Source: the stop-token table in [sft-results.md][res]. The doc also says
this puts `<|im_end|>` first in 159 of 200 reference positions.

**The earlier figures were close but not the same.** Part 1 said a base rank
of "about 115,000th" and a probability of "0.70". The saved base median
rank is 123,031. The 0.70 was probably a mean, since the saved SFT mean is
0.6335 and the median 0.7417, but it came from the Part 1 adapter on other
rows and was never saved, so which statistic it was is not known
([sft-results.md][res]). Use the saved figures only.

**What the probe does not show.** It is a next-token measure on reference
text. It is not a count of generated answers that stop; the eval harness
will measure that ([sft-results.md][res]). A 20-row scratch smoke on the
Part 1 adapter is not quotable ([PR #20 body][pr20]). The probe's `main()`
and row selection are untested (same PR).

## Decisions and why

- Train the two chat-token rows ([decisions.md][rows]) and use `nll`
  ([decisions.md][nll]). Neither alone works.
- Save the probe instead of quoting memory. The probe tool refuses to write
  a non-scratch result from a dirty tree ([issue #19][i19]).
- Keep the base model. On an Instruct model this failure would never have
  shown, and we would not have learned where stopping comes from
  ([decisions.md][base]).

## How to reproduce it

```bash
uv run python tools/stop_token_probe.py outputs/sft
```

From [sft-results.md][res]. Not re-run for this chapter: it loads the model
onto the GPU, which a training run was using.

## What we would do differently

Probe the stop token before training, on the base model, and check that the
chosen rows receive a gradient with a one-step test. Either check would have
shown both causes before a full epoch. The v1 run took 857 s and ended
none of its samples ([sft-results.md][res]).

[i19]: https://github.com/graylayer-labs/llm-post-training-lab/issues/19
[pr20]: https://github.com/graylayer-labs/llm-post-training-lab/pull/20
[res]: ../sft-results.md
[nll]: ../decisions.md#use-the-plain-nll-loss
[rows]: ../decisions.md#train-the-chat-token-embedding-rows-in-full
[base]: ../decisions.md#start-from-the-qwen25-05b-base-model
