# 1. The problem and the data

*Last updated 2026-10-03 from issues #15, #11, #12*

## What we set out to do

Adapt a small open model to finance Q&A and see what each post-training
stage changes. There is no public test set for this task, the hardware is
one Apple M4 laptop with 24 GB, and the stages are run by hand
([INTENT.md](../../INTENT.md)). This chapter covers the model choice and the
data. Later chapters cover training and evaluation.

## What we found

**The data repeats itself.** The set is `gbharti/finance-alpaca`. After the
length filter (below), 55,835 rows remain. Of those, 7,587 (13.6%) are
duplicate prompts ([PR #17 body][pr17]; [decisions.md][dedup]). The Part 1
split did not remove them. In the review of the rubric work, 4 of the 200
held-out prompts turned out to also be among the 2,000 training rows
([issue #15][i15]). A held-out score on a prompt the model trained on
is not held-out.

**The split now de-duplicates first.** Repeated prompts are dropped (first
occurrence kept) before the seeded shuffle. The re-run records
`duplicates_dropped` of 7,587 in `outputs/sft/summary.json`
([sft-results.md][res]). We fixed the split at its source, not just the 4
rows, so DPO and the eval harness inherit the fix.

**The cost of the fix.** The reshuffle draws different rows. Only 2 of the
200 new held-out prompts were in the old set, so Part 1 numbers cannot be
compared with the re-run ([decisions.md][dedup]; measured, not saved).

**The training data is mostly general instructions, not finance.** The set
mixes finance Q&A with general Alpaca instructions
([decisions.md][slice]). The rubric work let us count it. The rubric classes
a row as finance by its prompt. On the 2,200 reference answers (2,000
training plus 200 held-out), 283 rows (12.9%) are finance
(`outputs/rubric_references/v1/summary.json`, commit `723c41a`, clean tree;
[decisions.md][rubric]). An earlier scratch run, before de-duplication, found
23.4% finance rows. De-duplication cut the share to 12.9% because
finance-alpaca's finance rows were heavily duplicated ([decisions.md][rubric]).
That scratch figure was not saved as a quotable result.

So "fine-tune on finance Q&A" is closer to "fine-tune on a general
instruction set with a finance slice". The sample answers in
[sft-results.md][res] show this: one is a Python function, one a character
sketch.

## Decisions and why

- **A filtered 2,000-row slice.** Seeded 2,000 train and 200 eval rows,
  keeping answers of 40 to 1,500 characters. Short answers teach little, and
  very long ones are truncated at 512 tokens. 2,000 rows keep one epoch near
  15 minutes on the laptop ([decisions.md][slice]).
- **De-duplicate by prompt.** The prompt is instruction plus input with
  whitespace collapsed. The first occurrence is kept
  ([decisions.md][dedup]).
- **Base model, not Instruct.** `Qwen/Qwen2.5-0.5B` base, not
  Qwen2.5-0.5B-Instruct. An Instruct model already follows the chat format
  and stops, which would hide what SFT changes. On the base model, SFT's
  effect on format and stopping is visible ([decisions.md][base]). Chapter 3
  shows how much that choice exposed.

## How to reproduce it

The split is built by `make_splits` in `src/post_training/data/finance.py`
and is exercised by `tests/test_data.py`. The numbers above come from saved
files under `outputs/`, which is gitignored, so they exist only on the
machine that ran them. I did not re-run the split or the dataset download
for this chapter (it needs the Hub, and a training run holds the GPU).

## What we would do differently

Count the duplicates and the finance share before the first training run,
not after the review. Both were one pass over the data. We also would not
call the set "finance" in the plan without checking, since 12.9% of the
rows are.

[i15]: https://github.com/graylayer-labs/llm-post-training-lab/issues/15
[pr17]: https://github.com/graylayer-labs/llm-post-training-lab/pull/17
[res]: ../sft-results.md
[dedup]: ../decisions.md#de-duplicate-prompts-before-splitting
[slice]: ../decisions.md#train-on-a-filtered-2000-row-slice-of-finance-alpaca
[base]: ../decisions.md#start-from-the-qwen25-05b-base-model
[rubric]: ../decisions.md#class-finance-rows-by-the-prompt-and-match-figures-as-value-sets
