# 4. Trustworthy runs

*Last updated 2026-10-03 from issues #10, #11, #15, #21, #23, #12*

## What we set out to do

Make every number traceable to a run, and every run survivable. The
project's standard is that a claim needs a saved run, a held-out evaluation
and a plain statement of its limits ([INTENT.md](../../INTENT.md)). This
chapter is the machinery that supports that, and the mistakes that made us
build it.

## 1. Provenance (#10)

**Found.** Part 1's final SFT run came from a dirty working tree and
recorded no commit, so its code could not be traced ([issue #10][i10]).
**Decision.** Every `summary.json` carries a `provenance` block: the commit,
whether the tree was dirty (untracked files count), the branch, the machine,
and the torch, transformers, trl, peft and datasets versions. A dirty run is
marked `scratch` ([decisions.md][prov]).
**Review.** The reviewer caught provenance being captured after training.
The fix has a test that failed before and passes after ([PR #16 body][pr16]).
**Rule.** Only runs from a clean commit on `main` are quoted. Files:
`src/post_training/run.py`, `tests/test_run.py`.

## 2. De-duplication (#15)

Covered in chapter 1: 7,587 duplicate prompts dropped, after 4 of 200
held-out prompts were found among the training rows ([decisions.md][dedup]).
It belongs here too, because it is a trust fix, not a data choice.

## 3. The rubric and its limits (#11)

Five rules score an answer: format, clean stop, repetition, domain terms
and ungrounded numbers. One module serves DPO pair building and the eval
harness ([issue #11][i11]). Finance rows are classed by the prompt, and
figures match as value sets, so `5%` equals 0.05 ([decisions.md][rubric]).

On the 2,200 reference answers, 2,170 pass (98.6%) and 283 (12.9%) are
finance rows. Per rule: `domain_terms` 96.8% (274 of 283), `repetition`
99.1%, `format` 99.95% (2,199 of 2,200). Source: `tools/rubric_on_references.py`
at commit `723c41a`, clean, saved in
`outputs/rubric_references/v1/summary.json` ([decisions.md][rubric]).

**Limit: grounding is not truth.** `ungrounded_numbers` checks that a figure
appears in the prompt or the reference, not that it is right. A correct
figure the reference does not state fails it, for example "$52.50" for
$50 after a year at 5%. With the reference removed as a number source, 105
of the 283 finance references (37.1%) fail it ([decisions.md][rubric]). So
a good answer that works out its own figures is often marked ungrounded.
For that reason DPO's rejection rule uses only format and repetition
([decisions.md][reject]). An earlier, narrower classifier gave 38.2%,
measured but not saved; use 37.1%.

## 4. The killed run, checkpoints and resume (#21)

The first attempt at the clean re-run was killed at step 18 of 125 by
another agent's `pkill -f`, which matched its process. Nothing had been
saved, so it restarted from step 0. The log is
`outputs/failed/sft-killed-step18/run.log` ([sft-results.md][res]). That led
to a machine-wide guard against pattern kills and to checkpointing
([issue #21][i21]).

**Decision.** SFT checkpoints every 25 steps. `lab sft --resume` continues
from the newest complete checkpoint, refuses to start over existing ones,
and requires the same commit and config on a clean tree unless `--scratch`
([decisions.md][ckpt]).

**A bug the review caught.** The Trainer's own `training_loss` is wrong
after a resume: its running total restarts at 0 but is divided by the full
step count. On a tiny CPU run it gave 3.3802 against 4.1598 uninterrupted
([PR #22 body][pr22]). `train_loss` is now the mean of the logged step
losses, and `tests/test_resume_equivalence.py` shows resumed and
uninterrupted runs agree. The review also found a `--scratch` resume was
not marked scratch; fixed with tests (same PR).

**Limits.** MPS RNG state is not restored, so a resumed MPS run is
statistically equivalent, not bit-identical; on CPU it matched exactly. A
resumed run's wall time and peak memory cover only the resumed part
([decisions.md][ckpt]). No MPS kill-and-resume was run after the review
fixes. The #12 re-run ran with checkpoints on MPS (`save_steps` 25) but was
not interrupted ([PR #22 body][pr22]; [issue #21][i21]).

## 5. Batch size is a generation setting (#10)

In bf16 on MPS, padding in a batched greedy decode flips near-tied tokens.
On the first 5 held-out prompts, batch size 1 reproduced all 10 saved greedy
answers exactly. Batch size 5 matched the 5 short answers but diverged on 4
of the 5 that ran to the token limit, in one case after 22 characters.
Measured on 2026-10-03, not saved ([decisions.md][batch]; [PR #16 body][pr16]).
So the batch size is recorded beside the other generation settings and held
fixed when systems are compared.

## 6. The eval harness's cache identity (#23)

The harness scores base, SFT and DPO on identical held-out prompts and
settings, in one table: perplexity on answer tokens with SFT's masking,
ROUGE-L, and rubric per-rule rates with bootstrap intervals. Its
generation cache is crash-safe and keyed on adapter hashes, the chat
template, the tokenizer and the base-model revision ([issue #23][i23]).
That key exists so a changed adapter or template cannot reuse stale answers.
It reads held-out rows from the SFT run's `eval_rows.json`, because the
Part 1 split cannot be rebuilt after the de-duplication
([decisions.md][heldout]).

**Gap.** There is no model judge, the owner's choice to keep cost inside
the subscription. Nothing scores whether an answer is correct or helpful
beyond the rubric's rules ([decisions.md][nojudge]).

**Not yet run.** The full harness on the re-run adapter and on DPO. The
only run so far is a 10-prompt smoke, not quotable ([issue #23][i23]). No
harness results are quoted in this guide yet.

## How to reproduce it

The documented commands are the quality gate and the ones in chapters 2 and
3. Not re-run here: those need the GPU, which a training run was using.
The rubric on the references is a command in
`tools/rubric_on_references.py`; I did not re-run it.

## What we would do differently

Start with provenance and checkpoints. Three of the fixes in this chapter
(provenance, de-dup, resume) came from a review or a failure, not from
planning. The cost was a Part 1 run that now stands only as history.

[i10]: https://github.com/graylayer-labs/llm-post-training-lab/issues/10
[i11]: https://github.com/graylayer-labs/llm-post-training-lab/issues/11
[i21]: https://github.com/graylayer-labs/llm-post-training-lab/issues/21
[i23]: https://github.com/graylayer-labs/llm-post-training-lab/issues/23
[pr16]: https://github.com/graylayer-labs/llm-post-training-lab/pull/16
[pr22]: https://github.com/graylayer-labs/llm-post-training-lab/pull/22
[res]: ../sft-results.md
[prov]: ../decisions.md#record-each-runs-commit-tree-state-and-library-versions
[dedup]: ../decisions.md#de-duplicate-prompts-before-splitting
[rubric]: ../decisions.md#class-finance-rows-by-the-prompt-and-match-figures-as-value-sets
[reject]: ../decisions.md#reject-samples-on-format-and-repetition-only
[ckpt]: ../decisions.md#checkpoint-training-runs-and-resume-after-a-crash
[batch]: ../decisions.md#treat-the-generation-batch-size-as-a-setting
[heldout]: ../decisions.md#read-the-held-out-rows-from-the-sft-runs-saved-file
[nojudge]: ../decisions.md#evaluate-without-a-model-judge
