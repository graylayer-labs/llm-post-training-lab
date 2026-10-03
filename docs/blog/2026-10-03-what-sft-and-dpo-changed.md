# SFT taught a 0.5B model to stop; DPO taught it to stop every time, and to be short

*2026-10-03 · Draft for the owner to edit*

We ran LoRA SFT and then DPO on `Qwen/Qwen2.5-0.5B`, on one laptop, with
the preference pairs built from the project's own data, and scored base,
SFT and DPO on the same 200 held-out prompts with no benchmark and no paid
model judge. The rule-based rubric pass rate went from 35.0% for base to 73.0%
after SFT and 89.0% after DPO, and every DPO answer stopped cleanly
([eval-results.md, "The table"](../eval-results.md#the-table)). Most of
that gain is one behaviour, "do not loop", and DPO also learned something
the pairs did not mean to teach: be short. If you are adapting an open
model to your own data with no test set, the shape of this result, and the
holes in it, are the useful part.

| System | Rubric pass, overall | Stopped % | Mean new tokens | Perplexity of references | ROUGE-L F1 |
|---|---|---|---|---|---|
| base | 35.0 (70/200) | 39.5 | 270.0 | 8.910 | 0.153 |
| sft | 73.0 (146/200) | 86.5 | 111.6 | 6.245 | 0.297 |
| dpo | 89.0 (178/200) | 100.0 | 51.9 | 6.555 | 0.296 |

*Base, SFT and DPO on the same 200 held-out prompts, decoded greedily at
batch size 16, one seed; copied from
[eval-results.md, "The table"](../eval-results.md#the-table).*

## The result

One run per stage, one seed, 200 prompts. The rubric is rule-based: format,
a clean stop, no repetition, and two finance-only rules that apply to 24 of
the 200 prompts ([eval-results.md, "How to read it"](../eval-results.md#how-to-read-it)).
Nothing in the table scores whether an answer is correct; a blind,
model-graded check on 50 of the prompts, further down, does.

**SFT taught the model to end a turn.** The base model keeps the chat format
but stops on only 39.5% of prompts and runs 121 answers to the 384-token
limit. After one epoch of LoRA it stops on 86.5%, with 27 answers at the
limit. That needed training the two chat-token embedding rows in full, because LoRA on the projections could not reach them; the
stop-token story is in the guide. Its answers moved closer to the
references, ROUGE-L 0.153 to 0.297. Not every change was forward: SFT
fixes 88 prompts that base failed and breaks 12, all of them now
repetition loops
([eval-results.md, "SFT against base"](../eval-results.md#sft-against-base)).

**DPO cut looping.** The pairs came from SFT's own training prompts:
chosen is the dataset's reference, rejected is SFT's greedy answer when the
rubric fails it. Of 105 pairs, 104 were rejected for repetition, and DPO
trained on them for 18 optimiser steps
([dpo-results.md, "Building the pairs"](../dpo-results.md#building-the-pairs)
and "Training results").

Every DPO answer stops, none reaches the limit, and the repetition pass
rate rises from 75.0% to 92.5%. That still leaves 15 of 200 answers that
repeat, and a short loop can stay under the rule's threshold (index 29 in
the same doc). DPO fixes 36 prompts and breaks 4
([eval-results.md, "DPO against SFT"](../eval-results.md#dpo-against-sft)).

**DPO also learned "shorter", and this run cannot separate the two.** The
rejected answers averaged 356.4 tokens and the chosen 124.1, about 2.9
times longer, so "do not loop" and "be short" were the same signal. The
expected outcome, if DPO had only learned to break loops, was that answers SFT
already finished would keep their length. They did not. DPO is shorter on
144 of 200 prompts; on the 173 where SFT already stopped, DPO averages
50.4 tokens against 69.1, and is slightly further from the references
there, ROUGE-L 0.314 against 0.330. 48 DPO answers are under 20 tokens,
against 27 for SFT. The repetition rule counts repeats, so a shorter answer
passes it more easily, and part of DPO's rubric gain is length
([eval-results.md, "Is it just 'shorter'?"](../eval-results.md#is-it-just-shorter)).

**The rubric passes a wrong answer.** Asked to simplify
`3x + 4xy + y - 2x - 3y`, where the reference ends `= 4xy + x - 2y`, DPO
answered "The simplified expression is 2x + 2y" in 12 tokens. It is wrong
and passes every rule. Asked to name and describe four types of renewable
energy, SFT did both in 87 tokens and DPO named them in 17 and stopped; the
rubric counts that as a DPO fix. Perplexity of the references rose from
6.245 to 6.555 under DPO, as expected when a model is pushed towards short
answers, and it says nothing about the quality of DPO's own answers either
way ([eval-results.md, "What each stage changed"](../eval-results.md#what-each-stage-changed)).

**Graded blind, DPO is correct more often, but over half of the gain is
where SFT looped.** Two Claude graders, Opus and Sonnet, graded 50 of the
prompts against the reference without knowing which system wrote which
answer. This is grading by models, not a person. Correct: base 10.0% and
12.0%, SFT 30.0% and 30.0%, DPO 44.0% and 48.0% (Opus and Sonnet), with
the graders agreeing on 84.0% of answers (kappa 0.756). Passing the rubric
is not being correct: only 34.3% and 40.0% of SFT's rubric passes are
graded correct. DPO's paired gain over SFT is +14.0 points [+0.0, +28.0]
and +18.0 [+6.0, +28.0]. A loop is graded wrong, so the same comparison was
run on only the 35 prompts where the rubric passes both: +8.6 [−8.6,
+22.9] and +11.4 [+0.0, +25.7]. Still DPO's way, but both intervals reach
zero. The first Sonnet grader was discarded for judging from length and
the first ~300 characters; the kept grades are a rerun that read every
answer ([eval-results.md, "Blind correctness check"](../eval-results.md#blind-correctness-check)).

## What it means

If you have one machine and no benchmark, SFT and DPO on your own data can
fix the failures you can write a rule for: not stopping, looping, leaking
chat markers. A rule-based rubric with bootstrap intervals will measure
those, and it can also choose the DPO rejected answers, so training and
evaluation agree on what "bad" means. Watch the pairs for a length gap,
because the model will learn it.

What this result shows about correctness is narrow. On 50 blind-graded
prompts DPO is graded correct more often than SFT, and over half of that
gain is on prompts where SFT's answer failed the rubric, mostly by
looping. Where SFT already gave a clean answer, DPO is ahead by 3 or 4
prompts of 35, which is not enough to separate from zero. The rubric
intervals are several points wide ([66.5, 79.0] for SFT and [84.5, 93.0]
for DPO, from [eval-results.md](../eval-results.md#limits)), the grading
was by two models that may share biases, it is one seed, and the finance
rules rest on 24 prompts. The honest claim is "DPO stops and loops less,
and is graded correct more often, over half of that where SFT looped". A
person
grading a sample of the answers is the next thing to add.

## The detail

The step-by-step account, with the commands and the mistakes, is in the
[follow-along guide](../guide/README.md):
[chapter 2, supervised fine-tuning](../guide/02-supervised-fine-tuning.md);
[chapter 3, the stop-token bug](../guide/03-the-stop-token-bug.md);
[chapter 5, preference optimisation](../guide/05-preference-optimisation.md);
and [chapter 6, evaluating without a benchmark](../guide/06-evaluating-without-a-benchmark.md).
The full write-up is
[what-each-stage-changed.md](../what-each-stage-changed.md).
