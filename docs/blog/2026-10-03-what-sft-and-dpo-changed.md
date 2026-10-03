# SFT taught a 0.5B model to stop; DPO taught it to stop every time, and to be short

*2026-10-03 · Draft for the owner to edit*

We ran LoRA SFT and then DPO on `Qwen/Qwen2.5-0.5B`, on one laptop, with
the preference pairs built from the project's own data, and scored base,
SFT and DPO on the same 200 held-out prompts with no benchmark and no model
judge. The rule-based rubric pass rate went from
[35.0%](../eval-results.md#the-table) for base to
[73.0%](../eval-results.md#the-table) after SFT and
[89.0%](../eval-results.md#the-table) after DPO, and every DPO answer
stopped cleanly. Most of that gain is one behaviour, "do not loop", and DPO
also learned something the pairs did not mean to teach: be short. If you
are adapting an open model to your own data with no test set, the shape of
this result, and the holes in it, are the useful part.

| System | Rubric pass, overall | Stopped % | Mean new tokens | Perplexity of references | ROUGE-L F1 |
|---|---|---|---|---|---|
| base | [35.0 (70/200)](../eval-results.md#the-table) | [39.5](../eval-results.md#the-table) | [270.0](../eval-results.md#the-table) | [8.910](../eval-results.md#the-table) | [0.153](../eval-results.md#the-table) |
| sft | [73.0 (146/200)](../eval-results.md#the-table) | [86.5](../eval-results.md#the-table) | [111.6](../eval-results.md#the-table) | [6.245](../eval-results.md#the-table) | [0.297](../eval-results.md#the-table) |
| dpo | [89.0 (178/200)](../eval-results.md#the-table) | [100.0](../eval-results.md#the-table) | [51.9](../eval-results.md#the-table) | [6.555](../eval-results.md#the-table) | [0.296](../eval-results.md#the-table) |

*Base, SFT and DPO on the same 200 held-out prompts, decoded greedily at
batch size 16, one seed; copied from
[eval-results.md, "The table"](../eval-results.md#the-table).*

## The result

One run per stage, one seed, 200 prompts. The rubric is rule-based: format,
a clean stop, no repetition, and two finance-only rules that apply to
[24](../eval-results.md#how-to-read-it) of the 200 prompts. Nothing in the
table scores whether an answer is correct.

**SFT taught the model to end a turn.** The base model keeps the chat format
but stops on only [39.5%](../eval-results.md#sft-against-base) of prompts and
runs [121](../eval-results.md#sft-against-base) answers to the 384-token
limit. After one epoch of LoRA on
[2,000](../what-each-stage-changed.md#what-sft-changed) rows it stops on
[86.5%](../eval-results.md#sft-against-base), with
[27](../eval-results.md#sft-against-base) answers at the limit. That needed
training the two chat-token embedding rows in full, because LoRA on the
projections could not reach them; the stop-token story is in the guide.
ROUGE-L against the reference nearly doubled,
[0.153 to 0.297](../eval-results.md#sft-against-base). Not every change was
forward: SFT fixes [88](../eval-results.md#sft-against-base) prompts that
base failed and breaks [12](../eval-results.md#sft-against-base), all of
them now repetition loops.

**DPO removed the loops.** The pairs came from SFT's own training prompts:
chosen is the dataset's reference, rejected is SFT's greedy answer when the
rubric fails it. Of [105](../dpo-results.md#counts) pairs,
[104](../dpo-results.md#counts) were rejected for repetition. After
[18](../dpo-results.md#training-results) optimiser steps, every DPO answer
stops, [none](../eval-results.md#dpo-against-sft) reaches the limit, and the
repetition pass rate rises from [75.0% to 92.5%](../eval-results.md#dpo-against-sft).
DPO fixes [36](../eval-results.md#dpo-against-sft) prompts and breaks
[4](../eval-results.md#dpo-against-sft).

**DPO also learned "shorter", and this run cannot separate the two.** The
rejected answers averaged [356.4](../dpo-results.md#limits) tokens and the
chosen [124.1](../dpo-results.md#limits), about
[2.9](../dpo-results.md#limits) times longer, so "do not loop" and "be
short" were the same signal. The expected outcome, if DPO had only learned
to break loops, was that answers SFT already finished would keep their
length. They did not. DPO is shorter on
[144 of 200](../eval-results.md#is-it-just-shorter) prompts; on the
[173](../eval-results.md#is-it-just-shorter) where SFT already stopped, DPO
averages [50.4](../eval-results.md#is-it-just-shorter) tokens against
[69.1](../eval-results.md#is-it-just-shorter), and is slightly further from
the references there, ROUGE-L
[0.314 against 0.330](../eval-results.md#is-it-just-shorter).
[48](../eval-results.md#is-it-just-shorter) DPO answers are under 20 tokens,
against [27](../eval-results.md#is-it-just-shorter) for SFT. The repetition
rule counts repeats, so a shorter answer passes it more easily, and part of
DPO's rubric gain is length.

**The rubric passes a wrong answer.** Asked to simplify
`3x + 4xy + y - 2x - 3y`, where the reference ends `= 4xy + x - 2y`, DPO
answered "The simplified expression is 2x + 2y" in
[12](../eval-results.md#is-it-just-shorter) tokens. It is wrong and passes
every rule. Asked to name and describe four types of renewable energy, SFT
did both in [87](../eval-results.md#three-examples) tokens and DPO named them
in [17](../eval-results.md#three-examples) and stopped; the rubric counts
that as a DPO fix. Perplexity of the references rose from
[6.245 to 6.555](../eval-results.md#dpo-against-sft) under DPO, as expected
when a model is pushed towards short answers, and it says nothing about the
quality of DPO's own answers either way.

## What it means

If you have one machine and no benchmark, SFT and DPO on your own data can
fix the failures you can write a rule for: not stopping, looping, leaking
chat markers. A rule-based rubric with bootstrap intervals will measure
those, and it can also choose the DPO rejected answers, so training and
evaluation agree on what "bad" means. Watch the pairs for a length gap,
because the model will learn it.

What this result does not show is that DPO answers better. Nothing here
scores correctness or completeness, the intervals are several points wide
([66.5, 79.0] for SFT and
[84.5, 93.0] for DPO, from [eval-results.md](../eval-results.md#limits)),
it is one seed, and the finance rules rest on 24 prompts. The honest claim
is "DPO stops and loops less", and a correctness check, even a small
hand-graded set, is the next thing to add.

## The detail

The step-by-step account, with the commands and the mistakes, is in the
[follow-along guide](../guide/README.md):
[chapter 2, supervised fine-tuning](../guide/02-supervised-fine-tuning.md);
[chapter 3, the stop-token bug](../guide/03-the-stop-token-bug.md);
[chapter 5, preference optimisation](../guide/05-preference-optimisation.md);
and [chapter 6, evaluating without a benchmark](../guide/06-evaluating-without-a-benchmark.md).
The full write-up is
[what-each-stage-changed.md](../what-each-stage-changed.md).
