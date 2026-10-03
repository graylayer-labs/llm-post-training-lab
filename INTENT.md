# Intent

Why this project exists and how decisions get made. This file changes rarely.
Plans, tasks and results live elsewhere (see "Where things live").

## Purpose

Build a small but complete post-training pipeline for an open LLM, and
understand each stage end to end.

## The problem

Organisations want to adapt open LLMs to their own domain data. Three things
usually stand in the way:

- **No benchmark.** Their task has no public test set, so "is it better?" has
  no ready answer.
- **Limited hardware.** They may have one machine, or a mix of machines, not a
  GPU cluster.
- **Opaque stages.** SFT and preference optimisation are often run as recipes,
  without a clear view of what each one changes.

The project runs every stage by hand on a small model, so that each effect
can be seen and measured:

- what SFT changes, and what DPO changes on top of it;
- where the memory goes during training;
- how to evaluate when no off-the-shelf benchmark exists.

## What value means

Value = learning + a careful reader trusting the work.

- **Learning:** the owner understands each stage well enough to explain why it
  worked or failed.
- **Trust:** a careful technical reader sees sound engineering and an honest
  evaluation. Numbers are traceable to a saved run, and unknowns are stated.

Work that produces neither is out of scope.

## Standard

- Understanding over scale. A 0.5B model on one laptop is enough to show what
  each stage does.
- A claim needs a saved run, a held-out evaluation and a plain statement of
  its limits.
- Failures and bugs are written up with the same care as results.
- Every run is described by one YAML file and a commit.

## Technical direction

1. **SFT:** LoRA fine-tune of a base model on domain Q&A, with the prompt
   masked from the loss.
2. **DPO:** preference optimisation on pairs built from the project's own
   data, since real users rarely have preference data.
3. **Evaluation:** loss-based metrics, a rule-based rubric and a model judge,
   on the same held-out prompts for every stage.
4. **Write-up:** what each stage changed, the memory breakdown, and what would
   differ at much larger scale.

## Compute

One Apple M4 laptop with 24 GB of unified memory, using MPS. There is no CUDA,
so CUDA-only tools such as bitsandbytes are out. Runs stay short enough to
finish on the laptop.

## Non-goals

- Models other than Qwen2.5-0.5B.
- Public benchmark suites or leaderboard numbers.
- Multi-GPU training, reward models and PPO-style RL.
- A production assistant.

## Where things live

| What | Where |
|---|---|
| Why (this file) | `INTENT.md` |
| How agents work in this repo | `CLAUDE.md` |
| What is planned and in progress | [Project board](https://github.com/orgs/graylayer-labs/projects/3) and issues |
| Why each choice was made | `docs/decisions.md` |
| What happened and what we learned | `docs/` results write-ups |
| The owner's own log | `NOTES.md` |
