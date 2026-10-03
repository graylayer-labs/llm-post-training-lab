# Follow-along guide

How this project was built, step by step, including what went wrong. It is
written for a capable engineer who was not there.

It is not the results. Each stage's numbers live in its results doc under
`docs/` (for example [`sft-results.md`](../sft-results.md)), and short
result-led posts live in [`docs/blog/`](../blog/README.md). The guide is the
process: the data, the design choices, the mistakes and the commands.

## Rules

- **Every number links to its source:** a results doc, an issue or a handoff
  comment. It is the number written there, not a rounded one.
- **A command appears only if it works on `main`.** The chapter says when it
  was last run.
- **Failures are written as plainly as successes.**
- **Each chapter says when it was last updated and from which issues.**
- **Where a chapter and its source disagree, the source wins.**

Each chapter has the same shape: what we set out to do, what we found, the
decisions and why, how to reproduce it, and what we would do differently.

## Chapters

Chapters are written with `/guide` once a stage's results land. The DPO and
evaluation chapters come later.

| # | Chapter | Last updated | From issues |
|---|---|---|---|
| 1 | [The problem and the data](01-the-problem-and-the-data.md) | 2026-10-03 | #15, #11, #12 |
| 2 | [Supervised fine-tuning](02-supervised-fine-tuning.md) | 2026-10-03 | #12, #1 |
| 3 | [The stop-token bug](03-the-stop-token-bug.md) | 2026-10-03 | #19, #12, #1 |

## Notes

`notes/<issue>.md` holds a short note per closed issue, written at handoff.
Each note is folded into a chapter and then deleted.
