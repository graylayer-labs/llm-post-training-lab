---
name: guide
description: Update the follow-along guide in docs/guide/ from issues closed since it was last updated. Use after a stage's results land, or whenever several issues have closed.
---

# Guide

1. Start from an up-to-date `main` on a branch `docs/<issue>-guide-<date>`,
   in a worktree.
2. Delegate to the `guide-writer` agent. It finds new closed issues itself.
   Give it the date, the branch, and anything the owner asked for. Do not
   tell it what to write.
3. Read its report. Open one changed chapter and trace five numbers to the
   sources it links. If any cannot be found, send the chapter back.
4. Run one command from a changed chapter yourself, on a scratch path.
5. Docs-only change: the lead reads the diff and says so in the PR. Ask
   `github-manager` to push and open the PR.

No number without a source. A gap is fixed at the source, not guessed.
