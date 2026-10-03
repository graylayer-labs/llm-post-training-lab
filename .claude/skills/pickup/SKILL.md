---
name: pickup
description: Start a session by recovering project state from the GitHub board and issues. Use at the beginning of every session, or whenever unsure what is in progress.
---

# Pickup

Recover the project's state from the board, not from memory.

1. Read `INTENT.md`.
2. Check local state:
   ```bash
   git status -sb && git branch --no-merged main && git worktree list
   ```
   Uncommitted changes or unmerged branches mean a session stopped mid-task.
3. Read the board and open PRs. Retry `gh project` on `unknown owner type`.
   ```bash
   gh project item-list 3 --owner graylayer-labs --format json --limit 100 \
     --jq '.items[] | [.status, .content.number, .title] | @tsv'
   gh pr list --state open
   ```
4. Read the current epic (label `epic`). Its "Order of work" says what is
   next; "Decisions so far" is its short history.
5. Read every In Progress issue in full, body and comments. The last handoff
   comment says where work stopped.
6. Report to the owner in the update format in `CLAUDE.md`: what is in
   progress and where it stopped, what is next, and anything that looks wrong
   (a branch with no issue, an issue in progress with no branch).
7. Start the next issue unless it needs the owner.

Trust issue comments and `git log` over anything recalled. If the board and
the repo disagree, say so.
