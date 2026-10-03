---
name: handoff
description: End work on an issue by writing its state onto the GitHub issue so the next agent can continue. Use before stopping, finished or not.
---

# Handoff

Write what the next agent needs. Assume they have never seen this session.

1. Make sure the work is committed on its branch in logical units. If
   something is uncommitted, say what and why.
2. Run the quality gate and note the result.
3. Draft the comment below. Ask `github-manager` to post it on the issue,
   and to update the board status if it changed.
4. If the session found follow-up work, ask `github-manager` to create issues
   for it from `.github/ISSUE_TEMPLATE/task.md`.

```markdown
## Handoff <date>

**State:** done | in progress | blocked
**Branch:** `<branch>` (pushed: yes/no) · **PR:** #<n>

**Done**
- ...

**Evidence**
- Commands and their output as it appeared. For a result: the numbers, the
  config, the seed, the commit and the run directory.

**Decisions and why**
- ...

**Tried and did not work**
- What, and the likely reason.

**Next step**
- The single next action.
```

## When the issue is finished

- Add the merged PR link, and what was found but left alone with the issue
  that now tracks it.
- Ask `github-manager` to add one line to the epic's "Decisions so far": the
  date, the outcome in a sentence, and a link to the issue.
- Write `docs/guide/notes/<issue>.md`, ten lines or fewer: what the issue set
  out to do, what it found, the decision and why, and which files hold the
  numbers. No number without its source beside it. Commit it in the issue's
  PR when there is one.
- A new decision gets its own section in `docs/decisions.md`; a result or a
  new capability gets a `CHANGELOG.md` entry.

Write whole sentences. Include failures; an unwritten dead end gets repeated.
