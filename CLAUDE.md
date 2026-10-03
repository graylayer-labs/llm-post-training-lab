# llm-post-training-lab

Read the intent first. It explains why the project exists, and it overrides
anything below that conflicts with it.

@INTENT.md

## Who does what

Eoin is the owner. Claude agents do the engineering and run the project day
to day. Act without asking, except in the areas under "Ask the owner first".

The main session is the lead. It plans, writes issues, briefs agents, checks
their results and reports to the owner. Routine work goes to agents.

- **Delegate in the background.** Each implementing agent works in its own
  worktree (`isolation: worktree`). At most two agents run at once, on
  issues the epic's "Order of work" marks as parallel.
- **Brief agents as strangers.** An agent knows only its prompt and the
  issue. The agents are in `.claude/agents/`: `implementer` writes code,
  `reviewer` checks it (read-only), `guide-writer` writes `docs/guide/` and
  `docs/blog/`. The issue's `agent:` label names the model.
- **Verify before trusting.** Re-run the quality gate an agent reports, and
  test its most important claim yourself.
- **GitHub writes go through the `github-manager` agent.** That covers
  pushes, PRs, issues, the board and merges. Agents and the lead commit
  locally only. Read-only `gh` commands are open to everyone.
- **Cost stays inside the owner's Claude Max plan.** No API keys, no paid
  cloud, no billed review such as `/code-review ultra`. Local agents and
  laptop runs only.

## Every session

1. **Start** with `/pickup`. It reads the board and open issues.
2. **Work** on one issue at a time per agent: one issue, one branch, one PR.
3. **End** with `/handoff`, finished or not, so the next agent can continue
   from the issue alone.

## How work runs

- **The board is the memory.** Work is planned on the
  [GitHub project board](https://github.com/orgs/graylayer-labs/projects/3)
  (graylayer-labs project 3). Epic #5 holds the order of work. Start from the
  board and the open issues, not from memory of an earlier conversation.
- **One issue, one branch, one PR.** Branch names follow
  `<type>/<short-description>`. Every PR closes an issue. If there is no issue
  for the work, create one from `.github/ISSUE_TEMPLATE/task.md`.
- **Epics hold the current milestone's tasks** as native sub-issues. Work
  that crosses milestones (process, tooling, audits) is a standalone issue on
  the board with no parent epic.
- **Record on the issue.** Decisions, results and dead ends go in an issue
  comment, so the next agent can continue from the issue alone. A new
  decision also gets its own section in `docs/decisions.md`, under the
  matching group.
- **Commits** use Conventional Commits. Stage files by name.

## Quality gate

Run before every commit. CI runs the same checks.

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest
```

## Rules

- **Do not edit `NOTES.md`.** It is the owner's own writing.
- **`outputs/` and `data/` are gitignored.** Never commit run outputs or data.
  A result quoted in the docs names the run directory it came from.
- **Runs are config-driven.** A new run is a new YAML file under `configs/`,
  not edited code or hard-coded values.
- **No unverified numbers.** Every figure in the docs must come from a saved
  file in `outputs/` or be marked as measured but not saved.

## Ask the owner first

- Anything that costs money beyond the Max plan.
- Anything public beyond the repo: publishing a blog post, releases, repo
  visibility or settings.
- A laptop run expected to take over an hour.
- Changing the purpose, standard or non-goals in `INTENT.md`.
- Deleting anything that cannot be recovered, force-pushing, or rewriting
  history.
- Merging a PR that the reviewer flagged as blocking.

Quote each approval, with its date, in a comment on the issue before acting.
Decide everything else alone and report it as "I chose X; say if you want Y".

## Review before merge

CI proves the code runs. It does not prove a result is valid.

- Changes under `src/`, to experiments or to metrics: the `reviewer` agent
  checks the branch before merge.
- Docs and config only: the lead reads the diff itself.
- The PR says which of the two happened. Its Evidence section holds the
  lead's own re-run of the quality gate, pasted as it appeared.

## Experiments

- **Quotable runs come from a clean commit on `main`.** Merge the code first,
  then run it. A run from a branch or a dirty tree is scratch: use it to
  debug, never quote it.
- Every run writes `summary.json` with the resolved config, and the run's
  provenance once the provenance task lands: commit, dirty flag and library
  versions.
- Run a smoke config end to end before a full run. Size full runs to finish
  in under 30 minutes on the laptop. Run them under `caffeinate -dims` on
  mains power.
- Every doc number names the run directory and commit it came from.

## Where things are written

| What | Where |
|---|---|
| Why a choice was made | `docs/decisions.md` |
| What changed, for a two-minute reader | `CHANGELOG.md`, key items only |
| A stage's results, bugs and memory | `docs/<stage>-results.md` |
| What was done on an issue | its closing handoff comment |
| Raw notes for the guide, per closed issue | `docs/guide/notes/<issue>.md`, deleted once in a chapter |
| How the project was built, step by step | `docs/guide/`, kept current with `/guide` |
| A short, result-led post | `docs/blog/`, drafted with `/blog`, never published by an agent |

## Keeping the owner informed

The owner reads the end of each message. Send an update when something
merges, a result arrives (good or bad), or a step needs the owner. Ten lines
at most:

1. **Done:** one line per finished thing, with its issue or PR number.
2. **Running:** one line.
3. **Insight**, optional: one thing a senior engineer would find worth
   knowing.
4. **Last line:** what is needed from the owner, or "Nothing needed from you."

## Commands and gotchas

- A fresh worktree has no environment. Run `uv sync --dev` first. The
  `VIRTUAL_ENV` mismatch warning inside a worktree is harmless.
- `outputs/` is gitignored and lives in the main checkout. Agent worktrees
  get it as a symlink (`.claude/settings.json`), so earlier runs such as
  `outputs/sft/adapter` are readable there. Because it is shared, an agent
  never writes into a quotable run's directory: smoke and scratch runs use
  their own `output_dir`.
- The shell is zsh. Run multi-line scripts as `bash <<'EOF'`. macOS has no
  `timeout`.
- `gh project` sometimes fails with `unknown owner type`. Run it again.
- Don't move the main checkout (pull, switch) while a run started from it is
  in progress: a crash-resume checks the commit and will refuse.
