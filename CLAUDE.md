# llm-post-training-lab

Read the intent first. It explains why the project exists, and it overrides
anything below that conflicts with it.

@INTENT.md

## How work runs

- **The board is the memory.** Work is planned on the
  [GitHub project board](https://github.com/orgs/graylayer-labs/projects/3)
  (graylayer-labs project 3). Epic #5 holds the order of work. Start from the
  board and the open issues, not from memory of an earlier conversation.
- **One issue, one branch, one PR.** Branch names follow
  `<type>/<short-description>`. Every PR closes an issue. If there is no issue
  for the work, create one from `.github/ISSUE_TEMPLATE/task.md`.
- **Record on the issue.** Decisions, results and dead ends go in an issue
  comment, so the next agent can continue from the issue alone. A new
  decision also gets one line in `docs/decisions.md`.
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
