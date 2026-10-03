---
name: blog
description: Draft a result-led blog post from the guide and the results docs, for the owner to edit. Use after /guide at the end of an epic. Never publishes.
---

# Blog

1. Check that `/guide` has run since the epic's last issue closed. If not,
   run it first.
2. Delegate the draft to the `guide-writer` agent, naming the epic and the
   branch. It reads the guide, the results docs and `docs/blog/TEMPLATE.md`,
   and writes `docs/blog/<YYYY-MM-DD>-<slug>.md`.
3. Read the draft. Check its shape: a one-paragraph hook, one figure or
   table, the result, what it means, a link to the guide. Trace five numbers
   to their sources.
4. Show the owner the hook and the headline result in your message, and open
   the file. Ask for their edits.
5. Ask `github-manager` to push and open a PR for the draft.

The post is a draft. Publishing it anywhere needs the owner. Report negative
results as plainly as positive ones.
