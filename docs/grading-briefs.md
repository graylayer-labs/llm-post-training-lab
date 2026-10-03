# Grader briefs

The instructions given to the model graders of the blind correctness check
([eval-results.md, "Blind correctness check"](eval-results.md#blind-correctness-check)),
copied from the lead session on 2026-10-04. Each grader was a separate
Claude Code subagent with no other context. It was given its own copy of
`sheet.jsonl` in a directory without `key.json`, and told to read nothing
else.

| Grader | Model | Brief | Used |
|---|---|---|---|
| opus | Claude Opus | Brief 1 | yes |
| sonnet, first attempt | Claude Sonnet | Brief 1 | no: it reported judging mostly from the first ~300 characters and the length, partly by script; its grades are in `outputs/grading/v1/discarded/` |
| sonnet | Claude Sonnet | Brief 2 | yes |

Paths in the briefs were scratch directories and are shown here as `<dir>`.

## Brief 1

> You are a blind grader of answers to questions. Read ONLY this file:
> `<dir>/sheet.jsonl`. Do not open, list or search any other file or
> directory. Do not try to find out which model wrote which answer. Your job
> is only to judge each answer on its merits.
>
> Each line is one prompt: {"index", "prompt", "reference", "answers": {"A",
> "B", "C"}}. The reference is one human-written answer from a dataset, some
> of it forum-style; it is a guide, not the only correct answer.
>
> Grade every answer (50 prompts × 3 answers = 150 grades) as exactly one of:
> - "correct": answers the question and is factually and logically right;
>   any substantive claims or calculations are right. Brevity is fine if the
>   question is answered.
> - "partly": partly right, e.g. on-topic with a real error, incomplete in an
>   important way, or right content buried in heavy repetition or junk.
> - "wrong": factually or logically wrong, off-topic, doesn't answer, empty,
>   or mostly degenerate (looping, garbage text, chat-markup leaks).
>
> Judge correctness and usefulness, not resemblance to the reference wording.
> Be consistent across all 150, and do not let length alone decide a grade.
>
> Work through the file in chunks so nothing is skipped. Write your grades to
> `<dir>/grades_<grader>.jsonl`, one JSON object per line, exactly:
> {"index": <int>, "label": "A"|"B"|"C", "grade": "correct"|"partly"|"wrong",
> "reason": "<one short sentence>"}. There must be exactly 150 lines, every
> (index, label) once. Before finishing, verify the count and that every
> index × label is present, with a short script that reads only your grades
> file and the sheet.

## Brief 2

Brief 2 has the same grade definitions as Brief 1 and adds a strict process:

> STRICT PROCESS. These rules are required, and a previous attempt was
> discarded for breaking them:
> - READ EACH ANSWER IN FULL before grading it. Never grade from the first few
>   hundred characters.
> - NEVER use answer length, or any automatic heuristic, to decide a grade. Do
>   not write code that assigns grades; code may only print the sheet or
>   check your output file.
> - Work in batches of 10 prompts (lines 1–10, 11–20, …). For each batch,
>   print the full text of those prompts and answers, read them, decide all
>   30 grades yourself, and append them to the output file before moving on.
> - Each "reason" must be specific to that answer (e.g. "states 3 strands
>   correctly", "loops after first sentence", "gives 2x+2y, wrong
>   expansion"). Never generic.

The Opus grader got Brief 1 without these rules. Its 150 reasons include
126 distinct ones (`grades_opus.jsonl`), which suggests it read the answers
individually, but the rules were not imposed on it.
