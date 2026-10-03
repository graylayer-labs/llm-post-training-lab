# Changelog

Key changes to the project, newest first. Routine fixes and wording changes
are left out; the git log has those. Entries are grouped by epic and date, and
each links to the issue that holds the detail.

## Epic: Hands-on LLM post-training and evaluation (in progress)

### 2026-10-03

**Added**
- `tools/compare_systems.py` compares saved eval generations between
  systems on CPU: length distributions, rubric rates among answers that
  stopped, and like-for-like subsets. Output carries run provenance.
  ([#3](https://github.com/graylayer-labs/llm-post-training-lab/issues/3))
- Every `summary.json` now records a `provenance` block (commit, tree state,
  library versions) and marks a dirty-tree run as scratch. A shared batched
  generator was added alongside it. ([#10](https://github.com/graylayer-labs/llm-post-training-lab/issues/10))
- Rule-based rubric with five rules (format, clean stop, repetition, domain
  terms, ungrounded numbers) and a finance-row classifier. ([#11](https://github.com/graylayer-labs/llm-post-training-lab/issues/11))
- `tools/stop_token_probe.py` saves the probability and rank of `<|im_end|>`
  at the end of reference answers, for base and SFT. ([#19](https://github.com/graylayer-labs/llm-post-training-lab/issues/19))
- SFT writes checkpoints every 25 steps, and `lab sft --resume` continues a
  crashed run only from the same commit and config. ([#21](https://github.com/graylayer-labs/llm-post-training-lab/issues/21))
- `lab eval` harness: perplexity, ROUGE-L and rubric rates on the same
  held-out prompts, with a crash-safe generation cache. The model judge was
  dropped. ([#23](https://github.com/graylayer-labs/llm-post-training-lab/issues/23))
- Project docs: README, intent, decisions and the Part 1 results write-up.
- `tools/compare_generations.py` decodes greedily from the base model and
  from base + adapter, and reports whether each answer ended on a stop token.
  `tools/memory_lora_vs_full.py` measures peak memory for LoRA at any rank
  against a full fine-tune.
  ([#1](https://github.com/graylayer-labs/llm-post-training-lab/issues/1))
- `lab sft --config` trains LoRA adapters with a completion-only loss, plus
  the `<|im_start|>` and `<|im_end|>` embedding rows.
  ([#1](https://github.com/graylayer-labs/llm-post-training-lab/issues/1))
- Project scaffold: uv project, CI quality gate, typed YAML configs and the
  finance-alpaca loader.

**Result**
- Full eval at commit `76c9086`, 200 held-out prompts, greedy, batch 16,
  one seed. Rubric pass overall: base 35.0%, SFT 73.0%, DPO 89.0%. Stopped:
  39.5%, 86.5%, 100%. Mean new tokens: 270.0, 111.6, 51.9. Perplexity of
  the references: 8.910, 6.245, 6.555. ROUGE-L: 0.153, 0.297, 0.296. Wall
  time 1,562.8 s, contended by other GPU jobs. No model judge.
  ([results](docs/eval-results.md), [#3](https://github.com/graylayer-labs/llm-post-training-lab/issues/3))
- Length check on the saved generations at commit `b872427`: DPO's answers
  are shorter than SFT's even where SFT already stopped (50.4 against 69.1
  mean tokens on 173 prompts), with slightly lower ROUGE-L there (0.314
  against 0.330). DPO's gain is plausibly partly "shorter", not only "no
  loops". ([results](docs/eval-results.md), [#3](https://github.com/graylayer-labs/llm-post-training-lab/issues/3))
- DPO on top of SFT at commit `0263788`: 105 pairs from 500 greedy SFT
  answers on training prompts, 104 of them rejected for repetition loops.
  On 10 held-out pairs, a fit check and not the eval, DPO loss fell from
  0.693 to 0.058. On those pairs the rejected answers' log-probability
  fell while the chosen answers' barely moved. The reference is the merged SFT
  model, because TRL's own reference copy skips the trained chat-token
  rows. Peak memory 13.19 GiB. One seed.
  ([results](docs/dpo-results.md), [#2](https://github.com/graylayer-labs/llm-post-training-lab/issues/2))
- Clean SFT re-run at commit `76739e6` on the de-duplicated split: eval loss
  fell from 2.169 to 1.714 on 200 held-out answers. The saved probe puts
  `<|im_end|>` first in 0.795 of 200 reference positions for SFT, against
  0.00 for the base model. The wall time of 1,282.7 s was contended by other
  GPU jobs. ([results](docs/sft-results.md), ([#12](https://github.com/graylayer-labs/llm-post-training-lab/issues/12)))
- Rubric on the 2,200 reference answers: 98.6% pass overall, with 12.9% finance
  rows. (([#11](https://github.com/graylayer-labs/llm-post-training-lab/issues/11)))
- First SFT run: eval loss on 200 held-out answers fell from 2.28 to 1.83,
  and 4 of 5 sample answers now end on the stop token, against 0 of 5 with
  LoRA alone. Repetition loops remain. One seed.
  ([results](docs/sft-results.md),
  [#1](https://github.com/graylayer-labs/llm-post-training-lab/issues/1))

**Fixed**
- Four of the 200 held-out prompts also appeared in training because
  finance-alpaca repeats prompts. The split now drops repeated prompts before
  shuffling (7,587 dropped), so train and eval are disjoint. ([#15](https://github.com/graylayer-labs/llm-post-training-lab/issues/15))
- The first SFT re-run was killed at step 18 of 125 by another agent's
  `pkill -f`. It led to checkpoints (([#21](https://github.com/graylayer-labs/llm-post-training-lab/issues/21))) and a machine-wide guard
  against pattern kills; the re-run completed. ([#12](https://github.com/graylayer-labs/llm-post-training-lab/issues/12))
- Training on MPS ran out of memory as the allocator cache grew every step.
  The cache is now emptied after each step.
  ([#1](https://github.com/graylayer-labs/llm-post-training-lab/issues/1))
