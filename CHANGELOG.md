# Changelog

Key changes to the project, newest first. Routine fixes and wording changes
are left out; the git log has those. Entries are grouped by epic and date, and
each links to the issue that holds the detail.

## Epic: Hands-on LLM post-training and evaluation (in progress)

### 2026-10-03

**Added**
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
- First SFT run: eval loss on 200 held-out answers fell from 2.28 to 1.83,
  and 4 of 5 sample answers now end on the stop token, against 0 of 5 with
  LoRA alone. Repetition loops remain. One seed.
  ([results](docs/sft-results.md),
  [#1](https://github.com/graylayer-labs/llm-post-training-lab/issues/1))

**Fixed**
- Training on MPS ran out of memory as the allocator cache grew every step.
  The cache is now emptied after each step.
  ([#1](https://github.com/graylayer-labs/llm-post-training-lab/issues/1))
