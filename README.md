# llm-post-training-lab

Config-driven LoRA supervised fine-tuning, DPO preference optimisation and an
evaluation harness for a small open LLM, runnable on a single Apple-silicon
laptop or one consumer GPU.

```bash
uv sync --dev
uv run lab sft  --config configs/sft.yaml
uv run lab dpo  --config configs/dpo.yaml
uv run lab eval --config configs/eval.yaml
```

Design notes and findings are in `NOTES.md`.
