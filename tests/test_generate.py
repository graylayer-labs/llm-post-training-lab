from typing import Any

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from post_training.generate import generate, stop_token_ids, trim_at_stop

IM_END, EOT, PAD = 1, 2, 0


class StubTok:
    """Char-level tokenizer; ids 0-2 are pad / <|im_end|> / <|endoftext|>."""

    padding_side = "right"
    pad_token_id = PAD
    eos_token_id = EOT

    def convert_tokens_to_ids(self, t: str) -> int:
        return {"<|im_end|>": IM_END, "<|endoftext|>": EOT}[t]

    def apply_chat_template(
        self, messages: Any, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        return "".join(m["content"] for m in messages)

    def __call__(
        self, texts: list[str], return_tensors: str = "pt", padding: bool = True
    ) -> dict[str, torch.Tensor]:
        ids = [[3 + ord(c) % 60 for c in t] for t in texts]
        n = max(map(len, ids))
        rows, masks = [], []
        for x in ids:
            pad = [PAD] * (n - len(x))
            if self.padding_side == "left":
                rows.append(pad + x)
                masks.append([0] * len(pad) + [1] * len(x))
            else:
                rows.append(x + pad)
                masks.append([1] * len(x) + [0] * len(pad))
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.tensor(masks),
        }

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        kept = [int(i) for i in ids if not (skip_special_tokens and int(i) < 3)]
        return "".join(chr(97 + i % 26) for i in kept)


def _model() -> Any:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        pad_token_id=PAD,
    )
    return LlamaForCausalLM(cfg).eval()


def _msgs(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


PROMPTS = [_msgs("hi"), _msgs("a much longer prompt here"), _msgs("mid length")]


def test_stop_token_ids_cover_both_markers() -> None:
    assert stop_token_ids(StubTok()) == [IM_END, EOT]


def test_trim_at_stop_counts_through_first_stop() -> None:
    ids, stopped = trim_at_stop(torch.tensor([9, 8, IM_END, 7, PAD]), [IM_END, EOT])
    assert ids.tolist() == [9, 8, IM_END]
    assert stopped is True


def test_trim_at_stop_without_stop_keeps_all() -> None:
    ids, stopped = trim_at_stop(torch.tensor([9, 8, 7]), [IM_END, EOT])
    assert ids.tolist() == [9, 8, 7]
    assert stopped is False


def test_batched_greedy_equals_unbatched() -> None:
    model, tok = _model(), StubTok()
    one = generate(model, tok, PROMPTS, max_new_tokens=8, batch_size=1)
    many = generate(model, tok, PROMPTS, max_new_tokens=8, batch_size=3)
    assert [r["text"] for r in one] == [r["text"] for r in many]
    assert [r["new_tokens"] for r in one] == [r["new_tokens"] for r in many]
    assert all(set(r) == {"text", "new_tokens", "stopped"} for r in many)


def test_padding_side_is_restored() -> None:
    tok = StubTok()
    generate(_model(), tok, PROMPTS, max_new_tokens=2, batch_size=2)
    assert tok.padding_side == "right"


def test_new_tokens_stop_at_first_stop_token() -> None:
    model = _model()
    first = generate(model, StubTok(), PROMPTS[:1], max_new_tokens=4, batch_size=1)
    assert first[0]["new_tokens"] == 4 and first[0]["stopped"] is False

    # Make the first greedy token a stop token: generation must end at 1 token.
    tok = StubTok()
    enc = tok(["".join(m["content"] for m in PROMPTS[0])])
    out = model.generate(**enc, max_new_tokens=1, do_sample=False)
    tok.eos_token_id = int(out[0, -1])
    res = generate(model, tok, PROMPTS[:1], max_new_tokens=4, batch_size=1)
    assert res[0]["new_tokens"] == 1 and res[0]["stopped"] is True


def test_sampling_is_seeded() -> None:
    model, tok = _model(), StubTok()
    a = generate(model, tok, PROMPTS, 6, 2, do_sample=True, top_p=0.9, seed=7)
    b = generate(model, tok, PROMPTS, 6, 2, do_sample=True, top_p=0.9, seed=7)
    assert a == b
