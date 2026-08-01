#!/usr/bin/env python3
"""Generate chat-template and encoding fixtures for `ramvamp_core::tokenizer`.

Downloads the pinned Qwen3-30B-A3B-Instruct-2507 tokenizer, renders and
tokenizes a set of chat conversations with the reference `transformers`
implementation, and writes:

- crates/core/src/tokenizer/fixtures/chat_fixtures.json
- crates/core/src/tokenizer/fixtures/tokenizer/{tokenizer.json,
  tokenizer_config.json, generation_config.json}  (copied verbatim from the
  pinned revision; these are the offline test vectors)

Run (network required, fish shell):

    uv run --with transformers,jinja2 scripts/gen_tokenizer_fixtures.py

The Rust tests in crates/core/src/tokenizer/ replay these fixtures with no
network access. Re-run this script only if the model pin changes.
"""

import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "crates" / "core" / "src" / "tokenizer" / "fixtures"
TOKENIZER_DIR = FIXTURES_DIR / "tokenizer"

LONG_CODE_MESSAGE = """Please review this function:

```rust
fn top_k(logits: &mut [f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(k);
    idx
}
```

Two questions:

1. Is the `unwrap` on `partial_cmp` safe for NaN logits?
2. What is the complexity, and is there an O(n) selection alternative?

Thanks!"""

# name -> (messages, add_generation_prompt)
CHAT_CASES = [
    (
        "single_user",
        [
            {
                "role": "user",
                "content": "Give me a short introduction to large language models.",
            }
        ],
        True,
    ),
    (
        "system_user",
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        True,
    ),
    (
        "three_turn",
        [
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "Tell me a joke about compilers."},
        ],
        True,
    ),
    (
        "system_multi_turn",
        [
            {
                "role": "system",
                "content": "You are a terse assistant. Answer in one sentence.",
            },
            {"role": "user", "content": "Why is the sky blue?"},
            {
                "role": "assistant",
                "content": "Rayleigh scattering sends short blue wavelengths every which way.",
            },
            {"role": "user", "content": "And sunsets?"},
        ],
        True,
    ),
    (
        "arabic",
        [{"role": "user", "content": "ما هي عاصمة مصر؟"}],
        True,
    ),
    (
        "emoji_cjk",
        [
            {
                "role": "user",
                "content": "こんにちは 👋🌍! 你好，世界 — can you mix 🚀 emoji and 漢字?",
            }
        ],
        True,
    ),
    (
        "empty_system",
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Ping"},
        ],
        True,
    ),
    (
        "long_code_block",
        [{"role": "user", "content": LONG_CODE_MESSAGE}],
        True,
    ),
    (
        # Non-leading system message: the 2507 template renders it in place.
        "mid_conversation_system",
        [
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi there."},
            {"role": "system", "content": "From now on answer only in French."},
            {"role": "user", "content": "How are you?"},
        ],
        True,
    ),
    (
        # Full conversation ending with the assistant, no generation prompt.
        "full_conversation_no_gen_prompt",
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Name one prime number."},
            {"role": "assistant", "content": "Two."},
        ],
        False,
    ),
]

TEXT_CASES = [
    "Hello, world!",
    "ما هي عاصمة مصر؟",
    "こんにちは 👋🌍! 你好，世界 — mixed 🚀 text with 漢字.",
    "<|im_start|>user\nhi<|im_end|>\n",
    "line one\nline two\n\n    indented code\n\ttab",
    "",
]


def main() -> int:
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION)

    chat_fixtures = []
    for name, messages, add_generation_prompt in CHAT_CASES:
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        tokenized = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt
        )
        # Depending on the transformers version this is a flat id list or a
        # BatchEncoding.
        ids = tokenized["input_ids"] if not isinstance(tokenized, list) else tokenized
        # The Rust side implements encode_chat as render + encode. Pin that
        # equivalence here so a mismatch fails at generation time, loudly.
        re_encoded = tok(rendered, add_special_tokens=False)["input_ids"]
        assert ids == re_encoded, f"{name}: apply_chat_template != render+encode"
        chat_fixtures.append(
            {
                "name": name,
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "rendered": rendered,
                "token_ids": ids,
            }
        )

    text_fixtures = []
    for text in TEXT_CASES:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        decoded = tok.decode(ids, skip_special_tokens=False)
        assert decoded == text, f"decode round-trip failed for {text!r}"
        text_fixtures.append({"text": text, "token_ids": ids})

    fixtures = {
        "repo": REPO,
        "revision": REVISION,
        "chat_cases": chat_fixtures,
        "text_cases": text_fixtures,
    }

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    out_path = FIXTURES_DIR / "chat_fixtures.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {out_path} ({len(chat_fixtures)} chat, {len(text_fixtures)} text)")

    for fname in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        src = hf_hub_download(REPO, fname, revision=REVISION)
        dst = TOKENIZER_DIR / fname
        shutil.copyfile(src, dst)
        print(f"copied {fname} -> {dst} ({dst.stat().st_size} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
