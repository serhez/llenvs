"""Offline checks with the plan's exact Qwen tokenizer; no model weights."""

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from llenvs.integrations.skyrl._rendering import TextRenderer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
FILES = {
    "config.json": "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
}


@pytest.fixture(scope="module")
def qwen_tokenizer():
    explicit = os.environ.get("LLENVS_SKYRL_TOKENIZER")
    transformers = pytest.importorskip("transformers")
    if explicit:
        root = Path(explicit)
        assert root.is_absolute() and root.is_dir(), (
            "LLENVS_SKYRL_TOKENIZER must be a local absolute directory"
        )
    else:
        hub = pytest.importorskip("huggingface_hub")
        cached = hub.try_to_load_from_cache(MODEL, "tokenizer_config.json", revision=REVISION)
        if not isinstance(cached, str):
            pytest.skip(
                "pinned Qwen tokenizer is not cached; set LLENVS_SKYRL_TOKENIZER (no downloads)"
            )
        root = Path(cached).parent
    for name, expected in FILES.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, (
            f"tokenizer snapshot mismatch: {name}"
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    return tokenizer, json.loads((root / "config.json").read_text())["vocab_size"]


@pytest.mark.parametrize("system", [None, "Follow the task exactly.\n"])
@pytest.mark.parametrize(
    "observation",
    ["", " leading\tspace\n\n", "résumé 中文 🧪", "<tool_response>\n{}\n</tool_response>"],
)
def test_real_template_append_matches_explicit_qwen_tokens(qwen_tokenizer, system, observation):
    tokenizer, vocabulary = qwen_tokenizer
    renderer = TextRenderer(tokenizer, vocab_size=vocabulary, chat_template_kwargs={})
    opening = ([{"role": "system", "content": system}] if system is not None else []) + [
        {"role": "user", "content": "Task"}
    ]
    before = copy.deepcopy(opening)
    recorded = (
        renderer.initial(opening)
        + tokenizer.encode("reasoning\n", add_special_tokens=False)
        + [tokenizer.eos_token_id]
    )
    suffix = tokenizer.encode(
        f"\n<|im_start|>user\n{observation}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    actual = renderer.extend(recorded, {"role": "user", "content": observation})
    assert actual == recorded + suffix
    assert opening == before
    assert vocabulary > len(tokenizer)  # Actual output vocabulary includes extra entries.


def test_real_template_never_roundtrips_generated_byte_tokens(qwen_tokenizer):
    tokenizer, vocabulary = qwen_tokenizer
    renderer = TextRenderer(tokenizer, vocab_size=vocabulary, chat_template_kwargs={})
    # A standalone UTF-8 continuation byte decodes to U+FFFD and cannot be
    # reconstructed from text. It is still a genuine sampled vocabulary ID.
    sampled = [tokenizer.convert_tokens_to_ids("©"), tokenizer.eos_token_id]
    assert tokenizer.encode(tokenizer.decode(sampled), add_special_tokens=False) != sampled
    recorded = renderer.initial([{"role": "user", "content": "Task"}]) + sampled
    for text in ("first observation", "second observation"):
        extended = renderer.extend(recorded, {"role": "user", "content": text})
        assert extended[: len(recorded)] == recorded
        recorded = extended + sampled


def history(arguments, content=""):
    return [
        {"role": "system", "content": "Use the lookup tool."},
        {"role": "user", "content": "Look up Zürich."},
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": "call0",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": arguments},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call0", "content": "result"},
    ]


@pytest.mark.parametrize("content", ["", None])
def test_initial_tool_history_uses_template_objects_without_mutating_export(
    qwen_tokenizer, content
):
    tokenizer, vocabulary = qwen_tokenizer
    messages = history('{"city": "Zürich", "limit": 2}', content)
    original = copy.deepcopy(messages)
    renderer = TextRenderer(tokenizer, vocab_size=vocabulary, chat_template_kwargs={})
    ids = renderer.initial(messages)
    rendered = tokenizer.decode(ids, skip_special_tokens=False)
    call = json.loads(rendered.split("<tool_call>\n", 1)[1].split("\n</tool_call>", 1)[0])
    assert call == {"name": "lookup", "arguments": {"city": "Zürich", "limit": 2}}
    assert messages == original
    assert rendered.endswith("<|im_start|>assistant\n")
    assert "<tool_response>\nresult\n</tool_response>" in rendered


@pytest.mark.parametrize("arguments", ["not JSON", "[]", '{"x": 1, "x": 2}', '{"x": NaN}'])
def test_invalid_initial_tool_arguments_are_not_silently_repaired(qwen_tokenizer, arguments):
    tokenizer, vocabulary = qwen_tokenizer
    renderer = TextRenderer(tokenizer, vocab_size=vocabulary, chat_template_kwargs={})
    with pytest.raises(ValueError, match="argument|JSON|finite"):
        renderer.initial(history(arguments))
