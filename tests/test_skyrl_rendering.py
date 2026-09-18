"""Fixed-base observation tokenization never re-encodes sampled assistant text."""

import copy
import importlib

import pytest

from tests.test_skyrl_tokenizer import history


class Tokenizer:
    eos_token_id = 0

    def __init__(self):
        self.calls = []

    def __len__(self):
        return 256

    def apply_chat_template(
        self, messages, *, add_generation_prompt, tokenize, return_dict, **kwargs
    ):
        assert tokenize is True and return_dict is False
        self.calls.append((copy.deepcopy(messages), kwargs))
        result = []
        for message in messages:
            result += [1, {"system": 2, "user": 3, "assistant": 4}[message["role"]]]
            result += [ord(c) for c in message["content"]]
            result += [0, 5]
        if add_generation_prompt:
            result += [1, 4]
        return result


@pytest.fixture
def rendering():
    return importlib.import_module("llenvs.integrations.skyrl._rendering")


def test_append_uses_exact_sampled_prefix_even_when_text_cannot_roundtrip(rendering):
    tokenizer = Tokenizer()
    renderer = rendering.TextRenderer(
        tokenizer, vocab_size=256, chat_template_kwargs={"enable_thinking": True}
    )
    opening = [{"role": "user", "content": "Task"}]
    initial = renderer.initial(opening)
    # Arbitrary actual sampled IDs, including a valid EOS=0; never decode them.
    trace = initial + [200, 201, 0]
    actual = renderer.extend(trace, {"role": "user", "content": "Next"})
    assert actual == trace + [5, 1, 3, *map(ord, "Next"), 0, 5, 1, 4]
    assert all(m["role"] != "assistant" for messages, _ in tokenizer.calls for m in messages)
    assert all(kwargs == {"enable_thinking": True} for _, kwargs in tokenizer.calls)
    assert opening == [{"role": "user", "content": "Task"}]


def test_prefix_sensitive_template_is_rejected_not_sliced_blindly(rendering):
    class Bad(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            if len(messages) > 2:
                result[0] = 10
            return result

    renderer = rendering.TextRenderer(Bad(), vocab_size=256, chat_template_kwargs={})
    with pytest.raises(ValueError, match="prefix"):
        renderer.extend([1, 0], {"role": "user", "content": "Next"})


@pytest.mark.parametrize("content", ["", None])
def test_hf_history_boundary_converts_arguments_only_on_owned_messages(rendering, content):
    class Capture(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(copy.deepcopy(messages))
            return [1, 0]

    tokenizer = Capture()
    messages = history('{"n": 2}', content)
    original = copy.deepcopy(messages)
    renderer = rendering.TextRenderer(tokenizer, vocab_size=256, chat_template_kwargs={})
    renderer.initial(messages)
    assert tokenizer.calls[-1][2]["tool_calls"][0]["function"]["arguments"] == {"n": 2}
    assert tokenizer.calls[-1][2]["content"] == ""
    assert messages == original


@pytest.mark.parametrize("arguments", ["broken", "[]", '{"x": 1, "x": 2}', '{"x": NaN}'])
def test_invalid_history_arguments_fail_before_template_encoding(rendering, arguments):
    tokenizer = Tokenizer()
    renderer = rendering.TextRenderer(tokenizer, vocab_size=256, chat_template_kwargs={})
    with pytest.raises(ValueError, match="tool arguments"):
        renderer.initial(history(arguments))
    assert len(tokenizer.calls) == 1  # Only the fixed-base check, not bad history.


@pytest.mark.parametrize("damage", ["no_eos", "reserved_kwargs", "image"])
def test_unverified_rendering_paths_fail_closed(rendering, damage):
    tokenizer = Tokenizer()
    kwargs = {}
    if damage == "no_eos":
        tokenizer.eos_token_id = None
    elif damage == "reserved_kwargs":
        kwargs["tokenize"] = False
    with pytest.raises(ValueError):
        renderer = rendering.TextRenderer(tokenizer, vocab_size=256, chat_template_kwargs=kwargs)
        if damage == "image":
            renderer.initial(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Task"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,eA=="},
                            },
                        ],
                    }
                ]
            )
