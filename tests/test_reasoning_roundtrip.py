"""Reasoning must survive API parsing without becoming an answer/action."""

import pickle
from copy import deepcopy
from types import SimpleNamespace

import pytest

from llenvs.inference.backends.api import OpenRouterBackend
from llenvs.inference.protocol import ChatMessage, SamplingParams


def _response(**fields):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", **fields),
                finish_reason="length",
                logprobs=None,
            )
        ],
        model="qwen/test",
        id="offline-response",
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=8192),
    )


def _backend():
    # Exercise the real parser/request builder without creating an SDK client.
    backend = object.__new__(OpenRouterBackend)
    backend._model = "qwen/test"
    return backend


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"reasoning": None, "reasoning_content": None, "reasoning_details": None},
        {"reasoning": "existing thought"},
        {"reasoning_content": "existing thought"},
        {"reasoning_details": [{"type": "reasoning.text", "text": "existing thought"}]},
    ],
)
def test_real_sdk_response_preserves_optional_reasoning(fields):
    from openai.types.chat import ChatCompletion

    response = ChatCompletion.model_validate(
        {
            "id": "offline-response",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen/test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "answer", **fields},
                }
            ],
        }
    )
    result = _backend()._chat_result(response)
    assert result.text == "answer"
    if fields.get("reasoning_details"):
        assert result.metadata["reasoning_details"] == fields["reasoning_details"]
    elif fields.get("reasoning") or fields.get("reasoning_content"):
        assert result.metadata["reasoning"] == "existing thought"
    else:
        assert not result.metadata.get("reasoning_present")
        assert not result.metadata.get("reasoning_details_present")


@pytest.mark.parametrize(
    "fields",
    [
        {"reasoning": {"not": "a string"}},
        {"reasoning_details": "not a list of blocks"},
        {"reasoning_details": ["not a mapping"]},
    ],
)
def test_malformed_reasoning_is_not_silently_discarded_or_stringified(fields):
    with pytest.raises(ValueError, match="(?i)reasoning"):
        _backend()._chat_result(_response(**fields))


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_openrouter_preserves_plain_reasoning_separately(field):
    reasoning = "Consider candidate 3, then candidate 1. Still working..."
    result = _backend()._chat_result(_response(**{field: reasoning}))

    assert result.text == ""
    assert result.metadata["reasoning"] == reasoning
    assert result.metadata["reasoning_chars"] == len(reasoning)
    assert result.to_agent_action().text == ""


def test_openrouter_preserves_details_order_signatures_and_unknown_fields():
    details = [
        {
            "type": "reasoning.text",
            "text": "unfinished thought",
            "index": 0,
            "signature": "opaque-signature",
            "provider_field": {"keep": True},
        },
        {"type": "reasoning.encrypted", "data": "opaque-data", "index": 1},
    ]
    expected = deepcopy(details)
    result = _backend()._chat_result(_response(reasoning_details=details))

    assert result.text == ""
    assert result.metadata["reasoning_details"] == expected
    details[0]["provider_field"]["keep"] = False
    assert result.metadata["reasoning_details"] == expected
    assert pickle.loads(pickle.dumps(result)).metadata["reasoning_details"] == expected


def test_sdk_reasoning_blocks_are_plain_data_with_nulls_and_extras_preserved():
    from pydantic import BaseModel, ConfigDict

    class Block(BaseModel):
        model_config = ConfigDict(extra="allow")
        type: str
        text: str | None = None

    block = Block(type="reasoning.encrypted", data="opaque", signature="signature")
    result = _backend()._chat_result(_response(reasoning_details=[block]))
    assert result.metadata["reasoning_details"] == [
        {
            "type": "reasoning.encrypted",
            "text": None,
            "data": "opaque",
            "signature": "signature",
        }
    ]
    block.signature = "mutated"
    assert result.metadata["reasoning_details"][0]["signature"] == "signature"


def test_assistant_reasoning_reaches_request_without_changing_output_policy():
    reasoning = "The candidate labelled 4 is unsafe."
    message = ChatMessage(role="assistant", content="", reasoning=reasoning)
    params = SamplingParams(max_tokens=512, disable_thinking=True)
    request = _backend()._chat_kwargs([message], params)

    assert request["messages"][0]["reasoning"] == reasoning
    assert request["messages"][0]["content"] == ""
    assert request["max_tokens"] == 512
    assert request["extra_body"]["reasoning"] == {"effort": "none"}


def test_structured_reasoning_reaches_request_as_intact_blocks():
    details = ({"type": "reasoning.text", "text": "partial", "signature": "sig"},)
    message = ChatMessage(role="assistant", content="", reasoning_details=details)
    request = _backend()._chat_kwargs([message], SamplingParams(disable_thinking=True))

    assert request["messages"][0]["reasoning_details"] == list(details)
    assert "reasoning" not in request["messages"][0]
    assert "partial" not in request["messages"][0]["content"]


def test_plain_messages_keep_their_existing_wire_representation():
    message = ChatMessage(role="assistant", content="answer")
    assert message.to_dict() == {"role": "assistant", "content": "answer"}


def test_old_pickled_chat_message_without_reasoning_fields_remains_readable():
    message = ChatMessage(role="assistant", content="old answer")
    message.__dict__.pop("reasoning", None)
    message.__dict__.pop("reasoning_details", None)
    restored = pickle.loads(pickle.dumps(message))
    assert restored.to_dict() == {"role": "assistant", "content": "old answer"}


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_litellm_also_retains_plain_reasoning(field):
    pytest.importorskip("litellm")
    from llenvs.inference.backends.litellm import LiteLLMBackend

    backend = object.__new__(LiteLLMBackend)
    backend._model = "offline/test"
    result = backend._parse_response(_response(**{field: "available reasoning"}))
    assert result.text == ""
    assert result.metadata["reasoning"] == "available reasoning"
