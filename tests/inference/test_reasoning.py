"""Reasoning control and extraction preserve the native sampled sequence."""

from copy import deepcopy

import pytest

from llenvs.inference.backends.api import OpenAIBackend
from llenvs.inference.protocol import ChatMessage, ResponseSchema, SamplingParams
from llenvs.inference.reasoning import local_template_kwargs, reasoning_view


def backend(model="gpt-6-luna", flavor="openai"):
    result = OpenAIBackend.__new__(OpenAIBackend)
    result._model = model
    result._api_flavor = flavor
    return result


def request(params, model="gpt-6-luna"):
    return backend(model)._request_kwargs([ChatMessage("user", "question")], params)


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", None])
def test_luna_effort_and_completion_budget(effort):
    body = request(SamplingParams(reasoning_effort=effort, temperature=None, top_p=None))
    assert body["max_completion_tokens"] == 2048
    assert "max_tokens" not in body
    assert "temperature" not in body and "top_p" not in body
    if effort is None:
        assert "reasoning_effort" not in body
    else:
        assert body["reasoning_effort"] == effort


@pytest.mark.parametrize("control", [{"temperature": 0}, {"top_p": 1}, {"logprobs": True}])
@pytest.mark.parametrize("effort", [None, "medium", "xhigh"])
def test_luna_reasoning_requires_explicit_omission(control, effort):
    with pytest.raises(ValueError):
        request(
            SamplingParams(
                **({"temperature": None, "top_p": None, "reasoning_effort": effort} | control)
            )
        )


def test_luna_none_permits_decoding_controls():
    assert request(SamplingParams(reasoning_effort="none"))["temperature"] == 0


def test_luna_max_is_rejected_before_dispatch():
    with pytest.raises(ValueError, match="Unsupported Luna reasoning_effort"):
        request(SamplingParams(reasoning_effort="max", temperature=None, top_p=None))


@pytest.mark.parametrize("model,effort", [("gpt-6-luna", "bogus"), ("unknown", "medium")])
def test_unknown_effort_or_family_raises(model, effort):
    with pytest.raises(ValueError):
        request(SamplingParams(reasoning_effort=effort, temperature=None, top_p=None), model)


@pytest.mark.parametrize("effort", [None, "none", "low", "medium", "xhigh"])
def test_qwen_template_mapping(effort):
    defaults = {"enable_thinking": False, "preserve_thinking": True}
    before = deepcopy(defaults)
    actual = local_template_kwargs("qwen3.8", effort, defaults)
    assert actual["preserve_thinking"] is True
    if effort is not None:
        assert actual["enable_thinking"] == (effort != "none")
    if effort not in (None, "none"):
        assert actual["reasoning_effort"] == effort
    assert defaults == before


def test_qwen_rejects_high_and_conflicting_request_extras():
    with pytest.raises(ValueError):
        local_template_kwargs("qwen3.8", "high", {})
    with pytest.raises(ValueError):
        local_template_kwargs("qwen3.8", "low", {}, {"enable_thinking": False})


def test_unqualified_local_effort_family_is_rejected():
    with pytest.raises(ValueError, match="family"):
        local_template_kwargs("qwen3.5", "low", {})


def test_local_qwen_history_uses_native_reasoning_field_without_mutation():
    from llenvs.inference.reasoning import local_chat_messages

    message = ChatMessage("assistant", "answer", reasoning="retained thought")
    assert local_chat_messages([message], "qwen3.8") == [
        {"role": "assistant", "content": "answer", "reasoning_content": "retained thought"}
    ]
    assert message.to_dict()["reasoning"] == "retained thought"
    assert local_chat_messages([message], None) == [message.to_dict()]


def test_local_qwen_rejects_opaque_provider_reasoning_instead_of_dropping_it():
    from llenvs.inference.reasoning import local_chat_messages

    message = ChatMessage(
        "assistant", "answer", reasoning_details=({"type": "encrypted", "data": "opaque"},)
    )
    with pytest.raises(ValueError, match="reasoning_details"):
        local_chat_messages([message], "qwen3.8")


def test_reasoning_view_has_exact_answer_span_and_retains_closure():
    view = reasoning_view([4, 5, 99, 6, 7], close_id=99, decode=lambda ids: str(list(ids)))
    assert view["answer_ids"] == [6, 7]
    assert view["answer_span"] == [3, 5]
    assert view["reasoning_ids"] == [4, 5]
    assert view["raw_token_ids"] == [4, 5, 99, 6, 7]
    assert not view["reasoning_truncated"]


def test_unclosed_reasoning_never_becomes_the_answer():
    view = reasoning_view([4, 5], close_id=99, decode=lambda ids: str(list(ids)))
    assert view["answer_ids"] == [] and view["reasoning_truncated"]


def test_schema_is_detached_and_maps_to_native_chat_format():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    spec = ResponseSchema(schema, name="prediction")
    schema["properties"].clear()
    body = request(SamplingParams(reasoning_effort="none", response_schema=spec))
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "prediction", "schema": spec.schema, "strict": True},
    }
    assert "a" in spec.schema["properties"]


def test_closed_api_rejects_insertion_before_dispatch():
    with pytest.raises(ValueError, match="after_reasoning_text"):
        request(SamplingParams(reasoning_effort="none", after_reasoning_text="Return JSON."))


def test_config_roundtrip_preserves_controlled_answer_fields():
    from llenvs.core.config import EvalConfig, create_sampling_params

    values = {
        "reasoning_effort": "none",
        "after_reasoning_text": "Return JSON.",
        "response_schema": {"name": "pad", "schema": {"type": "object"}},
    }
    config = EvalConfig.from_dict({"inference": values})
    params = create_sampling_params(config.inference)
    assert params.reasoning_effort == "none"
    assert params.after_reasoning_text == "Return JSON."
    assert params.response_schema == ResponseSchema({"type": "object"}, name="pad")
    assert EvalConfig.from_dict(config.to_dict()).inference == config.inference


def test_local_answer_view_retains_full_native_logprobs():
    from llenvs.inference.protocol import GenerationResult, TokenLogprob
    from llenvs.inference.reasoning import extract_local_reasoning

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [98 if text == "<think>" else 99]

        def decode(self, ids, **kwargs):
            return str(list(ids))

    native = tuple(TokenLogprob(str(i), i, -0.1) for i in [4, 99, 5])
    result = extract_local_reasoning(
        GenerationResult(text="raw", token_logprobs=native),
        [4, 99, 5],
        Tokenizer(),
        reasoning_enabled=True,
    )
    assert result.text == "[5]" and result.token_logprobs == native[2:]
    assert result.metadata["raw_text"] == "raw"
    assert result.metadata["raw_token_logprobs"][1]["token_id"] == 99
    assert result.reasoning_tokens == 1


def test_hf_explicitly_rejects_schema_guarantee():
    from llenvs.inference.backends.huggingface import HuggingFaceBackend

    instance = HuggingFaceBackend.__new__(HuggingFaceBackend)
    with pytest.raises(ValueError, match="schema"):
        instance._to_generate_kwargs(
            SamplingParams(response_schema=ResponseSchema({"type": "object"}))
        )


def test_schema_field_names_are_preserved_in_public_request_records():
    spec = ResponseSchema(
        {
            "type": "object",
            "properties": {"input": {"type": "integer"}, "api_key": {"type": "string"}},
        }
    )
    params = SamplingParams(reasoning_effort="none", response_schema=spec)
    instance = backend()
    body = instance._request_kwargs([ChatMessage("user", "private question")], params)
    metadata = instance._request_metadata(body, params)
    assert metadata["request"]["response_format"] == body["response_format"]
    assert metadata["requested_sampling"]["response_schema"]["schema"] == spec.schema
    assert "messages" not in metadata["request"]


@pytest.mark.parametrize("backend_name", ["HuggingFaceBackend", "VLLMBackend"])
def test_local_effort_requires_a_chat_template_not_an_already_rendered_prompt(backend_name):
    from llenvs.inference import backends

    cls = getattr(backends, backend_name)
    instance = cls.__new__(cls)
    with pytest.raises(ValueError, match="chat"):
        instance.generate(["already rendered"], SamplingParams(reasoning_effort="medium"))


def test_provider_reasoning_usage_takes_precedence_over_visible_id_counts():
    from llenvs.inference.protocol import GenerationResult
    from llenvs.inference.reasoning import extract_local_reasoning

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [99 if text == "</think>" else 98]

        def decode(self, ids, **kwargs):
            return str(list(ids))

    result = extract_local_reasoning(
        GenerationResult(text="raw", completion_tokens=30, reasoning_tokens=17),
        [4, 99, 5],
        Tokenizer(),
        reasoning_enabled=True,
    )
    assert result.reasoning_tokens == 17
    assert result.metadata["reasoning_tokens_source"] == "provider"
    assert result.metadata["reasoning_token_ids_count"] == 1
