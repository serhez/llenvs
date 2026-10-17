"""Shared sampling validation and detached, credential-free request records."""

import json
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from llenvs.inference.protocol import SamplingParams

DECODING_FIELDS = (
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
)
_OWNED_KEYS = set(DECODING_FIELDS) | {
    "reasoning_effort",
    "response_format",
    "structured_outputs",
    "reasoning",
    "thinking",
    "n",
    "num_return_sequences",
    "max_new_tokens",
    "max_completion_tokens",
    "max_length",
    "do_sample",
    "generation_config",
    "stop",
    "stop_sequences",
    "logprobs",
    "top_logprobs",
    "num_logprobs",
    "output_scores",
    "return_dict_in_generate",
    "model",
    "messages",
    "prompt",
    "input",
    "input_ids",
    "attention_mask",
    "tools",
    "tool_choice",
    "stream",
    "stop_strings",
    "api_key",
    "api_base",
    "base_url",
    "client",
    "async_client",
    "http_client",
}
_PRIVATE_KEYS = {
    "messages",
    "prompt",
    "input",
    "input_ids",
    "attention_mask",
    "pixel_values",
    "api_key",
    "authorization",
    "proxy-authorization",
    "access_token",
    "password",
    "extra_headers",
    "headers",
    "http_client",
    "client",
    "async_client",
    "callbacks",
    "callback",
    "logger_fn",
    "logits_processor",
    "logits_processors",
}
_OMIT = object()


def validate_extra(extra: Mapping[str, Any], *, owned: tuple[str, ...] = ()) -> None:
    """Reject provider aliases of typed controls, including merged extra bodies."""
    keys = _OWNED_KEYS | set(owned)
    for mapping in (extra, extra.get("extra_body", {})):
        if not isinstance(mapping, Mapping):
            raise ValueError("extra_body must be a mapping")
        collisions = keys & mapping.keys()
        if collisions:
            raise ValueError(f"extra cannot override owned request keys: {sorted(collisions)}")


def validate_sampling(
    params: SamplingParams,
    *,
    unsupported: Mapping[str, float] | None = None,
    single_result: bool = True,
) -> None:
    if single_result and params.n != 1:
        raise ValueError("n must be 1 for an operation returning one result per input")
    validate_extra(params.extra)
    for name, neutral in (unsupported or {}).items():
        value = getattr(params, name)
        if value is not None and value != neutral:
            raise ValueError(f"{name}={value!r} is unsupported by this backend")


def _record_value(value: Any, *, schema: bool = False) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _record_value(asdict(value), schema=schema)
    names = getattr(type(value), "__struct_fields__", None)
    if names is not None:
        return _record_value({name: getattr(value, name) for name in names}, schema=schema)
    if schema:
        return json.loads(json.dumps(value, allow_nan=False))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or key.lower() in _PRIVATE_KEYS:
                continue
            if key in {"api_base", "base_url", "endpoint", "url"} and isinstance(item, str):
                url = urlsplit(item)
                query = [
                    (k, v)
                    for k, v in parse_qsl(url.query, keep_blank_values=True)
                    if k.lower() not in _PRIVATE_KEYS | {"key", "token"}
                ]
                item = urlunsplit(
                    (url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, urlencode(query), "")
                )
            recorded = _record_value(
                item,
                schema=key
                in {"response_format", "response_schema", "structured_outputs", "parameters"},
            )
            if recorded is not _OMIT:
                result[key] = recorded
        return result
    if isinstance(value, (tuple, list)):
        return [recorded for item in value if (recorded := _record_value(item)) is not _OMIT]
    return _OMIT


def sampling_metadata(
    params: SamplingParams,
    request: Mapping[str, Any],
    *,
    omitted: tuple[str, ...] = (),
    capture: str | None = None,
) -> dict[str, Any]:
    """Copy JSON data only; never retain inputs, credentials or live objects."""
    metadata = {
        "request": _record_value(request),
        "requested_sampling": _record_value(
            {f.name: getattr(params, f.name) for f in fields(params)}
        ),
        "inherited_controls": [name for name in DECODING_FIELDS if getattr(params, name) is None],
        "omitted_controls": list(omitted),
        "response_schema_applied": params.response_schema is not None,
        "after_reasoning_text_applied": False,
    }
    metadata["requests"] = [metadata["request"]]
    if capture is not None:
        metadata["request_capture"] = capture
    return metadata
