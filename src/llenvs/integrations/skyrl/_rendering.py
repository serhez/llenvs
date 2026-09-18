"""Text observation rendering with SkyRL's fixed-base, append-only convention.

Only observations are encoded after the opening request. Sampled assistant IDs
are never decoded/re-encoded to recover training history. This does not certify
arbitrary templates or multimodal expansion; those need separate renderers.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from llenvs.integrations.skyrl._checks import integer, token_ids
from llenvs.integrations.skyrl.data import _canonical_json, _unique_object, _validate_messages


class TextRenderer:
    def __init__(
        self, tokenizer: Any, *, vocab_size: int, chat_template_kwargs: dict[str, Any]
    ) -> None:
        self.tokenizer = tokenizer
        self.kwargs = copy.deepcopy(chat_template_kwargs)
        if set(self.kwargs) - {"enable_thinking"}:
            raise ValueError("text rendering only admits the enable_thinking template kwarg")
        if "enable_thinking" in self.kwargs and not isinstance(
            self.kwargs["enable_thinking"], bool
        ):
            raise ValueError("enable_thinking must be a boolean")
        self.eos_id = integer(tokenizer.eos_token_id, "tokenizer.eos_token_id")
        self.vocab_size = integer(vocab_size, "model vocabulary size", minimum=1)
        self._base = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I am a user."},
        ]
        base_ids = self._encode(self._base, add_generation_prompt=False)
        if self.eos_id not in base_ids:
            raise ValueError("fixed-base text template must contain its EOS delimiter")
        boundary = len(base_ids) - base_ids[::-1].index(self.eos_id)
        self._base_prefix = base_ids[:boundary]

    def _encode(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
        _validate_messages(messages)
        messages = copy.deepcopy(messages)
        # Exported initial history uses OpenAI's serialized arguments; HF chat
        # templates expect objects. Convert only this owned rendering input,
        # never an already sampled assistant reply or the fingerprinted row.
        for message in messages:
            for call in message.get("tool_calls", []):
                try:
                    arguments = json.loads(
                        call["function"]["arguments"], object_pairs_hook=_unique_object
                    )
                    _canonical_json(arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be an object")
                except ValueError as exc:
                    raise ValueError(
                        "initial tool arguments require a finite JSON object with unique keys"
                    ) from exc
                call["function"]["arguments"] = arguments
            if message.get("content") is None:
                # _validate_messages admits null content only for a tool-call
                # assistant message, not for an observation or arbitrary role.
                message["content"] = ""
        if any(not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("image/content-part input requires a verified multimodal renderer")
        ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=False,
            **self.kwargs,
        )
        token_ids(ids, "rendered input")
        if not ids or any(token >= self.vocab_size for token in ids):
            raise ValueError("rendered input is empty or outside the vocabulary")
        return list(ids)

    def initial(self, messages: list[dict[str, Any]]) -> list[int]:
        return self._encode(messages, add_generation_prompt=True)

    def extend(self, recorded_ids: list[int], observation: dict[str, Any]) -> list[int]:
        token_ids(recorded_ids, "recorded input prefix")
        if not recorded_ids or any(token >= self.vocab_size for token in recorded_ids):
            raise ValueError("recorded prefix is empty or outside the vocabulary")
        if observation.get("role") != "user":
            raise ValueError("observation must be one user message")
        encoded = self._encode([*self._base, observation], add_generation_prompt=True)
        boundary = len(self._base_prefix)
        if encoded[:boundary] != self._base_prefix:
            raise ValueError("observation template changed its fixed-base token prefix")
        return [*recorded_ids, *encoded[boundary:]]
