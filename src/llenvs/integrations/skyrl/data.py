"""Deterministic task preparation and plain-JSON dataset contracts for SkyRL.

No policy, judge, SkyRL runtime, tokenizer, or training worker is constructed.
Environment construction/reset can still have the adapter's own side effects.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import hashlib
import inspect
import json
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llenvs.core.config import EnvironmentConfig, EnvironmentFactory, EvalConfig
from llenvs.core.tool_parsing import HermesToolCallParser
from llenvs.inference.prompts import resolve_system_prompt
from llenvs.inference.protocol import ChatMessage
from llenvs.integrations.dataset_provider import DatasetProvider, TaskItem
from llenvs.integrations.skyrl._checks import identifier, integer

_ROW_FIELDS = {"prompt", "env_class", "data_source", "llenvs"}
_IDENTITY_FIELDS = {
    "schema_version",
    "env_fingerprint",
    "task_index",
    "task_id",
    "initial_fingerprint",
}
_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


def _canonical_json(value: Any) -> str:
    # JSON itself coerces non-string keys and accepts some non-JSON containers.
    # Reject these rather than giving distinct configurations the same identity.
    def check(item: Any) -> None:
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise ValueError("task data and environment identity must be JSON-compatible")

    check(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _task_id(env_fingerprint: str, index: int) -> str:
    return _fingerprint({"env_fingerprint": env_fingerprint, "task_index": index})


def _validate_image_url(url: Any) -> None:
    if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
        raise ValueError("image input must be an inline base64 data URL")
    media_type, encoded = url[5:].split(";base64,", 1)
    if media_type not in _IMAGE_TYPES:
        raise ValueError(f"unsupported image MIME type: {media_type}")
    try:
        if not base64.b64decode(encoded, validate=True):
            raise ValueError("image base64 data cannot be empty")
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid image base64 data") from exc


def _validate_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("prompt must be a nonempty list of messages")
    pending_tools: set[str] = set()
    seen_tools: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or set(message) - {
            "role",
            "content",
            "tool_calls",
            "tool_call_id",
            "name",
        }:
            raise ValueError("unsupported initial message fields")
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("unsupported initial message role")
        content = message.get("content")
        calls = message.get("tool_calls")
        if content is None:
            if role != "assistant" or not calls:
                raise ValueError("message content must be text or inline text/image parts")
        elif isinstance(content, list):
            if not content:
                raise ValueError("message content parts cannot be empty")
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("invalid message content part")
                if part.get("type") == "text" and set(part) == {"type", "text"}:
                    if not isinstance(part["text"], str):
                        raise ValueError("message text must be a string")
                elif part.get("type") == "image_url" and set(part) == {"type", "image_url"}:
                    image = part["image_url"]
                    if not isinstance(image, dict) or set(image) != {"url"}:
                        raise ValueError("image_url must contain only the inline URL")
                    _validate_image_url(image["url"])
                else:
                    raise ValueError("unsupported message modality or content fields")
        elif not isinstance(content, str):
            raise ValueError("message content must be text or inline text/image parts")
        if "name" in message:
            identifier(message["name"], "message.name")
        if role == "tool":
            call_id = identifier(message.get("tool_call_id"), "tool_call_id")
            if call_id not in pending_tools:
                raise ValueError("tool result does not match a pending initial tool call")
            pending_tools.remove(call_id)
        elif "tool_call_id" in message or pending_tools:
            raise ValueError("initial tool history has unmatched tool calls/results")
        if calls is not None:
            if role != "assistant" or not isinstance(calls, list) or not calls:
                raise ValueError("tool_calls must be a nonempty assistant call list")
            for call in calls:
                if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
                    raise ValueError("initial tool calls require the OpenAI function-call schema")
                call_id = identifier(call["id"], "tool call id")
                function = call["function"]
                if (
                    call["type"] != "function"
                    or not isinstance(function, dict)
                    or set(function) != {"name", "arguments"}
                ):
                    raise ValueError("invalid initial tool function")
                identifier(function["name"], "tool function name")
                if not isinstance(function["arguments"], str):
                    raise ValueError("tool arguments must retain their serialized string")
                if call_id in seen_tools:
                    raise ValueError("duplicate initial tool call id")
                seen_tools.add(call_id)
                pending_tools.add(call_id)
    if pending_tools:
        raise ValueError("initial tool calls require matching results before generation")
    _canonical_json(messages)


def _initial_messages(item: TaskItem, system_prompt: str | None) -> list[dict[str, Any]]:
    if not isinstance(item.prompt, str):
        raise ValueError("task prompt must be text")
    if item.available_tools:
        _canonical_json([tool.to_openai_schema() for tool in item.available_tools])
        preamble = HermesToolCallParser().format_tools(item.available_tools)
        system_prompt = f"{system_prompt}\n\n{preamble}" if system_prompt else preamble
    messages = []
    if system_prompt is not None:
        messages.append(ChatMessage(role="system", content=system_prompt).to_dict())
    messages.append(ChatMessage(role="user", content=item.prompt, images=item.images.all).to_dict())
    messages.extend(copy.deepcopy(item.messages))
    _validate_messages(messages)
    return messages


def _export_config(
    config: EvalConfig, env_name: str | None
) -> tuple[EnvironmentConfig, str | None, str]:
    choices = [env for env in config.environments if env_name is None or env.name == env_name]
    if len(choices) != 1:
        names = ", ".join(env.name for env in config.environments)
        raise ValueError(f"select exactly one environment with env_name; available: {names}")
    env_config = copy.deepcopy(choices[0])
    for owner, names in (
        (config, ("model_profile", "prompt_template")),
        (env_config, ("prompt_template", "branching_strategy")),
    ):
        for name in names:
            if getattr(owner, name) is not None:
                raise ValueError(f"{name} is not supported by the SkyRL opening-message contract")
    value = (
        env_config.system_prompt if env_config.system_prompt is not None else config.system_prompt
    )
    system_prompt = resolve_system_prompt(value) if value is not None else None
    definition = asdict(env_config)
    definition.pop("judge")
    definition.pop("system_prompt")
    if env_config.difficulties is not None:
        definition["difficulties"] = sorted(env_config.difficulties)
    fingerprint = _fingerprint(
        {"environment": definition, "system_prompt": system_prompt, "initial_message_format": 1}
    )
    return env_config, system_prompt, fingerprint


def _close_environment(environment: Any) -> None:
    for name in ("close", "shutdown"):
        close = getattr(environment, name, None)
        if callable(close):
            if inspect.iscoroutinefunction(close):
                raise TypeError("synchronous task export requires synchronous environment cleanup")
            result = close()
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("synchronous task export requires synchronous environment cleanup")
            return


def export_prompt_data(
    config: EvalConfig | str | Path,
    output_path: str | Path,
    *,
    env_name: str | None = None,
    indices: Sequence[int] | None = None,
    num_tasks: int | None = None,
) -> int:
    """Export indexed tasks atomically, without overwriting an existing path.

    Selection order is preserved; omitted selectors mean all tasks. The adapter
    must support finite task indexing and synchronous reset/cleanup. The YAML's
    policy/judge settings are not instantiated; an environment-owned LLM can
    still be constructed when required by the selected environment.
    """
    return _export_prompt_data(
        config, output_path, env_name=env_name, indices=indices, num_tasks=num_tasks
    )


def _export_prompt_data(
    config: EvalConfig | str | Path,
    output_path: str | Path,
    *,
    env_name: str | None = None,
    indices: Sequence[int] | None = None,
    num_tasks: int | None = None,
    start: int = 0,
) -> int:
    output = Path(output_path)
    if os.path.lexists(output):
        raise FileExistsError(output)
    if indices is not None and num_tasks is not None:
        raise ValueError("indices and num_tasks are mutually exclusive")
    if num_tasks is not None:
        integer(num_tasks, "num_tasks", minimum=1)
    integer(start, "start")
    cfg = config if isinstance(config, EvalConfig) else EvalConfig.from_yaml(config)
    env_config, system_prompt, fingerprint = _export_config(cfg, env_name)
    environment = EnvironmentFactory.create(env_config)
    temporary: Path | None = None
    try:
        try:
            provider = DatasetProvider(environment)
            size = len(provider)
            if start >= size:
                raise ValueError("start is outside the environment's task range")
            if num_tasks is not None and num_tasks > size - start:
                raise ValueError(f"num_tasks exceeds environment length {size}")
            selected = (
                list(indices)
                if indices is not None
                else range(start, start + num_tasks if num_tasks is not None else size)
            )
            if not selected:
                raise ValueError("task selection cannot be empty")
            for index in selected:
                integer(index, "task_index")
                if index >= size:
                    raise ValueError(f"task_index {index} is outside environment length {size}")
            if len(set(selected)) != len(selected):
                raise ValueError("duplicate task indices")
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                for index in selected:
                    messages = _initial_messages(provider[index], system_prompt)
                    row = {
                        "prompt": messages,
                        "env_class": "llenvs",
                        "data_source": f"{env_config.adapter}/{env_config.name}",
                        "llenvs": {
                            "schema_version": 1,
                            "env_fingerprint": fingerprint,
                            "task_index": index,
                            "task_id": _task_id(fingerprint, index),
                            "initial_fingerprint": _fingerprint(messages),
                        },
                    }
                    stream.write(_canonical_json(row) + "\n")
        finally:
            _close_environment(environment)
        # A same-filesystem link publishes complete data and fails if a racer
        # created the destination. replace()/rename() could overwrite that file.
        os.link(temporary, output)
        return len(selected)
    finally:
        if temporary is not None:
            temporary.unlink()


def _validate_row(row: Any, env_fingerprint: str) -> None:
    if not isinstance(row, dict) or set(row) != _ROW_FIELDS or row["env_class"] != "llenvs":
        raise ValueError("invalid SkyRL task row fields or env_class")
    identifier(row["data_source"], "data_source")
    identity = row["llenvs"]
    if not isinstance(identity, dict) or set(identity) != _IDENTITY_FIELDS:
        raise ValueError("invalid llenvs task identity fields")
    if integer(identity["schema_version"], "schema_version") != 1:
        raise ValueError("unsupported task schema_version")
    for name in ("env_fingerprint", "task_id", "initial_fingerprint"):
        if not isinstance(identity[name], str) or not re.fullmatch("[0-9a-f]{64}", identity[name]):
            raise ValueError(f"invalid {name}")
    if identity["env_fingerprint"] != env_fingerprint:
        raise ValueError("task env_fingerprint differs from selected environment")
    index = integer(identity["task_index"], "task_index")
    if identity["task_id"] != _task_id(env_fingerprint, index):
        raise ValueError("task_id does not match environment and task_index")
    validate_initial_messages(row, row["prompt"])


def validate_initial_messages(row: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    """Reject a fresh reset whose visible inputs differ from the prepared task."""
    _validate_messages(messages)
    if _fingerprint(messages) != row["llenvs"]["initial_fingerprint"]:
        raise ValueError("initial reset messages differ from the prepared initial_fingerprint")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


class LlenvsPromptDataset:
    """Validated JSON rows with stable UIDs and SkyRL's list-of-dicts collation.

    No tokenizer filtering, model state, live environment, or positional UID is
    stored. Rendered budgets and train/eval split checks belong to the runtime.
    """

    def __init__(self, paths: Sequence[str | Path], *, env_fingerprint: str) -> None:
        if not isinstance(env_fingerprint, str) or not re.fullmatch(
            "[0-9a-f]{64}", env_fingerprint
        ):
            raise ValueError("invalid expected env_fingerprint")
        self._rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for path in paths:
            with Path(path).open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line, object_pairs_hook=_unique_object)
                        _validate_row(row, env_fingerprint)
                        task_id = row["llenvs"]["task_id"]
                        if task_id in seen:
                            raise ValueError("duplicate task_id across dataset rows/files")
                        seen.add(task_id)
                        self._rows.append(row)
                    except ValueError as exc:
                        raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not self._rows:
            raise ValueError("prompt dataset cannot be empty")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[list[dict[str, Any]], str, dict[str, Any], str]:
        row = copy.deepcopy(self._rows[index])
        return (
            row["prompt"],
            row["env_class"],
            {"data_source": row["data_source"], "llenvs": row["llenvs"]},
            row["llenvs"]["task_id"],
        )

    @staticmethod
    def collate_fn(
        items: list[tuple[list[dict[str, Any]], str, dict[str, Any], str]],
    ) -> list[dict[str, Any]]:
        return [
            {"prompt": prompt, "env_class": env_class, "env_extras": extras, "uid": uid}
            for prompt, env_class, extras, uid in items
        ]


def validate_task_splits(
    train: LlenvsPromptDataset, evaluation: LlenvsPromptDataset | None, *, train_batch_size: int
) -> None:
    """Refuse overlapping identities and native drop-last loss of a training tail."""
    integer(train_batch_size, "train_batch_size", minimum=1)
    if not len(train) or len(train) % train_batch_size:
        raise ValueError("training length must be positive and divisible by train_batch_size")
    if evaluation is not None:
        train_ids = {row["llenvs"]["task_id"] for row in train._rows}
        if any(row["llenvs"]["task_id"] in train_ids for row in evaluation._rows):
            raise ValueError("training/evaluation task identities overlap")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export finite indexed llenvs tasks for SkyRL.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--env", dest="env_name")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-tasks", type=int)
    args = parser.parse_args(argv)
    count = _export_prompt_data(
        args.config, args.output, env_name=args.env_name, start=args.start, num_tasks=args.num_tasks
    )
    print(f"Exported {count} tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
