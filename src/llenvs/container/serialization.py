"""JSON serialization for core types crossing the container boundary.

Provides bidirectional conversion between core dataclasses and JSON-compatible
dicts. All enums serialize as their name string. Tuples serialize as lists and
are reconstructed on deserialization using known field types.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import SignalBundle, Signal, RewardType
from llenvs.core.state import Action, ImageContent, Observation, State, StateMetadata
from llenvs.core.tools import (
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolResult,
    ToolResultStatus,
)


class OpaqueHidden:
    """Opaque hidden state supporting attribute access for container proxy.

    The client side doesn't know the concrete HiddenT type. This provides
    immutable attribute access over a plain dict, sufficient for host-side
    inspection (e.g., ``state.hidden.expected_answer``).
    """

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", data)
        for key, value in data.items():
            object.__setattr__(self, key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("OpaqueHidden is immutable")

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OpaqueHidden):
            return self._data == other._data
        return NotImplemented

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self._data.items())
        return f"OpaqueHidden({items})"


# ---------------------------------------------------------------------------
# Hidden state helpers
# ---------------------------------------------------------------------------


def _serialize_hidden(hidden: Any) -> dict[str, Any]:
    """Serialize hidden state to a JSON-compatible dict."""
    if isinstance(hidden, OpaqueHidden):
        return hidden.to_dict()
    if dataclasses.is_dataclass(hidden) and not isinstance(hidden, type):
        return _dataclass_to_dict(hidden)
    if isinstance(hidden, dict):
        return dict(hidden)
    # Fallback: try vars()
    try:
        return dict(vars(hidden))
    except TypeError:
        return {"_value": hidden}


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Recursively convert a dataclass to a dict, handling nested types."""
    result: dict[str, Any] = {}
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        result[f.name] = _make_json_safe(value)
    return result


def _make_json_safe(value: Any) -> Any:
    """Recursively convert a value to be JSON-serializable."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass_to_dict(value)
    if isinstance(value, RewardType):
        return value.name
    if isinstance(value, ToolParameterType):
        return value.value
    if isinstance(value, ToolResultStatus):
        return value.name
    # Fallback
    return str(value)


def reconstruct_hidden(hidden_dict: dict[str, Any], hidden_type: type) -> Any:
    """Reconstruct a concrete HiddenT from a dict using its dataclass type.

    Used server-side where the concrete type is known.
    """
    if not dataclasses.is_dataclass(hidden_type):
        return hidden_dict
    field_names = {f.name for f in dataclasses.fields(hidden_type)}
    kwargs = {k: v for k, v in hidden_dict.items() if k in field_names}
    return hidden_type(**kwargs)


# ---------------------------------------------------------------------------
# ToolResult / ToolCall serialization
# ---------------------------------------------------------------------------


def serialize_tool_call(tc: ToolCall) -> dict[str, Any]:
    return {"id": tc.id, "name": tc.name, "arguments": tc.arguments}


def deserialize_tool_call(data: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=data["id"],
        name=data["name"],
        arguments=data.get("arguments", {}),
    )


def serialize_tool_result(tr: ToolResult) -> dict[str, Any]:
    return {
        "call_id": tr.call_id,
        "tool_name": tr.tool_name,
        "status": tr.status.name,
        "output": tr.output,
        "error": tr.error,
    }


def deserialize_tool_result(data: dict[str, Any]) -> ToolResult:
    return ToolResult(
        call_id=data["call_id"],
        tool_name=data["tool_name"],
        status=ToolResultStatus[data["status"]],
        output=data.get("output", ""),
        error=data.get("error"),
    )


# ---------------------------------------------------------------------------
# ToolDefinition / ToolParameter serialization
# ---------------------------------------------------------------------------


def serialize_tool_parameter(tp: ToolParameter) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": tp.name,
        "type": tp.type.value,
        "description": tp.description,
        "required": tp.required,
    }
    if tp.enum is not None:
        result["enum"] = list(tp.enum)
    return result


def deserialize_tool_parameter(data: dict[str, Any]) -> ToolParameter:
    enum_val = data.get("enum")
    return ToolParameter(
        name=data["name"],
        type=ToolParameterType(data["type"]),
        description=data["description"],
        required=data.get("required", True),
        enum=tuple(enum_val) if enum_val is not None else None,
    )


def serialize_tool_definition(td: ToolDefinition) -> dict[str, Any]:
    return {
        "name": td.name,
        "description": td.description,
        "parameters": [serialize_tool_parameter(p) for p in td.parameters],
        "is_terminal": td.is_terminal,
    }


def deserialize_tool_definition(data: dict[str, Any]) -> ToolDefinition:
    return ToolDefinition(
        name=data["name"],
        description=data["description"],
        parameters=tuple(deserialize_tool_parameter(p) for p in data.get("parameters", [])),
        is_terminal=data.get("is_terminal", False),
    )


# ---------------------------------------------------------------------------
# Reward serialization
# ---------------------------------------------------------------------------


def serialize_reward_signal(sig: Signal) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": sig.name,
        "reward_type": sig.reward_type.name,
        "reward": sig.reward,
        "weight": sig.weight,
    }
    if sig.feedback is not None:
        d["feedback"] = sig.feedback
    if sig.metadata is not None:
        d["metadata"] = sig.metadata
    return d


def deserialize_reward_signal(data: dict[str, Any]) -> Signal:
    return Signal(
        name=data["name"],
        reward_type=RewardType[data["reward_type"]],
        reward=data.get("reward"),
        feedback=data.get("feedback"),
        metadata=data.get("metadata"),
        weight=data.get("weight", 1.0),
    )


def serialize_reward_bundle(bundle: SignalBundle) -> dict[str, Any]:
    return {"signals": [serialize_reward_signal(s) for s in bundle.signals]}


def deserialize_reward_bundle(data: dict[str, Any]) -> SignalBundle:
    return SignalBundle(signals=tuple(deserialize_reward_signal(s) for s in data["signals"]))


# ---------------------------------------------------------------------------
# Observation serialization
# ---------------------------------------------------------------------------


def serialize_observation(obs: Observation) -> dict[str, Any]:
    return {
        "prompt": obs.prompt,
        "messages": [dict(m) for m in obs.messages],
        "tool_results": [serialize_tool_result(tr) for tr in obs.tool_results],
        "available_tools": [serialize_tool_definition(td) for td in obs.available_tools],
        "images": [{"data": img.data, "media_type": img.media_type} for img in obs.images],
    }


def deserialize_observation(data: dict[str, Any]) -> Observation:
    return Observation(
        prompt=data["prompt"],
        messages=tuple(data.get("messages", ())),
        tool_results=tuple(deserialize_tool_result(tr) for tr in data.get("tool_results", ())),
        available_tools=tuple(
            deserialize_tool_definition(td) for td in data.get("available_tools", ())
        ),
        images=tuple(
            ImageContent(data=img["data"], media_type=img.get("media_type", "image/png"))
            for img in data.get("images", ())
        ),
    )


# ---------------------------------------------------------------------------
# StateMetadata serialization
# ---------------------------------------------------------------------------


def serialize_state_metadata(meta: StateMetadata) -> dict[str, Any]:
    return {
        "step": meta.step,
        "episode_id": meta.episode_id,
        "is_terminal": meta.is_terminal,
        "info": meta.info,
    }


def deserialize_state_metadata(data: dict[str, Any]) -> StateMetadata:
    return StateMetadata(
        step=data["step"],
        episode_id=data["episode_id"],
        is_terminal=data.get("is_terminal", False),
        info=data.get("info", {}),
    )


# ---------------------------------------------------------------------------
# State serialization
# ---------------------------------------------------------------------------


def serialize_state(state: State) -> dict[str, Any]:
    return {
        "observation": serialize_observation(state.observation),
        "hidden": _serialize_hidden(state.hidden),
        "metadata": serialize_state_metadata(state.metadata),
    }


def deserialize_state(data: dict[str, Any]) -> State[OpaqueHidden]:
    """Deserialize state on the client side (hidden becomes OpaqueHidden)."""
    return State(
        observation=deserialize_observation(data["observation"]),
        hidden=OpaqueHidden(data["hidden"]),
        metadata=deserialize_state_metadata(data["metadata"]),
    )


def deserialize_state_typed(data: dict[str, Any], hidden_type: type) -> State:
    """Deserialize state on the server side with a concrete hidden type."""
    return State(
        observation=deserialize_observation(data["observation"]),
        hidden=reconstruct_hidden(data["hidden"], hidden_type),
        metadata=deserialize_state_metadata(data["metadata"]),
    )


# ---------------------------------------------------------------------------
# Action serialization
# ---------------------------------------------------------------------------


def serialize_action(action: Action) -> dict[str, Any]:
    return {
        "text": action.text,
        "tool_calls": [serialize_tool_call(tc) for tc in action.tool_calls],
    }


def deserialize_action(data: dict[str, Any]) -> Action:
    return Action(
        text=data.get("text"),
        tool_calls=tuple(deserialize_tool_call(tc) for tc in data.get("tool_calls", ())),
    )


# ---------------------------------------------------------------------------
# StepResult serialization
# ---------------------------------------------------------------------------


def serialize_step_result(result: StepResult) -> dict[str, Any]:
    return {
        "next_state": serialize_state(result.next_state),
        "rewards": serialize_reward_bundle(result.rewards),
        "terminated": result.terminated,
        "truncated": result.truncated,
        "info": result.info,
    }


def deserialize_step_result(data: dict[str, Any]) -> StepResult[OpaqueHidden]:
    return StepResult(
        next_state=deserialize_state(data["next_state"]),
        rewards=deserialize_reward_bundle(data["rewards"]),
        terminated=data.get("terminated", False),
        truncated=data.get("truncated", False),
        info=data.get("info", {}),
    )


# ---------------------------------------------------------------------------
# EnvironmentSpec serialization
# ---------------------------------------------------------------------------


def serialize_env_spec(spec: EnvironmentSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "adapter": spec.adapter,
        "max_steps": spec.max_steps,
        "is_multi_turn": spec.is_multi_turn,
        "supports_task_index": spec.supports_task_index,
        "supports_len": spec.supports_len,
        "supports_seed": spec.supports_seed,
        "pure_step": spec.pure_step,
        "metadata": spec.metadata,
        # observation_type and action_type are type objects — not serialized
    }


def deserialize_env_spec(data: dict[str, Any]) -> EnvironmentSpec:
    return EnvironmentSpec(
        name=data["name"],
        adapter=data.get("adapter", ""),
        max_steps=data.get("max_steps"),
        is_multi_turn=data.get("is_multi_turn", False),
        supports_task_index=data.get("supports_task_index", True),
        supports_len=data.get("supports_len", True),
        supports_seed=data.get("supports_seed", True),
        pure_step=data.get("pure_step", False),
        metadata=data.get("metadata", {}),
    )
