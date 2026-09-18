"""Synchronous environment boundary, run serially per episode by the generator."""

from __future__ import annotations

import copy
import json
from typing import Any

from llenvs.core.config import EnvironmentFactory
from llenvs.core.environment import StepResult
from llenvs.core.reward import SignalBundle
from llenvs.core.state import Action, State
from llenvs.core.tool_parsing import HermesToolCallParser
from llenvs.inference.protocol import ChatMessage
from llenvs.integrations.dataset_provider import DatasetProvider, TaskItem
from llenvs.integrations.skyrl._checks import finite_number
from llenvs.integrations.skyrl._config import SelectedEnvironment
from llenvs.integrations.skyrl.data import (
    _canonical_json,
    _close_environment,
    _initial_messages,
    _validate_messages,
    _validate_row,
    validate_initial_messages,
)


def reward_total(bundle: SignalBundle) -> float:
    """Validate components, then preserve SignalBundle's weighted-sum semantics."""
    for signal in bundle.signals:
        weight = finite_number(signal.weight, f"{signal.name}.weight")
        if signal.reward is not None:
            value = finite_number(signal.reward, f"{signal.name}.reward")
            finite_number(value * weight, f"{signal.name}.weighted_reward")
    return finite_number(bundle.total, "environment reward total")


def feedback_text(state: State[Any]) -> str:
    """Preserve environment feedback selection without embedding image payloads."""
    observation = state.observation
    if observation.tool_results:
        blocks = []
        for result in observation.tool_results:
            content = result.output
            if not isinstance(content, dict):
                content = str(content) or result.error or "(no output)"
            payload = json.dumps({"name": result.tool_name, "content": content}, default=str)
            blocks.append(f"<tool_response>\n{payload}\n</tool_response>")
        text = "\n".join(blocks)
    elif observation.state is not None and observation.state.text:
        text = observation.state.text
    elif observation.messages and observation.messages[-1].get("role") == "user":
        text = observation.messages[-1].get("content", "")
    else:
        text = observation.prompt
    if not isinstance(text, str):
        raise ValueError("environment feedback must be text")
    return text


def feedback_message(state: State[Any]) -> dict[str, Any]:
    """One user observation; raw Hermes calls have no structured assistant node."""
    message = ChatMessage(
        role="user", content=feedback_text(state), images=state.observation.get_images().all
    ).to_dict()
    _validate_messages([message])
    return message


def _tools(state: State[Any]) -> str:
    return _canonical_json([tool.to_openai_schema() for tool in state.observation.available_tools])


class Episode:
    """One owned environment, with no policy, scoring, threading, or retry logic.

    The async owner must always close it, including when open/reset fails. The
    interface is deliberately serial; arbitrary thread-affine adapters are not
    certified by running these methods in a shared executor.
    """

    def __init__(self, selected: SelectedEnvironment, row: dict[str, Any]) -> None:
        _validate_row(row, selected.fingerprint)
        self._selected = selected
        self._row = copy.deepcopy(row)
        self._environment: Any = None
        self._opened = False
        self._ended = False
        self.state: State[Any] | None = None
        self._tool_schema = ""

    def open(self) -> list[dict[str, Any]]:
        if self._opened or self._ended:
            raise RuntimeError("episode cannot be opened twice or after it ended")
        self._opened = True
        self._environment = EnvironmentFactory.create(
            copy.deepcopy(self._selected.environment_config)
        )
        provider = DatasetProvider(self._environment)
        index = self._row["llenvs"]["task_index"]
        if index >= len(provider):
            raise ValueError("task_index exceeds the fresh environment's task count")
        state, _ = self._environment.reset(options={"task_index": index})
        self.state = state
        item = TaskItem(
            task_index=index,
            prompt=state.observation.prompt,
            messages=state.observation.messages,
            ground_truth=None,
            metadata={},
            images=state.observation.get_images(),
            available_tools=state.observation.available_tools,
        )
        messages = _initial_messages(item, self._selected.system_prompt)
        validate_initial_messages(self._row, messages)
        self._tool_schema = _tools(state)
        return messages

    def step(self, raw_reply: str) -> StepResult[Any]:
        if self.state is None or self._ended:
            raise RuntimeError("episode has not opened or has ended")
        if not isinstance(raw_reply, str):
            raise TypeError("policy reply must be raw text")
        tools = self.state.observation.available_tools
        action = Action.from_text(raw_reply)
        if tools:
            parsed = HermesToolCallParser().parse(raw_reply, tools)
            if parsed.tool_calls:
                action = Action(text=parsed.text, tool_calls=tuple(parsed.tool_calls))
        result = self._environment.step(self.state, action)
        reward_total(result.rewards)
        self.state = result.next_state
        self._ended = result.done
        if not result.done and _tools(self.state) != self._tool_schema:
            raise ValueError("available tool schema changed during the episode")
        return result

    def close(self) -> None:
        self._ended = True
        environment, self._environment = self._environment, None
        if environment is not None:
            _close_environment(environment)

    def reward_names(self) -> set[str]:
        """Names declared by the environment, not connector-added judges."""
        return {reward.name for reward in getattr(self._environment, "reward_functions", ())}
