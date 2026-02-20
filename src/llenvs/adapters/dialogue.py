"""Dialogue adapter — LLM-in-the-loop environments.

Provides ``DialogueEnvironment`` for alternating-turns patterns where
``step()`` calls a ``ModelBackend`` to generate observations. Useful for
20-questions, student-teacher, debate, and similar scenarios.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import RewardFunction, Signal, SignalBundle
from llenvs.core.state import Action, Observation, ObservationContent, State, StateMetadata
from llenvs.inference.protocol import ChatMessage, SamplingParams

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DialogueTask:
    """A single task for the dialogue environment.

    Attributes:
        prompt: Initial observation shown to the agent.
        context: Injected into the env LLM system prompt per-task.
        ground_truth: For reward computation (accessible via hidden state).
        metadata: Arbitrary per-task metadata.
    """

    prompt: str
    context: str = ""
    ground_truth: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DialogueHidden:
    """Hidden state for dialogue environments.

    Attributes:
        task_index: Index of the current task.
        task: The full DialogueTask (carries context, ground_truth).
        step_count: Number of steps taken so far.
    """

    task_index: int
    task: DialogueTask
    step_count: int

    @property
    def ground_truth(self) -> str:
        """Ground truth for reward computation (duck-typed by JudgeReward)."""
        return self.task.ground_truth


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

DIALOGUE_PRESETS: dict[str, dict[str, Any]] = {
    "twenty_questions": {
        "system_prompt_template": (
            "You are playing 20 Questions. {context}\n"
            "Answer the player's questions with only Yes, No, "
            "or Correct! (when they guess correctly)."
        ),
        "default_prompt": (
            "I'm thinking of something. You have 20 questions to figure "
            "out what it is. Ask yes/no questions to narrow it down."
        ),
        "default_max_steps": 20,
        "is_terminal": lambda env_resp, _action, _step: "correct" in env_resp.lower(),
    },
    "teacher": {
        "system_prompt_template": (
            "You are a patient teacher. {context}\n"
            "Review the student's answer and provide constructive feedback."
        ),
        "default_prompt": "{question}",
        "default_max_steps": 5,
        "is_terminal": None,
    },
}


# ---------------------------------------------------------------------------
# Default sampling params
# ---------------------------------------------------------------------------

_DEFAULT_SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=512)


# ---------------------------------------------------------------------------
# DialogueEnvironment
# ---------------------------------------------------------------------------


class DialogueEnvironment:
    """Environment where an LLM generates observations for the agent.

    Implements the ``Environment`` protocol. Each ``step()`` call builds a
    conversation from the observation history, calls the env LLM backend,
    and returns the LLM response as the next observation.

    Attributes:
        spec: Environment specification.
        reward_functions: Tuple of reward functions (only extra_rewards).
    """

    def __init__(
        self,
        backend: Any,
        tasks: tuple[DialogueTask, ...],
        system_prompt: str = "",
        sampling_params: SamplingParams | None = None,
        max_steps: int = 20,
        extra_rewards: tuple[RewardFunction, ...] = (),
        is_terminal: Callable[[str, str, int], bool] | None = None,
    ) -> None:
        """Initialize the dialogue environment.

        Args:
            backend: ModelBackend for generating env LLM responses.
            tasks: Tuple of DialogueTask instances.
            system_prompt: Base system prompt template. May contain
                ``{context}`` placeholder filled per-task.
            sampling_params: Sampling params for env LLM calls.
                Defaults to temperature=0, max_tokens=512.
            max_steps: Maximum steps before truncation.
            extra_rewards: Additional reward functions.
            is_terminal: Callback ``(env_response, agent_action, step) -> bool``
                to determine if the episode should terminate.
        """
        self._backend = backend
        self._tasks = tasks
        self._system_prompt = system_prompt
        self._sampling_params = sampling_params or _DEFAULT_SAMPLING_PARAMS
        self._max_steps = max_steps
        self._extra_rewards = extra_rewards
        self._is_terminal = is_terminal

    @property
    def prompts(self) -> dict[str, str]:
        return {}

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="dialogue",
            adapter="dialogue",
            max_steps=self._max_steps,
            is_multi_turn=True,
            pure_step=True,
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        return self._extra_rewards

    def __len__(self) -> int:
        return len(self._tasks)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[DialogueHidden], dict[str, Any]]:
        """Reset the environment and return the initial state.

        Args:
            seed: Random seed (unused — tasks are deterministic).
            options: Must contain ``task_index`` (defaults to 0).

        Returns:
            Tuple of (initial_state, info_dict).

        Raises:
            ValueError: If task_index is out of bounds.
        """
        options = options or {}
        task_index = options.get("task_index", 0)

        if task_index < 0 or task_index >= len(self._tasks):
            raise ValueError(f"task_index {task_index} out of bounds [0, {len(self._tasks)})")

        task = self._tasks[task_index]
        observation = Observation(
            prompt=task.prompt, messages=(), task=ObservationContent(text=task.prompt)
        )
        hidden = DialogueHidden(task_index=task_index, task=task, step_count=0)

        episode_id = options.get("episode_id", str(uuid.uuid4()))
        metadata = StateMetadata(
            step=0,
            episode_id=episode_id,
            is_terminal=False,
            info={"task_index": task_index},
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        info = {"task_index": task_index}
        return state, info

    def step(
        self,
        state: State[DialogueHidden],
        action: Action,
    ) -> StepResult[DialogueHidden]:
        """Take an action and generate the env LLM response.

        Args:
            state: Current state.
            action: Agent's response.

        Returns:
            StepResult with next state, rewards, and done flags.
        """
        task = state.hidden.task
        action_text = action.text or ""

        # 1. Build system prompt with task-specific context
        if task.context and "{context}" in self._system_prompt:
            system = self._system_prompt.format(context=task.context)
        else:
            system = self._system_prompt

        # 2. Build messages for env LLM
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))

        # Reconstruct conversation from observation messages
        for msg in state.observation.messages:
            messages.append(ChatMessage(role=msg["role"], content=msg["content"]))

        # Add agent's current action
        messages.append(ChatMessage(role="user", content=action_text))

        # 3. Call backend
        result = self._backend.generate_chat(messages, self._sampling_params)
        env_response = result.text or ""

        # 4. Check termination
        step_count = state.hidden.step_count + 1
        terminated = (
            self._is_terminal(env_response, action_text, step_count) if self._is_terminal else False
        )
        truncated = step_count >= self._max_steps

        # 5. Build next observation (append to message history)
        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": action_text},
            {"role": "user", "content": env_response},
        )
        next_observation = Observation(
            prompt=state.observation.prompt,
            messages=new_messages,
            task=state.observation.task,
            state=ObservationContent(text=env_response),
        )

        # 6. Build next hidden and state
        next_hidden = DialogueHidden(
            task_index=state.hidden.task_index,
            task=task,
            step_count=step_count,
        )
        next_metadata = StateMetadata(
            step=step_count,
            episode_id=state.metadata.episode_id,
            is_terminal=terminated or truncated,
            info={**state.metadata.info, "env_response": env_response},
        )
        next_state = State(
            observation=next_observation,
            hidden=next_hidden,
            metadata=next_metadata,
        )

        # 7. Compute rewards
        rewards = self.compute_rewards(state, action, next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info={"env_response": env_response, "step_count": step_count},
        )

    def compute_rewards(
        self,
        state: State[DialogueHidden],
        action: Action,
        next_state: State[DialogueHidden],
    ) -> SignalBundle:
        """Compute rewards for a transition.

        Args:
            state: State before action.
            action: Action taken.
            next_state: State after action.

        Returns:
            SignalBundle with signals from extra_rewards only.
        """
        signals: list[Signal] = []
        for reward_fn in self.reward_functions:
            signals.append(reward_fn.compute(state, action, next_state))
        return SignalBundle(signals=tuple(signals))


# ---------------------------------------------------------------------------
# DialogueAdapter
# ---------------------------------------------------------------------------


class DialogueAdapter:
    """Adapter for dialogue environments.

    Creates ``DialogueEnvironment`` instances from preset names and
    task specifications.
    """

    @property
    def name(self) -> str:
        return "dialogue"

    def list_environments(self) -> list[str]:
        return list(DIALOGUE_PRESETS.keys())

    def get_environment(
        self,
        name: str,
        *,
        env_llm: Any | None = None,
        tasks: list[dict[str, Any]] | None = None,
        words: list[str] | None = None,
        questions: list[dict[str, str]] | None = None,
        preset: str | None = None,
        max_steps: int | None = None,
        system_prompt: str | None = None,
        sampling_params: SamplingParams | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        answer_extractor: Any | None = None,
        **kwargs: Any,
    ) -> DialogueEnvironment:
        """Create a DialogueEnvironment.

        Args:
            name: Preset name or arbitrary environment name.
            env_llm: ModelBackend for env LLM (required).
            tasks: Direct list of task dicts with prompt/context/ground_truth.
            words: Shorthand for 20-questions (list of secret words).
            questions: Shorthand for teacher (list of {question, answer} dicts).
            preset: Built-in preset name (auto-detected from ``name``).
            max_steps: Override preset default max steps.
            system_prompt: Override preset system prompt template.
            sampling_params: Sampling params for env LLM.
            extra_rewards: Additional reward functions.
            answer_extractor: Ignored (dialogue has no native extraction).
            **kwargs: Ignored extra kwargs for registry compatibility.

        Returns:
            Configured DialogueEnvironment.

        Raises:
            ValueError: If no backend is provided.
        """
        if env_llm is None:
            raise ValueError("env_llm (ModelBackend) is required for dialogue environments")

        preset_name = preset or name
        config = DIALOGUE_PRESETS.get(preset_name, {})

        # Resolve system prompt
        resolved_system_prompt = system_prompt or config.get("system_prompt_template", "")

        # Resolve max_steps
        resolved_max_steps = max_steps or config.get("default_max_steps", 20)

        # Resolve is_terminal
        is_terminal = config.get("is_terminal")

        # Build tasks
        dialogue_tasks: list[DialogueTask] = []

        if tasks is not None:
            for t in tasks:
                dialogue_tasks.append(
                    DialogueTask(
                        prompt=t.get("prompt", ""),
                        context=t.get("context", ""),
                        ground_truth=t.get("ground_truth", ""),
                        metadata=t.get("metadata", {}),
                    )
                )
        elif words is not None:
            default_prompt = config.get("default_prompt", "Guess the word.")
            for word in words:
                dialogue_tasks.append(
                    DialogueTask(
                        prompt=default_prompt,
                        context=f"The secret word is: {word}. Answer only Yes, No, or Correct!",
                        ground_truth=word,
                    )
                )
        elif questions is not None:
            for q in questions:
                question_text = q["question"]
                answer_text = q.get("answer", "")
                prompt_template = config.get("default_prompt", "{question}")
                dialogue_tasks.append(
                    DialogueTask(
                        prompt=prompt_template.format(question=question_text),
                        context=f"Correct answer: {answer_text}. Give feedback on the student's attempt.",
                        ground_truth=answer_text,
                    )
                )
        else:
            raise ValueError("Must provide one of: tasks, words, or questions")

        return DialogueEnvironment(
            backend=env_llm,
            tasks=tuple(dialogue_tasks),
            system_prompt=resolved_system_prompt,
            sampling_params=sampling_params,
            max_steps=resolved_max_steps,
            extra_rewards=extra_rewards,
            is_terminal=is_terminal,
        )

    def get_native_answer_extractor(self, task_name: str) -> None:
        return None

    def get_default_system_prompt(self, name: str) -> None:
        return None

    def get_prompt_template(self, name: str) -> None:
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        preset = DIALOGUE_PRESETS.get(name, {})
        return {
            "name": name,
            "adapter": "dialogue",
            "type": "multi_turn",
            "max_steps": preset.get("default_max_steps", 20),
        }
