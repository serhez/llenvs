"""Iterative refinement environment wrapper.

Turns any single-turn environment into a multi-turn loop: the agent
submits a solution, receives feedback, and refines — repeating until
solved, max turns reached, or the agent submits early.

Can also be used standalone with ``IterativeTask`` instances, without
wrapping an inner environment.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.extraction import RawGenerationExtractor
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import Action, Observation, ObservationContent, State, StateMetadata

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IterativeTask:
    """A standalone task for the iterative environment.

    Attributes:
        prompt: The task description shown to the agent.
        ground_truth: Expected answer (for reward computation).
        test_code: Test code for code execution rewards.
        metadata: Arbitrary per-task metadata.
    """

    prompt: str
    ground_truth: str = ""
    test_code: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IterativeHidden:
    """Hidden state for the iterative environment.

    Attributes:
        task_index: Index of the current task.
        inner_hidden: Hidden state from the inner environment (or IterativeTask).
        task_prompt: The original task prompt text.
        turn: Current turn number (0 = initial, 1+ = after submission).
        submissions: All submissions so far.
        feedback_history: Feedback texts per turn.
        max_turns: Maximum turns allowed.
    """

    task_index: int
    inner_hidden: Any
    task_prompt: str
    turn: int
    submissions: tuple[str, ...]
    feedback_history: tuple[tuple[str, ...], ...]
    max_turns: int

    @property
    def ground_truth(self) -> str:
        """Duck-typed for JudgeReward compatibility."""
        for attr in ("expected_answer", "answer", "ground_truth", "target"):
            val = getattr(self.inner_hidden, attr, None)
            if val is not None:
                return str(val)
        if isinstance(self.inner_hidden, IterativeTask):
            return self.inner_hidden.ground_truth
        return ""

    def __getattr__(self, name: str) -> Any:
        """Proxy to inner_hidden for transparent duck-typing."""
        return getattr(self.inner_hidden, name)


# ---------------------------------------------------------------------------
# History summarizer
# ---------------------------------------------------------------------------

HistorySummarizer = Callable[[tuple[str, ...], tuple[tuple[str, ...], ...]], str]
"""Callable that summarizes submission/feedback history into a string."""


def _default_history_formatter(
    submissions: tuple[str, ...],
    feedback_history: tuple[tuple[str, ...], ...],
    entry_template: str,
) -> str:
    """Format history entries using the entry template."""
    parts: list[str] = []
    for i, (submission, feedbacks) in enumerate(zip(submissions, feedback_history)):
        feedback_text = "\n".join(feedbacks) if feedbacks else "(no feedback)"
        entry = entry_template.format(
            turn=i + 1,
            submission=submission,
            feedback=feedback_text,
        )
        parts.append(entry)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------

_DEFAULT_PROMPTS: dict[str, str] = {
    "initial": (
        "{task}\n\n"
        "You have {turns_remaining} turn(s) to solve this task.\n"
        "When you are satisfied with your solution, write SUBMIT at the start of your response."
    ),
    "feedback": (
        "## Feedback on your previous submission (Turn {turn}/{max_turns})\n\n"
        "{feedback}\n\n"
        "{history_section}"
        "You have {turns_remaining} turn(s) remaining.\n"
        "Revise your solution or write SUBMIT to finalize."
    ),
    "history_entry": (
        "### Turn {turn}\n**Submission:**\n{submission}\n\n**Feedback:**\n{feedback}"
    ),
}


# ---------------------------------------------------------------------------
# IterativeEnvironment
# ---------------------------------------------------------------------------


class IterativeEnvironment:
    """Multi-turn iterative refinement environment.

    Wraps a single-turn environment (or standalone tasks) into a multi-turn
    loop. On each turn, the agent's submission is evaluated, feedback is
    collected from reward signals, and the agent can refine.

    Termination conditions:
    - **Solved**: Best OUTCOME value >= ``solved_threshold``
    - **Early submit**: Agent starts response with ``submit_keyword``
    - **Max turns**: ``turn >= max_turns``
    """

    def __init__(
        self,
        inner: Any | None = None,
        tasks: tuple[IterativeTask, ...] | None = None,
        submission_extractor: Any | None = None,
        max_turns: int = 3,
        include_history: bool = True,
        history_summarizer: HistorySummarizer | None = None,
        submit_keyword: str | None = "SUBMIT",
        solved_threshold: float = 1.0,
        extra_rewards: tuple[RewardFunction, ...] = (),
        prompts: dict[str, str] | None = None,
    ) -> None:
        if inner is None and tasks is None:
            raise ValueError("Must provide either 'inner' or 'tasks'")

        self._inner = inner
        self._tasks = tasks
        self._submission_extractor = submission_extractor or RawGenerationExtractor()
        self._max_turns = max_turns
        self._include_history = include_history
        self._history_summarizer = history_summarizer
        self._submit_keyword = submit_keyword
        self._solved_threshold = solved_threshold
        self._extra_rewards = extra_rewards
        self._prompts = {**_DEFAULT_PROMPTS, **(prompts or {})}

    @property
    def prompts(self) -> dict[str, str]:
        return dict(self._prompts)

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        name = self._inner.spec.name if self._inner else "iterative"
        return EnvironmentSpec(
            name=f"{name}_iterative",
            adapter="iterative",
            max_steps=self._max_turns,
            is_multi_turn=True,
            pure_step=True,
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction, ...]:
        inner_rewards = self._inner.reward_functions if self._inner else ()
        return inner_rewards + self._extra_rewards

    def __len__(self) -> int:
        if self._tasks is not None:
            return len(self._tasks)
        if self._inner is not None and hasattr(self._inner, "__len__"):
            return len(self._inner)
        raise TypeError("Environment does not support len()")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[IterativeHidden], dict[str, Any]]:
        """Reset and return the initial state with the task prompt."""
        options = options or {}
        task_index = options.get("task_index", 0)

        if self._inner is not None:
            inner_state, inner_info = self._inner.reset(seed=seed, options=options)
            task_prompt = inner_state.observation.prompt
            inner_hidden = inner_state.hidden
            # Store initial state for pure_step re-evaluation
            self._inner_initial_states: dict[int, State] = getattr(
                self, "_inner_initial_states", {}
            )
            self._inner_initial_states[task_index] = inner_state
        else:
            assert self._tasks is not None
            if task_index < 0 or task_index >= len(self._tasks):
                raise ValueError(f"task_index {task_index} out of bounds [0, {len(self._tasks)})")
            task = self._tasks[task_index]
            task_prompt = task.prompt
            inner_hidden = task
            inner_info = {"task_index": task_index}

        # Format initial prompt
        initial_text = self._prompts["initial"].format(
            task=task_prompt,
            turns_remaining=self._max_turns,
        )

        hidden = IterativeHidden(
            task_index=task_index,
            inner_hidden=inner_hidden,
            task_prompt=task_prompt,
            turn=0,
            submissions=(),
            feedback_history=(),
            max_turns=self._max_turns,
        )

        episode_id = options.get("episode_id", str(uuid.uuid4()))
        state = State(
            observation=Observation(
                prompt=initial_text,
                messages=(),
                task=ObservationContent(text=initial_text),
            ),
            hidden=hidden,
            metadata=StateMetadata(
                step=0,
                episode_id=episode_id,
                is_terminal=False,
                info={"task_index": task_index},
            ),
        )

        return state, inner_info

    def step(
        self,
        state: State[IterativeHidden],
        action: Action,
    ) -> StepResult[IterativeHidden]:
        """Process a submission, evaluate, and provide feedback."""
        hidden = state.hidden
        action_text = action.text or ""

        # 1. Check for early submit
        early_submit = self._submit_keyword is not None and action_text.lstrip().upper().startswith(
            self._submit_keyword.upper()
        )

        # 2. Extract submission
        extracted, _ = self._submission_extractor.extract(action_text)
        submission = extracted if extracted is not None else action_text

        # 3. Evaluate via inner env and/or extra rewards
        eval_action = Action(text=submission)
        all_signals: list[Signal] = []

        # Inner environment rewards (single-turn, pure_step re-evaluation)
        if self._inner is not None:
            inner_initial = self._inner_initial_states.get(hidden.task_index)
            if inner_initial is not None:
                inner_result = self._inner.step(inner_initial, eval_action)
                all_signals.extend(inner_result.rewards.signals)

        # Extra rewards (on iterative state)
        iter_state = State(
            observation=state.observation,
            hidden=hidden,
            metadata=state.metadata,
        )
        iter_next = State(
            observation=state.observation,
            hidden=hidden,
            metadata=state.metadata,
        )
        for rf in self._extra_rewards:
            sig = rf.compute(iter_state, eval_action, iter_next)
            all_signals.append(sig)

        combined = SignalBundle(signals=tuple(all_signals))

        # 4. Collect feedback
        feedback_texts = combined.feedback_texts()

        # 5. Determine termination
        new_turn = hidden.turn + 1
        outcome_signals = combined.by_type(RewardType.OUTCOME)
        best_outcome = max(
            (s.reward for s in outcome_signals if s.reward is not None),
            default=0.0,
        )
        solved = best_outcome >= self._solved_threshold

        terminated = early_submit or solved
        truncated = not terminated and new_turn >= self._max_turns
        is_done = terminated or truncated

        # 6. Build next hidden
        new_submissions = hidden.submissions + (submission,)
        new_feedback = hidden.feedback_history + (feedback_texts,)

        new_hidden = IterativeHidden(
            task_index=hidden.task_index,
            inner_hidden=hidden.inner_hidden,
            task_prompt=hidden.task_prompt,
            turn=new_turn,
            submissions=new_submissions,
            feedback_history=new_feedback,
            max_turns=self._max_turns,
        )

        # 7. Build next observation
        if is_done:
            obs_text = "Episode complete."
            messages: tuple[dict[str, Any], ...] = ()
        else:
            obs_text = self._build_feedback_observation(new_hidden, feedback_texts)
            # Build messages for chat runners
            messages = self._build_messages(state, action_text, obs_text)

        next_state = State(
            observation=Observation(
                prompt=obs_text,
                messages=messages,
                task=state.observation.task,
                state=ObservationContent(text=obs_text),
            ),
            hidden=new_hidden,
            metadata=StateMetadata(
                step=state.metadata.step + 1,
                episode_id=state.metadata.episode_id,
                is_terminal=is_done,
                info={
                    "task_index": hidden.task_index,
                    "turn": new_turn,
                    "solved": solved,
                    "early_submit": early_submit,
                },
            ),
        )

        return StepResult(
            next_state=next_state,
            rewards=combined,
            terminated=terminated,
            truncated=truncated,
            resolved_action=submission,
            info={
                "turn": new_turn,
                "submission": submission,
                "feedback": feedback_texts,
                "solved": solved,
                "early_submit": early_submit,
            },
        )

    def compute_rewards(
        self,
        state: State[IterativeHidden],
        action: Action,
        next_state: State[IterativeHidden],
    ) -> SignalBundle:
        """Compute rewards for a transition."""
        signals: list[Signal] = []
        for rf in self.reward_functions:
            signals.append(rf.compute(state, action, next_state))
        return SignalBundle(signals=tuple(signals))

    def _build_feedback_observation(
        self,
        hidden: IterativeHidden,
        feedback_texts: tuple[str, ...],
    ) -> str:
        """Build the observation text with feedback and optional history."""
        feedback_str = (
            "\n".join(feedback_texts) if feedback_texts else "No specific feedback available."
        )

        # Build history section
        history_section = ""
        if self._include_history and len(hidden.submissions) > 1:
            if self._history_summarizer is not None:
                history_section = self._history_summarizer(
                    hidden.submissions[:-1], hidden.feedback_history[:-1]
                )
                history_section += "\n\n"
            else:
                history_section = (
                    "## Previous attempts\n\n"
                    + _default_history_formatter(
                        hidden.submissions[:-1],
                        hidden.feedback_history[:-1],
                        self._prompts["history_entry"],
                    )
                    + "\n\n"
                )

        return self._prompts["feedback"].format(
            turn=hidden.turn,
            max_turns=hidden.max_turns,
            feedback=feedback_str,
            history_section=history_section,
            turns_remaining=hidden.max_turns - hidden.turn,
        )

    def _build_messages(
        self,
        state: State[IterativeHidden],
        action_text: str,
        obs_text: str,
    ) -> tuple[dict[str, Any], ...]:
        """Build message history for chat-based runners."""
        messages = list(state.observation.messages)

        # Add agent's submission
        messages.append({"role": "assistant", "content": action_text})

        # Add feedback as user message
        messages.append({"role": "user", "content": obs_text})

        return tuple(messages)
