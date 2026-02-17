"""InterCode adapter - wraps interactive code generation environments.

InterCode (Princeton NLP, NeurIPS 2023) provides a framework for interactive
code generation where LLM agents write and execute code (Bash, SQL, Python)
in Docker containers and receive execution feedback.

Agents interact through free-form text commands — there is no tool schema.
The API follows a gym-compatible pattern: reset(index) → observation,
step(action_string) → (obs, reward, done, info).

Reference: https://github.com/princeton-nlp/intercode
"""

import uuid
from dataclasses import dataclass
from typing import Any

from llenvs.core.environment import EnvironmentSpec, StepResult, _StateContinuityTracker
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import Action, Observation, State, StateMetadata

INTERCODE_PRESETS: dict[str, dict[str, str]] = {
    "bash": {
        "env_class": "BashEnv",
        "module": "intercode.envs",
    },
    "sql": {
        "env_class": "SqlEnv",
        "module": "intercode.envs",
    },
    "python": {
        "env_class": "PythonEnv",
        "module": "intercode.envs",
    },
    "ctf": {
        "env_class": "CTFEnv",
        "module": "intercode.envs",
    },
}


@dataclass(frozen=True)
class InterCodeHidden:
    """Hidden state for InterCode environment.

    Attributes:
        task_index: Index of the current task.
        env_type: Environment type (bash, sql, python, ctf).
        query: The task query/prompt from the dataset.
        gold: The gold/reference solution.
        episode_step: Current step within the episode.
        last_action: The last action taken.
        cumulative_reward: Cumulative reward across the episode.
        trajectory: Tuple of actions taken so far.
    """

    task_index: int
    env_type: str
    query: str
    gold: str
    episode_step: int
    last_action: str | None
    cumulative_reward: float
    trajectory: tuple[str, ...]


@dataclass
class InterCodeReward:
    """Reward function for InterCode environments.

    Intermediate steps produce STEP rewards with reward=None.
    Terminal steps produce OUTCOME rewards with the reward value
    from InterCode's info dict.
    """

    _name: str = "intercode"

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return RewardType.STEP

    def compute(
        self,
        state: State[InterCodeHidden],
        action: Action,
        next_state: State[InterCodeHidden],
    ) -> Signal:
        """Compute reward from InterCode's native reward signal."""
        is_terminal = next_state.metadata.is_terminal

        if is_terminal:
            reward = next_state.metadata.info.get("reward", 0.0)
            return Signal(
                name=self.name,
                reward_type=RewardType.OUTCOME,
                reward=float(reward),
                metadata={"source": "intercode"},
            )
        else:
            return Signal(
                name=self.name,
                reward_type=RewardType.STEP,
                reward=None,
                metadata={"source": "intercode"},
            )


class InterCodeEnvironment:
    """MDP wrapper for InterCode interactive code generation environments.

    InterCode is a multi-turn environment where agents write and execute code
    (Bash, SQL, Python) in Docker containers. Actions are free-form text
    commands that get executed in the container.

    Example:
        >>> env = InterCodeEnvironment(intercode_env=ic_env)
        >>> state, _ = env.reset(options={"task_index": 0})
        >>> print(state.observation.prompt)
        # Shows task description and initial output

        >>> action = Action(text="ls -la")
        >>> result = env.step(state, action)
    """

    def __init__(
        self,
        intercode_env: Any,
        env_type: str = "bash",
        max_steps: int = 10,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> None:
        """Initialize InterCode environment wrapper.

        Args:
            intercode_env: A pre-created InterCode environment instance
                (e.g., BashEnv, SqlEnv). Must have reset(), step(), close(),
                and data_loader attributes.
            env_type: Environment type identifier (bash, sql, python, ctf).
            max_steps: Maximum steps per episode before truncation.
            extra_rewards: Additional reward functions appended after
                native rewards.
        """
        self._ic_env = intercode_env
        self._env_type = env_type
        self._max_steps = max_steps
        self._native_rewards: tuple[RewardFunction, ...] = (InterCodeReward(),)
        self._extra_rewards = extra_rewards
        self._state_tracker = _StateContinuityTracker()

    def __len__(self) -> int:
        return len(self._ic_env.data_loader)

    @property
    def prompts(self) -> dict[str, str]:
        """Named prompt components (none for InterCode)."""
        return {}

    @property
    def available_tools(self) -> tuple:
        """No tools available — InterCode uses text-based commands."""
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        """Get environment specification."""
        return EnvironmentSpec(
            name="intercode",
            adapter="intercode",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            supports_task_index=True,
            supports_len=True,
            supports_seed=False,
            pure_step=False,
            metadata={
                "env_type": self._env_type,
                "num_tasks": len(self),
                "description": f"InterCode {self._env_type} environment",
            },
        )

    @property
    def reward_functions(
        self,
    ) -> tuple[RewardFunction[InterCodeHidden], ...]:
        """Get reward functions used by this environment."""
        return self._native_rewards + self._extra_rewards

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[InterCodeHidden], dict[str, Any]]:
        """Reset environment and return initial state.

        Args:
            seed: Unused (InterCode does not support seeding).
            options: Environment-specific options.
                - task_index (required): Select specific task (0-indexed).
                - episode_id: Custom episode identifier.

        Returns:
            Tuple of (initial_state, info_dict).
        """
        options = options or {}

        if "task_index" not in options:
            raise ValueError("task_index is required in options for InterCode environments")

        task_index = options["task_index"]

        if task_index < 0 or task_index >= len(self):
            raise IndexError(f"task_index {task_index} out of range [0, {len(self)})")

        # Get task info from data_loader
        task = self._ic_env.data_loader[task_index]
        query = task.get("query", "")
        gold = task.get("gold", "")

        # Reset InterCode environment at this task index
        raw_obs = self._ic_env.reset(index=task_index)

        hidden = InterCodeHidden(
            task_index=task_index,
            env_type=self._env_type,
            query=query,
            gold=gold,
            episode_step=0,
            last_action=None,
            cumulative_reward=0.0,
            trajectory=(),
        )

        observation = Observation(prompt=str(raw_obs))

        metadata = StateMetadata(
            step=0,
            episode_id=options.get("episode_id", str(uuid.uuid4())),
            is_terminal=False,
            info={
                "task_index": task_index,
                "env_type": self._env_type,
                "query": query,
            },
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        return state, {
            "task_index": task_index,
            "env_type": self._env_type,
            "query": query,
        }

    def step(
        self,
        state: State[InterCodeHidden],
        action: Action,
    ) -> StepResult[InterCodeHidden]:
        """Take an action from the given state.

        Args:
            state: Current state.
            action: Action to take (free-form text command).

        Returns:
            StepResult containing next state, rewards, and done flags.
        """
        self._state_tracker.validate(state, "InterCodeEnvironment")

        # Step InterCode environment
        action_text = action.text or ""
        raw_obs, reward, done, info = self._ic_env.step(action_text)

        # Track step
        next_step = state.hidden.episode_step + 1
        terminated = bool(done)
        truncated = next_step >= self._max_steps and not terminated
        cumulative_reward = state.hidden.cumulative_reward + float(reward)

        new_hidden = InterCodeHidden(
            task_index=state.hidden.task_index,
            env_type=state.hidden.env_type,
            query=state.hidden.query,
            gold=state.hidden.gold,
            episode_step=next_step,
            last_action=action_text,
            cumulative_reward=cumulative_reward,
            trajectory=state.hidden.trajectory + (action_text,),
        )

        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": action_text},
            {"role": "user", "content": str(raw_obs)},
        )
        new_observation = Observation(prompt=state.observation.prompt, messages=new_messages)

        new_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=terminated or truncated,
            info={
                **state.metadata.info,
                "reward": reward,
                "done": done,
                "action": action_text,
                "cumulative_reward": cumulative_reward,
            },
        )

        next_state = State(
            observation=new_observation,
            hidden=new_hidden,
            metadata=new_metadata,
        )

        # Compute rewards
        rewards = self.compute_rewards(state, action, next_state)
        self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info={
                "reward": reward,
                "done": done,
                "action": action_text,
                "cumulative_reward": cumulative_reward,
            },
        )

    def compute_rewards(
        self,
        state: State[InterCodeHidden],
        action: Action,
        next_state: State[InterCodeHidden],
    ) -> SignalBundle:
        """Compute rewards for a transition."""
        signals = []

        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)

        return SignalBundle(signals=tuple(signals))

    def close(self) -> None:
        """Close the underlying InterCode environment."""
        if self._ic_env is not None:
            self._ic_env.close()


class InterCodeAdapter:
    """Adapter for InterCode interactive code generation environments.

    InterCode provides environments for interactive code generation
    (Bash, SQL, Python, CTF) in Docker containers.

    Requires the intercode package:
        pip install intercode-bench
        OR pip install git+https://github.com/princeton-nlp/intercode.git

    Example:
        >>> adapter = InterCodeAdapter()
        >>> env = adapter.get_environment("intercode:bash", intercode_env=ic_env)
        >>> state, _ = env.reset(options={"task_index": 0})
    """

    @property
    def name(self) -> str:
        """Adapter identifier."""
        return "intercode"

    def _get_intercode(self) -> Any:
        """Import and return the intercode module."""
        try:
            import intercode

            return intercode
        except ImportError as e:
            raise ImportError(
                "InterCode is required for InterCodeAdapter. "
                "Install with: pip install intercode-bench\n"
                "Or: pip install git+https://github.com/princeton-nlp/intercode.git\n"
                "See: https://github.com/princeton-nlp/intercode"
            ) from e

    def list_environments(self) -> list[str]:
        """List available environment variants.

        Returns:
            List of environment IDs (one per env type).
        """
        return [f"intercode:{name}" for name in sorted(INTERCODE_PRESETS.keys())]

    def get_environment(
        self,
        name: str = "intercode:bash",
        intercode_env: Any | None = None,
        data_path: str | None = None,
        image_name: str | None = None,
        max_steps: int = 10,
        extra_rewards: tuple[RewardFunction, ...] = (),
        **kwargs: Any,
    ) -> InterCodeEnvironment:
        """Create an InterCode environment.

        Args:
            name: Environment ID. Format: "intercode:{type}" where type is
                one of bash, sql, python, ctf.
            intercode_env: Pre-created InterCode environment instance.
                If provided, data_path and image_name are ignored.
            data_path: Path to the task dataset. Required if intercode_env
                is not provided.
            image_name: Docker image name. If not provided, uses the
                default for the env type.
            max_steps: Maximum steps per episode.
            extra_rewards: Additional reward functions.
            **kwargs: Additional arguments passed to InterCode env constructor.

        Returns:
            Configured InterCodeEnvironment.

        Raises:
            ImportError: If intercode is not installed.
            ValueError: If the env type is unknown or required args are missing.
        """
        # Parse env type from name
        env_type = "bash"
        if ":" in name:
            env_type = name.split(":", 1)[1]

        if env_type not in INTERCODE_PRESETS:
            raise ValueError(
                f"Unknown InterCode env type: {env_type!r}. "
                f"Available: {sorted(INTERCODE_PRESETS.keys())}"
            )

        if intercode_env is None:
            if data_path is None:
                raise ValueError(
                    "Either intercode_env or data_path must be provided. "
                    "Pass a pre-created InterCode environment with intercode_env=, "
                    "or provide data_path= to create one."
                )

            intercode = self._get_intercode()
            preset = INTERCODE_PRESETS[env_type]
            env_class_name = preset["env_class"]
            env_class = getattr(intercode, env_class_name, None)

            if env_class is None:
                # Try importing from submodule
                import importlib

                mod = importlib.import_module(preset["module"])
                env_class = getattr(mod, env_class_name)

            ctor_kwargs: dict[str, Any] = {"data_path": data_path}
            if image_name is not None:
                ctor_kwargs["image_name"] = image_name
            ctor_kwargs.update(kwargs)

            intercode_env = env_class(**ctor_kwargs)

        return InterCodeEnvironment(
            intercode_env=intercode_env,
            env_type=env_type,
            max_steps=max_steps,
            extra_rewards=extra_rewards,
        )

    def get_default_system_prompt(self, name: str) -> None:
        """InterCode environments provide their own context."""
        return None

    def get_prompt_template(self, name: str) -> None:
        """InterCode manages multi-turn prompts internally."""
        return None

    def get_native_answer_extractor(self, task_name: str) -> None:
        """InterCode does not provide native answer extraction.

        Args:
            task_name: Task name (unused).

        Returns:
            None (no native extraction available).
        """
        return None

    def get_environment_info(self, name: str = "intercode") -> dict[str, Any]:
        """Get metadata about the environment.

        Args:
            name: Environment ID.

        Returns:
            Dictionary with environment metadata.
        """
        return {
            "name": name,
            "adapter": self.name,
            "type": "multi_turn",
            "env_types": sorted(INTERCODE_PRESETS.keys()),
            "description": (
                "InterCode: Interactive code generation in Docker containers. "
                "Supports Bash, SQL, Python, and CTF environments."
            ),
            "reference": "https://github.com/princeton-nlp/intercode",
        }
