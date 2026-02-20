"""AgentGym adapter - wraps AgentGym multi-turn agent environments.

AgentGym (Xi et al., 2024) provides 14 diverse agent environments (web
navigation, text games, API tasks, etc.) behind a unified client-server
architecture. Each environment runs as a FastAPI server and Python clients
communicate via REST.

Reference: https://github.com/THUDM/AgentGym
"""

from __future__ import annotations

import atexit
import socket
import subprocess
import time
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

from llenvs.core.environment import EnvironmentSpec, StepResult, _StateContinuityTracker
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import Action, Observation, ObservationContent, State, StateMetadata

# ---------------------------------------------------------------------------
# Environment registry: env_name -> (client_class_name, cli_command)
# ---------------------------------------------------------------------------

ENV_REGISTRY: dict[str, tuple[str, str]] = {
    "webshop": ("WebshopEnvClient", "webshop"),
    "alfworld": ("AlfWorldEnvClient", "alfworld"),
    "babyai": ("BabyAIEnvClient", "babyai"),
    "maze": ("MazeEnvClient", "lmrlgym_maze"),
    "wordle": ("WordleEnvClient", "lmrlgym_wordle"),
    "sciworld": ("SciworldEnvClient", "sciworld"),
    "sqlgym": ("SqlGymEnvClient", "sqlgym"),
    "textcraft": ("TextCraftEnvClient", "textcraft"),
    "webarena": ("WebarenaEnvClient", "webarena"),
    "searchqa": ("SearchQAEnvClient", "searchqa"),
    "movie": ("MovieEnvClient", "movie"),
    "weather": ("WeatherEnvClient", "weather"),
    "academia": ("AcademiaEnvClient", "academia"),
    "todo": ("TodoEnvClient", "todo"),
    "sheet": ("SheetEnvClient", "sheet"),
}


# ---------------------------------------------------------------------------
# Hidden state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentGymHidden:
    """Hidden state for AgentGym environments.

    Attributes:
        task_index: Index of the current task.
        env_name: Name of the AgentGym environment.
        episode_step: Current step within the episode.
        last_action: The last action taken.
        available_actions: Actions available in the current state (if provided by the client).
    """

    task_index: int
    env_name: str
    episode_step: int
    last_action: str | None
    available_actions: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


@dataclass
class AgentGymReward:
    """Reward function wrapping AgentGym's native step/outcome reward.

    Returns RewardType.STEP for intermediate steps, RewardType.OUTCOME
    for terminal steps.
    """

    _name: str = "agentgym_native"

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return RewardType.OUTCOME

    def compute(
        self,
        state: State[AgentGymHidden],
        action: Action,
        next_state: State[AgentGymHidden],
    ) -> Signal:
        """Compute reward from AgentGym's native reward."""
        native_reward = next_state.metadata.info.get("agentgym_reward", 0.0)
        rtype = RewardType.OUTCOME if next_state.metadata.is_terminal else RewardType.STEP

        return Signal(
            name=self.name,
            reward_type=rtype,
            reward=float(native_reward),
            metadata={"source": "agentgym"},
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class AgentGymEnvironment:
    """MDP wrapper for AgentGym environments.

    AgentGym environments are multi-turn: an agent receives text
    observations and responds with text actions until the episode ends
    or max_steps is reached.

    Example:
        >>> env = AgentGymEnvironment(client, "maze", max_steps=20)
        >>> state, _ = env.reset(options={"task_index": 0})
        >>> result = env.step(state, Action(text="go north"))
    """

    def __init__(
        self,
        client: Any,
        env_name: str,
        max_steps: int = 20,
        extra_rewards: tuple[RewardFunction, ...] = (),
        prompts: dict[str, str] | None = None,
        action_format: str = "react",
    ) -> None:
        self._client = client
        self._env_name = env_name
        self._max_steps = max_steps
        self._action_format = action_format
        self._data_len = len(client)
        self._native_rewards: tuple[RewardFunction, ...] = (AgentGymReward(),)
        self._extra_rewards = extra_rewards
        self._prompts: dict[str, str] = dict(prompts) if prompts else {}
        self._state_tracker = _StateContinuityTracker()

    def __len__(self) -> int:
        """Number of tasks in the dataset."""
        return self._data_len

    @staticmethod
    def _coerce_observation(obs: Any) -> str:
        """Coerce diverse observation types to str.

        agentenv clients return str (6 clients) or dict (9 clients).
        """
        if isinstance(obs, str):
            return obs
        if isinstance(obs, dict):
            return obs.get("observation", str(obs))
        return str(obs)

    def _read_client_info(self) -> dict[str, Any]:
        """Read info dict from stateful clients, prefixed with ``client_``.

        Skips ``"observation"`` to avoid redundancy with the observation field.
        """
        raw = getattr(self._client, "info", None)
        if not isinstance(raw, dict):
            return {}
        return {f"client_{k}": v for k, v in raw.items() if k != "observation"}

    def _extract_available_actions(self) -> tuple[str, ...]:
        """Extract available actions from client info if present."""
        raw = getattr(self._client, "info", None)
        if not isinstance(raw, dict):
            return ()
        actions = raw.get("available_actions")
        if actions is None:
            return ()
        return tuple(str(a) for a in actions)

    @property
    def prompts(self) -> dict[str, str]:
        return dict(self._prompts)

    @property
    def available_tools(self) -> tuple:
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name=self._env_name,
            adapter="agentgym",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            metadata={
                "description": f"AgentGym {self._env_name} environment",
                "dataset_size": self._data_len,
                "action_format": self._action_format,
            },
        )

    @property
    def reward_functions(self) -> tuple[RewardFunction[AgentGymHidden], ...]:
        return self._native_rewards + self._extra_rewards

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[AgentGymHidden], dict[str, Any]]:
        options = options or {}
        task_index = options.get("task_index", 0)

        if task_index < 0 or task_index >= self._data_len:
            raise ValueError(f"task_index {task_index} out of bounds [0, {self._data_len})")

        # Reset and capture return value
        reset_result = self._client.reset(task_index)
        reset_data: dict[str, Any] = {}
        if isinstance(reset_result, dict):
            reset_data = reset_result
        elif isinstance(reset_result, (tuple, list)) and len(reset_result) >= 1:
            if isinstance(reset_result[0], dict):
                reset_data = reset_result[0]

        obs_raw = self._client.observe()
        obs_text = self._coerce_observation(obs_raw)

        # Read client info and available actions
        client_info = self._read_client_info()
        available_actions = self._extract_available_actions()

        hidden = AgentGymHidden(
            task_index=task_index,
            env_name=self._env_name,
            episode_step=0,
            last_action=None,
            available_actions=available_actions,
        )

        observation = Observation(
            prompt=obs_text,
            task=ObservationContent(text=obs_text),
            state=ObservationContent(text=obs_text),
        )

        # Build info dict
        info: dict[str, Any] = {
            "task_index": task_index,
            "env_name": self._env_name,
            "action_format": self._action_format,
            "dataset_size": self._data_len,
        }
        # Add reset return data with prefix
        for k, v in reset_data.items():
            info[f"reset_{k}"] = v
        # Add client info
        info.update(client_info)
        # Add available actions if present
        if available_actions:
            info["available_actions"] = available_actions

        metadata = StateMetadata(
            step=0,
            episode_id=options.get("episode_id", str(uuid.uuid4())),
            is_terminal=False,
            info=dict(info),
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)
        return state, info

    def step(
        self,
        state: State[AgentGymHidden],
        action: Action,
    ) -> StepResult[AgentGymHidden]:
        self._state_tracker.validate(state, "AgentGymEnvironment")
        step_output = self._client.step(action.text)

        next_step = state.hidden.episode_step + 1
        done = step_output.done
        truncated = next_step >= self._max_steps and not done

        obs_text = self._coerce_observation(step_output.state)

        # Read client info and available actions
        client_info = self._read_client_info()
        available_actions = self._extract_available_actions()

        new_hidden = AgentGymHidden(
            task_index=state.hidden.task_index,
            env_name=state.hidden.env_name,
            episode_step=next_step,
            last_action=action.text,
            available_actions=available_actions,
        )

        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": action.text or ""},
            {"role": "user", "content": obs_text},
        )
        new_observation = Observation(
            prompt=state.observation.prompt,
            messages=new_messages,
            task=state.observation.task,
            state=ObservationContent(text=obs_text),
        )

        new_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=done or truncated,
            info={
                **state.metadata.info,
                "agentgym_reward": step_output.reward,
                "last_action": action.text,
                **client_info,
            },
        )

        next_state = State(
            observation=new_observation,
            hidden=new_hidden,
            metadata=new_metadata,
        )

        rewards = self.compute_rewards(state, action, next_state)
        self._state_tracker.track(next_state)

        # Build step info dict
        step_info: dict[str, Any] = {
            "agentgym_reward": step_output.reward,
            "action": action.text,
            "done": done,
            "truncated": truncated,
            "episode_step": next_step,
            "env_name": self._env_name,
            **client_info,
        }
        if available_actions:
            step_info["available_actions"] = available_actions

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=done,
            truncated=truncated,
            info=step_info,
        )

    def compute_rewards(
        self,
        state: State[AgentGymHidden],
        action: Action,
        next_state: State[AgentGymHidden],
    ) -> SignalBundle:
        signals = []
        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)
        return SignalBundle(signals=tuple(signals))


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


class _ServerManager:
    """Manages AgentGym environment server processes.

    Auto-starts servers from installed server packages when needed.
    Tracks running processes so they can be reused and cleaned up.
    """

    _servers: dict[str, tuple[subprocess.Popen, int]] = {}
    _atexit_registered: bool = False

    @classmethod
    def get_or_start(cls, env_name: str, env_server_base: str | None = None) -> str:
        """Get URL for an environment server, starting one if needed.

        Args:
            env_name: Environment name from ENV_REGISTRY.
            env_server_base: Pre-existing server URL. If provided, skip
                starting a new server.

        Returns:
            Server base URL (e.g. "http://localhost:12345").
        """
        if env_server_base is not None:
            return env_server_base

        # Check if we already have a running server for this env
        if env_name in cls._servers:
            proc, port = cls._servers[env_name]
            if proc.poll() is None:  # still running
                return f"http://localhost:{port}"
            # Process died — remove stale entry
            del cls._servers[env_name]

        return cls._start_server(env_name)

    @classmethod
    def _start_server(cls, env_name: str) -> str:
        """Start a server subprocess for the given environment.

        Args:
            env_name: Environment name from ENV_REGISTRY.

        Returns:
            Server base URL.

        Raises:
            KeyError: If env_name not in ENV_REGISTRY.
            RuntimeError: If server fails to start.
        """
        if env_name not in ENV_REGISTRY:
            raise KeyError(f"Unknown AgentGym environment: {env_name}")

        _, cli_command = ENV_REGISTRY[env_name]
        port = cls._find_free_port()

        proc = subprocess.Popen(
            [cli_command, "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        url = f"http://localhost:{port}"

        cls._servers[env_name] = (proc, port)

        if not cls._atexit_registered:
            atexit.register(cls.shutdown)
            cls._atexit_registered = True

        cls._wait_for_ready(url, timeout=60)
        return url

    @classmethod
    def _wait_for_ready(cls, url: str, timeout: float = 60) -> None:
        """Poll server until it responds or timeout.

        Args:
            url: Server base URL.
            timeout: Maximum seconds to wait.

        Raises:
            RuntimeError: If server doesn't respond within timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(url, timeout=2)
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError(f"AgentGym server at {url} did not start within {timeout}s")

    @staticmethod
    def _find_free_port() -> int:
        """Find an available port by binding to port 0."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @classmethod
    def shutdown(cls) -> None:
        """Stop all managed servers."""
        for env_name, (proc, _port) in list(cls._servers.items()):
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        cls._servers.clear()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class AgentGymAdapter:
    """Adapter for AgentGym environments.

    AgentGym provides 14 diverse multi-turn agent environments behind a
    unified client-server architecture.

    Requires: pip install agentenv
    Plus server packages per environment (e.g. agentenv-alfworld).

    Example:
        >>> adapter = AgentGymAdapter()
        >>> env = adapter.get_environment("maze", max_steps=20)
        >>> state, _ = env.reset(options={"task_index": 0})
    """

    @property
    def name(self) -> str:
        return "agentgym"

    def _get_agentenv(self) -> Any:
        """Import and return the agentenv module."""
        try:
            import agentenv

            return agentenv
        except ImportError as e:
            raise ImportError(
                "agentenv is required for AgentGymAdapter. Install with: pip install agentenv"
            ) from e

    def _resolve_client_class(self, env_name: str) -> type:
        """Resolve the client class for a given environment name.

        Args:
            env_name: Environment name from ENV_REGISTRY.

        Returns:
            The client class.

        Raises:
            KeyError: If env_name not in ENV_REGISTRY.
            ImportError: If the client class cannot be imported.
        """
        if env_name not in ENV_REGISTRY:
            raise KeyError(f"Unknown AgentGym environment: {env_name}")

        class_name, _ = ENV_REGISTRY[env_name]

        from agentenv.envs import __dict__ as envs_dict

        if class_name in envs_dict:
            return envs_dict[class_name]

        # Fallback: try getattr
        import agentenv.envs as envs_module

        return getattr(envs_module, class_name)

    def list_environments(self) -> list[str]:
        """List all available AgentGym environment names."""
        return list(ENV_REGISTRY.keys())

    @staticmethod
    def _extract_conversation_prompts(
        client: Any,
        action_format: str,
    ) -> dict[str, str]:
        """Extract system prompt and assistant ack from client conversation start.

        agentenv clients store per-format conversation starters in
        ``_conversation_start`` (or ``conversation_start``). Each entry is
        a list of ``(role, message)`` tuples.

        Args:
            client: An agentenv client instance.
            action_format: The action format key (e.g. ``"react"``).

        Returns:
            Dict with ``"system_prompt"`` and/or ``"assistant_ack"`` keys.
        """
        raw = getattr(client, "_conversation_start", None)
        if raw is None:
            raw = getattr(client, "conversation_start", None)
        if not isinstance(raw, dict) or not raw:
            return {}

        # Look up by action_format, fall back to first available
        messages = raw.get(action_format)
        if messages is None:
            messages = next(iter(raw.values()))

        if not isinstance(messages, (list, tuple)):
            return {}

        prompts: dict[str, str] = {}
        for role, text in messages:
            if role == "human" and "system_prompt" not in prompts:
                prompts["system_prompt"] = text
            elif role == "gpt" and "assistant_ack" not in prompts:
                prompts["assistant_ack"] = text
        return prompts

    def get_environment(
        self,
        name: str,
        env_server_base: str | None = None,
        data_len: int = 100,
        max_steps: int = 20,
        timeout: int = 300,
        action_format: str = "react",
        extra_rewards: tuple[RewardFunction, ...] = (),
        prompts: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AgentGymEnvironment:
        """Create an AgentGym environment.

        Args:
            name: Environment name (e.g. "maze", "alfworld").
            env_server_base: URL of a pre-running server. If None, the
                adapter will auto-start one.
            data_len: Number of tasks to load on the server.
            max_steps: Maximum steps per episode before truncation.
            timeout: HTTP request timeout in seconds.
            action_format: Action format for the client. One of ``"react"``,
                ``"function_calling"``, ``"code_as_action"``.
            extra_rewards: Additional reward functions.
            prompts: Override prompt components. Merged with auto-extracted
                conversation prompts (user values take precedence).
            **kwargs: Additional arguments passed to the client constructor.

        Returns:
            Configured AgentGymEnvironment.

        Raises:
            ImportError: If agentenv is not installed.
            KeyError: If name is not a valid environment.
        """
        self._get_agentenv()

        server_url = _ServerManager.get_or_start(name, env_server_base)

        client_class = self._resolve_client_class(name)
        client = client_class(
            env_server_base=server_url,
            data_len=data_len,
            timeout=timeout,
            **kwargs,
        )

        # Extract conversation prompts and merge with user overrides
        auto_prompts = self._extract_conversation_prompts(client, action_format)
        merged_prompts = {**auto_prompts, **(prompts or {})}

        return AgentGymEnvironment(
            client=client,
            env_name=name,
            max_steps=max_steps,
            extra_rewards=extra_rewards,
            prompts=merged_prompts or None,
            action_format=action_format,
        )

    def get_native_answer_extractor(self, task_name: str) -> None:
        """AgentGym does not provide native answer extraction."""
        return None

    def get_default_system_prompt(self, name: str) -> None:
        """AgentGym environments manage prompts internally."""
        return None

    def get_prompt_template(self, name: str) -> None:
        """AgentGym manages multi-turn prompts internally."""
        return None

    def get_environment_info(self, name: str = "agentgym") -> dict[str, Any]:
        """Get metadata about an AgentGym environment.

        Args:
            name: Environment name.

        Returns:
            Dictionary with environment metadata.
        """
        return {
            "name": name,
            "adapter": self.name,
            "type": "multi_turn",
            "description": (
                f"AgentGym: {name} environment. Multi-turn agent "
                f"interaction via client-server architecture."
            ),
            "reference": "https://github.com/THUDM/AgentGym",
        }
