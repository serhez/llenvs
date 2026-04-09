"""GEM adapter - wraps GEM environments as MDP environments.

GEM (General Experience Maker) provides a suite of environments for training
and evaluating LLMs, including multi-turn games and single-turn benchmarks.
"""

import copy
import uuid
from dataclasses import dataclass
from typing import Any

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.extraction import AnswerExtractor, TagBasedExtractor
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import (
    Action,
    Observation,
    ObservationContent,
    State,
    StateMetadata,
)
from llenvs.core.tool_environment import BaseToolEnvironment
from llenvs.core.tools import (
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolParameterType,
    ToolResult,
    ToolResultStatus,
)

# Multi-turn environments (games) - these are natively multi-step
MULTI_TURN_ENVS = {
    # Games with difficulty variants
    "game:GuessTheNumber-v0",
    "game:GuessTheNumber-v0-easy",
    "game:GuessTheNumber-v0-hard",
    "game:GuessTheNumber-v0-random",
    "game:Sudoku-v0",
    "game:Sudoku-v0-easy",
    "game:Sudoku-v0-hard",
    "game:Sudoku-v0-random",
    "game:Wordle-v0",
    "game:Wordle-v0-easy",
    "game:Wordle-v0-hard",
    "game:Wordle-v0-random",
    "game:Mastermind-v0",
    "game:Mastermind-v0-easy",
    "game:Mastermind-v0-hard",
    "game:Mastermind-v0-random",
    "game:Minesweeper-v0",
    "game:Minesweeper-v0-easy",
    "game:Minesweeper-v0-hard",
    "game:Minesweeper-v0-random",
    "game:Game2048-v0",
    "game:Game2048-v0-easy",
    "game:Game2048-v0-hard",
    "game:Game2048-v0-random",
    "game:Hangman-v0",
    "game:Hangman-v0-easy",
    "game:Hangman-v0-hard",
    "game:Hangman-v0-random",
    "game:TowerOfHanoi-v0",
    "game:TowerOfHanoi-v0-easy",
    "game:TowerOfHanoi-v0-hard",
    "game:TowerOfHanoi-v0-random",
    "game:FifteenPuzzle-v0",
    "game:FifteenPuzzle-v0-easy",
    "game:FifteenPuzzle-v0-hard",
    "game:FifteenPuzzle-v0-random",
    "game:Crosswords-v0",
    "game:Crosswords-v0-easy",
    "game:Crosswords-v0-hard",
    "game:Crosswords-v0-random",
    "game:Sokoban-v0",
    "game:Sokoban-v0-easy",
    "game:Sokoban-v0-hard",
    "game:Sokoban-v0-random",
    "game:WordSearch-v0",
    "game:WordSearch-v0-easy",
    "game:WordSearch-v0-hard",
    "game:WordSearch-v0-random",
    "game:WikiGame-v0",
    "game:WikiGame-v0-easy",
    "game:WikiGame-v0-hard",
    "game:WikiGame-v0-random",
}

# Multi-turn environment prefixes for pattern matching
MULTI_TURN_PREFIXES = ("game:",)


def _is_multi_turn_env(env_id: str) -> bool:
    """Check if an environment is multi-turn."""
    if env_id in MULTI_TURN_ENVS:
        return True
    return any(env_id.startswith(prefix) for prefix in MULTI_TURN_PREFIXES)


def _freeze_dict(d: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Convert dict to frozen tuple of tuples for immutability."""
    return tuple(sorted(d.items()))


def _unfreeze_dict(frozen: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """Convert frozen tuple back to dict."""
    return dict(frozen)


# =============================================================================
# Tool Definitions for GEM's Built-in Tools
# =============================================================================

GEM_PYTHON_TOOL = ToolDefinition(
    name="python",
    description="Execute Python code. Returns stdout or error message.",
    parameters=(
        ToolParameter(
            name="code",
            type=ToolParameterType.STRING,
            description="Python code to execute",
        ),
    ),
)

GEM_SEARCH_TOOL = ToolDefinition(
    name="search",
    description="Search for information using a query.",
    parameters=(
        ToolParameter(
            name="query",
            type=ToolParameterType.STRING,
            description="Search query text",
        ),
    ),
)

GEM_SUBMIT_ANSWER_TOOL = ToolDefinition(
    name="submit_answer",
    description="Submit the final answer to complete the task",
    parameters=(
        ToolParameter(
            name="answer",
            type=ToolParameterType.STRING,
            description="The final answer",
        ),
    ),
    is_terminal=True,
)


@dataclass(frozen=True)
class GemHidden:
    """Hidden state for GEM environments.

    Snapshots GEM's mutable state into our immutable structure.

    Attributes:
        env_id: GEM environment identifier.
        gem_state: Serialized GEM env state. Frozen tuple for get_state() envs,
            deepcopy dict for stateless envs.
        task_index: Current task index (for dataset-based envs).
        is_multi_turn: Whether this is a multi-turn environment.
        episode_step: Current step within the episode.
    """

    env_id: str
    gem_state: Any
    task_index: int
    is_multi_turn: bool
    episode_step: int


@dataclass
class GemCorrectnessReward:
    """Reward function wrapping GEM's native reward.

    GEM returns a single float reward per step. This wraps it in our
    Signal format with appropriate type based on terminal status.
    """

    _name: str = "correctness"
    _reward_type: RewardType = RewardType.OUTCOME

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def compute(
        self,
        state: State[GemHidden],
        action: Action,
        next_state: State[GemHidden],
    ) -> Signal:
        """Compute reward from GEM's native reward."""
        gem_reward = next_state.metadata.info.get("gem_reward", 0.0)

        # Use STEP type for intermediate rewards, OUTCOME for final
        reward_type = self.reward_type if next_state.metadata.is_terminal else RewardType.STEP

        return Signal(
            name=self.name,
            reward_type=reward_type,
            reward=float(gem_reward),
            metadata={"source": "gem"},
        )


class GemEnvironment:
    """MDP wrapper for GEM environments.

    Wraps GEM's Gym-style environments in our stateless MDP interface.
    Supports both single-turn (math, code, QA) and multi-turn (games)
    environments.

    Key features:
    - State snapshotting via get_state()/set_state() for immutability
    - Automatic multi-turn detection based on environment type
    - Composable reward functions (correctness + format)

    Example:
        >>> adapter = GemAdapter()
        >>> env = adapter.get_environment("game:GuessTheNumber-v0")
        >>> state, _ = env.reset(seed=42)
        >>> while not state.metadata.is_terminal:
        ...     result = env.step(state, Action(text="50"))
        ...     state = result.next_state
    """

    def __init__(
        self,
        gem_env: Any,
        env_id: str,
        is_multi_turn: bool,
        answer_extractor: AnswerExtractor | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        max_steps: int | None = None,
    ) -> None:
        """Initialize GEM environment wrapper.

        Args:
            gem_env: The underlying GEM environment instance.
            env_id: GEM environment identifier (e.g., "game:Sudoku-v0").
            is_multi_turn: Whether this is a multi-turn environment.
            answer_extractor: Extractor for model responses. Defaults to TagBasedExtractor.
            extra_rewards: Additional reward functions appended after native rewards.
            max_steps: Maximum steps per episode (None = unlimited).
        """
        self._gem_env = gem_env
        self._env_id = env_id
        self._is_multi_turn = is_multi_turn
        self._answer_extractor = answer_extractor or TagBasedExtractor()
        self._max_steps = max_steps
        self._supports_state = hasattr(gem_env, "get_state") and hasattr(gem_env, "set_state")

        self._native_rewards: tuple[RewardFunction, ...] = (GemCorrectnessReward(),)
        self._extra_rewards = extra_rewards

        # Track current task for multi-episode use
        self._current_task_index = 0

    @property
    def answer_extractor(self):
        """The extractor used to parse agent responses in ``step()``."""
        return self._answer_extractor

    @answer_extractor.setter
    def answer_extractor(self, value):
        self._answer_extractor = value

    @property
    def prompts(self) -> dict[str, str]:
        """GEM manages its own observations."""
        return {}

    @property
    def available_tools(self) -> tuple[()]:
        """No tools available for standard GEM environments."""
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        """Get environment specification."""
        return EnvironmentSpec(
            name=self._env_id,
            adapter="gem",
            max_steps=self._max_steps if self._is_multi_turn else 1,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=self._is_multi_turn,
            pure_step=True,
            metadata={"env_id": self._env_id},
        )

    @property
    def reward_functions(
        self,
    ) -> tuple[RewardFunction[GemHidden], ...]:
        """Get reward functions used by this environment."""
        return self._native_rewards + self._extra_rewards

    def _snapshot_state(self) -> Any:
        """Capture GEM env state as frozen structure."""
        if not self._supports_state:
            return copy.deepcopy(self._gem_env.__dict__)
        state_dict = self._gem_env.get_state()
        return _freeze_dict(state_dict)

    def _restore_state(self, frozen_state: Any) -> None:
        """Restore GEM env from frozen structure."""
        if not self._supports_state:
            self._gem_env.__dict__.update(copy.deepcopy(frozen_state))
            return
        state_dict = _unfreeze_dict(frozen_state)
        self._gem_env.set_state(state_dict)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[GemHidden], dict[str, Any]]:
        """Reset environment and return initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Environment-specific options.
                - task_index: Select specific task (for dataset-based envs).
                - episode_id: Override the generated episode ID.

        Returns:
            Tuple of (initial_state, info_dict).
        """
        options = options or {}
        task_index = options.get("task_index", self._current_task_index)
        self._current_task_index = task_index

        # Reset GEM env
        obs, info = self._gem_env.reset(seed=seed)

        # Snapshot state
        gem_state = self._snapshot_state()

        hidden = GemHidden(
            env_id=self._env_id,
            gem_state=gem_state,
            task_index=task_index,
            is_multi_turn=self._is_multi_turn,
            episode_step=0,
        )

        observation = Observation(
            prompt=str(obs),
            task=ObservationContent(text=str(obs)),
            state=ObservationContent(text=str(obs)),
        )

        metadata = StateMetadata(
            step=0,
            episode_id=options.get("episode_id", str(uuid.uuid4())),
            is_terminal=False,
            info={"task_index": task_index, "gem_info": info},
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        return state, {"task_index": task_index, "gem_info": info}

    def step(
        self,
        state: State[GemHidden],
        action: Action,
    ) -> StepResult[GemHidden]:
        """Take an action from the given state.

        This is a pure function - the same state and action always
        produce the same result. GEM state is restored before stepping.

        Args:
            state: Current state.
            action: Action to take (model's response).

        Returns:
            StepResult containing next state, rewards, and done flags.
        """
        # Restore GEM state to ensure pure function semantics
        self._restore_state(state.hidden.gem_state)

        # Step GEM env
        obs, reward, terminated, truncated, info = self._gem_env.step(action.text)

        # Snapshot new state
        new_gem_state = self._snapshot_state()

        is_terminal = terminated or truncated

        new_hidden = GemHidden(
            env_id=state.hidden.env_id,
            gem_state=new_gem_state,
            task_index=state.hidden.task_index,
            is_multi_turn=state.hidden.is_multi_turn,
            episode_step=state.hidden.episode_step + 1,
        )

        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": action.text or ""},
            {"role": "user", "content": str(obs)},
        )
        new_observation = Observation(
            prompt=state.observation.prompt,
            messages=new_messages,
            task=state.observation.task,
            state=ObservationContent(text=str(obs)),
        )

        new_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=is_terminal,
            info={
                **state.metadata.info,
                "gem_reward": reward,
                "gem_info": info,
                "response": action.text,
            },
        )

        next_state = State(
            observation=new_observation,
            hidden=new_hidden,
            metadata=new_metadata,
        )

        rewards = self.compute_rewards(state, action, next_state)

        # Extract answer for info
        extracted, extraction_meta = self._answer_extractor.extract(action.text)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            extracted_action=extracted,
            resolved_action=extracted,
            info={
                "gem_reward": reward,
                "gem_info": info,
                "extracted_answer": extracted,
                "extraction_metadata": extraction_meta,
            },
        )

    def compute_rewards(
        self,
        state: State[GemHidden],
        action: Action,
        next_state: State[GemHidden],
    ) -> SignalBundle:
        """Compute rewards for a transition."""
        signals = []

        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)

        return SignalBundle(signals=tuple(signals))


class GemAdapter:
    """Adapter for the GEM environment suite.

    Provides access to all GEM environments through a common interface.
    GEM includes:
    - Multi-turn games (GuessTheNumber, Sudoku, Wordle, etc.)
    - Single-turn math benchmarks (GSM8K, MATH500, AIME24, etc.)
    - Single-turn code challenges (CodeContest, Taco8k, etc.)
    - Single-turn QA datasets (NaturalQuestions, HotpotQA, etc.)
    - Reasoning-gym tasks

    Example:
        >>> adapter = GemAdapter()
        >>> envs = adapter.list_environments()
        >>> env = adapter.get_environment("game:Sudoku-v0")
    """

    @property
    def name(self) -> str:
        """Adapter identifier."""
        return "gem"

    def _get_gem(self) -> Any:
        """Import and return the gem module."""
        try:
            import gem

            return gem
        except ImportError as e:
            raise ImportError(
                "GEM is required for GemAdapter. Install with: pip install gem-llm"
            ) from e

    def list_environments(self) -> list[str]:
        """List all available environment names.

        Returns all GEM environments including reasoning-gym.
        Users can access reasoning-gym via either this adapter
        or ReasoningGymAdapter based on preference.

        Returns:
            Sorted list of environment IDs.
        """
        gem = self._get_gem()
        # Return all registered GEM envs
        return sorted(gem.ENV_REGISTRY.keys())

    def get_environment(
        self,
        name: str,
        seed: int | None = None,
        answer_extractor: AnswerExtractor | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> "GemEnvironment | GemToolEnvironment":
        """Create an environment by name.

        When ``tool_types`` is passed in kwargs, returns a GemToolEnvironment
        with structured tool support. Otherwise returns a standard
        GemEnvironment.

        Args:
            name: Environment ID (e.g., "game:Sudoku-v0", "math:GSM8K").
            seed: Random seed for environment creation.
            answer_extractor: Extractor for model responses.
            extra_rewards: Additional reward functions appended after native rewards.
            max_steps: Maximum steps per episode (None = unlimited).
            **kwargs: Additional arguments. If ``tool_types`` is present,
                a GemToolEnvironment is created with those tool types.
                Remaining kwargs are passed to gem.make() or tool creation.

        Returns:
            GemEnvironment or GemToolEnvironment.

        Raises:
            ImportError: If GEM is not installed.
            ValueError: If the environment name is not recognized.
        """
        gem = self._get_gem()

        # Check for tool_types to decide which environment to create
        tool_types = kwargs.pop("tool_types", None)

        if tool_types is not None:
            # Create tool environment
            gem_env = gem.make(name, **kwargs)
            return GemToolEnvironment(
                gem_env=gem_env,
                env_id=name,
                tool_types=tool_types,
                max_steps=max_steps or 10,
                extra_rewards=extra_rewards,
                **kwargs,
            )

        # Create standard GEM env
        gem_env = gem.make(name, **kwargs)

        # Detect if multi-turn
        is_multi_turn = _is_multi_turn_env(name)

        return GemEnvironment(
            gem_env=gem_env,
            env_id=name,
            is_multi_turn=is_multi_turn,
            answer_extractor=answer_extractor,
            extra_rewards=extra_rewards,
            max_steps=max_steps,
        )

    def get_default_system_prompt(self, name: str) -> None:
        """GEM manages its own observations, no default system prompt."""
        return None

    def get_prompt_template(self, name: str) -> None:
        """GEM environments manage their own prompts."""
        return None

    def get_native_answer_extractor(self, task_name: str) -> None:
        """GEM does not provide native answer extraction.

        Args:
            task_name: Task name (unused).

        Returns:
            None (no native extraction available).
        """
        return None

    def get_environment_info(self, name: str) -> dict[str, Any]:
        """Get metadata about an environment without creating it.

        Args:
            name: Environment ID.

        Returns:
            Dictionary with environment metadata.
        """
        is_multi_turn = _is_multi_turn_env(name)
        return {
            "name": name,
            "adapter": self.name,
            "type": "multi_turn" if is_multi_turn else "single_turn",
            "description": f"GEM environment: {name}",
        }


# =============================================================================
# Tool Environment Support
# =============================================================================


@dataclass(frozen=True)
class GemToolHidden:
    """Hidden state for GEM tool environments.

    Attributes:
        env_id: GEM environment identifier.
        gem_state: Serialized GEM env state. Frozen tuple for get_state() envs,
            deepcopy dict for stateless envs.
        task_index: Current task index (for dataset-based envs).
        is_multi_turn: Whether this is a multi-turn environment.
        episode_step: Current step within the episode.
        tool_types: Tuple of enabled tool type names.
    """

    env_id: str
    gem_state: Any
    task_index: int
    is_multi_turn: bool
    episode_step: int
    tool_types: tuple[str, ...]


class GemToolExecutor:
    """Executor that bridges our ToolCall to GEM's XML-based tools.

    GEM tools use text-based XML tags for tool calls:
    - Python: <python>print(2+2)</python>
    - Search: <search>what is the capital of France</search>

    This executor converts our structured ToolCall objects to GEM's
    XML format and executes them.
    """

    def __init__(self, gem_tools: dict[str, Any]) -> None:
        """Initialize with GEM tool instances.

        Args:
            gem_tools: Dict mapping tool name to GEM tool instance
                       e.g., {"python": PythonCodeTool(), "search": SearchTool(...)}
        """
        self._gem_tools = gem_tools

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call.

        Args:
            call: The ToolCall to execute.

        Returns:
            ToolResult with execution outcome.
        """
        if call.name not in self._gem_tools:
            return ToolResult.from_error(call.id, call.name, f"Unknown tool: {call.name}")

        # Convert to GEM's XML format
        gem_tool = self._gem_tools[call.name]
        xml_action = self._to_xml_action(call)

        # Execute via GEM
        try:
            is_valid, has_error, observation, _ = gem_tool.execute_action(xml_action)
        except Exception as e:
            return ToolResult.from_error(call.id, call.name, f"Execution error: {e}")

        if not is_valid:
            return ToolResult.from_error(
                call.id,
                call.name,
                "Invalid tool call format",
                status=ToolResultStatus.INVALID_ARGUMENTS,
            )
        if has_error:
            return ToolResult.from_error(call.id, call.name, observation)

        return ToolResult.success(call.id, call.name, observation)

    def _to_xml_action(self, call: ToolCall) -> str:
        """Convert ToolCall to GEM's XML format.

        Args:
            call: The ToolCall to convert.

        Returns:
            XML string in GEM's expected format.
        """
        if call.name == "python":
            return f"<python>{call.arguments.get('code', '')}</python>"
        elif call.name == "search":
            return f"<search>{call.arguments.get('query', '')}</search>"
        elif call.name == "submit_answer":
            return call.arguments.get("answer", "")
        else:
            # Fallback for unknown tools
            return str(call.arguments)

    def execute_batch(self, calls: tuple[ToolCall, ...]) -> tuple[ToolResult, ...]:
        """Execute multiple tool calls sequentially.

        Args:
            calls: Tuple of tool calls to execute.

        Returns:
            Tuple of results in the same order as calls.
        """
        return tuple(self.execute(call) for call in calls)


class GemToolEnvironment(BaseToolEnvironment["GemToolHidden"]):
    """Tool-aware wrapper for GEM environments.

    Wraps GEM environments that support tools (math:*, qa:*) and exposes
    them via our ToolEnvironment protocol for structured function calling.

    This enables using GEM's Python execution and search tools through
    our standard ToolCall/ToolResult interface, which maps to OpenAI and
    Anthropic function calling APIs.

    Example:
        >>> env = GemToolEnvironment(gem_env, "math:GSM8K", tool_types=("python",))
        >>> state, _ = env.reset()
        >>> call = ToolCall(id="1", name="python", arguments={"code": "print(2+2)"})
        >>> action = Action(tool_calls=(call,))
        >>> result = env.step(state, action)
        >>> print(result.info["tool_results"][0].output)
        '4'
    """

    def __init__(
        self,
        gem_env: Any,
        env_id: str,
        tool_types: tuple[str, ...] = ("python",),
        max_steps: int | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        **tool_kwargs: Any,
    ) -> None:
        """Initialize tool environment wrapper.

        Args:
            gem_env: The underlying GEM environment instance.
            env_id: GEM environment identifier (e.g., "math:GSM8K").
            tool_types: Which tools to enable ("python", "search").
            max_steps: Maximum steps per episode (default 10).
            extra_rewards: Additional reward functions appended after native rewards.
            **tool_kwargs: Additional arguments for tool creation
                          (search_url, search_topk for search tool).
        """
        self._gem_env = gem_env
        self._env_id = env_id
        self._max_steps = max_steps or 10
        self._tool_types = tool_types
        self._supports_state = hasattr(gem_env, "get_state") and hasattr(gem_env, "set_state")

        # Initialize GEM tools
        self._gem_tools = self._create_gem_tools(tool_types, **tool_kwargs)

        # Create our ToolDefinitions
        self._tools = self._create_tool_definitions(tool_types)

        # Create executor
        self._executor = GemToolExecutor(self._gem_tools)

        # Reward functions
        self._native_rewards: tuple[RewardFunction, ...] = (
            GemCorrectnessReward(),
            *self._tool_monitoring_rewards(),
        )
        self._extra_rewards = extra_rewards

    def _get_gem(self) -> Any:
        """Import and return the gem module."""
        try:
            import gem

            return gem
        except ImportError as e:
            raise ImportError(
                "GEM is required for GemToolEnvironment. Install with: pip install gem-llm"
            ) from e

    def _create_gem_tools(self, tool_types: tuple[str, ...], **kwargs: Any) -> dict[str, Any]:
        """Create GEM tool instances.

        Args:
            tool_types: Which tools to create.
            **kwargs: Tool-specific arguments.

        Returns:
            Dict mapping tool name to GEM tool instance.
        """
        tools: dict[str, Any] = {}

        if "python" in tool_types:
            from gem.tools.python_code_tool import PythonCodeTool

            tools["python"] = PythonCodeTool()

        if "search" in tool_types:
            from gem.tools.search_tool import SearchTool

            search_url = kwargs.get("search_url", "http://localhost:8000/retrieve")
            topk = kwargs.get("search_topk", 3)
            tools["search"] = SearchTool(search_url=search_url, topk=topk)

        return tools

    def _create_tool_definitions(self, tool_types: tuple[str, ...]) -> tuple[ToolDefinition, ...]:
        """Create ToolDefinitions for the enabled tools.

        Args:
            tool_types: Which tools are enabled.

        Returns:
            Tuple of ToolDefinitions.
        """
        defs: list[ToolDefinition] = []
        if "python" in tool_types:
            defs.append(GEM_PYTHON_TOOL)
        if "search" in tool_types:
            defs.append(GEM_SEARCH_TOOL)
        defs.append(GEM_SUBMIT_ANSWER_TOOL)  # Always include submit
        return tuple(defs)

    @property
    def prompts(self) -> dict[str, str]:
        """GEM tool environments manage prompts internally."""
        return {}

    @property
    def available_tools(self) -> tuple[ToolDefinition, ...]:
        """Get the tools available in this environment."""
        return self._tools

    @property
    def spec(self) -> EnvironmentSpec:
        """Get environment specification."""
        return EnvironmentSpec(
            name=self._env_id,
            adapter="gem",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            pure_step=True,
            metadata={"env_id": self._env_id, "tools": self._tool_types},
        )

    @property
    def reward_functions(
        self,
    ) -> tuple[RewardFunction[GemToolHidden], ...]:
        """Get reward functions used by this environment."""
        return self._native_rewards + self._extra_rewards

    def _snapshot_state(self) -> Any:
        """Capture GEM env state as frozen structure."""
        if not self._supports_state:
            return copy.deepcopy(self._gem_env.__dict__)
        state_dict = self._gem_env.get_state()
        return _freeze_dict(state_dict)

    def _restore_state(self, frozen_state: Any) -> None:
        """Restore GEM env from frozen structure."""
        if not self._supports_state:
            self._gem_env.__dict__.update(copy.deepcopy(frozen_state))
            return
        state_dict = _unfreeze_dict(frozen_state)
        self._gem_env.set_state(state_dict)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[GemToolHidden], dict[str, Any]]:
        """Reset environment and return initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Environment-specific options.
                - task_index: Select specific task (for dataset-based envs).
                - episode_id: Override the generated episode ID.

        Returns:
            Tuple of (initial_state with Observation, info_dict).
        """
        options = options or {}
        task_index = options.get("task_index", 0)

        # Reset GEM env
        obs, info = self._gem_env.reset(seed=seed)

        # Snapshot state
        gem_state = self._snapshot_state()

        hidden = GemToolHidden(
            env_id=self._env_id,
            gem_state=gem_state,
            task_index=task_index,
            is_multi_turn=True,
            episode_step=0,
            tool_types=self._tool_types,
        )

        observation = Observation(
            prompt=str(obs),
            messages=(),
            tool_results=(),
            available_tools=self._tools,
            task=ObservationContent(text=str(obs)),
        )

        state = State(
            observation=observation,
            hidden=hidden,
            metadata=StateMetadata(
                step=0,
                episode_id=options.get("episode_id", str(uuid.uuid4())),
                is_terminal=False,
                info={"task_index": task_index},
            ),
        )

        return state, {"task_index": task_index}

    def step(
        self,
        state: State[GemToolHidden],
        action: Action,
    ) -> StepResult[GemToolHidden]:
        """Take an action from the given state.

        This is a pure function - the same state and action always
        produce the same result. GEM state is restored before stepping.

        Args:
            state: Current state.
            action: Action with optional tool calls.

        Returns:
            StepResult with next state containing tool results.
        """
        # Restore GEM state to ensure pure function semantics
        self._restore_state(state.hidden.gem_state)

        # Handle tool calls
        tool_results: tuple[ToolResult, ...] = ()
        gem_obs: str | None = None
        gem_reward = 0.0
        terminated = False

        if action.has_tool_calls:
            tool_results = self.execute_tools(action.tool_calls)

            # Check for submit_answer (terminal)
            for call in action.tool_calls:
                if call.name == "submit_answer":
                    answer = call.arguments.get("answer", "")
                    gem_obs, gem_reward, terminated, _, info = self._gem_env.step(answer)
                    break

            # For non-terminal tools, observation is the tool result
            if gem_obs is None:
                gem_obs = "\n".join(
                    f"[{r.tool_name}]: {r.output if r.is_success else r.error}"
                    for r in tool_results
                )

        elif action.text:
            # Text-only action - treat as direct answer
            gem_obs, gem_reward, terminated, _, info = self._gem_env.step(action.text)

        else:
            # Empty action - invalid
            gem_obs = "No action provided. Please use a tool or submit an answer."

        # Snapshot new state
        new_gem_state = self._snapshot_state()

        # Check terminal from tool calls
        if action.has_tool_calls:
            terminated = terminated or self._check_terminal_tools(action.tool_calls)

        # Build next observation
        state_text = "\n".join(
            str(r.output) if r.is_success else str(r.error) for r in tool_results
        )
        next_obs = self._build_next_observation(
            state.observation,
            action,
            tool_results,
            state_content=ObservationContent(text=state_text) if state_text else None,
        )

        next_hidden = GemToolHidden(
            env_id=state.hidden.env_id,
            gem_state=new_gem_state,
            task_index=state.hidden.task_index,
            is_multi_turn=True,
            episode_step=state.hidden.episode_step + 1,
            tool_types=state.hidden.tool_types,
        )

        truncated = state.metadata.step + 1 >= self._max_steps

        next_state = State(
            observation=next_obs,
            hidden=next_hidden,
            metadata=StateMetadata(
                step=state.metadata.step + 1,
                episode_id=state.metadata.episode_id,
                is_terminal=terminated or truncated,
                info={**state.metadata.info, "gem_reward": gem_reward},
            ),
        )

        # Compute rewards
        rewards = self._compute_rewards(state, action, next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info={"gem_reward": gem_reward, "tool_results": tool_results},
        )

    def execute_tools(
        self,
        calls: tuple[ToolCall, ...],
    ) -> tuple[ToolResult, ...]:
        """Execute tool calls and return results.

        Args:
            calls: Tuple of tool calls to execute.

        Returns:
            Tuple of results in the same order as calls.
        """
        results: list[ToolResult] = []

        for call in calls:
            # Validate the call
            validation_error = self._validate_tool_call(call)
            if validation_error is not None:
                results.append(validation_error)
                continue

            # Execute via executor
            results.append(self._executor.execute(call))

        return tuple(results)

    def _compute_rewards(
        self,
        state: State[GemToolHidden],
        action: Action,
        next_state: State[GemToolHidden],
    ) -> SignalBundle:
        """Compute rewards for a transition."""
        signals = []

        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)

        return SignalBundle(signals=tuple(signals))
