"""WebShop adapter - wraps WebShop e-commerce environment.

WebShop (Yao et al., NeurIPS 2022) is a simulated e-commerce website with
1.18M real products and 12K crowd-sourced instructions. Agents navigate
webpages using search and click actions to find and purchase products.

Reference: https://github.com/princeton-nlp/WebShop
"""

from dataclasses import dataclass
from typing import Any
import uuid

from llenvs.core.state import State, StateMetadata, Observation, Action
from llenvs.core.reward import SignalBundle, Signal, RewardType, RewardFunction
from llenvs.core.environment import StepResult, EnvironmentSpec, _StateContinuityTracker


@dataclass(frozen=True)
class WebShopHidden:
    """Hidden state for WebShop environment.

    Attributes:
        instruction: The shopping task instruction (what to buy).
        session_id: WebShop session identifier.
        task_index: Index of the current task.
        episode_step: Current step within the episode.
        last_action: The last action taken (for context).
        available_actions: Currently available clickable elements.
    """

    instruction: str
    session_id: str
    task_index: int
    episode_step: int
    last_action: str | None
    available_actions: tuple[str, ...]


@dataclass
class WebShopReward:
    """Reward function for WebShop based on purchase match quality.

    WebShop computes reward only at episode end, measuring how well
    the purchased product matches the instruction criteria.
    """

    _name: str = "purchase_match"
    _reward_type: RewardType = RewardType.OUTCOME

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def compute(
        self,
        state: State[WebShopHidden],
        action: Action,
        next_state: State[WebShopHidden],
    ) -> Signal:
        """Compute reward from WebShop's native reward."""
        webshop_reward = next_state.metadata.info.get("webshop_reward", 0.0)

        return Signal(
            name=self.name,
            reward_type=self.reward_type,
            reward=float(webshop_reward),
            metadata={"source": "webshop"},
        )


DEFAULT_WEBSHOP_PROMPTS: dict[str, str] = {
    "instruction_prefix": "Instruction: {instruction}",
    "step_format": "[Step {step}]",
    "action_hint": "Actions: search[keywords] or click[element]",
}


class WebShopEnvironment:
    """MDP wrapper for WebShop e-commerce environment.

    WebShop is a multi-turn environment where agents navigate a simulated
    e-commerce website to find and purchase products matching given
    instructions.

    Actions are text-based:
    - search[query]: Search for products with given keywords
    - click[element]: Click on a page element (product, button, option)

    The environment provides rich text observations with tagged interactive
    elements that the agent can click.

    Example:
        >>> env = WebShopEnvironment(webshop_env)
        >>> state, _ = env.reset(options={"task_index": 0})
        >>> print(state.observation.prompt)
        # Shows instruction and current page with clickable elements

        >>> action = Action(text="search[wireless headphones]")
        >>> result = env.step(state, action)
        # Returns search results page

        >>> action = Action(text="click[Buy Now]")
        >>> result = env.step(result.next_state, action)
        # Completes purchase, returns final reward
    """

    def __init__(
        self,
        webshop_env: Any,
        observation_mode: str = "text_rich",
        max_steps: int = 15,
        include_instruction_in_obs: bool = True,
        extra_rewards: tuple[RewardFunction, ...] = (),
        prompts: dict[str, str] | None = None,
    ) -> None:
        """Initialize WebShop environment wrapper.

        Args:
            webshop_env: The underlying WebShop gym environment.
            observation_mode: How to format observations. Options:
                - "text": Simple text with [SEP] separators
                - "text_rich": Tagged buttons and clickables (recommended)
                - "html": Raw HTML
            max_steps: Maximum steps per episode before truncation.
            include_instruction_in_obs: Whether to prepend instruction to
                each observation (helps model remember the goal).
            extra_rewards: Additional reward functions appended after native rewards.
            prompts: Override default prompt components. Keys:
                instruction_prefix, step_format, action_hint.
        """
        self._env = webshop_env
        self._observation_mode = observation_mode
        self._max_steps = max_steps
        self._include_instruction_in_obs = include_instruction_in_obs
        self._native_rewards: tuple[RewardFunction, ...] = (WebShopReward(),)
        self._extra_rewards = extra_rewards
        self._prompts = {**DEFAULT_WEBSHOP_PROMPTS}
        if prompts:
            self._prompts.update(prompts)
        self._state_tracker = _StateContinuityTracker()

        # Track current instruction for observation building
        self._current_instruction: str = ""

    @property
    def prompts(self) -> dict[str, str]:
        """Named prompt components used for building observations."""
        return dict(self._prompts)

    @property
    def available_tools(self) -> tuple:
        """No tools available in WebShop environments."""
        return ()

    @property
    def spec(self) -> EnvironmentSpec:
        """Get environment specification."""
        return EnvironmentSpec(
            name="webshop",
            adapter="webshop",
            max_steps=self._max_steps,
            observation_type=Observation,
            action_type=Action,
            is_multi_turn=True,
            metadata={
                "observation_mode": self._observation_mode,
                "description": "E-commerce product search and purchase",
            },
        )

    @property
    def reward_functions(
        self,
    ) -> tuple[RewardFunction[WebShopHidden], ...]:
        """Get reward functions used by this environment."""
        return self._native_rewards + self._extra_rewards

    def _extract_available_actions(self, observation: str) -> tuple[str, ...]:
        """Extract available clickable actions from observation.

        Args:
            observation: The text observation from WebShop.

        Returns:
            Tuple of clickable element texts.
        """
        actions: list[str] = []

        # In text_rich mode, clickables are tagged with [button] or similar
        # Also check the environment's text_to_clickable mapping
        if hasattr(self._env, "text_to_clickable"):
            actions.extend(self._env.text_to_clickable.keys())

        # Search is always available
        if "search" not in [a.lower() for a in actions]:
            actions.append("search[...]")

        return tuple(actions)

    def _build_observation_prompt(
        self, raw_obs: str, instruction: str, step: int
    ) -> str:
        """Build the full observation prompt for the model.

        Args:
            raw_obs: Raw observation from WebShop.
            instruction: The shopping instruction.
            step: Current step number.

        Returns:
            Formatted observation string.
        """
        parts = []

        if self._include_instruction_in_obs:
            prefix = self._prompts["instruction_prefix"]
            parts.append(prefix.format(instruction=instruction))
            parts.append("")

        step_fmt = self._prompts["step_format"]
        parts.append(step_fmt.format(step=step))
        parts.append(raw_obs)

        hint = self._prompts.get("action_hint", "")
        if hint:
            parts.append("")
            parts.append(hint)

        return "\n".join(parts)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[State[WebShopHidden], dict[str, Any]]:
        """Reset environment and return initial state.

        Args:
            seed: Random seed (passed to WebShop if supported).
            options: Environment-specific options.
                - task_index: Select specific task/session.
                - session: WebShop session ID.
                - instruction_text: Override instruction (for testing).

        Returns:
            Tuple of (initial_state, info_dict).
        """
        options = options or {}
        task_index = options.get("task_index", 0)
        session = options.get("session", task_index)

        # Reset WebShop environment
        reset_kwargs: dict[str, Any] = {}
        if session is not None:
            reset_kwargs["session"] = session
        if "instruction_text" in options:
            reset_kwargs["instruction_text"] = options["instruction_text"]

        raw_obs, _ = self._env.reset(**reset_kwargs)

        # Get instruction from environment state
        instruction = ""
        if hasattr(self._env, "state") and isinstance(self._env.state, dict):
            instruction = self._env.state.get("instruction_text", "")
        elif hasattr(self._env, "instruction_text"):
            instruction = self._env.instruction_text

        self._current_instruction = instruction

        # Extract available actions
        available = self._extract_available_actions(raw_obs)

        # Build observation
        obs_prompt = self._build_observation_prompt(raw_obs, instruction, 0)

        hidden = WebShopHidden(
            instruction=instruction,
            session_id=str(session),
            task_index=task_index,
            episode_step=0,
            last_action=None,
            available_actions=available,
        )

        observation = Observation(prompt=obs_prompt)

        metadata = StateMetadata(
            step=0,
            episode_id=options.get("episode_id", str(uuid.uuid4())),
            is_terminal=False,
            info={
                "task_index": task_index,
                "instruction": instruction,
                "session_id": str(session),
            },
        )

        state = State(observation=observation, hidden=hidden, metadata=metadata)
        self._state_tracker.track(state)

        return state, {
            "task_index": task_index,
            "instruction": instruction,
            "session_id": str(session),
        }

    def step(
        self,
        state: State[WebShopHidden],
        action: Action,
    ) -> StepResult[WebShopHidden]:
        """Take an action from the given state.

        Args:
            state: Current state.
            action: Action to take. Should be in format:
                - "search[keywords]" for searching
                - "click[element]" for clicking

        Returns:
            StepResult containing next state, rewards, and done flags.
        """
        self._state_tracker.validate(state, "WebShopEnvironment")

        # Step WebShop environment
        raw_obs, reward, done, info = self._env.step(action.text)

        # Extract available actions from new observation
        available = self._extract_available_actions(raw_obs)

        # Build next observation
        next_step = state.hidden.episode_step + 1
        obs_prompt = self._build_observation_prompt(
            raw_obs, state.hidden.instruction, next_step
        )

        # Check truncation
        truncated = next_step >= self._max_steps and not done

        new_hidden = WebShopHidden(
            instruction=state.hidden.instruction,
            session_id=state.hidden.session_id,
            task_index=state.hidden.task_index,
            episode_step=next_step,
            last_action=action.text,
            available_actions=available,
        )

        new_messages = tuple(state.observation.messages) + (
            {"role": "assistant", "content": action.text or ""},
            {"role": "user", "content": obs_prompt},
        )
        new_observation = Observation(prompt=state.observation.prompt, messages=new_messages)

        new_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=done or truncated,
            info={
                **state.metadata.info,
                "webshop_reward": reward,
                "last_action": action.text,
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
            terminated=done,
            truncated=truncated,
            info={
                "webshop_reward": reward,
                "action": action.text,
                "done": done,
            },
        )

    def compute_rewards(
        self,
        state: State[WebShopHidden],
        action: Action,
        next_state: State[WebShopHidden],
    ) -> SignalBundle:
        """Compute rewards for a transition."""
        signals = []

        for reward_fn in self.reward_functions:
            signal = reward_fn.compute(state, action, next_state)
            signals.append(signal)

        return SignalBundle(signals=tuple(signals))


class WebShopAdapter:
    """Adapter for the WebShop e-commerce environment.

    WebShop simulates an e-commerce website where agents must find and
    purchase products matching natural language instructions.

    Requires the webshop package: pip install webshop

    Note: WebShop requires initial setup to download product data and
    build the search index. See: https://github.com/princeton-nlp/WebShop

    Example:
        >>> adapter = WebShopAdapter()
        >>> env = adapter.get_environment(observation_mode="text_rich")
        >>> state, _ = env.reset(options={"task_index": 0})
    """

    @property
    def name(self) -> str:
        """Adapter identifier."""
        return "webshop"

    def _get_webshop(self) -> Any:
        """Import and return the webshop module."""
        try:
            import gym

            # WebShop registers itself with gym on import
            from web_agent_site.envs import WebAgentTextEnv

            return gym, WebAgentTextEnv
        except ImportError as e:
            raise ImportError(
                "WebShop is required for WebShopAdapter. "
                "Install with: pip install webshop\n"
                "Then run setup: https://github.com/princeton-nlp/WebShop#setup"
            ) from e

    def list_environments(self) -> list[str]:
        """List available environment variants.

        Returns:
            List of environment IDs (observation modes).
        """
        return [
            "webshop:text",
            "webshop:text_rich",
            "webshop:html",
        ]

    def get_environment(
        self,
        name: str = "webshop:text_rich",
        observation_mode: str | None = None,
        max_steps: int = 15,
        num_products: int | None = None,
        human_goals: bool = True,
        prompts: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> WebShopEnvironment:
        """Create a WebShop environment.

        Args:
            name: Environment ID. Format: "webshop:{mode}" where mode is
                text, text_rich, or html. Defaults to text_rich.
            observation_mode: Override observation mode from name.
            max_steps: Maximum steps per episode.
            num_products: Number of products to load (None = all).
            human_goals: Use human-written goals (True) or templates.
            **kwargs: Additional arguments passed to WebAgentTextEnv.

        Returns:
            Configured WebShopEnvironment.

        Raises:
            ImportError: If webshop is not installed.
        """
        gym, WebAgentTextEnv = self._get_webshop()

        # Parse observation mode from name if not explicitly provided
        if observation_mode is None:
            if ":" in name:
                observation_mode = name.split(":", 1)[1]
            else:
                observation_mode = "text_rich"

        # Validate observation mode
        valid_modes = {"text", "text_rich", "html", "url"}
        if observation_mode not in valid_modes:
            raise ValueError(
                f"Invalid observation_mode: {observation_mode}. "
                f"Must be one of: {valid_modes}"
            )

        # Create WebShop gym environment
        env_kwargs = {
            "observation_mode": observation_mode,
            "human_goals": human_goals,
            **kwargs,
        }
        if num_products is not None:
            env_kwargs["num_products"] = num_products

        webshop_env = gym.make("WebAgentTextEnv-v0", **env_kwargs)

        return WebShopEnvironment(
            webshop_env=webshop_env,
            observation_mode=observation_mode,
            max_steps=max_steps,
            prompts=prompts,
        )

    def get_default_system_prompt(self, name: str) -> None:
        """WebShop observations include built-in instructions."""
        return None

    def get_prompt_template(self, name: str) -> None:
        """WebShop manages multi-turn prompts internally."""
        return None

    def get_native_answer_extractor(self, task_name: str) -> None:
        """WebShop does not provide native answer extraction.

        Args:
            task_name: Task name (unused).

        Returns:
            None (no native extraction available).
        """
        return None

    def get_environment_info(self, name: str = "webshop") -> dict[str, Any]:
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
            "description": (
                "WebShop: E-commerce product search and purchase. "
                "Agent navigates website to find and buy products matching instructions."
            ),
            "actions": ["search[keywords]", "click[element]"],
            "reference": "https://github.com/princeton-nlp/WebShop",
        }
