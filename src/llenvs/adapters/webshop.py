"""WebShop adapter - wraps WebShop e-commerce environment.

WebShop (Yao et al., NeurIPS 2022) is a simulated e-commerce website with
1.18M real products and 12K crowd-sourced instructions. Agents navigate
webpages using search and click actions to find and purchase products.

Reference: https://github.com/princeton-nlp/WebShop
"""

import re
import uuid
from dataclasses import dataclass
from typing import Any

from llenvs.core.environment import EnvironmentSpec, StepResult, _StateContinuityTracker
from llenvs.core.extraction import AnswerExtractor
from llenvs.core.reward import RewardFunction, RewardType, Signal, SignalBundle
from llenvs.core.state import Action, Observation, ObservationContent, State, StateMetadata


@dataclass(frozen=True)
class WebShopHidden:
    """Hidden state for WebShop environment.

    Attributes:
        instruction: The shopping task instruction (what to buy).
        session_id: WebShop session identifier.
        task_index: Index of the current task.
        task_name: Instruction text, used for identity validation during restore.
        episode_step: Current step within the episode.
        last_action: The last action taken (for context).
        available_actions: Currently available clickable elements.
        trajectory: Accumulated action history for replay-based restore.
    """

    instruction: str
    session_id: str
    task_index: int
    task_name: str
    episode_step: int
    last_action: str | None
    available_actions: tuple[str, ...]
    trajectory: tuple[str, ...] = ()


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
    "instruction_prefix": "<goal>{instruction}</goal>",
    "action_hint": "Actions: search[keywords] or click[element]",
}

DEFAULT_INVALID_ACTION_TEXT = "[invalid action]"
DEFAULT_INVALID_NOOP_COMMAND = "__invalid_action_noop__"
DEFAULT_WEBSHOP_INVALID_ACTION_OBSERVATION = (
    "The provided action was invalid. A turn was wasted. "
    "Provide exactly one action in the required format and use a valid action "
    "described above."
)


def _strip_webshop_instruction_prefix(raw_obs: str, instruction: str) -> str:
    """Remove WebShop's own "Instruction:…<text>…" header from raw_obs.

    WebShop's text_rich renderer emits an ``Instruction:[ \\t]*\\n<text>\\n``
    block on most pages (homepage additionally prefixed by ``WebShop\\n``
    and uses a trailing space in the label). text mode uses a single-line
    ``Instruction: [SEP] <text> [SEP] `` prefix. The adapter surfaces the
    instruction once via ``<goal>…</goal>`` in task text and doesn't want
    it duplicated inside ``<page>…</page>``. Returns raw_obs unchanged if
    no matching block is found.
    """
    if not instruction:
        return raw_obs
    text_mode_prefix = f"Instruction: [SEP] {instruction} [SEP] "
    if raw_obs.startswith(text_mode_prefix):
        return raw_obs[len(text_mode_prefix):]
    pattern = re.compile(rf"Instruction:[ \t]*\n{re.escape(instruction)}\n")
    return pattern.sub("", raw_obs, count=1)


def _classify_webshop_command(cmd: str, env: Any) -> str:
    """Classify a parsed WebShop command against the current page's validity.

    Mirrors WebShop's step() validity check (``web_agent_text_env.py:99-112``):
    verb ∈ {search, click}, non-empty arg, search requires a search bar,
    click target must be in ``env.text_to_clickable`` (case-insensitive).

    Returns one of: ``"valid"``, ``"unknown_verb"``, ``"empty_arg"``,
    ``"no_search_bar"``, ``"target_missing"``. A ``"valid"`` result promises
    the command will not be silently no-op'd by WebShop's step().
    """
    try:
        # WebShop's engine exports parse_action(action) -> (verb, arg).
        from web_agent_site.engine.engine import parse_action as _parse
    except ImportError:
        _parse = None

    if _parse is not None:
        verb, arg = _parse(cmd)
    else:
        # Fallback parser matching the WebShop regex: '(.+)\[(.+)\]'. Kept so
        # tests using MockWebShopEnv don't require the upstream repo to be
        # importable.
        m = re.match(r"(.+)\[(.*)\]", cmd)
        if m is None:
            verb, arg = cmd, None
        else:
            verb, arg = m.group(1), m.group(2)

    if verb not in ("search", "click") or arg is None:
        return "unknown_verb"
    if not arg.strip():
        return "empty_arg"

    avail = env.get_available_actions()
    has_search_bar = bool(avail.get("has_search_bar", False))

    if verb == "search":
        return "valid" if has_search_bar else "no_search_bar"

    # verb == "click"
    clickables = {k.lower() for k in env.text_to_clickable.keys()}
    return "valid" if arg.lower() in clickables else "target_missing"


def _normalize_webshop_instruction(instruction: str) -> str:
    """Strip WebShop's "Instruction:" label from the returned instruction.

    WebShop's ``get_instruction_text()`` extracts ``<h4>Instruction:<br>
    {text}</h4>`` via BeautifulSoup's ``.text``, which concatenates the
    ``"Instruction:"`` label with the actual goal. Strip the label so the
    stored instruction is the pure goal text.
    """
    return re.sub(r"^\s*Instruction:\s*", "", instruction).strip()


def _wrap_page(raw_obs: str, instruction: str) -> str:
    """Strip WebShop's embedded instruction header and wrap in <page> tags."""
    stripped = _strip_webshop_instruction_prefix(raw_obs, instruction).rstrip()
    return f"<page>\n{stripped}\n</page>"


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
        num_tasks: int | None = None,
        pure_step: bool = False,
        answer_extractor: AnswerExtractor | None = None,
        invalid_action_text: str | None = DEFAULT_INVALID_ACTION_TEXT,
        invalid_action_observation: str | None = None,
        advance_on_invalid: str | None = DEFAULT_INVALID_NOOP_COMMAND,
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
            num_tasks: Number of tasks available. Required for __len__.
            pure_step: When True, step() replays the trajectory on the gym
                env before executing, enabling branching from arbitrary
                states. WebShop's gym env is not picklable (Cython
                __cinit__ in pyjnius/pyserini) so replay is the only
                mechanism — expect O(N) step cost.
            answer_extractor: Extractor applied to raw action text before
                sending to WebShop. Strips reasoning tokens, etc.
            invalid_action_text: Assistant history text stored when no
                executable action could be extracted.
            invalid_action_observation: Optional custom reminder shown
                before the fallback env observation on malformed turns.
            advance_on_invalid: Real WebShop action executed when no
                action could be extracted. Defaults to a fixed sentinel
                invalid action so replay/snapshots stay aligned.
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
        self._num_tasks = num_tasks
        self._pure_step = pure_step
        self._answer_extractor = answer_extractor
        self._invalid_action_text = invalid_action_text
        self._invalid_action_observation = invalid_action_observation
        self._advance_on_invalid = advance_on_invalid
        self._state_tracker = None if pure_step else _StateContinuityTracker()

        # Track current instruction for observation building
        self._current_instruction: str = ""

    @property
    def answer_extractor(self):
        """The extractor used to parse agent responses in ``step()``."""
        return self._answer_extractor

    @answer_extractor.setter
    def answer_extractor(self, value):
        self._answer_extractor = value

    @property
    def prompts(self) -> dict[str, str]:
        """Named prompt components used for building observations."""
        return dict(self._prompts)

    @property
    def available_tools(self) -> tuple:
        """No tools available in WebShop environments."""
        return ()

    def __len__(self) -> int:
        """Return number of available tasks.

        Raises:
            TypeError: If num_tasks was not provided at construction.
        """
        if self._num_tasks is None:
            raise TypeError("WebShopEnvironment has no len(); pass num_tasks to the constructor")
        return self._num_tasks

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
            supports_task_index=True,
            pure_step=self._pure_step,
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

        # In text_rich mode, clickables are tagged with [button] or similar.
        # Also check the environment's text_to_clickable mapping — the attr
        # exists but is None until the first page is rendered, so hasattr()
        # alone isn't enough to guard the .keys() call.
        text_to_clickable = getattr(self._env, "text_to_clickable", None)
        if text_to_clickable is not None:
            actions.extend(text_to_clickable.keys())

        # Search is always available
        if "search" not in [a.lower() for a in actions]:
            actions.append("search[...]")

        return tuple(actions)

    def _build_observation_prompt(
        self,
        raw_obs: str,
        instruction: str,
        step: int,
        invalid_action_notice: str | None = None,
    ) -> str:
        """Build the full observation prompt for the model.

        Layout: ``<goal>…</goal>`` (optional), the invalid-action notice if
        any, ``<page>…</page>`` with WebShop's own duplicated instruction
        header stripped, and the static action hint.
        """
        parts = []

        if self._include_instruction_in_obs:
            prefix = self._prompts["instruction_prefix"]
            parts.append(prefix.format(instruction=instruction))
            parts.append("")

        if invalid_action_notice:
            parts.append(invalid_action_notice)
            parts.append("")

        parts.append(_wrap_page(raw_obs, instruction))

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
        instruction = _normalize_webshop_instruction(instruction)

        self._current_instruction = instruction

        # Extract available actions
        available = self._extract_available_actions(raw_obs)

        # Build observation
        obs_prompt = self._build_observation_prompt(raw_obs, instruction, 0)

        # Task = static <goal>…</goal> for the whole episode.
        # State = dynamic page content, same shape on every turn.
        task_text = self._prompts["instruction_prefix"].format(instruction=instruction)
        state_text = _wrap_page(raw_obs, instruction)

        hidden = WebShopHidden(
            instruction=instruction,
            session_id=str(session),
            task_index=task_index,
            task_name=instruction,
            episode_step=0,
            last_action=None,
            available_actions=available,
        )

        observation = Observation(
            prompt=obs_prompt,
            task=ObservationContent(text=task_text),
            state=ObservationContent(text=state_text),
        )

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
        if self._state_tracker is not None:
            self._state_tracker.track(state)

        return state, {
            "task_index": task_index,
            "instruction": instruction,
            "session_id": str(session),
        }

    def _text_for_history(
        self,
        raw_text: str,
        extracted_cmd: str | None,
        *,
        invalid_action_format: bool = False,
    ) -> str:
        """Return text for the assistant turn in conversation history.

        Uses extracted command when available. On extraction failure, applies
        the extractor's pre-cleaners to strip reasoning tokens from history.
        """
        if extracted_cmd is not None:
            return extracted_cmd
        if invalid_action_format and self._invalid_action_text is not None:
            return self._invalid_action_text
        if self._answer_extractor is None:
            return raw_text
        from llenvs.core.extraction import CleanedExtractor

        if isinstance(self._answer_extractor, CleanedExtractor):
            cleaned = raw_text
            for cleaner in self._answer_extractor.pre_cleaners:
                cleaned = cleaner(cleaned)
            return cleaned
        return raw_text

    def _invalid_action_notice(self) -> str:
        if self._invalid_action_observation is not None:
            return self._invalid_action_observation
        return DEFAULT_WEBSHOP_INVALID_ACTION_OBSERVATION

    def _combine_invalid_observation(self, env_feedback: str) -> str:
        notice = self._invalid_action_notice()
        if not env_feedback:
            return notice
        return f"{notice}\n\n{env_feedback}"

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
        if self._state_tracker is not None:
            self._state_tracker.validate(state, "WebShopEnvironment")

        action_text = action.text or ""

        # Extract clean command (strips reasoning tokens etc.)
        extracted_cmd: str | None = None
        extraction_metadata: dict[str, Any] = {}
        invalid_action_format = False
        if self._answer_extractor is not None:
            extracted_cmd, extraction_metadata = self._answer_extractor.extract(action_text)
            if extracted_cmd is not None:
                extracted_cmd = extracted_cmd.strip()
                if not extracted_cmd:
                    extracted_cmd = None
        if self._answer_extractor is not None and extracted_cmd is None:
            cmd_for_env = self._advance_on_invalid
            invalid_action_format = True
        else:
            cmd_for_env = extracted_cmd or action_text

        # For pure_step, replay the trajectory on a fresh gym env to reach
        # `state` before executing this step. WebShop's env isn't picklable,
        # so replay (O(N)) is the only mechanism.
        if self._pure_step:
            reset_kwargs = (
                {"session": state.hidden.session_id}
                if state.hidden.session_id is not None
                else {}
            )
            self._env.reset(**reset_kwargs)
            for past_action in state.hidden.trajectory:
                self._env.step(past_action)

        # Semantic validation: WebShop silently no-ops unknown verbs, empty
        # args, searches without a search bar, and clicks on targets that
        # aren't on the current page. Classify here so the adapter can
        # surface meta-feedback via invalid_action_text/observation instead.
        # Skipped if the extraction layer already flagged the turn.
        if not invalid_action_format and cmd_for_env is not None:
            classification = _classify_webshop_command(cmd_for_env, self._env)
            if classification != "valid":
                invalid_action_format = True
                cmd_for_env = self._advance_on_invalid

        if cmd_for_env is not None:
            # Step WebShop environment
            raw_obs, reward, done, info = self._env.step(cmd_for_env)
        else:
            raw_obs = state.observation.state.text if state.observation.state is not None else ""
            reward = 0.0
            done = False
            info = None

        # Extract available actions from new observation
        available = self._extract_available_actions(raw_obs)

        # Build next observation. Page content is wrapped in <page> tags;
        # the invalid-action notice (if any) is rendered outside the page
        # block so it reads as our meta-feedback, not WebShop's page output.
        next_step = state.hidden.episode_step + 1
        notice = (
            self._invalid_action_notice() if invalid_action_format else None
        )
        page_block = _wrap_page(raw_obs, state.hidden.instruction)
        obs_prompt = self._build_observation_prompt(
            raw_obs,
            state.hidden.instruction,
            next_step,
            invalid_action_notice=notice,
        )

        # Check truncation
        truncated = next_step >= self._max_steps and not done

        new_hidden = WebShopHidden(
            instruction=state.hidden.instruction,
            session_id=state.hidden.session_id,
            task_index=state.hidden.task_index,
            task_name=state.hidden.task_name,
            episode_step=next_step,
            last_action=cmd_for_env,
            available_actions=available,
            trajectory=(
                state.hidden.trajectory + (cmd_for_env,)
                if cmd_for_env is not None
                else state.hidden.trajectory
            ),
        )

        state_text = f"{notice}\n\n{page_block}" if notice else page_block

        new_messages = tuple(state.observation.messages) + (
            {
                "role": "assistant",
                "content": self._text_for_history(
                    action_text,
                    extracted_cmd,
                    invalid_action_format=invalid_action_format,
                ),
            },
            {"role": "user", "content": obs_prompt},
        )
        new_observation = Observation(
            prompt=state.observation.prompt,
            messages=new_messages,
            task=state.observation.task,
            state=ObservationContent(text=state_text),
        )

        new_metadata = StateMetadata(
            step=state.metadata.step + 1,
            episode_id=state.metadata.episode_id,
            is_terminal=done or truncated,
            info={
                **state.metadata.info,
                "webshop_reward": reward,
                "last_action": cmd_for_env,
                "invalid_action_format": invalid_action_format,
                **({"extraction_metadata": extraction_metadata} if extraction_metadata else {}),
            },
        )

        next_state = State(
            observation=new_observation,
            hidden=new_hidden,
            metadata=new_metadata,
        )

        # Compute rewards
        rewards = self.compute_rewards(state, action, next_state)
        if self._state_tracker is not None:
            self._state_tracker.track(next_state)

        return StepResult(
            next_state=next_state,
            rewards=rewards,
            terminated=done,
            truncated=truncated,
            extracted_action=extracted_cmd,
            resolved_action=(
                self._invalid_action_text
                if invalid_action_format and self._invalid_action_text is not None
                else extracted_cmd
            ),
            info={
                "webshop_reward": reward,
                "action": cmd_for_env,
                "done": done,
                "invalid_action_format": invalid_action_format,
                **({"extraction_metadata": extraction_metadata} if extraction_metadata else {}),
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


def webshop_restore(
    env: WebShopEnvironment,
    state: State[WebShopHidden],
) -> State[WebShopHidden]:
    """Restore a WebShop env to a saved state via action replay.

    WebShop's gym env is not picklable, so this is the only mechanism
    available — O(N) in the trajectory length.

    Args:
        env: A fresh WebShopEnvironment instance (from env_factory).
        state: The target state to restore to (from a prior trajectory).

    Returns:
        The restored state, ready for continued stepping.

    Raises:
        ValueError: If instruction at task_index doesn't match the saved state.
    """
    return _restore_from_replay(env, state)


def _restore_from_replay(
    env: WebShopEnvironment,
    state: State[WebShopHidden],
) -> State[WebShopHidden]:
    """Restore by resetting and replaying the action history."""
    current, _ = env.reset(
        options={
            "task_index": state.hidden.task_index,
            "session": state.hidden.session_id,
            "episode_id": state.metadata.episode_id,
        }
    )

    # Validate instruction identity
    if state.hidden.task_name and current.hidden.task_name:
        if state.hidden.task_name != current.hidden.task_name:
            raise ValueError(
                f"Instruction mismatch at task_index {state.hidden.task_index}: "
                f"expected {state.hidden.task_name!r}, "
                f"got {current.hidden.task_name!r}"
            )

    # Replay action history
    for action_text in state.hidden.trajectory:
        result = env.step(current, Action(text=action_text))
        current = result.next_state

    return current


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
        pure_step: bool = False,
        num_tasks: int | None = None,
        answer_extractor: AnswerExtractor | None = None,
        invalid_action_text: str | None = DEFAULT_INVALID_ACTION_TEXT,
        invalid_action_observation: str | None = None,
        advance_on_invalid: str | None = DEFAULT_INVALID_NOOP_COMMAND,
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
            pure_step: When True, enable state save/restore via
                pickle for branching from arbitrary states.
            num_tasks: Number of tasks for ``__len__``.
            answer_extractor: Extractor applied to raw action text before
                sending to WebShop. Strips reasoning tokens, etc.
            invalid_action_text: Assistant history text stored for
                malformed responses.
            invalid_action_observation: Optional custom reminder shown
                before the fallback env observation on malformed turns.
            advance_on_invalid: Real WebShop action executed when no
                action could be extracted.
            **kwargs: Additional arguments passed to WebAgentTextEnv.

        Returns:
            Configured WebShopEnvironment.

        Raises:
            ImportError: If webshop is not installed.
        """
        gym, _web_agent_text_env = self._get_webshop()

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
                f"Invalid observation_mode: {observation_mode}. Must be one of: {valid_modes}"
            )

        # Create WebShop gym environment. WebAgentTextEnv predates gym 0.26's
        # requirement that envs declare `action_space`, so the PassiveEnvChecker
        # that gym.make wraps envs in by default trips an AssertionError on
        # construction. Bypass it; WebShop defines its own action parser.
        env_kwargs = {
            "observation_mode": observation_mode,
            "human_goals": human_goals,
            "disable_env_checker": True,
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
            pure_step=pure_step,
            num_tasks=num_tasks,
            answer_extractor=answer_extractor,
            invalid_action_text=invalid_action_text,
            invalid_action_observation=invalid_action_observation,
            advance_on_invalid=advance_on_invalid,
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
