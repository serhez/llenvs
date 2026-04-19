"""Tests for the WebShop adapter."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llenvs.adapters.webshop import (
    DEFAULT_WEBSHOP_PROMPTS,
    WebShopAdapter,
    WebShopEnvironment,
    WebShopHidden,
    WebShopReward,
    _classify_webshop_command,
    _normalize_webshop_instruction,
    _strip_webshop_instruction_prefix,
    webshop_restore,
)
from llenvs.core.reward import RewardType
from llenvs.core.state import Action, Observation, ObservationContent


class MockWebShopEnv:
    """Mock WebShop gym environment for testing."""

    def __init__(
        self,
        observation_mode: str = "text_rich",
        num_products: int = 1000,
        human_goals: bool = True,
    ):
        self.observation_mode = observation_mode
        self.num_products = num_products
        self.human_goals = human_goals

        # Mock state
        self.state = {
            "url": "http://webshop/",
            "html": "<html>...</html>",
            "instruction_text": "Find a red wireless headphone under $50",
        }

        # Mock clickable elements
        self.text_to_clickable = {
            "search": "search_button",
            "back to search": "back_button",
        }

        # Whether the current page has a search bar. Real WebShop derives this
        # from ``find(id='search_input')`` in ``get_available_actions``.
        self.has_search_bar = True

        self._step_count = 0
        self._done = False

    @property
    def instruction_text(self) -> str:
        return self.state.get("instruction_text", "")

    def reset(
        self,
        session: int | str | None = None,
        instruction_text: str | None = None,
    ) -> tuple[str, None]:
        """Reset the mock environment."""
        self._step_count = 0
        self._done = False
        self.has_search_bar = True
        self.text_to_clickable = {
            "search": "search_button",
            "back to search": "back_button",
        }

        if instruction_text:
            self.state["instruction_text"] = instruction_text

        # Return mock observation based on mode
        if self.observation_mode == "text_rich":
            obs = (
                "Instruction: Find a red wireless headphone under $50\n"
                "[button] Search [button_]\n"
                "[Search bar]\n"
                "Welcome to WebShop!"
            )
        else:
            obs = "Welcome to WebShop! [SEP] Search [SEP]"

        return obs, None

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any] | None]:
        """Take a step in the mock environment."""
        self._step_count += 1

        # Parse action
        if action.startswith("search["):
            # Search action
            query = action[7:-1]  # Extract query
            obs = (
                f"Search results for '{query}':\n"
                "[button] Product 1 - Red Headphones $45 [button_]\n"
                "[button] Product 2 - Blue Headphones $30 [button_]\n"
                "[button] Back to Search [button_]"
            )
            self.text_to_clickable = {
                "product 1 - red headphones $45": "product_1",
                "product 2 - blue headphones $30": "product_2",
                "back to search": "back_button",
            }
            self.has_search_bar = False
            return obs, 0.0, False, None

        elif action.startswith("click["):
            element = action[6:-1]  # Extract element

            if "Product 1" in element:
                # Clicked on matching product
                obs = (
                    "Product: Red Wireless Headphones\n"
                    "Price: $45\n"
                    "Color: Red\n"
                    "[button] Buy Now [button_]\n"
                    "[button] Back to Search [button_]"
                )
                self.text_to_clickable = {
                    "buy now": "buy_button",
                    "back to search": "back_button",
                }
                self.has_search_bar = False
                return obs, 0.0, False, None

            elif element == "Buy Now":
                # Purchase - episode ends with reward
                obs = "Thank you for your purchase!"
                self._done = True
                self.has_search_bar = False
                # Reward based on match quality (mocked as 0.8)
                return obs, 0.8, True, None

            else:
                # Other click
                obs = f"Clicked on: {element}"
                return obs, 0.0, False, None

        else:
            # Invalid action
            obs = "Invalid action format. Use search[query] or click[element]."
            return obs, 0.0, False, None

    def get_available_actions(self) -> dict[str, Any]:
        """Mirror WebShop's ``get_available_actions`` contract."""
        return {
            "has_search_bar": self.has_search_bar,
            "clickables": list(self.text_to_clickable.keys()),
        }


class SimpleTagExtractor:
    """Minimal extractor that pulls text from <answer>...</answer>."""

    def extract(self, text: str | None) -> tuple[str | None, dict]:
        if not text:
            return None, {}
        import re

        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip(), {}
        return None, {}


@pytest.fixture
def mock_webshop_env() -> MockWebShopEnv:
    """Create mock WebShop environment."""
    return MockWebShopEnv()


class TestWebShopHidden:
    """Tests for WebShopHidden state."""

    def test_creation(self):
        """Test hidden state creation."""
        hidden = WebShopHidden(
            instruction="Find red headphones",
            session_id="123",
            task_index=0,
            task_name="Find red headphones",
            episode_step=2,
            last_action="search[headphones]",
            available_actions=("Buy Now", "Back"),
        )

        assert hidden.instruction == "Find red headphones"
        assert hidden.session_id == "123"
        assert hidden.task_index == 0
        assert hidden.task_name == "Find red headphones"
        assert hidden.episode_step == 2
        assert hidden.last_action == "search[headphones]"
        assert "Buy Now" in hidden.available_actions

    def test_immutability(self):
        """Test that hidden state is frozen."""
        hidden = WebShopHidden(
            instruction="test",
            session_id="0",
            task_index=0,
            task_name="test",
            episode_step=0,
            last_action=None,
            available_actions=(),
        )
        with pytest.raises(AttributeError):
            hidden.episode_step = 1  # type: ignore

    def test_trajectory_default(self):
        """Test trajectory defaults to empty tuple."""
        hidden = WebShopHidden(
            instruction="test",
            session_id="0",
            task_index=0,
            task_name="test",
            episode_step=0,
            last_action=None,
            available_actions=(),
        )
        assert hidden.trajectory == ()

class TestWebShopReward:
    """Tests for WebShopReward."""

    def test_reward_name(self):
        """Test reward function name."""
        reward_fn = WebShopReward()
        assert reward_fn.name == "purchase_match"

    def test_reward_type(self):
        """Test reward function type."""
        reward_fn = WebShopReward()
        assert reward_fn.reward_type == RewardType.OUTCOME


class TestWebShopEnvironment:
    """Tests for WebShopEnvironment."""

    def test_creation(self, mock_webshop_env):
        """Test environment creation."""
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            observation_mode="text_rich",
            max_steps=15,
        )

        assert env.spec.name == "webshop"
        assert env.spec.adapter == "webshop"
        assert env.spec.is_multi_turn is True
        assert env.spec.max_steps == 15
        assert env.spec.pure_step is False

    def test_step_raises_on_stale_state(self, mock_webshop_env):
        """Replaying the initial state after a step raises NotImplementedError."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state_0, _ = env.reset()

        env.step(state_0, Action(text="search[headphones]"))

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            env.step(state_0, Action(text="search[headphones]"))

    def test_reset(self, mock_webshop_env):
        """Test environment reset."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, info = env.reset(options={"task_index": 0})

        # Check observation
        assert isinstance(state.observation, Observation)
        assert "<goal>" in state.observation.prompt
        assert "red wireless headphone" in state.observation.prompt.lower()
        assert "<page>" in state.observation.prompt

        # Check hidden state
        assert isinstance(state.hidden, WebShopHidden)
        assert "red wireless headphone" in state.hidden.instruction.lower()
        assert state.hidden.episode_step == 0

        # Check metadata
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False

        # Check info
        assert "instruction" in info

    def test_step_search(self, mock_webshop_env):
        """Test search action."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset()

        action = Action(text="search[red headphones]")
        result = env.step(state, action)

        # Check not terminated
        assert result.terminated is False
        assert result.next_state.metadata.is_terminal is False

        # Check observation in messages (step content goes into messages, not prompt)
        last_msg = result.next_state.observation.messages[-1]["content"]
        assert "Search results" in last_msg
        assert "Red Headphones" in last_msg

        # Check step count
        assert result.next_state.hidden.episode_step == 1
        assert result.next_state.hidden.last_action == "search[red headphones]"

    def test_step_click(self, mock_webshop_env):
        """Test click action."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset()

        # Search first
        action1 = Action(text="search[headphones]")
        result1 = env.step(state, action1)

        # Click on product
        action2 = Action(text="click[Product 1 - Red Headphones $45]")
        result2 = env.step(result1.next_state, action2)

        # Check not terminated
        assert result2.terminated is False

        # Check observation in messages shows product details
        last_msg = result2.next_state.observation.messages[-1]["content"]
        assert "Red Wireless Headphones" in last_msg
        assert "Buy Now" in last_msg

    def test_step_purchase(self, mock_webshop_env):
        """Test purchase completes episode with reward."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset()

        # Search
        action1 = Action(text="search[headphones]")
        result1 = env.step(state, action1)

        # Click product
        action2 = Action(text="click[Product 1 - Red Headphones $45]")
        result2 = env.step(result1.next_state, action2)

        # Buy
        action3 = Action(text="click[Buy Now]")
        result3 = env.step(result2.next_state, action3)

        # Check terminated
        assert result3.terminated is True
        assert result3.next_state.metadata.is_terminal is True

        # Check reward
        assert result3.info["webshop_reward"] == 0.8
        reward_signal = result3.rewards.by_name("purchase_match")
        assert reward_signal is not None
        assert reward_signal.reward == 0.8

    def test_truncation(self, mock_webshop_env):
        """Test episode truncates at max_steps."""
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            max_steps=2,
        )
        state, _ = env.reset()

        # Take max_steps without buying
        action = Action(text="search[headphones]")
        result1 = env.step(state, action)
        result2 = env.step(result1.next_state, action)

        # Should be truncated
        assert result2.truncated is True
        assert result2.next_state.metadata.is_terminal is True

    def test_observation_includes_instruction(self, mock_webshop_env):
        """Test observation includes instruction when configured."""
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            include_instruction_in_obs=True,
        )
        state, _ = env.reset()

        # Instruction surfaces inside a <goal> tag, not as bare text
        assert "<goal>" in state.observation.prompt
        assert "red wireless headphone" in state.observation.prompt.lower()

    def test_observation_no_step_number(self, mock_webshop_env):
        """Test observation does not embed step numbers (handled by runner)."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset()

        assert "[Step 0]" not in state.observation.prompt

        action = Action(text="search[test]")
        result = env.step(state, action)

        assert "[Step 1]" not in result.next_state.observation.messages[-1]["content"]

    def test_action_format_hint(self, mock_webshop_env):
        """Test observation includes action format hint."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset()

        assert "search[" in state.observation.prompt.lower()
        assert "click[" in state.observation.prompt.lower()


class TestWebShopMessageHistory:
    """Tests for message history accumulation in WebShop."""

    def test_message_history_accumulates(self, mock_webshop_env):
        """Message history grows by 2 each turn; prompt stays as initial observation."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset(options={"task_index": 0})
        initial_prompt = state.observation.prompt
        assert state.observation.messages == ()

        # Turn 1: search
        result = env.step(state, Action(text="search[headphones]"))
        state = result.next_state
        assert len(state.observation.messages) == 2
        assert state.observation.messages[0] == {
            "role": "assistant",
            "content": "search[headphones]",
        }
        assert "Search results" in state.observation.messages[1]["content"]
        assert state.observation.prompt == initial_prompt

        # Turn 2: click
        result = env.step(state, Action(text="click[Product 1 - Red Headphones $45]"))
        state = result.next_state
        assert len(state.observation.messages) == 4
        assert state.observation.messages[2] == {
            "role": "assistant",
            "content": "click[Product 1 - Red Headphones $45]",
        }
        assert "Red Wireless Headphones" in state.observation.messages[3]["content"]
        assert state.observation.prompt == initial_prompt

    def test_message_history_on_terminal_step(self, mock_webshop_env):
        """Messages accumulate even on terminal step (Buy Now)."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset(options={"task_index": 0})

        # search → click → buy
        result = env.step(state, Action(text="search[headphones]"))
        result = env.step(result.next_state, Action(text="click[Product 1 - Red Headphones $45]"))
        result = env.step(result.next_state, Action(text="click[Buy Now]"))

        assert result.terminated is True
        assert len(result.next_state.observation.messages) == 6
        assert result.next_state.observation.messages[-2] == {
            "role": "assistant",
            "content": "click[Buy Now]",
        }
        assert "Thank you" in result.next_state.observation.messages[-1]["content"]


class TestWebShopAdapter:
    """Tests for WebShopAdapter."""

    def test_adapter_name(self):
        """Test adapter name property."""
        adapter = WebShopAdapter()
        assert adapter.name == "webshop"

    def test_list_environments(self):
        """Test list_environments returns variants."""
        adapter = WebShopAdapter()
        envs = adapter.list_environments()

        assert "webshop:text" in envs
        assert "webshop:text_rich" in envs
        assert "webshop:html" in envs

    def test_get_environment_info(self):
        """Test get_environment_info returns metadata."""
        adapter = WebShopAdapter()
        info = adapter.get_environment_info()

        assert info["name"] == "webshop"
        assert info["adapter"] == "webshop"
        assert info["type"] == "multi_turn"
        assert "actions" in info
        assert "search[keywords]" in info["actions"]
        assert "click[element]" in info["actions"]

    @patch("llenvs.adapters.webshop.WebShopAdapter._get_webshop")
    def test_get_environment(self, mock_get_webshop):
        """Test creating environment via adapter."""
        # Setup mock
        mock_gym = MagicMock()
        mock_env_class = MagicMock()
        mock_gym.make.return_value = MockWebShopEnv()
        mock_get_webshop.return_value = (mock_gym, mock_env_class)

        adapter = WebShopAdapter()
        env = adapter.get_environment(
            observation_mode="text_rich",
            max_steps=10,
        )

        assert isinstance(env, WebShopEnvironment)
        assert env.spec.max_steps == 10
        mock_gym.make.assert_called_once()

    @patch("llenvs.adapters.webshop.WebShopAdapter._get_webshop")
    def test_get_environment_parses_name(self, mock_get_webshop):
        """Test environment name parsing."""
        mock_gym = MagicMock()
        mock_env_class = MagicMock()
        mock_gym.make.return_value = MockWebShopEnv()
        mock_get_webshop.return_value = (mock_gym, mock_env_class)

        adapter = WebShopAdapter()

        # Name with mode
        adapter.get_environment(name="webshop:html")

        # Check that html mode was used
        call_kwargs = mock_gym.make.call_args[1]
        assert call_kwargs["observation_mode"] == "html"

    @patch("llenvs.adapters.webshop.WebShopAdapter._get_webshop")
    def test_invalid_observation_mode(self, mock_get_webshop):
        """Test error on invalid observation mode."""
        mock_gym = MagicMock()
        mock_env_class = MagicMock()
        mock_get_webshop.return_value = (mock_gym, mock_env_class)

        adapter = WebShopAdapter()

        with pytest.raises(ValueError, match="Invalid observation_mode"):
            adapter.get_environment(observation_mode="invalid")


class TestCreateWebShopEnvironment:
    """Tests for the factory function."""

    @patch("llenvs.adapters.webshop.WebShopAdapter._get_webshop")
    def test_create_environment(self, mock_get_webshop):
        """Test factory function creates environment."""
        mock_gym = MagicMock()
        mock_env_class = MagicMock()
        mock_gym.make.return_value = MockWebShopEnv()
        mock_get_webshop.return_value = (mock_gym, mock_env_class)

        env = WebShopAdapter().get_environment(
            observation_mode="text_rich",
            max_steps=20,
            num_products=1000,
        )

        assert isinstance(env, WebShopEnvironment)
        assert env.spec.max_steps == 20


class TestWebShopMultiStepEpisode:
    """Integration tests for multi-step episodes."""

    def test_full_shopping_episode(self, mock_webshop_env):
        """Test a complete shopping episode."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, info = env.reset(options={"task_index": 0})

        # Verify initial state
        assert "red wireless headphone" in info["instruction"].lower()
        assert state.metadata.step == 0

        # task = static instruction, state = dynamic step content
        assert isinstance(state.observation.task, ObservationContent)
        assert "red wireless headphone" in state.observation.task.text.lower()
        assert isinstance(state.observation.state, ObservationContent)
        assert "[Step 0]" not in state.observation.state.text
        # task and state are distinct
        assert state.observation.task.text != state.observation.state.text
        reset_task = state.observation.task

        # Step 1: Search
        action1 = Action(text="search[red wireless headphones]")
        result1 = env.step(state, action1)
        assert not result1.terminated
        assert result1.next_state.metadata.step == 1

        # task carried forward, state updated with step content
        assert result1.next_state.observation.task is reset_task
        assert isinstance(result1.next_state.observation.state, ObservationContent)
        assert "[Step 1]" not in result1.next_state.observation.state.text
        assert "Search results" in result1.next_state.observation.state.text

        # Step 2: Click on product
        action2 = Action(text="click[Product 1 - Red Headphones $45]")
        result2 = env.step(result1.next_state, action2)
        assert not result2.terminated
        assert result2.next_state.metadata.step == 2

        # task still the same object, state updated to new obs
        assert result2.next_state.observation.task is reset_task
        assert "[Step 2]" not in result2.next_state.observation.state.text

        # Step 3: Buy
        action3 = Action(text="click[Buy Now]")
        result3 = env.step(result2.next_state, action3)
        assert result3.terminated
        assert result3.next_state.metadata.step == 3

        # task still carried forward on terminal step
        assert result3.next_state.observation.task is reset_task

        # Check final reward
        assert result3.info["webshop_reward"] == 0.8

    def test_episode_with_wrong_product(self, mock_webshop_env):
        """Test episode where agent buys wrong product."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset()

        # Search
        action1 = Action(text="search[headphones]")
        result1 = env.step(state, action1)

        # Click wrong product (blue instead of red)
        action2 = Action(text="click[Product 2 - Blue Headphones $30]")
        result2 = env.step(result1.next_state, action2)

        # Episode continues (mock doesn't simulate wrong product details)
        assert not result2.terminated


class TestWebShopPrompts:
    """Tests for WebShop configurable prompt components."""

    def test_default_prompts(self, mock_webshop_env):
        """Test that default prompts are set."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        prompts = env.prompts

        assert "instruction_prefix" in prompts
        assert "action_hint" in prompts

    def test_default_prompts_match_constants(self, mock_webshop_env):
        """Test that default prompts match DEFAULT_WEBSHOP_PROMPTS."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        assert env.prompts == DEFAULT_WEBSHOP_PROMPTS

    def test_prompts_returns_copy(self, mock_webshop_env):
        """Test that prompts property returns a copy."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        p1 = env.prompts
        p2 = env.prompts
        assert p1 == p2
        assert p1 is not p2  # Different dict instances

    def test_custom_prompts_override(self, mock_webshop_env):
        """Test overriding specific prompt components."""
        custom = {"action_hint": "Navigate using search[q] or click[e]."}
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            prompts=custom,
        )
        prompts = env.prompts

        # Overridden
        assert prompts["action_hint"] == "Navigate using search[q] or click[e]."
        # Defaults preserved
        assert prompts["instruction_prefix"] == DEFAULT_WEBSHOP_PROMPTS["instruction_prefix"]
        assert "step_format" not in prompts

    def test_custom_instruction_prefix(self, mock_webshop_env):
        """Test custom instruction prefix appears in observation."""
        custom = {"instruction_prefix": "Your goal: {instruction}"}
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            prompts=custom,
        )
        state, _ = env.reset(options={"task_index": 0})

        assert "Your goal:" in state.observation.prompt
        # Default <goal>...</goal> wrapper should NOT appear when overridden
        assert "<goal>" not in state.observation.prompt

    def test_custom_action_hint(self, mock_webshop_env):
        """Test custom action hint appears in observation."""
        custom = {"action_hint": "Use search[q] or click[e] to navigate."}
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            prompts=custom,
        )
        state, _ = env.reset(options={"task_index": 0})

        assert "Use search[q] or click[e] to navigate." in state.observation.prompt

    def test_empty_action_hint(self, mock_webshop_env):
        """Test that empty action hint is omitted."""
        custom = {"action_hint": ""}
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            prompts=custom,
        )
        state, _ = env.reset(options={"task_index": 0})

        # Default action hint should not appear
        assert "Actions: search[keywords]" not in state.observation.prompt


class TestWebShopAnswerExtractor:
    def test_extracted_action_is_used_for_valid_steps(self, mock_webshop_env):
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            answer_extractor=SimpleTagExtractor(),
            pure_step=True,
        )
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="<answer>search[headphones]</answer>"))

        assert result.extracted_action == "search[headphones]"
        assert result.resolved_action == "search[headphones]"
        assert result.next_state.hidden.trajectory == ("search[headphones]",)
        assistant_msg = result.next_state.observation.messages[-2]
        assert assistant_msg["content"] == "search[headphones]"

    def test_invalid_extraction_executes_sentinel_and_consumes_turn(self, mock_webshop_env):
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            answer_extractor=SimpleTagExtractor(),
            pure_step=True,
        )
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="search[headphones]"))

        assert result.extracted_action is None
        assert result.resolved_action == "[invalid action]"
        assert result.info["invalid_action_format"] is True
        assert result.info["action"] == "__invalid_action_noop__"
        assert result.next_state.metadata.step == 1
        assert result.next_state.hidden.last_action == "__invalid_action_noop__"
        assert result.next_state.hidden.trajectory == ("__invalid_action_noop__",)
        assert "invalid" in result.next_state.observation.state.text.lower()
        assert "Invalid action format" in result.next_state.observation.state.text

    def test_none_invalid_action_text_preserves_raw_history(self, mock_webshop_env):
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            answer_extractor=SimpleTagExtractor(),
            invalid_action_text=None,
            pure_step=True,
        )
        state, _ = env.reset(options={"task_index": 0})

        result = env.step(state, Action(text="search[headphones]"))

        assert result.extracted_action is None
        assert result.resolved_action is None
        assistant_msg = result.next_state.observation.messages[-2]
        assert assistant_msg["content"] == "search[headphones]"


class TestWebShopAdapterDefaultSystemPrompt:
    """Tests for WebShopAdapter.get_default_system_prompt."""

    def test_returns_none(self):
        """WebShop adapter returns None for default system prompt."""
        adapter = WebShopAdapter()
        assert adapter.get_default_system_prompt("webshop") is None


class TestWebShopTrajectoryTracking:
    """Tests for trajectory accumulation in hidden state."""

    def test_reset_sets_empty_trajectory(self, mock_webshop_env):
        """Reset initializes empty trajectory."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset(options={"task_index": 0})

        assert state.hidden.trajectory == ()

    def test_reset_sets_task_name(self, mock_webshop_env):
        """Reset sets task_name from instruction."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset(options={"task_index": 0})

        assert state.hidden.task_name == state.hidden.instruction
        assert "red wireless headphone" in state.hidden.task_name.lower()

    def test_step_appends_to_trajectory(self, mock_webshop_env):
        """Each step appends the action text to trajectory."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset(options={"task_index": 0})

        result1 = env.step(state, Action(text="search[headphones]"))
        assert result1.next_state.hidden.trajectory == ("search[headphones]",)

        result2 = env.step(
            result1.next_state,
            Action(text="click[Product 1 - Red Headphones $45]"),
        )
        assert result2.next_state.hidden.trajectory == (
            "search[headphones]",
            "click[Product 1 - Red Headphones $45]",
        )

    def test_trajectory_preserved_through_full_episode(self, mock_webshop_env):
        """Trajectory accumulates through an entire episode."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset(options={"task_index": 0})

        actions = [
            "search[red headphones]",
            "click[Product 1 - Red Headphones $45]",
            "click[Buy Now]",
        ]
        current = state
        for action_text in actions:
            result = env.step(current, Action(text=action_text))
            current = result.next_state

        assert current.hidden.trajectory == tuple(actions)
        assert current.metadata.is_terminal is True


class TestWebShopRestore:
    """Tests for webshop_restore() (replay-based)."""

    def _run_trajectory(self, env, actions):
        """Helper: reset and step through a list of actions."""
        state, _ = env.reset(options={"task_index": 0})
        current = state
        for action_text in actions:
            result = env.step(current, Action(text=action_text))
            current = result.next_state
        return current

    def test_restore_from_replay(self, mock_webshop_env):
        """Replay-based restore reaches same observation."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        actions = ["search[headphones]", "click[Product 1 - Red Headphones $45]"]
        original_state = self._run_trajectory(env, actions)

        # Create fresh env and restore via replay
        fresh_env = WebShopEnvironment(webshop_env=MockWebShopEnv())
        restored = webshop_restore(fresh_env, original_state)

        assert restored.hidden.episode_step == original_state.hidden.episode_step
        assert restored.observation.state.text == original_state.observation.state.text

    def test_restore_validates_instruction(self, mock_webshop_env):
        """Restore raises ValueError on instruction mismatch."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset(options={"task_index": 0})

        # Tamper with task_name to simulate catalog change
        from dataclasses import replace

        bad_hidden = replace(state.hidden, task_name="Buy a different product entirely")
        bad_state = replace(state, hidden=bad_hidden)

        fresh_env = WebShopEnvironment(webshop_env=MockWebShopEnv())
        with pytest.raises(ValueError, match="Instruction mismatch"):
            webshop_restore(fresh_env, bad_state)

    def test_restore_replay_continues_stepping(self, mock_webshop_env):
        """After replay restore, can continue stepping normally."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        actions = ["search[headphones]"]
        original_state = self._run_trajectory(env, actions)

        fresh_env = WebShopEnvironment(webshop_env=MockWebShopEnv())
        restored = webshop_restore(fresh_env, original_state)

        # Should be able to continue from restored state
        result = fresh_env.step(
            restored,
            Action(text="click[Product 1 - Red Headphones $45]"),
        )
        assert result.next_state.hidden.episode_step == 2
        assert "Red Wireless Headphones" in result.next_state.observation.state.text


class TestWebShopLen:
    """Tests for __len__ on WebShopEnvironment."""

    def test_len_with_num_tasks(self, mock_webshop_env):
        """__len__ returns num_tasks when provided."""
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            num_tasks=500,
        )
        assert len(env) == 500

    def test_len_not_supported_without_num_tasks(self, mock_webshop_env):
        """__len__ raises TypeError when num_tasks not provided."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        with pytest.raises(TypeError):
            len(env)


class TestWebShopSpecFlags:
    """Tests for EnvironmentSpec flags."""

    def test_supports_task_index(self, mock_webshop_env):
        """Spec reports supports_task_index=True."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        assert env.spec.supports_task_index is True

    def test_pure_step_false(self, mock_webshop_env):
        """Spec reports pure_step=False."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        assert env.spec.pure_step is False


class TestWebShopPureStep:
    """Tests for pure_step=True support."""

    def test_pure_step_spec_flag(self, mock_webshop_env):
        """Spec reports pure_step=True when enabled."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env, pure_step=True)
        assert env.spec.pure_step is True

    def test_pure_step_no_state_tracker_error(self, mock_webshop_env):
        """Stepping from a stale state works with pure_step=True."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env, pure_step=True)
        state_0, _ = env.reset()

        # First step
        env.step(state_0, Action(text="search[headphones]"))

        # Replay from state_0 — should NOT raise
        result = env.step(state_0, Action(text="search[shoes]"))
        assert result.next_state.hidden.episode_step == 1

    def test_pure_step_branch_from_earlier_state(self, mock_webshop_env):
        """Branching from an earlier state produces correct independent results."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env, pure_step=True)
        state, _ = env.reset()

        # Branch A: search headphones
        result_a = env.step(state, Action(text="search[headphones]"))
        last_msg_a = result_a.next_state.observation.messages[-1]["content"]

        # Branch B: search shoes (from same initial state)
        result_b = env.step(state, Action(text="search[shoes]"))
        last_msg_b = result_b.next_state.observation.messages[-1]["content"]

        # Different search queries → different observations
        assert "headphones" in last_msg_a.lower()
        assert "shoes" in last_msg_b.lower()
        assert last_msg_a != last_msg_b

    def test_pure_step_default_false_still_has_tracker(self, mock_webshop_env):
        """Default pure_step=False still enforces state continuity."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state_0, _ = env.reset()

        env.step(state_0, Action(text="search[headphones]"))

        with pytest.raises(NotImplementedError, match="pure_step=False"):
            env.step(state_0, Action(text="search[shoes]"))


class TestStripWebShopInstructionPrefix:
    """Unit tests for ``_strip_webshop_instruction_prefix``.

    Exercises the concrete text_rich formats WebShop emits (derived from its
    Jinja templates): search_page has ``WebShop\\n`` + ``Instruction: \\n``,
    other pages use ``Instruction:\\n`` with no leading title.
    """

    INSTR = "i am looking for blue color toothbrushes under 50 dollars"

    def test_homepage_search_page_has_trailing_space(self):
        raw = (
            "WebShop\n"
            f"Instruction: \n{self.INSTR}\n"
            "[button] Search [button_]\n"
        )
        out = _strip_webshop_instruction_prefix(raw, self.INSTR)
        assert "Instruction:" not in out
        assert self.INSTR not in out
        assert out == "WebShop\n[button] Search [button_]\n"

    def test_results_page_no_trailing_space(self):
        raw = (
            f"Instruction:\n{self.INSTR}\n"
            "[button] Back to Search [button_]\n"
            "Page 1 (Total results: 50)\n"
        )
        out = _strip_webshop_instruction_prefix(raw, self.INSTR)
        assert "Instruction:" not in out
        assert self.INSTR not in out
        assert out.startswith("[button] Back to Search [button_]\n")

    def test_text_mode_sep_format(self):
        raw = f"Instruction: [SEP] {self.INSTR} [SEP] some page content"
        out = _strip_webshop_instruction_prefix(raw, self.INSTR)
        assert out == "some page content"

    def test_empty_instruction_returns_unchanged(self):
        raw = "Instruction:\nsomething\n[button] X [button_]\n"
        assert _strip_webshop_instruction_prefix(raw, "") == raw

    def test_no_match_returns_unchanged(self):
        raw = "[button] Buy Now [button_]\n"
        assert _strip_webshop_instruction_prefix(raw, self.INSTR) == raw


class TestNormalizeWebShopInstruction:
    """Unit tests for ``_normalize_webshop_instruction``.

    WebShop's ``get_instruction_text()`` returns the entire
    ``<h4>Instruction:<br>{text}</h4>`` element via ``.text``, so the value
    it surfaces includes the ``"Instruction: "`` label concatenated with
    the goal. The adapter must strip the label before using the string as
    the episode instruction; otherwise the strip in
    ``_strip_webshop_instruction_prefix`` can't match.
    """

    def test_strips_label_with_space(self):
        raw = "Instruction: find a red shirt under $20"
        assert _normalize_webshop_instruction(raw) == "find a red shirt under $20"

    def test_strips_label_without_space(self):
        raw = "Instruction:find a red shirt"
        assert _normalize_webshop_instruction(raw) == "find a red shirt"

    def test_strips_leading_whitespace_before_label(self):
        raw = "   Instruction: find me a thing"
        assert _normalize_webshop_instruction(raw) == "find me a thing"

    def test_no_label_returns_trimmed(self):
        assert _normalize_webshop_instruction("find me a thing  ") == "find me a thing"

    def test_empty_string(self):
        assert _normalize_webshop_instruction("") == ""


class TestWebShopLabeledInstructionIntegration:
    """End-to-end check that the adapter cleanly handles a WebShop-style
    ``instruction_text`` value (``"Instruction: <goal>"``).

    Under the real WebShop env, this is the shape the adapter receives.
    The adapter must: (1) normalize to the pure goal, (2) surface a single
    ``<goal>…</goal>`` tag in ``obs.task.text``, (3) strip the duplicated
    header from inside ``<page>…</page>`` in ``obs.state.text``.
    """

    def test_reset_handles_prefixed_instruction_and_homepage_obs(self):
        env_mock = MagicMock()
        env_mock.reset.return_value = (
            "WebShop\nInstruction: \nfind me a red shirt\n[button] Search [button_]\n",
            None,
        )
        env_mock.state = {
            "instruction_text": "Instruction: find me a red shirt",
            "url": "http://x",
        }
        env_mock.text_to_clickable = {}

        env = WebShopEnvironment(webshop_env=env_mock)
        state, _ = env.reset(options={"task_index": 0})

        task_text = state.observation.task.text
        assert "<goal>find me a red shirt</goal>" in task_text

        state_text = state.observation.state.text
        assert "<goal>" not in state_text
        assert "<page>" in state_text and "</page>" in state_text
        page_block = state_text[state_text.index("<page>"): state_text.index("</page>") + len("</page>")]
        assert "Instruction:" not in page_block
        assert "find me a red shirt" not in page_block
        assert state.hidden.instruction == "find me a red shirt"


class TestClassifyWebShopCommand:
    """Unit tests for ``_classify_webshop_command``.

    Mirrors WebShop's own step() validity check
    (``web_agent_text_env.py:99-112``): verb ∈ {search, click}, non-empty arg,
    search requires a search bar, click target must be in text_to_clickable.
    """

    def test_valid_search_with_search_bar(self, mock_webshop_env):
        mock_webshop_env.has_search_bar = True
        assert _classify_webshop_command("search[red shoes]", mock_webshop_env) == "valid"

    def test_valid_click_existing_target(self, mock_webshop_env):
        mock_webshop_env.text_to_clickable = {"buy now": "btn", "search": "btn"}
        assert _classify_webshop_command("click[Buy Now]", mock_webshop_env) == "valid"

    def test_click_case_insensitive(self, mock_webshop_env):
        """WebShop lowercases the click arg before key lookup."""
        mock_webshop_env.text_to_clickable = {"buy now": "btn"}
        assert _classify_webshop_command("click[BUY NOW]", mock_webshop_env) == "valid"

    def test_unknown_verb(self, mock_webshop_env):
        assert _classify_webshop_command("buy[now]", mock_webshop_env) == "unknown_verb"

    def test_no_brackets_is_unknown_verb(self, mock_webshop_env):
        assert _classify_webshop_command("foo", mock_webshop_env) == "unknown_verb"

    def test_empty_search_arg(self, mock_webshop_env):
        assert _classify_webshop_command("search[]", mock_webshop_env) == "empty_arg"

    def test_whitespace_search_arg(self, mock_webshop_env):
        assert _classify_webshop_command("search[   ]", mock_webshop_env) == "empty_arg"

    def test_search_without_search_bar(self, mock_webshop_env):
        mock_webshop_env.has_search_bar = False
        assert _classify_webshop_command("search[red shoes]", mock_webshop_env) == "no_search_bar"

    def test_click_target_not_on_page(self, mock_webshop_env):
        mock_webshop_env.text_to_clickable = {"buy now": "btn"}
        assert _classify_webshop_command("click[B00NOTAREALPRODUCT]", mock_webshop_env) == "target_missing"


class TestWebShopSemanticInvalidAction:
    """Tests that semantically invalid actions (verb/arg wrong, click target
    missing, search without a search bar) flow through the same invalid-action
    machinery as extraction failure.

    Baseline today: adapter only flags extraction failure; semantically
    invalid commands reach WebShop's step() which silently no-ops, so the
    agent sees no feedback. These tests assert the new behavior.
    """

    @pytest.fixture
    def env(self, mock_webshop_env):
        # No answer_extractor: action.text is forwarded verbatim, so only the
        # new semantic classifier decides validity.
        return WebShopEnvironment(
            webshop_env=mock_webshop_env,
            answer_extractor=None,
            invalid_action_text="[invalid action]",
            invalid_action_observation="NOTICE: bad action.",
            advance_on_invalid="__invalid_action_noop__",
        )

    def test_valid_search_no_notice(self, env):
        state, _ = env.reset()
        result = env.step(state, Action(text="search[red headphones]"))
        assert result.next_state.metadata.info.get("invalid_action_format") is False
        assert "NOTICE: bad action." not in result.next_state.observation.state.text

    def test_valid_click_no_notice(self, env, mock_webshop_env):
        state, _ = env.reset()
        # Homepage has "search" clickable
        result = env.step(state, Action(text="click[Search]"))
        assert result.next_state.metadata.info.get("invalid_action_format") is False

    def test_unknown_verb_flags_invalid(self, env):
        state, _ = env.reset()
        result = env.step(state, Action(text="buy[now]"))
        assert result.next_state.metadata.info["invalid_action_format"] is True
        assert "NOTICE: bad action." in result.next_state.observation.state.text

    def test_no_brackets_flags_invalid(self, env):
        state, _ = env.reset()
        result = env.step(state, Action(text="foo"))
        assert result.next_state.metadata.info["invalid_action_format"] is True

    def test_empty_search_flags_invalid(self, env):
        state, _ = env.reset()
        result = env.step(state, Action(text="search[]"))
        assert result.next_state.metadata.info["invalid_action_format"] is True

    def test_click_target_missing_flags_invalid(self, env):
        state, _ = env.reset()
        result = env.step(state, Action(text="click[NONEXISTENT]"))
        assert result.next_state.metadata.info["invalid_action_format"] is True
        assert "NOTICE: bad action." in result.next_state.observation.state.text

    def test_search_without_search_bar_flags_invalid(self, env, mock_webshop_env):
        """After clicking a product, item page has no search bar."""
        state, _ = env.reset()
        r1 = env.step(state, Action(text="search[red headphones]"))
        r2 = env.step(r1.next_state, Action(text="click[Product 1 - Red Headphones $45]"))
        # Now on item page, no search bar
        r3 = env.step(r2.next_state, Action(text="search[anything]"))
        assert r3.next_state.metadata.info["invalid_action_format"] is True

    def test_invalid_action_sends_advance_sentinel(self, env, mock_webshop_env):
        """The sentinel (not the agent's bad command) is what reaches WebShop."""
        state, _ = env.reset()
        observed: list[str] = []
        original_step = mock_webshop_env.step

        def spy(action_str):
            observed.append(action_str)
            return original_step(action_str)

        mock_webshop_env.step = spy
        env.step(state, Action(text="click[NONEXISTENT]"))
        assert observed == ["__invalid_action_noop__"]

    def test_invalid_history_text_in_assistant_message(self, env):
        """History assistant message stores the invalid_action_text sentinel."""
        state, _ = env.reset()
        result = env.step(state, Action(text="foo"))
        messages = result.next_state.observation.messages
        assistant_msgs = [
            m for m in messages
            if (m.get("role") if isinstance(m, dict) else m.role) == "assistant"
        ]
        assert assistant_msgs
        last = assistant_msgs[-1]
        content = last.get("content") if isinstance(last, dict) else last.content
        assert content == "[invalid action]"

    def test_extraction_failure_still_flags_invalid(self, mock_webshop_env):
        """Regression: extraction-failure path still works alongside the new
        semantic path (same end-state)."""
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            answer_extractor=SimpleTagExtractor(),
            invalid_action_text="[invalid action]",
            invalid_action_observation="NOTICE: bad action.",
            advance_on_invalid="__invalid_action_noop__",
        )
        state, _ = env.reset()
        # No <answer> tag at all → extractor returns None → invalid.
        result = env.step(state, Action(text="here is some prose with no tag"))
        assert result.next_state.metadata.info["invalid_action_format"] is True


class TestWebShopGoalOnlyInTaskText:
    """Goal never appears in ``obs.state.text`` — only in ``obs.task.text``.

    ``state.text`` is ``<page>...</page>`` on every turn (plus the
    invalid-action notice when applicable). The goal lives in
    ``obs.task.text`` (per-episode static). Evaluator/ranking prompts in
    value-bench opt in to prepend ``task.text`` at the top of user prompts
    via the ``inject_task_text_in_prompts`` env-context flag; the adapter
    itself is agnostic to that.
    """

    def test_turn_0_state_omits_goal(self, mock_webshop_env):
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset()
        assert "<goal>" not in state.observation.state.text
        assert "</goal>" not in state.observation.state.text

    def test_turn_1_state_omits_goal(self, mock_webshop_env):
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset()
        result = env.step(state, Action(text="search[red headphones]"))
        assert "<goal>" not in result.next_state.observation.state.text
        assert "</goal>" not in result.next_state.observation.state.text

    def test_turn_2_state_omits_goal(self, mock_webshop_env):
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state0, _ = env.reset()
        r1 = env.step(state0, Action(text="search[red headphones]"))
        r2 = env.step(r1.next_state, Action(text="click[Product 1 - Red Headphones $45]"))
        assert "<goal>" not in r2.next_state.observation.state.text

    def test_invalid_action_turn_still_omits_goal(self, mock_webshop_env):
        """Invalid-action notice appears in state.text but goal still does not."""
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            answer_extractor=None,
            invalid_action_observation="NOTICE",
        )
        state, _ = env.reset()
        result = env.step(state, Action(text="click[NONEXISTENT]"))
        text = result.next_state.observation.state.text
        assert "NOTICE" in text
        assert "<goal>" not in text

    def test_task_text_still_carries_goal(self, mock_webshop_env):
        """obs.task.text keeps the goal on every turn (stable, llenvs-style)."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state0, _ = env.reset()
        r1 = env.step(state0, Action(text="search[red headphones]"))
        assert "<goal>" in state0.observation.task.text
        assert "<goal>" in r1.next_state.observation.task.text

    def test_task_text_is_goal_only(self, mock_webshop_env):
        """obs.task.text is strictly ``<goal>…</goal>`` — no action hint, no
        other content. The system prompt covers action grammar; duplicating
        it in task.text would double-print once value-bench injects task
        text at the top of evaluator prompts."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset()
        task_text = state.observation.task.text
        assert "Actions:" not in task_text
        assert "search[" not in task_text
        assert task_text.startswith("<goal>")
        assert task_text.rstrip().endswith("</goal>")


class TestFilterTerminalPage:
    """``_filter_terminal_page`` strips WebShop's noisy terminal-state dump.

    WebShop's text_rich observation on terminal steps dumps the full
    internal state summary — most fields are ``None`` or empty labels,
    plus an MTurk crowd-sourcing artifact, a Target block that duplicates
    the goal, and a Reward / Reward Details block that would leak GT to
    Q-value evaluators on the Buy-Now transition. The filter keeps only
    the episode-complete signal and the purchase details.
    """

    NOISY_TERMINAL_RAW = (
        "Thank you for shopping with us!\n"
        "Your code: \n"
        "None\n"
        " (Paste it in your MTurk interface.)\n"
        "Purchased\n"
        "asin\n"
        "B08ZRR3DZT\n"
        "options\n"
        "{}\n"
        "attrs\n"
        "None\n"
        "category\n"
        "None\n"
        "query\n"
        "None\n"
        "product category\n"
        "None\n"
        "Target\n"
        "asin\n"
        "options\n"
        "attrs\n"
        "price upper\n"
        "instuction text\n"
        "category\n"
        "product category\n"
        "query\n"
        "Goal \n"
        "None\n"
        "Reward\n"
        "Your score (min 0.0, max 1.0)\n"
        "0.0\n"
        "Reward Details \n"
        "None"
    )

    def test_keeps_purchase_confirmation(self):
        from llenvs.adapters.webshop import _filter_terminal_page

        out = _filter_terminal_page(self.NOISY_TERMINAL_RAW)
        assert "Thank you for shopping with us!" in out

    def test_keeps_purchased_asin_and_options(self):
        from llenvs.adapters.webshop import _filter_terminal_page

        out = _filter_terminal_page(self.NOISY_TERMINAL_RAW)
        assert "Purchased" in out
        assert "asin" in out
        assert "B08ZRR3DZT" in out
        assert "options" in out
        assert "{}" in out

    def test_drops_mturk_block(self):
        from llenvs.adapters.webshop import _filter_terminal_page

        out = _filter_terminal_page(self.NOISY_TERMINAL_RAW)
        assert "Your code" not in out
        assert "MTurk" not in out
        assert "Paste it" not in out

    def test_drops_target_section(self):
        from llenvs.adapters.webshop import _filter_terminal_page

        out = _filter_terminal_page(self.NOISY_TERMINAL_RAW)
        # Target header gone, as is everything beneath it until the next section.
        assert "Target" not in out
        assert "price upper" not in out  # Target-only label
        assert "instuction text" not in out  # Target-only label (upstream typo)

    def test_drops_reward_section(self):
        from llenvs.adapters.webshop import _filter_terminal_page

        out = _filter_terminal_page(self.NOISY_TERMINAL_RAW)
        assert "Reward" not in out
        assert "Your score" not in out
        # The numeric reward value should not leak to evaluators either.
        assert "0.0" not in out

    def test_drops_none_valued_rows_within_purchased(self):
        """Purchased's label/None rows (attrs, category, query, product
        category) should all disappear, leaving only non-None pairs."""
        from llenvs.adapters.webshop import _filter_terminal_page

        out = _filter_terminal_page(self.NOISY_TERMINAL_RAW)
        # No bare "None" token survives anywhere.
        assert "\nNone" not in out
        # Specific None-valued labels from Purchased are gone.
        assert "attrs" not in out
        assert "category" not in out
        assert "query" not in out

    def test_preserves_non_none_purchased_fields(self):
        """If attrs / category have actual values (non-None), they survive."""
        from llenvs.adapters.webshop import _filter_terminal_page

        raw = (
            "Thank you for shopping with us!\n"
            "Purchased\n"
            "asin\n"
            "B01\n"
            "options\n"
            "{'size': 'small', 'color': 'blue'}\n"
            "attrs\n"
            "['cotton', 'machine washable']\n"
        )
        out = _filter_terminal_page(raw)
        assert "attrs" in out
        assert "['cotton', 'machine washable']" in out
        assert "{'size': 'small', 'color': 'blue'}" in out

    def test_simple_terminal_passthrough(self):
        """A minimal terminal obs (no noise sections) passes through unchanged."""
        from llenvs.adapters.webshop import _filter_terminal_page

        raw = "Thank you for your purchase!"
        assert _filter_terminal_page(raw) == raw

    def test_empty_input(self):
        from llenvs.adapters.webshop import _filter_terminal_page

        assert _filter_terminal_page("") == ""


class TestTerminalPageIntegration:
    """End-to-end: terminal-step observation in ``state.text`` is filtered."""

    @pytest.fixture
    def noisy_mock_env(self, mock_webshop_env):
        """Mock env that returns WebShop's noisy terminal dump on Buy Now."""
        noisy = TestFilterTerminalPage.NOISY_TERMINAL_RAW
        original_step = mock_webshop_env.step

        def patched_step(action):
            obs, reward, done, info = original_step(action)
            if done:
                return noisy, reward, done, info
            return obs, reward, done, info

        mock_webshop_env.step = patched_step
        return mock_webshop_env

    def test_terminal_state_text_filters_noise(self, noisy_mock_env):
        env = WebShopEnvironment(webshop_env=noisy_mock_env)
        state0, _ = env.reset()
        r1 = env.step(state0, Action(text="search[red headphones]"))
        r2 = env.step(r1.next_state, Action(text="click[Product 1 - Red Headphones $45]"))
        r3 = env.step(r2.next_state, Action(text="click[Buy Now]"))

        text = r3.next_state.observation.state.text
        assert r3.terminated is True
        # Useful signal survives
        assert "Thank you for shopping" in text
        assert "B08ZRR3DZT" in text
        # Noise does not
        assert "Your code" not in text
        assert "Target" not in text
        assert "Reward" not in text
        assert "\nNone" not in text

    def test_non_terminal_state_text_unchanged(self, mock_webshop_env):
        """Filter only applies to terminal steps; search/click pages pass through."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state0, _ = env.reset()
        r1 = env.step(state0, Action(text="search[red headphones]"))
        # Non-terminal search-results page; should be wrapped as-is (no filter).
        text = r1.next_state.observation.state.text
        assert "<page>" in text
        assert "Product 1 - Red Headphones $45" in text
