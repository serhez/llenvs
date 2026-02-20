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
            "Search": "search_button",
            "Back to Search": "back_button",
        }

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
                "Product 1 - Red Headphones $45": "product_1",
                "Product 2 - Blue Headphones $30": "product_2",
                "Back to Search": "back_button",
            }
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
                    "Buy Now": "buy_button",
                    "Back to Search": "back_button",
                }
                return obs, 0.0, False, None

            elif element == "Buy Now":
                # Purchase - episode ends with reward
                obs = "Thank you for your purchase!"
                self._done = True
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
            episode_step=2,
            last_action="search[headphones]",
            available_actions=("Buy Now", "Back"),
        )

        assert hidden.instruction == "Find red headphones"
        assert hidden.session_id == "123"
        assert hidden.task_index == 0
        assert hidden.episode_step == 2
        assert hidden.last_action == "search[headphones]"
        assert "Buy Now" in hidden.available_actions

    def test_immutability(self):
        """Test that hidden state is frozen."""
        hidden = WebShopHidden(
            instruction="test",
            session_id="0",
            task_index=0,
            episode_step=0,
            last_action=None,
            available_actions=(),
        )
        with pytest.raises(AttributeError):
            hidden.episode_step = 1  # type: ignore


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
        assert "Instruction:" in state.observation.prompt
        assert "red wireless headphone" in state.observation.prompt.lower()

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

        # Instruction should be in observation
        assert "Instruction:" in state.observation.prompt
        assert "red wireless headphone" in state.observation.prompt.lower()

    def test_observation_step_number(self, mock_webshop_env):
        """Test observation includes step number."""
        env = WebShopEnvironment(webshop_env=mock_webshop_env)
        state, _ = env.reset()

        assert "[Step 0]" in state.observation.prompt

        action = Action(text="search[test]")
        result = env.step(state, action)

        assert "[Step 1]" in result.next_state.observation.messages[-1]["content"]

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
        env = adapter.get_environment(name="webshop:html")

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

        # task and state are set on reset
        assert isinstance(state.observation.task, ObservationContent)
        assert state.observation.task.text == state.observation.prompt
        assert isinstance(state.observation.state, ObservationContent)
        assert state.observation.state.text == state.observation.prompt
        reset_task = state.observation.task

        # Step 1: Search
        action1 = Action(text="search[red wireless headphones]")
        result1 = env.step(state, action1)
        assert not result1.terminated
        assert result1.next_state.metadata.step == 1

        # task carried forward, state updated
        assert result1.next_state.observation.task is reset_task
        assert isinstance(result1.next_state.observation.state, ObservationContent)
        step1_obs = result1.next_state.observation.messages[-1]["content"]
        assert result1.next_state.observation.state.text == step1_obs

        # Step 2: Click on product
        action2 = Action(text="click[Product 1 - Red Headphones $45]")
        result2 = env.step(result1.next_state, action2)
        assert not result2.terminated
        assert result2.next_state.metadata.step == 2

        # task still the same object, state updated to new obs
        assert result2.next_state.observation.task is reset_task
        step2_obs = result2.next_state.observation.messages[-1]["content"]
        assert result2.next_state.observation.state.text == step2_obs

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
        assert "step_format" in prompts
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
        assert prompts["step_format"] == DEFAULT_WEBSHOP_PROMPTS["step_format"]

    def test_custom_instruction_prefix(self, mock_webshop_env):
        """Test custom instruction prefix appears in observation."""
        custom = {"instruction_prefix": "Your goal: {instruction}"}
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            prompts=custom,
        )
        state, _ = env.reset(options={"task_index": 0})

        assert "Your goal:" in state.observation.prompt
        # Default "Instruction:" should NOT appear
        assert not state.observation.prompt.startswith("Instruction:")

    def test_custom_step_format(self, mock_webshop_env):
        """Test custom step format appears in observation."""
        custom = {"step_format": "Step #{step}:"}
        env = WebShopEnvironment(
            webshop_env=mock_webshop_env,
            prompts=custom,
        )
        state, _ = env.reset(options={"task_index": 0})

        assert "Step #0:" in state.observation.prompt

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


class TestWebShopAdapterDefaultSystemPrompt:
    """Tests for WebShopAdapter.get_default_system_prompt."""

    def test_returns_none(self):
        """WebShop adapter returns None for default system prompt."""
        adapter = WebShopAdapter()
        assert adapter.get_default_system_prompt("webshop") is None
