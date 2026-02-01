"""Tests for GEM tool environment integration."""

import pytest
from typing import Any
from unittest.mock import MagicMock, patch

from llenvs.core.state import AgentObservation, AgentAction
from llenvs.core.tools import (
    ToolCall,
    ToolDefinition,
    ToolParameterType,
    ToolResult,
    ToolResultStatus,
)
from llenvs.core.reward import RewardType
from llenvs.adapters.gem import (
    GemToolEnvironment,
    GemToolHidden,
    GemToolExecutor,
    GEM_PYTHON_TOOL,
    GEM_SEARCH_TOOL,
    GEM_SUBMIT_ANSWER_TOOL,
)


class MockPythonCodeTool:
    """Mock GEM Python code tool for testing."""

    def execute_action(self, action: str) -> tuple[bool, bool, str, Any]:
        """Execute Python code in XML format.

        Args:
            action: XML-formatted action like "<python>print(2+2)</python>"

        Returns:
            Tuple of (is_valid, has_error, observation, parsed_action)
        """
        # Parse XML
        if not action.startswith("<python>") or not action.endswith("</python>"):
            return False, True, "Invalid format", None

        code = action[8:-9]  # Extract code between tags

        # Simple simulation
        try:
            # Execute very basic expressions
            if "print(" in code:
                # Extract the expression inside print()
                import re

                match = re.search(r"print\((.+)\)", code)
                if match:
                    expr = match.group(1)
                    result = str(eval(expr))
                    return True, False, result, code
            elif code.strip():
                result = str(eval(code))
                return True, False, result, code
            return True, False, "", code
        except Exception as e:
            return True, True, str(e), code


class MockSearchTool:
    """Mock GEM search tool for testing."""

    def __init__(self, search_url: str = "", topk: int = 3):
        self.search_url = search_url
        self.topk = topk

    def execute_action(self, action: str) -> tuple[bool, bool, str, Any]:
        """Execute search query in XML format.

        Args:
            action: XML-formatted action like "<search>query</search>"

        Returns:
            Tuple of (is_valid, has_error, observation, parsed_action)
        """
        if not action.startswith("<search>") or not action.endswith("</search>"):
            return False, True, "Invalid format", None

        query = action[8:-9]  # Extract query between tags

        # Return mock results
        return (
            True,
            False,
            f"Search results for '{query}': Result 1, Result 2, Result 3",
            query,
        )


class MockGemEnvWithTools:
    """Mock GEM environment for tool testing."""

    def __init__(self, env_id: str = "math:GSM8K"):
        self.env_id = env_id
        self.problem = "What is 15% of 80?"
        self.answer = "12"
        self._state: dict[str, Any] = {}

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        """Reset with a math problem."""
        self._state = {"problem": self.problem, "answer": self.answer}
        return self.problem, {"answer": self.answer}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Check answer (always terminates)."""
        # Check if answer is correct
        if "12" in action:
            reward = 1.0
            obs = "Correct!"
        else:
            reward = 0.0
            obs = f"Incorrect. The answer was {self.answer}."

        return obs, reward, True, False, {"submitted_answer": action}

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    def set_state(self, state: dict[str, Any]) -> None:
        self._state = dict(state)


@pytest.fixture
def mock_python_tool() -> MockPythonCodeTool:
    """Create mock Python tool."""
    return MockPythonCodeTool()


@pytest.fixture
def mock_search_tool() -> MockSearchTool:
    """Create mock search tool."""
    return MockSearchTool()


@pytest.fixture
def mock_gem_env() -> MockGemEnvWithTools:
    """Create mock GEM environment."""
    return MockGemEnvWithTools()


class TestToolDefinitions:
    """Tests for tool definitions."""

    def test_python_tool_definition(self):
        """Test Python tool definition structure."""
        assert GEM_PYTHON_TOOL.name == "python"
        assert "Execute Python code" in GEM_PYTHON_TOOL.description
        assert len(GEM_PYTHON_TOOL.parameters) == 1
        assert GEM_PYTHON_TOOL.parameters[0].name == "code"
        assert GEM_PYTHON_TOOL.parameters[0].type == ToolParameterType.STRING
        assert GEM_PYTHON_TOOL.is_terminal is False

    def test_search_tool_definition(self):
        """Test Search tool definition structure."""
        assert GEM_SEARCH_TOOL.name == "search"
        assert "Search" in GEM_SEARCH_TOOL.description
        assert len(GEM_SEARCH_TOOL.parameters) == 1
        assert GEM_SEARCH_TOOL.parameters[0].name == "query"
        assert GEM_SEARCH_TOOL.parameters[0].type == ToolParameterType.STRING
        assert GEM_SEARCH_TOOL.is_terminal is False

    def test_submit_answer_tool_definition(self):
        """Test submit_answer tool definition structure."""
        assert GEM_SUBMIT_ANSWER_TOOL.name == "submit_answer"
        assert "final answer" in GEM_SUBMIT_ANSWER_TOOL.description.lower()
        assert len(GEM_SUBMIT_ANSWER_TOOL.parameters) == 1
        assert GEM_SUBMIT_ANSWER_TOOL.parameters[0].name == "answer"
        assert GEM_SUBMIT_ANSWER_TOOL.is_terminal is True

    def test_tool_to_openai_schema(self):
        """Test conversion to OpenAI schema format."""
        schema = GEM_PYTHON_TOOL.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "python"
        assert "code" in schema["function"]["parameters"]["properties"]

    def test_tool_to_anthropic_schema(self):
        """Test conversion to Anthropic schema format."""
        schema = GEM_PYTHON_TOOL.to_anthropic_schema()

        assert schema["name"] == "python"
        assert "code" in schema["input_schema"]["properties"]


class TestGemToolExecutor:
    """Tests for the GEM tool executor."""

    def test_python_tool_execution(self, mock_python_tool):
        """Test converting ToolCall to Python XML and executing."""
        executor = GemToolExecutor({"python": mock_python_tool})

        call = ToolCall(id="1", name="python", arguments={"code": "print(2+2)"})
        result = executor.execute(call)

        assert result.is_success
        assert result.call_id == "1"
        assert result.tool_name == "python"
        assert result.output == "4"

    def test_search_tool_execution(self, mock_search_tool):
        """Test converting ToolCall to Search XML and executing."""
        executor = GemToolExecutor({"search": mock_search_tool})

        call = ToolCall(id="2", name="search", arguments={"query": "capital of France"})
        result = executor.execute(call)

        assert result.is_success
        assert result.call_id == "2"
        assert result.tool_name == "search"
        assert "capital of France" in result.output

    def test_unknown_tool(self, mock_python_tool):
        """Test handling of unknown tool names."""
        executor = GemToolExecutor({"python": mock_python_tool})

        call = ToolCall(id="3", name="unknown", arguments={})
        result = executor.execute(call)

        assert result.is_error
        assert result.status == ToolResultStatus.ERROR
        assert "Unknown tool" in result.error

    def test_execute_batch(self, mock_python_tool, mock_search_tool):
        """Test batch execution of multiple tool calls."""
        executor = GemToolExecutor(
            {"python": mock_python_tool, "search": mock_search_tool}
        )

        calls = (
            ToolCall(id="1", name="python", arguments={"code": "print(3*3)"}),
            ToolCall(id="2", name="search", arguments={"query": "test"}),
        )
        results = executor.execute_batch(calls)

        assert len(results) == 2
        assert results[0].is_success
        assert results[0].output == "9"
        assert results[1].is_success

    def test_xml_conversion_python(self, mock_python_tool):
        """Test XML conversion for Python tool."""
        executor = GemToolExecutor({"python": mock_python_tool})

        call = ToolCall(id="1", name="python", arguments={"code": "x = 5"})
        xml = executor._to_xml_action(call)

        assert xml == "<python>x = 5</python>"

    def test_xml_conversion_search(self, mock_search_tool):
        """Test XML conversion for Search tool."""
        executor = GemToolExecutor({"search": mock_search_tool})

        call = ToolCall(id="1", name="search", arguments={"query": "test query"})
        xml = executor._to_xml_action(call)

        assert xml == "<search>test query</search>"

    def test_xml_conversion_submit_answer(self):
        """Test XML conversion for submit_answer tool."""
        executor = GemToolExecutor({})

        call = ToolCall(id="1", name="submit_answer", arguments={"answer": "42"})
        xml = executor._to_xml_action(call)

        assert xml == "42"


class TestGemToolHidden:
    """Tests for GemToolHidden state."""

    def test_creation(self):
        """Test hidden state creation."""
        hidden = GemToolHidden(
            env_id="math:GSM8K",
            gem_state=(("key", "value"),),
            task_index=0,
            is_multi_turn=True,
            episode_step=2,
            tool_types=("python", "search"),
        )

        assert hidden.env_id == "math:GSM8K"
        assert hidden.task_index == 0
        assert hidden.episode_step == 2
        assert hidden.tool_types == ("python", "search")

    def test_immutability(self):
        """Test that hidden state is frozen."""
        hidden = GemToolHidden(
            env_id="math:GSM8K",
            gem_state=(),
            task_index=0,
            is_multi_turn=True,
            episode_step=0,
            tool_types=("python",),
        )
        with pytest.raises(AttributeError):
            hidden.episode_step = 1  # type: ignore


class TestGemToolEnvironment:
    """Tests for the GEM tool environment wrapper."""

    @patch("llenvs.adapters.gem.GemToolEnvironment._create_gem_tools")
    def test_creation(self, mock_create_tools, mock_gem_env):
        """Test environment creation."""
        mock_create_tools.return_value = {"python": MockPythonCodeTool()}

        env = GemToolEnvironment(
            gem_env=mock_gem_env,
            env_id="math:GSM8K",
            tool_types=("python",),
            max_steps=10,
        )

        assert env.spec.name == "math:GSM8K"
        assert env.spec.adapter == "gem"
        assert env.spec.is_multi_turn is True
        assert env.spec.max_steps == 10

    @patch("llenvs.adapters.gem.GemToolEnvironment._create_gem_tools")
    def test_available_tools_property(self, mock_create_tools, mock_gem_env):
        """Test available_tools returns correct definitions."""
        mock_create_tools.return_value = {"python": MockPythonCodeTool()}

        env = GemToolEnvironment(
            gem_env=mock_gem_env,
            env_id="math:GSM8K",
            tool_types=("python",),
        )

        tools = env.available_tools
        tool_names = [t.name for t in tools]

        assert "python" in tool_names
        assert "submit_answer" in tool_names

    @patch("llenvs.adapters.gem.GemToolEnvironment._create_gem_tools")
    def test_reset_returns_agent_observation(self, mock_create_tools, mock_gem_env):
        """Test reset returns AgentObservation with tools."""
        mock_create_tools.return_value = {"python": MockPythonCodeTool()}

        env = GemToolEnvironment(
            gem_env=mock_gem_env,
            env_id="math:GSM8K",
            tool_types=("python",),
        )

        state, info = env.reset(options={"task_index": 0})

        # Check observation type
        assert isinstance(state.observation, AgentObservation)

        # Check observation content
        assert "15% of 80" in state.observation.prompt
        assert state.observation.messages == ()
        assert state.observation.tool_results == ()
        assert len(state.observation.available_tools) > 0

        # Check hidden state
        assert isinstance(state.hidden, GemToolHidden)
        assert state.hidden.tool_types == ("python",)
        assert state.hidden.episode_step == 0

        # Check metadata
        assert state.metadata.step == 0
        assert state.metadata.is_terminal is False

    @patch("llenvs.adapters.gem.GemToolEnvironment._create_gem_tools")
    def test_step_with_tool_call(self, mock_create_tools, mock_gem_env):
        """Test stepping with a tool call."""
        mock_python = MockPythonCodeTool()
        mock_create_tools.return_value = {"python": mock_python}

        env = GemToolEnvironment(
            gem_env=mock_gem_env,
            env_id="math:GSM8K",
            tool_types=("python",),
        )

        state, _ = env.reset()

        # Make a tool call
        call = ToolCall(id="1", name="python", arguments={"code": "print(0.15 * 80)"})
        action = AgentAction(tool_calls=(call,))
        result = env.step(state, action)

        # Check tool results
        assert "tool_results" in result.info
        tool_results = result.info["tool_results"]
        assert len(tool_results) == 1
        assert tool_results[0].is_success
        assert tool_results[0].output == "12.0"

        # Check not terminated (didn't submit answer)
        assert result.terminated is False

        # Check next observation includes tool results
        assert len(result.next_state.observation.tool_results) == 1

    @patch("llenvs.adapters.gem.GemToolEnvironment._create_gem_tools")
    def test_submit_answer_terminates(self, mock_create_tools, mock_gem_env):
        """Test submit_answer tool terminates episode."""
        mock_create_tools.return_value = {"python": MockPythonCodeTool()}

        env = GemToolEnvironment(
            gem_env=mock_gem_env,
            env_id="math:GSM8K",
            tool_types=("python",),
        )

        state, _ = env.reset()

        # Submit correct answer
        call = ToolCall(id="1", name="submit_answer", arguments={"answer": "12"})
        action = AgentAction(tool_calls=(call,))
        result = env.step(state, action)

        # Check terminated
        assert result.terminated is True
        assert result.next_state.metadata.is_terminal is True

        # Check reward
        assert result.info["gem_reward"] == 1.0

    @patch("llenvs.adapters.gem.GemToolEnvironment._create_gem_tools")
    def test_submit_wrong_answer(self, mock_create_tools, mock_gem_env):
        """Test submitting wrong answer."""
        mock_create_tools.return_value = {"python": MockPythonCodeTool()}

        env = GemToolEnvironment(
            gem_env=mock_gem_env,
            env_id="math:GSM8K",
            tool_types=("python",),
        )

        state, _ = env.reset()

        # Submit wrong answer
        call = ToolCall(id="1", name="submit_answer", arguments={"answer": "10"})
        action = AgentAction(tool_calls=(call,))
        result = env.step(state, action)

        # Still terminated
        assert result.terminated is True

        # But reward is 0
        assert result.info["gem_reward"] == 0.0

    @patch("llenvs.adapters.gem.GemToolEnvironment._create_gem_tools")
    def test_text_action(self, mock_create_tools, mock_gem_env):
        """Test text-only action (direct answer)."""
        mock_create_tools.return_value = {"python": MockPythonCodeTool()}

        env = GemToolEnvironment(
            gem_env=mock_gem_env,
            env_id="math:GSM8K",
            tool_types=("python",),
        )

        state, _ = env.reset()

        # Text action treated as direct answer
        action = AgentAction.from_text("12")
        result = env.step(state, action)

        assert result.terminated is True
        assert result.info["gem_reward"] == 1.0

    @patch("llenvs.adapters.gem.GemToolEnvironment._create_gem_tools")
    def test_invalid_tool_call(self, mock_create_tools, mock_gem_env):
        """Test handling of invalid tool names."""
        mock_create_tools.return_value = {"python": MockPythonCodeTool()}

        env = GemToolEnvironment(
            gem_env=mock_gem_env,
            env_id="math:GSM8K",
            tool_types=("python",),
        )

        state, _ = env.reset()

        # Call non-existent tool
        call = ToolCall(id="1", name="invalid_tool", arguments={})
        action = AgentAction(tool_calls=(call,))
        result = env.step(state, action)

        # Check tool result shows error
        tool_results = result.info["tool_results"]
        assert len(tool_results) == 1
        assert tool_results[0].is_error
        assert tool_results[0].status == ToolResultStatus.INVALID_TOOL

    @patch("llenvs.adapters.gem.GemToolEnvironment._create_gem_tools")
    def test_multi_step_episode(self, mock_create_tools, mock_gem_env):
        """Test multi-step episode with tools."""
        mock_create_tools.return_value = {"python": MockPythonCodeTool()}

        env = GemToolEnvironment(
            gem_env=mock_gem_env,
            env_id="math:GSM8K",
            tool_types=("python",),
        )

        state, _ = env.reset()

        # Step 1: Use Python to calculate
        call1 = ToolCall(id="1", name="python", arguments={"code": "print(0.15 * 80)"})
        action1 = AgentAction(tool_calls=(call1,))
        result1 = env.step(state, action1)

        assert not result1.terminated
        assert result1.next_state.hidden.episode_step == 1

        # Step 2: Submit the answer
        call2 = ToolCall(id="2", name="submit_answer", arguments={"answer": "12"})
        action2 = AgentAction(tool_calls=(call2,))
        result2 = env.step(result1.next_state, action2)

        assert result2.terminated
        assert result2.info["gem_reward"] == 1.0

    @patch("llenvs.adapters.gem.GemToolEnvironment._create_gem_tools")
    def test_truncation_at_max_steps(self, mock_create_tools, mock_gem_env):
        """Test environment truncates at max_steps."""
        mock_create_tools.return_value = {"python": MockPythonCodeTool()}

        env = GemToolEnvironment(
            gem_env=mock_gem_env,
            env_id="math:GSM8K",
            tool_types=("python",),
            max_steps=2,
        )

        state, _ = env.reset()

        # Take max_steps without submitting answer
        for _ in range(2):
            call = ToolCall(
                id="1", name="python", arguments={"code": "print('thinking...')"}
            )
            action = AgentAction(tool_calls=(call,))
            result = env.step(state, action)
            state = result.next_state

        # Should be truncated
        assert result.truncated is True
        assert result.next_state.metadata.is_terminal is True

    @patch("llenvs.adapters.gem.GemToolEnvironment._create_gem_tools")
    def test_reward_functions(self, mock_create_tools, mock_gem_env):
        """Test reward function composition."""
        mock_create_tools.return_value = {"python": MockPythonCodeTool()}

        env = GemToolEnvironment(
            gem_env=mock_gem_env,
            env_id="math:GSM8K",
            tool_types=("python",),
        )

        rewards = env.reward_functions
        reward_names = [r.name for r in rewards]

        assert "correctness" in reward_names
        assert "tool_validity" in reward_names

    @patch("llenvs.adapters.gem.GemToolEnvironment._create_gem_tools")
    def test_state_restoration(self, mock_create_tools, mock_gem_env):
        """Test that state is properly restored for replay."""
        mock_create_tools.return_value = {"python": MockPythonCodeTool()}

        env = GemToolEnvironment(
            gem_env=mock_gem_env,
            env_id="math:GSM8K",
            tool_types=("python",),
        )

        state, _ = env.reset()

        # Take a step
        call = ToolCall(id="1", name="python", arguments={"code": "print(1+1)"})
        action = AgentAction(tool_calls=(call,))
        result1 = env.step(state, action)

        # Replay from same state
        result2 = env.step(state, action)

        # Results should match
        assert (
            result1.info["tool_results"][0].output
            == result2.info["tool_results"][0].output
        )


class TestGemToolEnvironmentWithSearchTool:
    """Tests for GEM tool environment with search tool."""

    @patch("llenvs.adapters.gem.GemToolEnvironment._create_gem_tools")
    def test_search_tool_available(self, mock_create_tools, mock_gem_env):
        """Test search tool is available when enabled."""
        mock_create_tools.return_value = {
            "python": MockPythonCodeTool(),
            "search": MockSearchTool(),
        }

        env = GemToolEnvironment(
            gem_env=mock_gem_env,
            env_id="qa:HotpotQA",
            tool_types=("python", "search"),
        )

        tool_names = [t.name for t in env.available_tools]
        assert "search" in tool_names
        assert "python" in tool_names
        assert "submit_answer" in tool_names

    @patch("llenvs.adapters.gem.GemToolEnvironment._create_gem_tools")
    def test_search_tool_execution(self, mock_create_tools, mock_gem_env):
        """Test search tool execution."""
        mock_create_tools.return_value = {
            "python": MockPythonCodeTool(),
            "search": MockSearchTool(),
        }

        env = GemToolEnvironment(
            gem_env=mock_gem_env,
            env_id="qa:HotpotQA",
            tool_types=("python", "search"),
        )

        state, _ = env.reset()

        call = ToolCall(
            id="1", name="search", arguments={"query": "capital of France"}
        )
        action = AgentAction(tool_calls=(call,))
        result = env.step(state, action)

        tool_results = result.info["tool_results"]
        assert len(tool_results) == 1
        assert tool_results[0].is_success
        assert "capital of France" in tool_results[0].output


class TestGemAdapterToolEnvironment:
    """Tests for GemAdapter.get_tool_environment method."""

    @patch("llenvs.adapters.gem.GemAdapter._get_gem")
    def test_get_tool_environment(self, mock_get_gem):
        """Test creating tool environment via adapter."""
        mock_gem = MagicMock()
        mock_gem.make.return_value = MockGemEnvWithTools()
        mock_get_gem.return_value = mock_gem

        from llenvs.adapters.gem import GemAdapter

        adapter = GemAdapter()

        with patch.object(
            GemToolEnvironment,
            "_create_gem_tools",
            return_value={"python": MockPythonCodeTool()},
        ):
            env = adapter.get_tool_environment(
                "math:GSM8K",
                tool_types=("python",),
                max_steps=5,
            )

        assert env.spec.name == "math:GSM8K"
        assert env.spec.max_steps == 5

    @patch("llenvs.adapters.gem.GemAdapter._get_gem")
    def test_create_gem_tool_environment_factory(self, mock_get_gem):
        """Test factory function for creating tool environments."""
        mock_gem = MagicMock()
        mock_gem.make.return_value = MockGemEnvWithTools()
        mock_get_gem.return_value = mock_gem

        from llenvs.adapters.gem import create_gem_tool_environment

        with patch.object(
            GemToolEnvironment,
            "_create_gem_tools",
            return_value={"python": MockPythonCodeTool()},
        ):
            env = create_gem_tool_environment(
                "math:GSM8K",
                tool_types=("python",),
                max_steps=10,
            )

        assert isinstance(env, GemToolEnvironment)
        assert env.spec.name == "math:GSM8K"
