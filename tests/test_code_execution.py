"""Tests for code execution reward."""

import textwrap

import pytest
from llenvs.core.code_execution import (
    CodeExecutionResult,
    CodeExecutionReward,
    SubprocessCodeExecutor,
)
from llenvs.core.reward import RewardType, Signal
from llenvs.core.state import Action, Observation, State, StateMetadata


# ---------------------------------------------------------------------------
# CodeExecutionResult
# ---------------------------------------------------------------------------


class TestCodeExecutionResult:
    def test_all_passed(self):
        result = CodeExecutionResult(passed=5, total=5)
        assert result.score == 1.0
        assert result.all_passed

    def test_partial(self):
        result = CodeExecutionResult(passed=3, total=5)
        assert result.score == pytest.approx(0.6)
        assert not result.all_passed

    def test_none_passed(self):
        result = CodeExecutionResult(passed=0, total=5)
        assert result.score == 0.0
        assert not result.all_passed

    def test_compilation_error(self):
        result = CodeExecutionResult(
            passed=0, total=0, compilation_error="SyntaxError: invalid syntax"
        )
        assert result.score == 0.0
        assert not result.all_passed

    def test_with_errors(self):
        result = CodeExecutionResult(
            passed=2,
            total=3,
            errors=(("test_edge_case", "AssertionError: 5 != 6"),),
        )
        assert result.score == pytest.approx(2 / 3)
        assert len(result.errors) == 1

    def test_zero_total(self):
        result = CodeExecutionResult(passed=0, total=0)
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# SubprocessCodeExecutor
# ---------------------------------------------------------------------------


class TestSubprocessCodeExecutor:
    def test_passing_code(self):
        executor = SubprocessCodeExecutor()
        code = "def add(a, b): return a + b"
        test_code = textwrap.dedent("""\
            assert add(1, 2) == 3
            assert add(0, 0) == 0
            assert add(-1, 1) == 0
        """)
        result = executor.execute(code, test_code)
        assert result.all_passed
        assert result.passed == 3
        assert result.total == 3

    def test_failing_test(self):
        executor = SubprocessCodeExecutor()
        code = "def add(a, b): return a - b"  # intentionally wrong
        test_code = textwrap.dedent("""\
            assert add(1, 2) == 3
            assert add(0, 0) == 0
        """)
        result = executor.execute(code, test_code)
        assert result.passed == 1  # add(0,0)==0 passes since 0-0=0
        assert result.total == 2
        assert not result.all_passed
        assert len(result.errors) == 1

    def test_syntax_error(self):
        executor = SubprocessCodeExecutor()
        code = "def add(a, b) return a + b"  # missing colon
        test_code = "assert add(1, 2) == 3"
        result = executor.execute(code, test_code)
        assert result.compilation_error is not None
        assert result.score == 0.0

    def test_runtime_error_in_code(self):
        executor = SubprocessCodeExecutor()
        code = "def divide(a, b): return a / b"
        test_code = textwrap.dedent("""\
            assert divide(4, 2) == 2.0
            assert divide(1, 0) == 0
        """)
        result = executor.execute(code, test_code)
        # First test passes, second raises ZeroDivisionError
        assert result.passed == 1
        assert result.total == 2

    def test_timeout(self):
        executor = SubprocessCodeExecutor()
        code = textwrap.dedent("""\
            import time
            def slow(): time.sleep(100)
        """)
        test_code = "slow()"
        result = executor.execute(code, test_code, timeout=1.0)
        assert result.compilation_error is not None
        assert "timeout" in result.compilation_error.lower() or "timed out" in result.compilation_error.lower()

    def test_multiline_test(self):
        executor = SubprocessCodeExecutor()
        code = "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)"
        test_code = textwrap.dedent("""\
            assert fib(0) == 0
            assert fib(1) == 1
            assert fib(5) == 5
            assert fib(10) == 55
        """)
        result = executor.execute(code, test_code)
        assert result.all_passed
        assert result.passed == 4

    def test_test_list_format(self):
        """Test with list-of-assertions format (MBPP style)."""
        executor = SubprocessCodeExecutor()
        code = "def double(x): return x * 2"
        test_code = "assert double(2) == 4\nassert double(0) == 0\nassert double(-1) == -2"
        result = executor.execute(code, test_code)
        assert result.all_passed
        assert result.passed == 3


# ---------------------------------------------------------------------------
# CodeExecutionReward
# ---------------------------------------------------------------------------


class TestCodeExecutionReward:
    def _make_state(self, hidden=None):
        """Create a minimal state for testing."""
        return State(
            observation=Observation(prompt="Write a function"),
            hidden=hidden,
            metadata=StateMetadata(step=0, episode_id="test"),
        )

    def test_correct_code(self):
        """Test reward for correct code with passing tests."""
        executor = SubprocessCodeExecutor()
        reward_fn = CodeExecutionReward(executor=executor)

        class Hidden:
            tests = "assert add(1, 2) == 3\nassert add(0, 0) == 0"

        state = self._make_state(Hidden())
        action = Action(text="```python\ndef add(a, b): return a + b\n```")
        signal = reward_fn.compute(state, action, state)

        assert signal.reward == 1.0
        assert signal.feedback is not None
        assert signal.name == "code_execution"
        assert signal.reward_type == RewardType.OUTCOME

    def test_wrong_code(self):
        """Test reward for incorrect code."""
        executor = SubprocessCodeExecutor()
        reward_fn = CodeExecutionReward(executor=executor)

        class Hidden:
            tests = "assert sub(5, 3) == 2\nassert sub(0, 0) == 0"

        state = self._make_state(Hidden())
        action = Action(text="```python\ndef sub(a, b): return a + b\n```")
        signal = reward_fn.compute(state, action, state)

        assert signal.reward is not None
        assert signal.reward < 1.0
        assert signal.feedback is not None

    def test_no_code_extracted(self):
        """Test reward when no code can be extracted."""
        executor = SubprocessCodeExecutor()
        reward_fn = CodeExecutionReward(executor=executor)

        class Hidden:
            tests = "assert True"

        state = self._make_state(Hidden())
        action = Action(text="I don't know how to solve this.")
        signal = reward_fn.compute(state, action, state)

        assert signal.reward == 0.0
        assert signal.feedback is not None

    def test_custom_test_extractor(self):
        """Test with custom test_extractor function."""
        executor = SubprocessCodeExecutor()

        def extract_tests(hidden):
            return hidden.my_tests

        reward_fn = CodeExecutionReward(
            executor=executor,
            test_extractor=extract_tests,
        )

        class Hidden:
            my_tests = "assert multiply(3, 4) == 12"

        state = self._make_state(Hidden())
        action = Action(text="```python\ndef multiply(a, b): return a * b\n```")
        signal = reward_fn.compute(state, action, state)

        assert signal.reward == 1.0

    def test_custom_name_and_type(self):
        """Test custom name and reward type."""
        executor = SubprocessCodeExecutor()
        reward_fn = CodeExecutionReward(
            executor=executor,
            name="code_test",
            reward_type=RewardType.STEP,
            weight=0.5,
        )

        assert reward_fn.name == "code_test"
        assert reward_fn.reward_type == RewardType.STEP

    def test_mbpp_style_hidden(self):
        """Test with MBPP-style hidden state (entry dict with test_list)."""
        executor = SubprocessCodeExecutor()
        reward_fn = CodeExecutionReward(executor=executor)

        class Hidden:
            entry = {
                "test_list": [
                    "assert double(2) == 4",
                    "assert double(0) == 0",
                ]
            }

        state = self._make_state(Hidden())
        action = Action(text="```python\ndef double(x): return x * 2\n```")
        signal = reward_fn.compute(state, action, state)

        assert signal.reward == 1.0

    def test_humaneval_style_hidden(self):
        """Test with HumanEval-style hidden state (entry dict with test + entry_point)."""
        executor = SubprocessCodeExecutor()
        reward_fn = CodeExecutionReward(executor=executor)

        class Hidden:
            entry = {
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n\ncheck(add)",
                "entry_point": "add",
            }

        state = self._make_state(Hidden())
        action = Action(text="```python\ndef add(a, b): return a + b\n```")
        signal = reward_fn.compute(state, action, state)

        assert signal.reward == 1.0
