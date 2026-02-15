"""Code execution and test-based reward computation.

Provides a ``CodeExecutor`` protocol and ``SubprocessCodeExecutor`` for
running code + tests locally. ``CodeExecutionReward`` wraps an executor
as a ``RewardFunction`` that produces ``Signal`` with both numeric score
and textual feedback.

For sandboxed execution, wrap the inner environment in a
``ContainerEnvironment`` — the subprocess then runs inside the container.
"""

from __future__ import annotations

import subprocess
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from llenvs.core.extraction import CodeBlockExtractor
from llenvs.core.reward import RewardType, Signal
from llenvs.core.state import State


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeExecutionResult:
    """Result of executing code against a test suite.

    Attributes:
        passed: Number of tests that passed.
        total: Total number of tests.
        errors: Tuples of (test_name, error_message) for failed tests.
        compilation_error: Error message if code failed to compile/load.
        stdout: Captured standard output.
        stderr: Captured standard error.
    """

    passed: int
    total: int
    errors: tuple[tuple[str, str], ...] = ()
    compilation_error: str | None = None
    stdout: str = ""
    stderr: str = ""

    @property
    def score(self) -> float:
        """Fraction of tests passed (0.0 on compilation error or zero total)."""
        if self.compilation_error is not None or self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def all_passed(self) -> bool:
        """Whether every test passed with no errors."""
        return (
            self.compilation_error is None
            and self.total > 0
            and self.passed == self.total
        )


# ---------------------------------------------------------------------------
# Executor protocol
# ---------------------------------------------------------------------------


class CodeExecutor(Protocol):
    """Protocol for executing code against tests."""

    def execute(
        self, code: str, test_code: str, *, timeout: float = 30.0
    ) -> CodeExecutionResult: ...


# ---------------------------------------------------------------------------
# Subprocess executor
# ---------------------------------------------------------------------------

_RUNNER_TEMPLATE = textwrap.dedent("""\
import json, sys, traceback

# --- Load user code ---
try:
    exec(compile({code!r}, "<user_code>", "exec"), globals())
except Exception as e:
    print(json.dumps({{"compilation_error": str(e)}}))
    sys.exit(0)

# --- Run tests ---
tests = {tests!r}
results = []
for i, test in enumerate(tests):
    name = f"test_{{i}}"
    try:
        exec(compile(test.strip(), f"<test_{{i}}>", "exec"), globals())
        results.append({{"name": name, "passed": True}})
    except Exception as e:
        results.append({{"name": name, "passed": False, "error": str(e)}})

print(json.dumps({{"results": results}}))
""")


class SubprocessCodeExecutor:
    """Execute code + tests in a subprocess with timeout."""

    def execute(
        self, code: str, test_code: str, *, timeout: float = 30.0
    ) -> CodeExecutionResult:
        """Run *code* then *test_code* in a fresh Python subprocess.

        Each assertion line in *test_code* is run independently so partial
        results are captured even when some tests fail.

        Args:
            code: The code to test (function definitions, etc.).
            test_code: Test assertions, one per line or as a block.
            timeout: Maximum execution time in seconds.

        Returns:
            CodeExecutionResult with per-test pass/fail information.
        """
        tests = _split_tests(test_code)
        script = _RUNNER_TEMPLATE.format(code=code, tests=tests)

        try:
            proc = subprocess.run(
                ["python3", "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return CodeExecutionResult(
                passed=0,
                total=len(tests),
                compilation_error=f"Timed out after {timeout}s",
            )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if not stdout:
            return CodeExecutionResult(
                passed=0,
                total=len(tests),
                compilation_error=stderr or "No output from subprocess",
                stderr=stderr,
            )

        import json

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return CodeExecutionResult(
                passed=0,
                total=len(tests),
                compilation_error=f"Invalid JSON output: {stdout[:200]}",
                stdout=stdout,
                stderr=stderr,
            )

        if "compilation_error" in data:
            return CodeExecutionResult(
                passed=0,
                total=len(tests),
                compilation_error=data["compilation_error"],
                stdout=stdout,
                stderr=stderr,
            )

        results = data.get("results", [])
        passed = sum(1 for r in results if r.get("passed"))
        errors = tuple(
            (r["name"], r.get("error", ""))
            for r in results
            if not r.get("passed")
        )

        return CodeExecutionResult(
            passed=passed,
            total=len(results),
            errors=errors,
            stdout=stdout,
            stderr=stderr,
        )


def _split_tests(test_code: str) -> list[str]:
    """Split test code into individual test cases.

    Handles both simple assertion lists and structured test functions.
    """
    lines = test_code.strip().splitlines()
    if not lines:
        return []

    # If it looks like individual assert statements, split them
    if all(
        line.strip().startswith("assert ") or not line.strip()
        for line in lines
        if line.strip()
    ):
        return [line.strip() for line in lines if line.strip()]

    # Otherwise treat the whole block as a single test
    return [test_code.strip()]


# ---------------------------------------------------------------------------
# Default test extractor
# ---------------------------------------------------------------------------


def _default_test_extractor(hidden: Any) -> str | None:
    """Extract test code from hidden state via duck-typing.

    Supports common HF dataset formats:
    - ``hidden.entry["test_list"]`` — MBPP (list of assertion strings)
    - ``hidden.entry["test"]`` + ``hidden.entry["entry_point"]`` — HumanEval
    - ``hidden.tests`` — generic attribute
    - ``hidden.entry.get("tests", "")`` — fallback
    """
    # Try entry dict first (HF datasets)
    entry = getattr(hidden, "entry", None)
    if isinstance(entry, dict):
        # MBPP style: list of assertion strings
        test_list = entry.get("test_list")
        if test_list and isinstance(test_list, list):
            return "\n".join(test_list)

        # HumanEval style: test function + entry_point
        test_fn = entry.get("test")
        if test_fn:
            return str(test_fn)

        # Generic fallback
        tests = entry.get("tests", "")
        if tests:
            return str(tests)

    # Try direct attribute
    tests_attr = getattr(hidden, "tests", None)
    if tests_attr is not None:
        return str(tests_attr)

    return None


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def _format_feedback(result: CodeExecutionResult) -> str:
    """Format execution result as human-readable feedback."""
    if result.compilation_error:
        return f"Code failed to compile/run: {result.compilation_error}"

    parts = [f"{result.passed}/{result.total} tests passed."]

    if result.errors:
        parts.append("Failures:")
        for name, error in result.errors:
            parts.append(f"  - {name}: {error}")

    return "\n".join(parts)


@dataclass
class CodeExecutionReward:
    """Reward function that runs code against tests.

    Extracts code from the model response, runs it against test cases
    from the hidden state, and returns a Signal with both a numeric
    score (fraction of tests passed) and textual feedback.

    Implements the ``RewardFunction`` protocol.
    """

    _name: str = "code_execution"
    _reward_type: RewardType = RewardType.OUTCOME

    def __init__(
        self,
        executor: CodeExecutor,
        code_extractor: Any | None = None,
        test_extractor: Callable[[Any], str | None] | None = None,
        name: str = "code_execution",
        reward_type: RewardType = RewardType.OUTCOME,
        weight: float = 1.0,
    ) -> None:
        self._executor = executor
        self._code_extractor = code_extractor or CodeBlockExtractor(language="python")
        self._test_extractor = test_extractor or _default_test_extractor
        self._name = name
        self._reward_type = reward_type
        self._weight = weight

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward_type(self) -> RewardType:
        return self._reward_type

    def compute(
        self,
        state: State[Any],
        action: Any,
        next_state: State[Any],
    ) -> Signal:
        """Extract code, run tests, return scored signal with feedback."""
        # Extract code from response
        code, _ = self._code_extractor.extract(action.text or "")
        if code is None:
            return Signal(
                name=self._name,
                reward_type=self._reward_type,
                reward=0.0,
                feedback="No code block found in response.",
                weight=self._weight,
                metadata={"extraction_failed": True},
            )

        # Extract test code from hidden state
        test_code = self._test_extractor(state.hidden)
        if not test_code:
            return Signal(
                name=self._name,
                reward_type=self._reward_type,
                reward=0.0,
                feedback="No test code available for this task.",
                weight=self._weight,
                metadata={"no_tests": True},
            )

        # Execute
        result = self._executor.execute(code, test_code)
        feedback = _format_feedback(result)

        return Signal(
            name=self._name,
            reward_type=self._reward_type,
            reward=result.score,
            feedback=feedback,
            weight=self._weight,
            metadata={
                "passed": result.passed,
                "total": result.total,
                "errors": result.errors,
                "compilation_error": result.compilation_error,
            },
        )
