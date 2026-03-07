"""Tests for second elicitation on MAX_TOKENS."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    SamplingParams,
    StopReason,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(
    sampling_params: SamplingParams | None = None,
    backend: Any = None,
) -> Any:
    """Create a TrajectoryRunner with a mock environment and backend."""
    from llenvs.core.environment import EnvironmentSpec
    from llenvs.evaluation.runner import TrajectoryRunner

    env = MagicMock()
    env.spec = EnvironmentSpec(name="test")

    if backend is None:
        backend = MagicMock()

    return TrajectoryRunner(
        environment=env,
        backend=backend,
        sampling_params=sampling_params or SamplingParams(),
    )


def _gen_result(
    text: str = "some output",
    finish_reason: StopReason = StopReason.MAX_TOKENS,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    metadata: dict[str, Any] | None = None,
) -> GenerationResult:
    return GenerationResult(
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfigFields:
    def test_inference_config_defaults(self) -> None:
        from llenvs.core.config import InferenceConfig

        cfg = InferenceConfig()
        assert cfg.second_elicitation_suffix is None
        assert cfg.second_elicitation_max_tokens == 256

    def test_create_sampling_params_disabled(self) -> None:
        from llenvs.core.config import InferenceConfig, create_sampling_params

        cfg = InferenceConfig()
        sp = create_sampling_params(cfg)
        assert sp.second_elicitation_suffix is None

    def test_create_sampling_params_enabled(self) -> None:
        from llenvs.core.config import InferenceConfig, create_sampling_params

        cfg = InferenceConfig(
            second_elicitation_suffix="Wrap up.",
            second_elicitation_max_tokens=128,
        )
        sp = create_sampling_params(cfg)
        assert sp.second_elicitation_suffix == "Wrap up."
        assert sp.second_elicitation_max_tokens == 128

    def test_create_sampling_params_preserves_existing_extra(self) -> None:
        from llenvs.core.config import InferenceConfig, create_sampling_params

        cfg = InferenceConfig(
            second_elicitation_suffix="Go.",
            extra={"some_key": 42},
        )
        sp = create_sampling_params(cfg)
        assert sp.extra["some_key"] == 42
        assert sp.second_elicitation_suffix == "Go."


# ---------------------------------------------------------------------------
# Runner unit tests
# ---------------------------------------------------------------------------


class TestSecondElicitation:
    def test_disabled_by_default(self) -> None:
        """No second call when feature is off."""
        runner = _make_runner()
        first = _gen_result(finish_reason=StopReason.MAX_TOKENS)

        runner.backend.generate_chat.return_value = first
        action, result = runner._generate_action(
            MagicMock(
                observation=MagicMock(available_tools=[]),
                metadata=MagicMock(is_terminal=False),
            )
        )
        # Only one call to generate_chat
        assert runner.backend.generate_chat.call_count == 1
        assert result is first

    def test_no_elicitation_on_stop_sequence(self) -> None:
        """Enabled but finish_reason=STOP_SEQUENCE → no second call."""
        params = SamplingParams(second_elicitation_suffix="wrap up")
        runner = _make_runner(sampling_params=params)
        first = _gen_result(finish_reason=StopReason.STOP_SEQUENCE)
        runner.backend.generate_chat.return_value = first

        action, result = runner._generate_action(
            MagicMock(
                observation=MagicMock(available_tools=[]),
                metadata=MagicMock(is_terminal=False),
            )
        )
        assert runner.backend.generate_chat.call_count == 1
        assert result is first

    def test_no_elicitation_on_end_of_text(self) -> None:
        """Enabled but finish_reason=END_OF_TEXT → no second call."""
        params = SamplingParams(second_elicitation_suffix="wrap up")
        runner = _make_runner(sampling_params=params)
        first = _gen_result(finish_reason=StopReason.END_OF_TEXT)
        runner.backend.generate_chat.return_value = first

        action, result = runner._generate_action(
            MagicMock(
                observation=MagicMock(available_tools=[]),
                metadata=MagicMock(is_terminal=False),
            )
        )
        assert runner.backend.generate_chat.call_count == 1

    def test_elicitation_on_max_tokens(self) -> None:
        """Enabled + MAX_TOKENS → second call, texts concatenated."""
        params = SamplingParams(
            second_elicitation_suffix="\nAnswer:",
            second_elicitation_max_tokens=128,
        )
        runner = _make_runner(sampling_params=params)

        first = _gen_result(text="partial output", prompt_tokens=100, completion_tokens=50)
        second = _gen_result(
            text="The answer is 42.",
            finish_reason=StopReason.END_OF_TEXT,
            prompt_tokens=200,
            completion_tokens=20,
        )
        runner.backend.generate_chat.side_effect = [first, second]

        action, result = runner._generate_action(
            MagicMock(
                observation=MagicMock(available_tools=[]),
                metadata=MagicMock(is_terminal=False),
            )
        )
        assert runner.backend.generate_chat.call_count == 2
        assert "partial output" in result.text
        assert "The answer is 42." in result.text
        assert result.finish_reason == StopReason.END_OF_TEXT
        assert result.prompt_tokens == 300
        assert result.completion_tokens == 70

    def test_custom_suffix(self) -> None:
        """Custom suffix used in continuation messages."""
        custom_suffix = "\n\nPlease answer now.\n"
        params = SamplingParams(second_elicitation_suffix=custom_suffix)
        runner = _make_runner(sampling_params=params)

        first = _gen_result(text="thinking...")
        second = _gen_result(text="42", finish_reason=StopReason.END_OF_TEXT)
        runner.backend.generate_chat.side_effect = [first, second]

        action, result = runner._generate_action(
            MagicMock(
                observation=MagicMock(available_tools=[]),
                metadata=MagicMock(is_terminal=False),
            )
        )
        # Check the continuation messages sent to the second call
        second_call_msgs = runner.backend.generate_chat.call_args_list[1][0][0]
        assistant_msg = second_call_msgs[-2]
        assert assistant_msg.role == "assistant"
        assert assistant_msg.content == "thinking..." + custom_suffix

        user_msg = second_call_msgs[-1]
        assert user_msg.role == "user"
        assert user_msg.content == "Please provide your final answer."

        # Check merged text
        assert result.text == "thinking..." + custom_suffix + "42"

    def test_custom_max_tokens(self) -> None:
        """Second call uses configured budget."""
        params = SamplingParams(
            max_tokens=4096,
            second_elicitation_suffix="wrap up",
            second_elicitation_max_tokens=64,
        )
        runner = _make_runner(sampling_params=params)

        first = _gen_result(text="partial")
        second = _gen_result(text="done", finish_reason=StopReason.END_OF_TEXT)
        runner.backend.generate_chat.side_effect = [first, second]

        action, result = runner._generate_action(
            MagicMock(
                observation=MagicMock(available_tools=[]),
                metadata=MagicMock(is_terminal=False),
            )
        )
        # Check that the second call used max_tokens=64
        second_call_params = runner.backend.generate_chat.call_args_list[1][0][1]
        assert second_call_params.max_tokens == 64
        # And that second_elicitation is disabled (no recursion)
        assert second_call_params.second_elicitation_suffix is None

    def test_merged_metadata(self) -> None:
        """Combined result has second_elicitation: True in metadata."""
        params = SamplingParams(second_elicitation_suffix="wrap up")
        runner = _make_runner(sampling_params=params)

        first = _gen_result(text="a", metadata={"model": "test"})
        second = _gen_result(
            text="b",
            finish_reason=StopReason.END_OF_TEXT,
            metadata={"latency": 0.5},
        )
        runner.backend.generate_chat.side_effect = [first, second]

        action, result = runner._generate_action(
            MagicMock(
                observation=MagicMock(available_tools=[]),
                metadata=MagicMock(is_terminal=False),
            )
        )
        assert result.metadata["second_elicitation"] is True
        assert result.metadata["model"] == "test"
        assert result.metadata["latency"] == 0.5


# ---------------------------------------------------------------------------
# Batch tests
# ---------------------------------------------------------------------------


class TestBatchSecondElicitation:
    def test_batch_mixed(self) -> None:
        """Batch with some MAX_TOKENS and some STOP → only truncated ones get second call."""
        params = SamplingParams(
            second_elicitation_suffix="wrap up",
            second_elicitation_max_tokens=64,
        )
        runner = _make_runner(sampling_params=params)

        # Simulate three results: first truncated, second complete, third truncated
        gen_results = [
            _gen_result(text="truncated1", finish_reason=StopReason.MAX_TOKENS),
            _gen_result(text="complete", finish_reason=StopReason.END_OF_TEXT),
            _gen_result(text="truncated2", finish_reason=StopReason.MAX_TOKENS),
        ]

        suffix = runner._resolve_elicitation_suffix()

        # Elicitation results for the two truncated ones
        elicitation_results = [
            _gen_result(text="answer1", finish_reason=StopReason.END_OF_TEXT),
            _gen_result(text="answer2", finish_reason=StopReason.END_OF_TEXT),
        ]

        # Test the helpers directly
        messages_batch = [
            [ChatMessage(role="user", content="q1")],
            [ChatMessage(role="user", content="q2")],
            [ChatMessage(role="user", content="q3")],
        ]

        needs_elicitation = [
            (i, gen)
            for i, gen in enumerate(gen_results)
            if gen.finish_reason == StopReason.MAX_TOKENS
        ]
        assert len(needs_elicitation) == 2
        assert needs_elicitation[0][0] == 0
        assert needs_elicitation[1][0] == 2

        elicitation_msgs = [
            runner._build_elicitation_messages(messages_batch[i], gen, suffix)
            for i, gen in needs_elicitation
        ]
        assert len(elicitation_msgs) == 2

        # Check that elicitation messages have the right structure
        for msgs in elicitation_msgs:
            assert msgs[-2].role == "assistant"
            assert msgs[-1].role == "user"
            assert msgs[-1].content == "Please provide your final answer."

        # Merge
        for (i, first_gen), second_gen in zip(needs_elicitation, elicitation_results):
            gen_results[i] = runner._merge_elicitation(first_gen, second_gen, suffix)

        # Verify results
        assert "truncated1" in gen_results[0].text
        assert "answer1" in gen_results[0].text
        assert gen_results[0].metadata["second_elicitation"] is True

        # Complete one unchanged
        assert gen_results[1].text == "complete"
        assert "second_elicitation" not in gen_results[1].metadata

        assert "truncated2" in gen_results[2].text
        assert "answer2" in gen_results[2].text
        assert gen_results[2].metadata["second_elicitation"] is True

    def test_batch_no_truncation(self) -> None:
        """Batch where nothing is truncated → no elicitation calls."""
        params = SamplingParams(second_elicitation_suffix="wrap up")
        _make_runner(sampling_params=params)

        gen_results = [
            _gen_result(text="ok1", finish_reason=StopReason.END_OF_TEXT),
            _gen_result(text="ok2", finish_reason=StopReason.STOP_SEQUENCE),
        ]

        needs = [
            (i, g) for i, g in enumerate(gen_results) if g.finish_reason == StopReason.MAX_TOKENS
        ]
        assert len(needs) == 0

    def test_elicitation_params_disable_recursion(self) -> None:
        """Elicitation params disable second_elicitation to prevent recursion."""
        params = SamplingParams(
            max_tokens=4096,
            temperature=0.5,
            second_elicitation_suffix="wrap up",
            second_elicitation_max_tokens=128,
            thinking_budget=512,
        )
        runner = _make_runner(sampling_params=params)
        ep = runner._elicitation_params()
        assert ep.max_tokens == 128
        assert ep.temperature == 0.5
        assert ep.second_elicitation_suffix is None
        assert ep.thinking_budget == 512
