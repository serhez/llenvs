"""Tests for batched generation and concurrent inference."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from llenvs.core.environment import EnvironmentSpec, StepResult
from llenvs.core.reward import SignalBundle
from llenvs.core.segmentation import LineSegmenter, TokenSegmenter
from llenvs.core.segmented_environment import SegmentedEnvironment
from llenvs.core.state import (
    Observation,
    State,
    StateMetadata,
)
from llenvs.core.tools import ToolDefinition
from llenvs.evaluation.continuation import (
    BoundaryContinuationStrategy,
    SegmentContext,
    TokenContinuationStrategy,
)
from llenvs.evaluation.runner import (
    COMPLETE,
    ForceAction,
    MultiEvalEntry,
    SegmentedTrajectoryRunner,
    TrajectoryRunner,
    run_multi_evaluation,
)
from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    ModelBackend,
    SamplingParams,
    StopReason,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(text: str) -> GenerationResult:
    """Create a simple GenerationResult."""
    return GenerationResult(
        text=text,
        finish_reason=StopReason.END_OF_TEXT,
        prompt_tokens=10,
        completion_tokens=5,
    )


def _make_messages(content: str) -> list[ChatMessage]:
    """Create a single-user-message conversation."""
    return [ChatMessage(role="user", content=content)]


class RecordingBackend(ModelBackend):
    """Backend that records calls and returns canned responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or []
        self._call_index = 0
        self.generate_calls: list[list[str]] = []
        self.generate_chat_calls: list[list[ChatMessage]] = []

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(supports_chat=True)

    @property
    def model_name(self) -> str:
        return "recording-backend"

    def generate(self, prompts: list[str], params: SamplingParams) -> list[GenerationResult]:
        self.generate_calls.append(prompts)
        return [self._next_result() for _ in prompts]

    def generate_chat(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> GenerationResult:
        self.generate_chat_calls.append(messages)
        return self._next_result()

    def _next_result(self) -> GenerationResult:
        if self._call_index < len(self._responses):
            text = self._responses[self._call_index]
        else:
            text = f"response_{self._call_index}"
        self._call_index += 1
        return _make_result(text)


# ===========================================================================
# Phase 1a: Protocol — default generate_chat_batch
# ===========================================================================


class TestDefaultGenerateChatBatch:
    """Test that ModelBackend.generate_chat_batch default impl works."""

    def test_default_calls_generate_chat_sequentially(self):
        """Default generate_chat_batch loops over generate_chat."""
        backend = RecordingBackend(["a", "b", "c"])
        messages_batch = [
            _make_messages("q1"),
            _make_messages("q2"),
            _make_messages("q3"),
        ]
        params = SamplingParams()

        results = backend.generate_chat_batch(messages_batch, params)

        assert len(results) == 3
        assert results[0].text == "a"
        assert results[1].text == "b"
        assert results[2].text == "c"
        # Should have made 3 individual generate_chat calls
        assert len(backend.generate_chat_calls) == 3

    def test_empty_batch(self):
        """Empty batch returns empty list."""
        backend = RecordingBackend()
        results = backend.generate_chat_batch([], SamplingParams())
        assert results == []

    def test_single_item_batch(self):
        """Single-item batch works."""
        backend = RecordingBackend(["only"])
        results = backend.generate_chat_batch([_make_messages("q")], SamplingParams())
        assert len(results) == 1
        assert results[0].text == "only"


class TestDefaultGenerateWithToolsBatch:
    """Test that ModelBackend.generate_with_tools_batch default impl works."""

    def test_default_calls_generate_with_tools_sequentially(self):
        """Default loops over generate_with_tools."""
        backend = RecordingBackend(["a", "b"])

        # Mock generate_with_tools on the instance
        call_count = 0
        original_results = [_make_result("tool_a"), _make_result("tool_b")]

        def mock_generate_with_tools(messages, tools, params, tool_choice="auto"):
            nonlocal call_count
            result = original_results[call_count]
            call_count += 1
            return result

        backend.generate_with_tools = mock_generate_with_tools

        messages_batch = [_make_messages("q1"), _make_messages("q2")]
        results = backend.generate_with_tools_batch(
            messages_batch, tools=[], params=SamplingParams()
        )

        assert len(results) == 2
        assert results[0].text == "tool_a"
        assert results[1].text == "tool_b"
        assert call_count == 2


# ===========================================================================
# Phase 1b/1c: vLLM and HuggingFace batch overrides
# ===========================================================================


class TestVLLMBatchChat:
    """Test vLLM generate_chat_batch routes through generate()."""

    def test_batch_routes_through_generate(self):
        """generate_chat_batch should convert all messages to prompts and call generate() once."""
        from llenvs.inference.backends.vllm import VLLMBackend

        # Create a mock that avoids actual vLLM initialization
        backend = object.__new__(VLLMBackend)

        # Mock tokenizer with chat template
        mock_tokenizer = MagicMock()
        prompts_seen = []

        def fake_apply_template(messages, tokenize=False, add_generation_prompt=True, **kwargs):
            prompt = f"formatted:{messages[0]['content']}"
            prompts_seen.append(prompt)
            return prompt

        mock_tokenizer.apply_chat_template = fake_apply_template
        backend._tokenizer = mock_tokenizer
        backend._chat_template_kwargs = {}
        backend._is_vlm = False

        # Mock generate to return results
        def fake_generate(prompts, params):
            return [_make_result(f"reply_to_{p}") for p in prompts]

        backend.generate = fake_generate

        messages_batch = [
            _make_messages("hello"),
            _make_messages("world"),
            _make_messages("test"),
        ]
        params = SamplingParams()

        results = backend.generate_chat_batch(messages_batch, params)

        # Should have called apply_chat_template 3 times
        assert len(prompts_seen) == 3
        # Should return 3 results
        assert len(results) == 3
        assert results[0].text == "reply_to_formatted:hello"
        assert results[1].text == "reply_to_formatted:world"


class TestHuggingFaceBatchChat:
    """Test HuggingFace generate_chat_batch routes through generate()."""

    def test_batch_routes_through_generate(self):
        """generate_chat_batch should convert messages to prompts and call generate() once."""
        from llenvs.inference.backends.huggingface import HuggingFaceBackend

        backend = object.__new__(HuggingFaceBackend)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = "some_template"
        prompts_seen = []

        def fake_apply_template(messages, tokenize=False, add_generation_prompt=True, **kwargs):
            prompt = f"hf:{messages[0]['content']}"
            prompts_seen.append(prompt)
            return prompt

        mock_tokenizer.apply_chat_template = fake_apply_template
        backend._tokenizer = mock_tokenizer
        backend._chat_template_kwargs = {}

        # Mock generate
        def fake_generate(prompts, params):
            return [_make_result(f"hf_reply_{i}") for i in range(len(prompts))]

        backend.generate = fake_generate

        messages_batch = [_make_messages("a"), _make_messages("b")]
        results = backend.generate_chat_batch(messages_batch, SamplingParams())

        assert len(prompts_seen) == 2
        assert len(results) == 2

    def test_batch_fallback_without_chat_template(self):
        """When no chat template, uses fallback formatter."""
        from llenvs.inference.backends.huggingface import HuggingFaceBackend

        backend = object.__new__(HuggingFaceBackend)

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None  # No chat template
        backend._tokenizer = mock_tokenizer

        def fake_generate(prompts, params):
            return [_make_result(f"reply_{i}") for i in range(len(prompts))]

        backend.generate = fake_generate

        messages_batch = [_make_messages("q1"), _make_messages("q2")]
        results = backend.generate_chat_batch(messages_batch, SamplingParams())

        assert len(results) == 2


# ===========================================================================
# Phase 1d: API backend async concurrency
# ===========================================================================


class TestOpenAIBatchChat:
    """Test OpenAI generate_chat_batch uses async concurrency."""

    def test_batch_returns_correct_results(self):
        """Batch should return one result per conversation."""
        from llenvs.inference.backends.api import OpenAIBackend

        backend = object.__new__(OpenAIBackend)
        backend._model = "gpt-4o"
        backend._max_concurrency = 10

        # Track calls via async client mock
        call_count = 0

        def _make_openai_response(text: str) -> MagicMock:
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = text
            resp.choices[0].finish_reason = "stop"
            resp.choices[0].logprobs = None
            resp.usage = MagicMock()
            resp.usage.prompt_tokens = 5
            resp.usage.completion_tokens = 3
            resp.model = "gpt-4o"
            resp.id = "test-id"
            return resp

        async def fake_create(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            return _make_openai_response(f"openai_{idx}")

        mock_async_client = MagicMock()
        mock_async_client.chat.completions.create = fake_create
        backend._async_client = mock_async_client

        messages_batch = [_make_messages(f"q{i}") for i in range(5)]
        results = backend.generate_chat_batch(messages_batch, SamplingParams())

        assert len(results) == 5
        assert call_count == 5

    def test_empty_batch(self):
        """Empty batch returns empty list."""
        from llenvs.inference.backends.api import OpenAIBackend

        backend = object.__new__(OpenAIBackend)
        backend._model = "gpt-4o"
        backend._max_concurrency = 10

        results = backend.generate_chat_batch([], SamplingParams())
        assert results == []

    def test_concurrency_with_async_client(self):
        """Verify that async client is used for concurrent generation."""
        from llenvs.inference.backends.api import OpenAIBackend

        backend = object.__new__(OpenAIBackend)
        backend._model = "gpt-4o"
        backend._max_concurrency = 2

        # Mock the async client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "async_result"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].logprobs = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.model = "gpt-4o"
        mock_response.id = "test-id"

        mock_async_client = MagicMock()
        mock_async_client.chat = MagicMock()
        mock_async_client.chat.completions = MagicMock()
        mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        backend._async_client = mock_async_client

        messages_batch = [_make_messages("q1"), _make_messages("q2"), _make_messages("q3")]
        results = backend.generate_chat_batch(messages_batch, SamplingParams())

        assert len(results) == 3
        # Async client should have been called 3 times
        assert mock_async_client.chat.completions.create.await_count == 3


class TestAnthropicBatchChat:
    """Test Anthropic generate_chat_batch uses async concurrency."""

    def test_batch_returns_correct_results(self):
        """Batch should return one result per conversation."""
        from llenvs.inference.backends.api import AnthropicBackend

        backend = object.__new__(AnthropicBackend)
        backend._model = "claude-sonnet-4-20250514"
        backend._max_concurrency = 10

        call_count = 0

        def _make_anthropic_response(text: str) -> MagicMock:
            block = MagicMock()
            block.type = "text"
            block.text = text
            resp = MagicMock()
            resp.content = [block]
            resp.stop_reason = "end_turn"
            resp.usage = MagicMock()
            resp.usage.input_tokens = 5
            resp.usage.output_tokens = 3
            resp.model = "claude-sonnet-4-20250514"
            resp.id = "test-id"
            return resp

        async def fake_create(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            return _make_anthropic_response(f"anthropic_{idx}")

        mock_async_client = MagicMock()
        mock_async_client.messages.create = fake_create
        backend._async_client = mock_async_client

        messages_batch = [_make_messages(f"q{i}") for i in range(4)]
        results = backend.generate_chat_batch(messages_batch, SamplingParams())

        assert len(results) == 4
        assert call_count == 4

    def test_concurrency_with_async_client(self):
        """Verify that async client is used for concurrent generation."""
        from llenvs.inference.backends.api import AnthropicBackend

        backend = object.__new__(AnthropicBackend)
        backend._model = "claude-sonnet-4-20250514"
        backend._max_concurrency = 2

        # Mock Anthropic async response
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "async_anthropic"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 3
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.id = "test-id"

        mock_async_client = MagicMock()
        mock_async_client.messages = MagicMock()
        mock_async_client.messages.create = AsyncMock(return_value=mock_response)
        backend._async_client = mock_async_client

        messages_batch = [_make_messages("q1"), _make_messages("q2")]
        results = backend.generate_chat_batch(messages_batch, SamplingParams())

        assert len(results) == 2
        assert mock_async_client.messages.create.await_count == 2


class TestOpenRouterBatchChat:
    """Test OpenRouter generate_chat_batch uses async concurrency."""

    def test_batch_returns_correct_results(self):
        """Batch should return one result per conversation."""
        from llenvs.inference.backends.api import OpenRouterBackend

        backend = object.__new__(OpenRouterBackend)
        backend._model = "anthropic/claude-sonnet-4-20250514"
        backend._max_concurrency = 10

        call_count = 0

        def _make_openai_response(text: str) -> MagicMock:
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = text
            resp.choices[0].finish_reason = "stop"
            resp.choices[0].logprobs = None
            resp.usage = MagicMock()
            resp.usage.prompt_tokens = 5
            resp.usage.completion_tokens = 3
            resp.model = "anthropic/claude-sonnet-4-20250514"
            resp.id = "test-id"
            return resp

        async def fake_create(**kwargs):
            nonlocal call_count
            idx = call_count
            call_count += 1
            return _make_openai_response(f"openrouter_{idx}")

        mock_async_client = MagicMock()
        mock_async_client.chat.completions.create = fake_create
        backend._async_client = mock_async_client

        messages_batch = [_make_messages(f"q{i}") for i in range(3)]
        results = backend.generate_chat_batch(messages_batch, SamplingParams())

        assert len(results) == 3
        assert call_count == 3


class TestSemaphoreConcurrency:
    """Test that max_concurrency is respected via semaphore."""

    def test_max_concurrency_limits_parallel_calls(self):
        """At most max_concurrency calls should be in-flight simultaneously."""
        from llenvs.inference.backends.api import OpenAIBackend

        backend = object.__new__(OpenAIBackend)
        backend._model = "gpt-4o"
        backend._max_concurrency = 2

        # Track concurrent calls
        max_concurrent = 0
        current_concurrent = 0

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].logprobs = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.model = "gpt-4o"
        mock_response.id = "test-id"

        async def tracked_create(**kwargs):
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)  # Simulate network delay
            current_concurrent -= 1
            return mock_response

        mock_async_client = MagicMock()
        mock_async_client.chat = MagicMock()
        mock_async_client.chat.completions = MagicMock()
        mock_async_client.chat.completions.create = tracked_create
        backend._async_client = mock_async_client

        messages_batch = [_make_messages(f"q{i}") for i in range(6)]
        results = backend.generate_chat_batch(messages_batch, SamplingParams())

        assert len(results) == 6
        assert max_concurrent <= 2, f"Max concurrent was {max_concurrent}, expected <= 2"


# ===========================================================================
# Phase 1e: BackendCapabilities
# ===========================================================================


class TestBackendCapabilities:
    """Test updated backend capabilities."""

    def test_max_concurrency_field(self):
        """BackendCapabilities should have max_concurrency field."""
        caps = BackendCapabilities(max_concurrency=32)
        assert caps.max_concurrency == 32

    def test_max_concurrency_default_none(self):
        """Default max_concurrency should be None."""
        caps = BackendCapabilities()
        assert caps.max_concurrency is None


# ===========================================================================
# Phase 2: Lockstep runner batching
# ===========================================================================


# --- Mock environments for runner tests ---


class MockSingleTurnEnv:
    """Single-step environment: every task completes in one step."""

    def __init__(self, num_tasks: int = 5):
        self._num_tasks = num_tasks

    @property
    def spec(self):
        return EnvironmentSpec(name="mock_single", max_steps=1)

    @property
    def reward_functions(self):
        return ()

    @property
    def prompts(self):
        return {}

    def __len__(self):
        return self._num_tasks

    def reset(self, *, seed=None, options=None):
        idx = (options or {}).get("task_index", 0)
        return State(
            observation=Observation(prompt=f"Question {idx}?"),
            hidden={"answer": str(idx)},
            metadata=StateMetadata(step=0, episode_id=f"ep_{idx}"),
        ), {"task_index": idx}

    def step(self, state, action):
        return StepResult(
            next_state=state.with_metadata(step=1, is_terminal=True),
            rewards=SignalBundle.single(reward=1.0, name="correctness"),
            terminated=True,
        )

    def compute_rewards(self, state, action, next_state):
        return SignalBundle.single(reward=1.0, name="correctness")


class MockMultiTurnEnv:
    """Multi-turn environment: tasks take variable numbers of steps."""

    def __init__(self, steps_per_task: dict[int, int]):
        self._steps_per_task = steps_per_task

    @property
    def spec(self):
        return EnvironmentSpec(name="mock_multi", max_steps=100)

    @property
    def reward_functions(self):
        return ()

    @property
    def prompts(self):
        return {}

    def __len__(self):
        return max(self._steps_per_task.keys()) + 1

    def reset(self, *, seed=None, options=None):
        idx = (options or {}).get("task_index", 0)
        return State(
            observation=Observation(prompt=f"Q{idx}"),
            hidden={"task_index": idx, "target_steps": self._steps_per_task.get(idx, 1)},
            metadata=StateMetadata(step=0, episode_id=f"ep_{idx}"),
        ), {"task_index": idx}

    def step(self, state, action):
        next_step = state.metadata.step + 1
        target = state.hidden["target_steps"]
        done = next_step >= target
        return StepResult(
            next_state=state.with_metadata(step=next_step, is_terminal=done),
            rewards=SignalBundle.single(reward=1.0 if done else 0.0, name="correctness"),
            terminated=done,
        )

    def compute_rewards(self, state, action, next_state):
        return SignalBundle.empty()


class MockNonPureSingleTurnEnv(MockSingleTurnEnv):
    """Non-pure variant that enforces episode consistency."""

    def __init__(self, num_tasks: int = 5):
        super().__init__(num_tasks=num_tasks)
        self._current_episode_id: str | None = None

    @property
    def spec(self):
        return EnvironmentSpec(name="mock_non_pure_single", max_steps=1, pure_step=False)

    def reset(self, *, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)
        self._current_episode_id = state.metadata.episode_id
        return state, info

    def step(self, state, action):
        if state.metadata.episode_id != self._current_episode_id:
            raise RuntimeError("episode mismatch")
        result = super().step(state, action)
        self._current_episode_id = result.next_state.metadata.episode_id
        return result


class BatchTrackingBackend(ModelBackend):
    """Backend that tracks batch vs individual generation calls."""

    def __init__(self) -> None:
        self._call_index = 0
        self.batch_call_sizes: list[int] = []
        self.individual_call_count: int = 0

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(supports_chat=True, supports_batching=True)

    @property
    def model_name(self) -> str:
        return "batch-tracking"

    def generate(self, prompts: list[str], params: SamplingParams) -> list[GenerationResult]:
        return [self._next_result() for _ in prompts]

    def generate_chat(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> GenerationResult:
        self.individual_call_count += 1
        return self._next_result()

    def generate_chat_batch(
        self, messages_batch: list[list[ChatMessage]], params: SamplingParams
    ) -> list[GenerationResult]:
        self.batch_call_sizes.append(len(messages_batch))
        return [self._next_result() for _ in messages_batch]

    def _next_result(self) -> GenerationResult:
        text = f"response_{self._call_index}"
        self._call_index += 1
        return _make_result(text)


# --- TrajectoryRunner batch tests ---


class TestTrajectoryRunnerBatch:
    """Phase 2: TrajectoryRunner.run_batch() lockstep batching."""

    def test_single_turn_uses_batch_generation(self):
        """Single-turn tasks should use one batch call for all trajectories."""
        env = MockSingleTurnEnv(num_tasks=5)
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2, 3, 4])

        assert len(result.trajectory_results) == 5
        assert backend.individual_call_count == 0
        assert backend.batch_call_sizes == [5]

    def test_non_pure_env_factory_allows_parallel_batch(self):
        """Non-pure envs should run in parallel when env_factory is provided."""
        base_env = MockNonPureSingleTurnEnv(num_tasks=3)
        backend = BatchTrackingBackend()

        def env_factory():
            return MockNonPureSingleTurnEnv(num_tasks=3)

        runner = TrajectoryRunner(
            environment=base_env,
            backend=backend,
            env_factory=env_factory,
        )

        result = runner.run_batch([0, 1, 2])

        assert len(result.trajectory_results) == 3
        assert backend.batch_call_sizes == [3]

    def test_multi_turn_decreasing_batch_sizes(self):
        """Batch sizes decrease as trajectories finish."""
        # Task 0: 1 step, task 1: 2 steps, task 2: 3 steps
        env = MockMultiTurnEnv(steps_per_task={0: 1, 1: 2, 2: 3})
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2])

        assert len(result.trajectory_results) == 3
        # Step 1: 3 active, step 2: 2 active (task 0 done), step 3: 1 active
        assert backend.batch_call_sizes == [3, 2, 1]
        assert backend.individual_call_count == 0

    def test_preserves_result_order(self):
        """Results correspond to task_indices order, not completion order."""
        # Task 0: finishes second, task 1: finishes first, task 2: finishes third
        env = MockMultiTurnEnv(steps_per_task={0: 2, 1: 1, 2: 3})
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2])

        assert result.trajectory_results[0].metadata["task_index"] == 0
        assert result.trajectory_results[1].metadata["task_index"] == 1
        assert result.trajectory_results[2].metadata["task_index"] == 2

    def test_all_trajectories_successful(self):
        """All trajectories should report success when rewards are 1.0."""
        env = MockSingleTurnEnv(num_tasks=3)
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2])

        assert result.success_rate == 1.0
        for tr in result.trajectory_results:
            assert tr.success

    def test_num_steps_in_metadata(self):
        """Trajectory metadata includes correct step count."""
        env = MockMultiTurnEnv(steps_per_task={0: 2, 1: 3})
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1])

        assert result.trajectory_results[0].metadata["num_steps"] == 2
        assert result.trajectory_results[1].metadata["num_steps"] == 3

    def test_progress_callback_final_report(self):
        """Progress callback final call should report all done."""
        env = MockMultiTurnEnv(steps_per_task={0: 1, 1: 2, 2: 3})
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        progress_reports: list[tuple[int, int]] = []
        runner.run_batch(
            [0, 1, 2],
            progress_callback=lambda c, t: progress_reports.append((c, t)),
        )

        assert progress_reports[-1] == (3, 3)
        # Completed count should be non-decreasing
        completions = [p[0] for p in progress_reports]
        assert completions == sorted(completions)

    def test_empty_task_list(self):
        """Empty task list returns empty BatchResult."""
        env = MockSingleTurnEnv()
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([])

        assert len(result.trajectory_results) == 0
        assert result.success_rate == 0.0

    def test_single_task(self):
        """Degenerate batch of one task."""
        env = MockSingleTurnEnv()
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0])

        assert len(result.trajectory_results) == 1
        assert backend.batch_call_sizes == [1]

    def test_reset_error_handled_per_trajectory(self):
        """Reset failure for one task doesn't stop others."""

        class FailingResetEnv(MockSingleTurnEnv):
            def reset(self, *, seed=None, options=None):
                idx = (options or {}).get("task_index", 0)
                if idx == 1:
                    raise ValueError("bad task")
                return super().reset(seed=seed, options=options)

        env = FailingResetEnv(num_tasks=3)
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2])

        assert len(result.trajectory_results) == 3
        assert result.trajectory_results[0].success
        assert not result.trajectory_results[1].success
        assert "error" in result.trajectory_results[1].metadata
        assert result.trajectory_results[2].success

    def test_step_error_handled_per_trajectory(self):
        """Step failure for one trajectory marks it failed, others continue."""

        class FailingStepEnv(MockMultiTurnEnv):
            def step(self, state, action):
                if state.hidden["task_index"] == 1 and state.metadata.step == 0:
                    raise RuntimeError("step failed")
                return super().step(state, action)

        env = FailingStepEnv(steps_per_task={0: 1, 1: 2, 2: 1})
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2])

        assert len(result.trajectory_results) == 3
        assert result.trajectory_results[0].success
        assert not result.trajectory_results[1].success
        assert "error" in result.trajectory_results[1].metadata
        assert result.trajectory_results[2].success


# --- TrajectoryRunner tool batch tests ---


class ToolBatchTrackingBackend(BatchTrackingBackend):
    """Backend that also tracks tool batch calls."""

    def __init__(self) -> None:
        super().__init__()
        self.tools_batch_call_sizes: list[int] = []

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_chat=True,
            supports_batching=True,
            supports_function_calling=True,
        )

    def generate_with_tools(self, messages, tools, params, tool_choice="auto"):
        return self._next_result()

    def generate_with_tools_batch(self, messages_batch, tools, params, tool_choice="auto"):
        self.tools_batch_call_sizes.append(len(messages_batch))
        return [self._next_result() for _ in messages_batch]


class MockSingleTurnToolEnv:
    """Single-step tool environment."""

    def __init__(self, num_tasks: int = 3):
        self._num_tasks = num_tasks
        self._tools = (ToolDefinition(name="search", description="Search"),)

    @property
    def spec(self):
        return EnvironmentSpec(name="mock_tool", max_steps=5, is_multi_turn=True)

    @property
    def reward_functions(self):
        return ()

    @property
    def available_tools(self):
        return self._tools

    @property
    def prompts(self):
        return {}

    def __len__(self):
        return self._num_tasks

    def reset(self, *, seed=None, options=None):
        idx = (options or {}).get("task_index", 0)
        return State(
            observation=Observation(
                prompt=f"Tool question {idx}?",
                available_tools=self._tools,
            ),
            hidden={"answer": str(idx)},
            metadata=StateMetadata(step=0, episode_id=f"tool_ep_{idx}"),
        ), {"task_index": idx}

    def step(self, state, action):
        return StepResult(
            next_state=state.with_metadata(step=1, is_terminal=True),
            rewards=SignalBundle.single(reward=1.0, name="correctness"),
            terminated=True,
        )

    def execute_tools(self, calls):
        return ()


class TestTrajectoryRunnerToolBatch:
    """Phase 2: TrajectoryRunner.run_batch() with tools lockstep batching."""

    def test_uses_generate_with_tools_batch(self):
        """Should use generate_with_tools_batch when tools are available."""
        env = MockSingleTurnToolEnv(num_tasks=3)
        backend = ToolBatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2])

        assert len(result.trajectory_results) == 3
        assert backend.tools_batch_call_sizes == [3]
        assert backend.batch_call_sizes == []  # Should NOT use chat batch
        assert backend.individual_call_count == 0

    def test_falls_back_to_chat_batch_without_function_calling(self):
        """Without function calling support, should use generate_chat_batch."""
        env = MockSingleTurnToolEnv(num_tasks=3)
        backend = BatchTrackingBackend()  # No function calling
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2])

        assert len(result.trajectory_results) == 3
        assert backend.batch_call_sizes == [3]
        assert backend.individual_call_count == 0


# ===========================================================================
# Phase 3: Segmented runner batching
# ===========================================================================


# --- Helpers for segmented tests ---


class _SimpleTokenizer:
    """1-token-per-character tokenizer for testing."""

    def __init__(self) -> None:
        self._text: str = ""

    def encode(self, text: str) -> list[int]:
        self._text = text
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return self._text[: len(tokens)]


class MockBaseEnvForSegmented:
    """Simple single-step base environment for SegmentedEnvironment testing."""

    def __init__(self, num_tasks: int = 5):
        self._num_tasks = num_tasks

    @property
    def spec(self):
        return EnvironmentSpec(name="mock_base", max_steps=1)

    @property
    def reward_functions(self):
        return ()

    def __len__(self):
        return self._num_tasks

    def reset(self, *, seed=None, options=None):
        idx = (options or {}).get("task_index", 0)
        return State(
            observation=Observation(prompt=f"Question {idx}?"),
            hidden={"task_index": idx},
            metadata=StateMetadata(step=0, episode_id=f"ep_{idx}"),
        ), {"task_index": idx}

    def step(self, state, action):
        return StepResult(
            next_state=state.with_metadata(step=1, is_terminal=True),
            rewards=SignalBundle.single(reward=1.0, name="correctness"),
            terminated=True,
        )

    def compute_rewards(self, state, action, next_state):
        return SignalBundle.empty()


class SegmentScriptedBackend(ModelBackend):
    """Backend with per-round scripted batch results for segment testing.

    Call add_round() to script each batch call. Each round provides a list
    of (text, finish_reason) tuples matching the batch size for that round.
    """

    def __init__(self) -> None:
        self._rounds: list[list[tuple[str, StopReason]]] = []
        self._round_idx: int = 0
        self.batch_call_sizes: list[int] = []
        self.individual_call_count: int = 0

    def add_round(self, responses: list[tuple[str, StopReason]]) -> None:
        self._rounds.append(responses)

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(supports_chat=True, supports_batching=True)

    @property
    def model_name(self) -> str:
        return "scripted-segment"

    def generate(self, prompts: list[str], params: SamplingParams) -> list[GenerationResult]:
        return [GenerationResult(text="x", finish_reason=StopReason.END_OF_TEXT) for _ in prompts]

    def generate_chat(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> GenerationResult:
        self.individual_call_count += 1
        return GenerationResult(
            text="completed",
            finish_reason=StopReason.END_OF_TEXT,
            prompt_tokens=10,
            completion_tokens=5,
        )

    def generate_chat_batch(
        self, messages_batch: list[list[ChatMessage]], params: SamplingParams
    ) -> list[GenerationResult]:
        self.batch_call_sizes.append(len(messages_batch))
        if self._round_idx < len(self._rounds):
            round_data = self._rounds[self._round_idx]
            self._round_idx += 1
            return [
                GenerationResult(
                    text=text,
                    finish_reason=reason,
                    prompt_tokens=10,
                    completion_tokens=5,
                )
                for text, reason in round_data
            ]
        return [
            GenerationResult(
                text="default",
                finish_reason=StopReason.END_OF_TEXT,
                prompt_tokens=10,
                completion_tokens=5,
            )
            for _ in messages_batch
        ]


# --- Continuation strategy batch tests ---


class TestTokenContinuationStrategyBatch:
    """Phase 3: TokenContinuationStrategy.generate_segment_batch()."""

    def test_batch_uses_generate_chat_batch(self):
        """Should make one generate_chat_batch call for all contexts."""
        backend = BatchTrackingBackend()
        strategy = TokenContinuationStrategy(backend=backend, token_size=64)
        contexts = [
            SegmentContext(
                messages=[ChatMessage(role="user", content="q1")],
                accumulated_text="",
                buffer="",
            ),
            SegmentContext(
                messages=[ChatMessage(role="user", content="q2")],
                accumulated_text="some_text",
                buffer="",
            ),
        ]

        results = strategy.generate_segment_batch(contexts, SamplingParams())

        assert len(results) == 2
        assert backend.batch_call_sizes == [2]
        assert backend.individual_call_count == 0
        for seg, buf, gr in results:
            assert isinstance(seg, str)
            assert buf == ""  # Token strategy always returns empty buffer


class TestBoundaryContinuationStrategyBatch:
    """Phase 3: BoundaryContinuationStrategy.generate_segment_batch()."""

    def test_buffer_only_no_backend_call(self):
        """When all buffers already have boundaries, no backend call is made."""
        backend = BatchTrackingBackend()
        segmenter = LineSegmenter()
        strategy = BoundaryContinuationStrategy(backend=backend, segmenter=segmenter)
        contexts = [
            SegmentContext(
                messages=[ChatMessage(role="user", content="q1")],
                accumulated_text="",
                buffer="line one\nline two",
            ),
            SegmentContext(
                messages=[ChatMessage(role="user", content="q2")],
                accumulated_text="prev",
                buffer="first\nsecond",
            ),
        ]

        results = strategy.generate_segment_batch(contexts, SamplingParams())

        assert len(results) == 2
        assert backend.batch_call_sizes == []  # No backend calls needed
        assert results[0][0] == "line one\n"
        assert results[0][1] == "line two"
        assert results[1][0] == "first\n"
        assert results[1][1] == "second"

    def test_mixed_buffer_and_generation(self):
        """Some contexts have buffer boundaries, others need generation."""
        backend = SegmentScriptedBackend()
        # Only 1 context needs generation, response has a boundary
        backend.add_round([("generated\nmore", StopReason.MAX_TOKENS)])

        segmenter = LineSegmenter()
        strategy = BoundaryContinuationStrategy(backend=backend, segmenter=segmenter)
        contexts = [
            SegmentContext(
                messages=[ChatMessage(role="user", content="q1")],
                accumulated_text="",
                buffer="buffered\nrest",  # Has boundary
            ),
            SegmentContext(
                messages=[ChatMessage(role="user", content="q2")],
                accumulated_text="",
                buffer="no_boundary_yet",  # No boundary, needs generation
            ),
        ]

        results = strategy.generate_segment_batch(contexts, SamplingParams())

        assert len(results) == 2
        # First: from buffer
        assert results[0][0] == "buffered\n"
        assert results[0][1] == "rest"
        # Second: from generation (buffer + generated text)
        assert "generated\n" in results[1][0] or results[1][0].endswith("\n")
        # Backend was called once for the 1 context needing generation
        assert backend.batch_call_sizes == [1]


# --- SegmentedTrajectoryRunner batch tests ---


class TestSegmentedTrajectoryRunnerBatch:
    """Phase 3: SegmentedTrajectoryRunner.run_batch() lockstep batching."""

    def _make_env(self, num_tasks: int = 5) -> SegmentedEnvironment:
        base = MockBaseEnvForSegmented(num_tasks=num_tasks)
        tokenizer = _SimpleTokenizer()
        segmenter = TokenSegmenter(tokenizer=tokenizer, token_size=64)
        return SegmentedEnvironment(base, segmenter)

    def test_uses_batch_generation(self):
        """Should use generate_chat_batch instead of generate_chat."""
        env = self._make_env(num_tasks=3)
        backend = SegmentScriptedBackend()
        backend.add_round(
            [
                ("answer_0", StopReason.END_OF_TEXT),
                ("answer_1", StopReason.END_OF_TEXT),
                ("answer_2", StopReason.END_OF_TEXT),
            ]
        )
        runner = SegmentedTrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2])

        assert len(result.trajectory_results) == 3
        assert backend.batch_call_sizes == [3]
        assert backend.individual_call_count == 0

    def test_decreasing_batch_sizes(self):
        """Batch sizes decrease as trajectories finish generation."""
        env = self._make_env(num_tasks=3)
        backend = SegmentScriptedBackend()
        # Round 0: task 1 finishes (EOS), tasks 0 and 2 continue
        backend.add_round(
            [
                ("s0", StopReason.MAX_TOKENS),
                ("s1", StopReason.END_OF_TEXT),
                ("s2", StopReason.MAX_TOKENS),
            ]
        )
        # Round 1: task 0 finishes, task 2 continues
        backend.add_round(
            [
                ("s0b", StopReason.END_OF_TEXT),
                ("s2b", StopReason.MAX_TOKENS),
            ]
        )
        # Round 2: task 2 finishes
        backend.add_round(
            [
                ("s2c", StopReason.END_OF_TEXT),
            ]
        )
        runner = SegmentedTrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2])

        assert len(result.trajectory_results) == 3
        assert backend.batch_call_sizes == [3, 2, 1]

    def test_preserves_result_order(self):
        """Results correspond to task_indices order, not completion order."""
        env = self._make_env(num_tasks=3)
        backend = SegmentScriptedBackend()
        # Task 1 finishes first, task 0 second, task 2 last
        backend.add_round(
            [
                ("s", StopReason.MAX_TOKENS),
                ("s", StopReason.END_OF_TEXT),
                ("s", StopReason.MAX_TOKENS),
            ]
        )
        backend.add_round(
            [
                ("s", StopReason.END_OF_TEXT),
                ("s", StopReason.MAX_TOKENS),
            ]
        )
        backend.add_round(
            [
                ("s", StopReason.END_OF_TEXT),
            ]
        )
        runner = SegmentedTrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2])

        assert result.trajectory_results[0].metadata["task_index"] == 0
        assert result.trajectory_results[1].metadata["task_index"] == 1
        assert result.trajectory_results[2].metadata["task_index"] == 2

    def test_empty_task_list(self):
        """Empty task list returns empty BatchResult."""
        env = self._make_env()
        backend = SegmentScriptedBackend()
        runner = SegmentedTrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([])

        assert len(result.trajectory_results) == 0
        assert result.success_rate == 0.0

    def test_finalize_produces_rewards(self):
        """Non-terminal trajectories get finalized with correctness rewards."""
        env = self._make_env(num_tasks=1)
        backend = SegmentScriptedBackend()
        backend.add_round([("the_answer", StopReason.END_OF_TEXT)])
        runner = SegmentedTrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0])

        tr = result.trajectory_results[0]
        # Should have generation transition + finalize transition
        assert len(tr.trajectory.transitions) >= 2
        last_info = tr.trajectory.transitions[-1].info
        assert last_info.get("finalize") is True

    def test_reset_error_handled(self):
        """Reset failure for one task doesn't stop others."""

        class FailResetBase(MockBaseEnvForSegmented):
            def reset(self, *, seed=None, options=None):
                idx = (options or {}).get("task_index", 0)
                if idx == 1:
                    raise ValueError("bad task 1")
                return super().reset(seed=seed, options=options)

        base = FailResetBase(num_tasks=3)
        segmenter = TokenSegmenter(tokenizer=_SimpleTokenizer(), token_size=64)
        env = SegmentedEnvironment(base, segmenter)
        backend = SegmentScriptedBackend()
        backend.add_round(
            [
                ("a0", StopReason.END_OF_TEXT),
                ("a2", StopReason.END_OF_TEXT),
            ]
        )
        runner = SegmentedTrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2])

        assert len(result.trajectory_results) == 3
        assert not result.trajectory_results[1].success
        assert "error" in result.trajectory_results[1].metadata

    def test_progress_callback(self):
        """Progress callback reports increasing completion."""
        env = self._make_env(num_tasks=2)
        backend = SegmentScriptedBackend()
        backend.add_round(
            [
                ("s", StopReason.END_OF_TEXT),
                ("s", StopReason.MAX_TOKENS),
            ]
        )
        backend.add_round(
            [
                ("s", StopReason.END_OF_TEXT),
            ]
        )
        runner = SegmentedTrajectoryRunner(environment=env, backend=backend)

        reports: list[tuple[int, int]] = []
        runner.run_batch(
            [0, 1],
            progress_callback=lambda c, t: reports.append((c, t)),
        )

        assert reports[-1] == (2, 2)
        completions = [r[0] for r in reports]
        assert completions == sorted(completions)

    def test_callback_feedback_injection(self):
        """step_callback returning a string injects a user message."""
        env = self._make_env(num_tasks=1)
        backend = SegmentScriptedBackend()
        backend.add_round([("first_seg", StopReason.MAX_TOKENS)])
        backend.add_round([("after_feedback", StopReason.END_OF_TEXT)])
        runner = SegmentedTrajectoryRunner(environment=env, backend=backend)

        call_count = 0

        def callback(step_result):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Please reconsider."
            return None

        result = runner.run_batch([0], step_callback=callback)

        assert len(result.trajectory_results) == 1
        assert backend.batch_call_sizes == [1, 1]

    def test_callback_force_action(self):
        """step_callback returning ForceAction uses forced text as segment."""
        env = self._make_env(num_tasks=1)
        backend = SegmentScriptedBackend()
        backend.add_round([("first", StopReason.MAX_TOKENS)])
        # After forced segment (no backend call), generation resumes
        backend.add_round([("after_force", StopReason.END_OF_TEXT)])
        runner = SegmentedTrajectoryRunner(environment=env, backend=backend)

        call_count = 0

        def callback(step_result):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ForceAction(text="FORCED")
            return None

        result = runner.run_batch([0], step_callback=callback)

        tr = result.trajectory_results[0]
        segments = [t.action.text for t in tr.trajectory.transitions if t.action.text]
        assert "FORCED" in segments

    def test_callback_complete(self):
        """step_callback returning COMPLETE finishes via one-shot completion."""
        env = self._make_env(num_tasks=1)
        backend = SegmentScriptedBackend()
        backend.add_round([("first", StopReason.MAX_TOKENS)])
        runner = SegmentedTrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0], step_callback=lambda _: COMPLETE)

        assert len(result.trajectory_results) == 1
        # _complete_remainder uses individual generate_chat
        assert backend.individual_call_count >= 1


# ===========================================================================
# Phase 4: batch_size chunking
# ===========================================================================


class TestBatchSizeChunking:
    """Phase 4: batch_size parameter on run_batch() methods."""

    def test_trajectory_runner_batch_size_chunks(self):
        """TrajectoryRunner.run_batch with batch_size=2 and 5 tasks makes 3 batch calls."""
        env = MockSingleTurnEnv(num_tasks=5)
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2, 3, 4], batch_size=2)

        assert len(result.trajectory_results) == 5
        # 5 tasks / batch_size=2 => 3 chunks: [0,1], [2,3], [4]
        assert backend.batch_call_sizes == [2, 2, 1]
        assert result.success_rate == 1.0

    def test_trajectory_runner_batch_size_none_all_at_once(self):
        """batch_size=None runs all tasks in a single batch."""
        env = MockSingleTurnEnv(num_tasks=3)
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2], batch_size=None)

        assert len(result.trajectory_results) == 3
        assert backend.batch_call_sizes == [3]

    def test_trajectory_runner_batch_size_exceeds_task_count(self):
        """batch_size larger than task count runs a single batch."""
        env = MockSingleTurnEnv(num_tasks=3)
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2], batch_size=100)

        assert len(result.trajectory_results) == 3
        assert backend.batch_call_sizes == [3]

    def test_trajectory_runner_batch_size_progress_callback(self):
        """Progress callback reports correct global offsets across chunks."""
        env = MockSingleTurnEnv(num_tasks=4)
        backend = BatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        reports: list[tuple[int, int]] = []
        result = runner.run_batch(
            [0, 1, 2, 3],
            batch_size=2,
            progress_callback=lambda c, t: reports.append((c, t)),
        )

        assert len(result.trajectory_results) == 4
        # All reports should use total=4
        for _, total in reports:
            assert total == 4
        # Last report should show all done
        assert reports[-1] == (4, 4)
        # Completed count should be non-decreasing
        completions = [c for c, _ in reports]
        assert completions == sorted(completions)

    def test_tool_runner_batch_size_chunks(self):
        """TrajectoryRunner.run_batch with batch_size chunks correctly."""
        env = MockSingleTurnToolEnv(num_tasks=4)
        backend = ToolBatchTrackingBackend()
        runner = TrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2, 3], batch_size=2)

        assert len(result.trajectory_results) == 4
        # 4 tasks / batch_size=2 => 2 chunks
        assert backend.tools_batch_call_sizes == [2, 2]
        assert result.success_rate == 1.0

    def test_segmented_runner_batch_size_chunks(self):
        """SegmentedTrajectoryRunner.run_batch with batch_size chunks correctly."""
        base = MockBaseEnvForSegmented(num_tasks=4)
        segmenter = TokenSegmenter(tokenizer=_SimpleTokenizer(), token_size=64)
        env = SegmentedEnvironment(base, segmenter)

        backend = SegmentScriptedBackend()
        # Chunk 1: tasks 0, 1
        backend.add_round(
            [
                ("a0", StopReason.END_OF_TEXT),
                ("a1", StopReason.END_OF_TEXT),
            ]
        )
        # Chunk 2: tasks 2, 3
        backend.add_round(
            [
                ("a2", StopReason.END_OF_TEXT),
                ("a3", StopReason.END_OF_TEXT),
            ]
        )
        runner = SegmentedTrajectoryRunner(environment=env, backend=backend)

        result = runner.run_batch([0, 1, 2, 3], batch_size=2)

        assert len(result.trajectory_results) == 4
        assert backend.batch_call_sizes == [2, 2]


# ===========================================================================
# Phase 5: Cross-environment batched evaluation
# ===========================================================================


class TestMultiEvaluation:
    """Tests for run_multi_evaluation() — cross-environment batched evaluation."""

    def test_basic_two_envs(self):
        """Two single-turn envs produce correct per-entry BatchResults."""
        env_a = MockSingleTurnEnv(num_tasks=3)
        env_b = MockSingleTurnEnv(num_tasks=2)
        backend = BatchTrackingBackend()

        runner_a = TrajectoryRunner(environment=env_a, backend=backend)
        runner_b = TrajectoryRunner(environment=env_b, backend=backend)

        results = run_multi_evaluation(
            entries=[
                MultiEvalEntry(runner=runner_a, task_indices=[0, 1, 2]),
                MultiEvalEntry(runner=runner_b, task_indices=[0, 1]),
            ],
        )

        assert len(results) == 2
        assert len(results[0].trajectory_results) == 3
        assert len(results[1].trajectory_results) == 2
        assert results[0].success_rate == 1.0
        assert results[1].success_rate == 1.0

    def test_shared_backend_validation(self):
        """Raises ValueError if entries have different backends."""
        env = MockSingleTurnEnv(num_tasks=2)
        backend_a = BatchTrackingBackend()
        backend_b = BatchTrackingBackend()

        runner_a = TrajectoryRunner(environment=env, backend=backend_a)
        runner_b = TrajectoryRunner(environment=env, backend=backend_b)

        with pytest.raises(ValueError, match="backend"):
            run_multi_evaluation(
                entries=[
                    MultiEvalEntry(runner=runner_a, task_indices=[0]),
                    MultiEvalEntry(runner=runner_b, task_indices=[0]),
                ],
            )

    def test_shared_sampling_params_validation(self):
        """Raises ValueError if entries have different sampling_params."""
        env = MockSingleTurnEnv(num_tasks=2)
        backend = BatchTrackingBackend()

        runner_a = TrajectoryRunner(
            environment=env,
            backend=backend,
            sampling_params=SamplingParams(temperature=0.0),
        )
        runner_b = TrajectoryRunner(
            environment=env,
            backend=backend,
            sampling_params=SamplingParams(temperature=1.0),
        )

        with pytest.raises(ValueError, match="sampling_params"):
            run_multi_evaluation(
                entries=[
                    MultiEvalEntry(runner=runner_a, task_indices=[0]),
                    MultiEvalEntry(runner=runner_b, task_indices=[0]),
                ],
            )

    def test_mixed_step_counts(self):
        """Env A finishes in 1 step, env B in 3 — lockstep handles dropout."""
        env_a = MockSingleTurnEnv(num_tasks=2)
        env_b = MockMultiTurnEnv(steps_per_task={0: 3, 1: 3})
        backend = BatchTrackingBackend()

        runner_a = TrajectoryRunner(environment=env_a, backend=backend)
        runner_b = TrajectoryRunner(environment=env_b, backend=backend)

        results = run_multi_evaluation(
            entries=[
                MultiEvalEntry(runner=runner_a, task_indices=[0, 1]),
                MultiEvalEntry(runner=runner_b, task_indices=[0, 1]),
            ],
        )

        assert len(results) == 2
        # Env A: 2 tasks, all finish in 1 step
        assert len(results[0].trajectory_results) == 2
        assert all(r.success for r in results[0].trajectory_results)
        # Env B: 2 tasks, finish in 3 steps
        assert len(results[1].trajectory_results) == 2
        assert all(r.success for r in results[1].trajectory_results)
        for r in results[1].trajectory_results:
            assert r.metadata["num_steps"] == 3

    def test_single_generate_batch_call_per_step(self):
        """One generate_chat_batch() call per step, batch size = sum of active."""
        env_a = MockSingleTurnEnv(num_tasks=2)  # finishes in 1 step
        env_b = MockMultiTurnEnv(steps_per_task={0: 2})  # finishes in 2 steps
        backend = BatchTrackingBackend()

        runner_a = TrajectoryRunner(environment=env_a, backend=backend)
        runner_b = TrajectoryRunner(environment=env_b, backend=backend)

        run_multi_evaluation(
            entries=[
                MultiEvalEntry(runner=runner_a, task_indices=[0, 1]),
                MultiEvalEntry(runner=runner_b, task_indices=[0]),
            ],
        )

        # Step 1: 3 active (2 from A + 1 from B), step 2: 1 active (only B)
        assert backend.batch_call_sizes == [3, 1]
        assert backend.individual_call_count == 0

    def test_batch_size_chunks(self):
        """batch_size=2, 5 total trajectories across 2 envs, correct chunking + results."""
        env_a = MockSingleTurnEnv(num_tasks=3)
        env_b = MockSingleTurnEnv(num_tasks=2)
        backend = BatchTrackingBackend()

        runner_a = TrajectoryRunner(environment=env_a, backend=backend)
        runner_b = TrajectoryRunner(environment=env_b, backend=backend)

        results = run_multi_evaluation(
            entries=[
                MultiEvalEntry(runner=runner_a, task_indices=[0, 1, 2]),
                MultiEvalEntry(runner=runner_b, task_indices=[0, 1]),
            ],
            batch_size=2,
        )

        assert len(results) == 2
        assert len(results[0].trajectory_results) == 3
        assert len(results[1].trajectory_results) == 2
        # 5 total / batch_size=2 => 3 chunks: [2], [2], [1]
        assert backend.batch_call_sizes == [2, 2, 1]

    def test_progress_callback(self):
        """Callback receives correct (completed, total) where total = sum of all task_indices."""
        env_a = MockSingleTurnEnv(num_tasks=2)
        env_b = MockMultiTurnEnv(steps_per_task={0: 2})
        backend = BatchTrackingBackend()

        runner_a = TrajectoryRunner(environment=env_a, backend=backend)
        runner_b = TrajectoryRunner(environment=env_b, backend=backend)

        reports: list[tuple[int, int]] = []
        run_multi_evaluation(
            entries=[
                MultiEvalEntry(runner=runner_a, task_indices=[0, 1]),
                MultiEvalEntry(runner=runner_b, task_indices=[0]),
            ],
            progress_callback=lambda c, t: reports.append((c, t)),
        )

        # Total should be 3 (2 + 1)
        for _, total in reports:
            assert total == 3
        assert reports[-1] == (3, 3)
        completions = [c for c, _ in reports]
        assert completions == sorted(completions)

    def test_empty_entries(self):
        """Empty entries list returns empty list."""
        results = run_multi_evaluation(entries=[])
        assert results == []

    def test_single_entry_equivalent(self):
        """Single entry matches TrajectoryRunner.run_batch() output."""
        env = MockMultiTurnEnv(steps_per_task={0: 2, 1: 1, 2: 3})
        backend_a = BatchTrackingBackend()
        backend_b = BatchTrackingBackend()

        runner_a = TrajectoryRunner(environment=env, backend=backend_a)
        runner_b = TrajectoryRunner(environment=env, backend=backend_b)

        # Run via multi-eval
        multi_results = run_multi_evaluation(
            entries=[MultiEvalEntry(runner=runner_a, task_indices=[0, 1, 2])],
        )

        # Run via normal run_batch
        direct_result = runner_b.run_batch([0, 1, 2])

        assert len(multi_results) == 1
        multi_result = multi_results[0]
        assert len(multi_result.trajectory_results) == len(direct_result.trajectory_results)
        assert multi_result.success_rate == direct_result.success_rate
        assert multi_result.mean_reward == direct_result.mean_reward
        for mr, dr in zip(multi_result.trajectory_results, direct_result.trajectory_results):
            assert mr.metadata["task_index"] == dr.metadata["task_index"]
            assert mr.metadata["num_steps"] == dr.metadata["num_steps"]
            assert mr.success == dr.success
