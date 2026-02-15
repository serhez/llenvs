"""Ready-to-use iterative coding environments.

Convenience factories that wire up HuggingFace coding datasets with
code execution feedback, optional LLM judge feedback, and iterative
refinement.

Usage::

    from llenvs.environments import IterativeCodingEnvironment

    env = IterativeCodingEnvironment.from_humaneval(max_turns=5)
    state, info = env.reset(options={"task_index": 0})
"""

from __future__ import annotations

from typing import Any

from llenvs.adapters.iterative import IterativeEnvironment
from llenvs.core.code_execution import (
    CodeExecutionReward,
    SubprocessCodeExecutor,
)
from llenvs.core.extraction import CodeBlockExtractor
from llenvs.core.reward import RewardFunction, RewardType


class IterativeCodingEnvironment:
    """Factory for iterative coding environments.

    Not an environment itself — each method returns a configured
    ``IterativeEnvironment`` wrapping an HF coding dataset with
    code execution and optional judge feedback.
    """

    @staticmethod
    def from_humaneval(
        *,
        max_turns: int = 5,
        timeout: float = 30.0,
        judge_backend: Any | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> IterativeEnvironment:
        """Create an iterative environment for OpenAI HumanEval.

        Args:
            max_turns: Maximum refinement turns.
            timeout: Code execution timeout in seconds.
            judge_backend: Optional ModelBackend for LLM judge feedback.
            extra_rewards: Additional reward functions.

        Returns:
            Configured IterativeEnvironment.
        """
        return IterativeCodingEnvironment.create(
            dataset="openai/openai_humaneval",
            language="python",
            max_turns=max_turns,
            timeout=timeout,
            question_column="prompt",
            answer_column="canonical_solution",
            test_column="test",
            judge_backend=judge_backend,
            extra_rewards=extra_rewards,
            split="test",
        )

    @staticmethod
    def from_mbpp(
        *,
        max_turns: int = 3,
        timeout: float = 30.0,
        judge_backend: Any | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
    ) -> IterativeEnvironment:
        """Create an iterative environment for MBPP.

        Args:
            max_turns: Maximum refinement turns.
            timeout: Code execution timeout in seconds.
            judge_backend: Optional ModelBackend for LLM judge feedback.
            extra_rewards: Additional reward functions.

        Returns:
            Configured IterativeEnvironment.
        """
        return IterativeCodingEnvironment.create(
            dataset="google-research-datasets/mbpp",
            language="python",
            max_turns=max_turns,
            timeout=timeout,
            question_column="text",
            answer_column="code",
            test_column="test_list",
            judge_backend=judge_backend,
            extra_rewards=extra_rewards,
            split="test",
        )

    @staticmethod
    def create(
        dataset: str,
        *,
        language: str = "python",
        max_turns: int = 3,
        timeout: float = 30.0,
        question_column: str = "prompt",
        answer_column: str = "canonical_solution",
        test_column: str | None = None,
        judge_backend: Any | None = None,
        judge_template: str = "iterative_feedback",
        submission_extractor: Any | None = None,
        extra_rewards: tuple[RewardFunction, ...] = (),
        **hf_kwargs: Any,
    ) -> IterativeEnvironment:
        """Create an iterative coding environment from any HF dataset.

        Args:
            dataset: HuggingFace dataset name.
            language: Programming language for code extraction.
            max_turns: Maximum refinement turns.
            timeout: Code execution timeout in seconds.
            question_column: Column containing task descriptions.
            answer_column: Column containing reference solutions.
            test_column: Column containing test code (optional).
            judge_backend: Optional ModelBackend for LLM judge feedback.
            judge_template: Judge prompt template name.
            submission_extractor: Custom extractor for submissions.
            extra_rewards: Additional reward functions.
            **hf_kwargs: Additional arguments for HuggingFaceAdapter.

        Returns:
            Configured IterativeEnvironment.
        """
        from llenvs.adapters.huggingface import HuggingFaceAdapter

        # 1. Create inner HF environment
        adapter = HuggingFaceAdapter()
        hf_env = adapter.get_environment(
            dataset,
            question_column=question_column,
            answer_column=answer_column,
            **hf_kwargs,
        )

        # 2. Build code execution reward
        executor = SubprocessCodeExecutor()
        code_extractor = CodeBlockExtractor(language=language)

        # Build test_extractor based on test_column
        test_extractor = None
        if test_column:
            col = test_column

            def _test_extractor(hidden: Any) -> str | None:
                entry = getattr(hidden, "entry", None)
                if not isinstance(entry, dict):
                    return None
                val = entry.get(col)
                if isinstance(val, list):
                    return "\n".join(str(v) for v in val)
                if val is not None:
                    return str(val)
                return None

            test_extractor = _test_extractor

        code_exec_reward = CodeExecutionReward(
            executor=executor,
            code_extractor=code_extractor,
            test_extractor=test_extractor,
        )

        # 3. Optionally create judge reward
        all_extra: list[RewardFunction] = [code_exec_reward]

        if judge_backend is not None:
            from llenvs.core.judge import JudgeReward

            judge = JudgeReward(
                backend=judge_backend,
                template=judge_template,
                reward_type=RewardType.PROCESS,
                name="judge_feedback",
            )
            all_extra.append(judge)

        all_extra.extend(extra_rewards)

        # 4. Wrap in IterativeEnvironment
        return IterativeEnvironment(
            inner=hf_env,
            max_turns=max_turns,
            submission_extractor=submission_extractor or code_extractor,
            extra_rewards=tuple(all_extra),
        )
