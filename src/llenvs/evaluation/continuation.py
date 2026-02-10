"""Continuation strategies for generation-time segmentation.

Strategies control how text is generated one segment at a time,
handling the resumption of generation after each segment boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from llenvs.core.segmentation import Segmenter, TokenSegmenter
from llenvs.inference.protocol import (
    ChatMessage,
    GenerationResult,
    ModelBackend,
    SamplingParams,
    StopReason,
)


@dataclass(frozen=True)
class SegmentContext:
    """Context for one trajectory in a batched segment generation call.

    Attributes:
        messages: Base conversation messages.
        accumulated_text: Text generated so far in the current assistant turn.
        buffer: Leftover text from the previous generation.
    """

    messages: list[ChatMessage]
    accumulated_text: str
    buffer: str


@runtime_checkable
class ContinuationStrategy(Protocol):
    """Protocol for segment-at-a-time generation strategies.

    Each call to generate_segment produces one segment of text,
    handling buffering and boundary detection as needed.
    """

    def generate_segment(
        self,
        messages: list[ChatMessage],
        accumulated_text: str,
        buffer: str,
        sampling_params: SamplingParams,
    ) -> tuple[str, str, GenerationResult]:
        """Generate the next segment of text.

        Args:
            messages: Base conversation messages (system + user, possibly
                with prior assistant/user turns from feedback).
            accumulated_text: Text generated so far in the current
                assistant turn.
            buffer: Leftover text from the previous generation that
                hasn't been emitted as a segment yet.
            sampling_params: Base sampling parameters.

        Returns:
            Tuple of (segment_text, remaining_buffer, gen_result).
            gen_result reflects the latest backend call (or a synthetic
            result if the segment came entirely from the buffer).
        """
        ...

    def generate_segment_batch(
        self,
        contexts: Sequence[SegmentContext],
        sampling_params: SamplingParams,
    ) -> list[tuple[str, str, GenerationResult]]:
        """Generate segments for multiple trajectories in parallel.

        Args:
            contexts: Segment contexts, one per trajectory.
            sampling_params: Base sampling parameters.

        Returns:
            List of (segment_text, remaining_buffer, gen_result) tuples.
        """
        ...

    def is_generation_done(self, gen_result: GenerationResult, buffer: str) -> bool:
        """Check if generation is complete.

        Args:
            gen_result: Result from the most recent generation call.
            buffer: Remaining buffered text.

        Returns:
            True if no more segments should be generated.
        """
        ...


def _build_continuation_messages(
    messages: list[ChatMessage],
    accumulated_text: str,
) -> list[ChatMessage]:
    """Append accumulated text as an assistant message for continuation.

    Args:
        messages: Base conversation messages.
        accumulated_text: Text generated so far in the current assistant turn.

    Returns:
        Messages with the accumulated assistant text appended.
    """
    if not accumulated_text:
        return list(messages)
    return list(messages) + [ChatMessage(role="assistant", content=accumulated_text)]


@dataclass
class TokenContinuationStrategy:
    """Continuation strategy for TokenSegmenter.

    Sets max_tokens to the token_size so each generation call naturally
    produces exactly one segment. No boundary detection needed.
    Buffer is always empty.
    """

    backend: ModelBackend
    token_size: int

    def generate_segment(
        self,
        messages: list[ChatMessage],
        accumulated_text: str,
        buffer: str,
        sampling_params: SamplingParams,
    ) -> tuple[str, str, GenerationResult]:
        """Generate one token-sized segment."""
        # Override max_tokens to produce exactly one segment
        params = SamplingParams(
            max_tokens=self.token_size,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            stop_sequences=sampling_params.stop_sequences,
            presence_penalty=sampling_params.presence_penalty,
            frequency_penalty=sampling_params.frequency_penalty,
            n=sampling_params.n,
            logprobs=sampling_params.logprobs,
            num_logprobs=sampling_params.num_logprobs,
            extra=sampling_params.extra,
        )

        cont_messages = _build_continuation_messages(messages, accumulated_text)
        gen_result = self.backend.generate_chat(cont_messages, params)

        segment = gen_result.text or ""
        return segment, "", gen_result

    def generate_segment_batch(
        self,
        contexts: Sequence[SegmentContext],
        sampling_params: SamplingParams,
    ) -> list[tuple[str, str, GenerationResult]]:
        """Generate one token-sized segment for each trajectory."""
        params = SamplingParams(
            max_tokens=self.token_size,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            stop_sequences=sampling_params.stop_sequences,
            presence_penalty=sampling_params.presence_penalty,
            frequency_penalty=sampling_params.frequency_penalty,
            n=sampling_params.n,
            logprobs=sampling_params.logprobs,
            num_logprobs=sampling_params.num_logprobs,
            extra=sampling_params.extra,
        )

        messages_batch = [
            _build_continuation_messages(ctx.messages, ctx.accumulated_text)
            for ctx in contexts
        ]
        gen_results = self.backend.generate_chat_batch(messages_batch, params)

        return [(gr.text or "", "", gr) for gr in gen_results]

    def is_generation_done(self, gen_result: GenerationResult, buffer: str) -> bool:
        """Done when EOS or empty output."""
        if gen_result.finish_reason in (StopReason.END_OF_TEXT, StopReason.STOP_SEQUENCE):
            return True
        if not gen_result.text:
            return True
        return False


# Sentinel for "no backend call was made"
_BUFFER_ONLY_RESULT = GenerationResult(
    text=None,
    finish_reason=StopReason.UNKNOWN,
    prompt_tokens=0,
    completion_tokens=0,
)


@dataclass
class BoundaryContinuationStrategy:
    """Continuation strategy for text-pattern segmenters.

    Generates text in chunks, uses the segmenter's find_boundary() to
    detect segment boundaries, and buffers overflow for the next call.
    Keeps generating more text when no boundary is found in the current chunk.
    """

    backend: ModelBackend
    segmenter: Segmenter
    chunk_max_tokens: int = 256

    def generate_segment(
        self,
        messages: list[ChatMessage],
        accumulated_text: str,
        buffer: str,
        sampling_params: SamplingParams,
    ) -> tuple[str, str, GenerationResult]:
        """Generate text until a segment boundary is found."""
        # Check buffer first — may already contain a boundary
        if buffer:
            boundary = self.segmenter.find_boundary(buffer)
            if boundary is not None:
                segment = buffer[:boundary]
                remaining = buffer[boundary:]
                return segment, remaining, _BUFFER_ONLY_RESULT

        # Generate new text
        working_buffer = buffer
        last_gen_result = _BUFFER_ONLY_RESULT

        params = SamplingParams(
            max_tokens=self.chunk_max_tokens,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            stop_sequences=sampling_params.stop_sequences,
            presence_penalty=sampling_params.presence_penalty,
            frequency_penalty=sampling_params.frequency_penalty,
            n=sampling_params.n,
            logprobs=sampling_params.logprobs,
            num_logprobs=sampling_params.num_logprobs,
            extra=sampling_params.extra,
        )

        max_attempts = 10  # safety limit
        for _ in range(max_attempts):
            full_accumulated = accumulated_text + working_buffer
            cont_messages = _build_continuation_messages(messages, full_accumulated)
            gen_result = self.backend.generate_chat(cont_messages, params)
            last_gen_result = gen_result

            new_text = gen_result.text or ""
            working_buffer += new_text

            # Try to find a boundary in the working buffer
            boundary = self.segmenter.find_boundary(working_buffer)
            if boundary is not None:
                segment = working_buffer[:boundary]
                remaining = working_buffer[boundary:]
                return segment, remaining, last_gen_result

            # No boundary found
            is_eos = gen_result.finish_reason in (
                StopReason.END_OF_TEXT,
                StopReason.STOP_SEQUENCE,
            )
            if is_eos or not new_text:
                # Generation ended — return everything as one segment
                return working_buffer, "", last_gen_result

        # Safety: exhausted attempts, return what we have
        return working_buffer, "", last_gen_result

    def generate_segment_batch(
        self,
        contexts: Sequence[SegmentContext],
        sampling_params: SamplingParams,
    ) -> list[tuple[str, str, GenerationResult]]:
        """Generate segments for multiple trajectories, batching backend calls.

        Checks existing buffers first (no backend call needed for those),
        then batch-generates for remaining contexts. Repeats up to
        max_attempts for contexts that don't find a boundary.
        """
        n = len(contexts)
        results: list[tuple[str, str, GenerationResult] | None] = [None] * n
        active_indices: list[int] = []
        working_buffers: dict[int, str] = {}
        last_gen_results: dict[int, GenerationResult] = {}

        # Phase 1: Check existing buffers for boundaries
        for i, ctx in enumerate(contexts):
            if ctx.buffer:
                boundary = self.segmenter.find_boundary(ctx.buffer)
                if boundary is not None:
                    results[i] = (ctx.buffer[:boundary], ctx.buffer[boundary:], _BUFFER_ONLY_RESULT)
                    continue
            active_indices.append(i)
            working_buffers[i] = ctx.buffer
            last_gen_results[i] = _BUFFER_ONLY_RESULT

        # Phase 2: Multi-round batch generation
        params = SamplingParams(
            max_tokens=self.chunk_max_tokens,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            stop_sequences=sampling_params.stop_sequences,
            presence_penalty=sampling_params.presence_penalty,
            frequency_penalty=sampling_params.frequency_penalty,
            n=sampling_params.n,
            logprobs=sampling_params.logprobs,
            num_logprobs=sampling_params.num_logprobs,
            extra=sampling_params.extra,
        )

        max_attempts = 10
        for _ in range(max_attempts):
            if not active_indices:
                break

            messages_batch = [
                _build_continuation_messages(
                    contexts[i].messages,
                    contexts[i].accumulated_text + working_buffers[i],
                )
                for i in active_indices
            ]
            gen_results = self.backend.generate_chat_batch(messages_batch, params)

            still_active: list[int] = []
            for idx, gen_result in zip(active_indices, gen_results):
                new_text = gen_result.text or ""
                working_buffers[idx] += new_text
                last_gen_results[idx] = gen_result

                boundary = self.segmenter.find_boundary(working_buffers[idx])
                if boundary is not None:
                    results[idx] = (
                        working_buffers[idx][:boundary],
                        working_buffers[idx][boundary:],
                        gen_result,
                    )
                elif gen_result.finish_reason in (StopReason.END_OF_TEXT, StopReason.STOP_SEQUENCE) or not new_text:
                    results[idx] = (working_buffers[idx], "", gen_result)
                else:
                    still_active.append(idx)

            active_indices = still_active

        # Safety: fill any remaining with what we have
        for idx in active_indices:
            results[idx] = (working_buffers[idx], "", last_gen_results[idx])

        return results  # type: ignore[return-value]

    def is_generation_done(self, gen_result: GenerationResult, buffer: str) -> bool:
        """Done when EOS/stop and buffer is empty."""
        if buffer:
            return False
        return gen_result.finish_reason in (
            StopReason.END_OF_TEXT,
            StopReason.STOP_SEQUENCE,
        )


def select_strategy(
    backend: ModelBackend,
    segmenter: Segmenter,
    chunk_max_tokens: int = 256,
) -> ContinuationStrategy:
    """Auto-select the appropriate continuation strategy.

    Args:
        backend: The model backend.
        segmenter: The segmenter used by the environment.
        chunk_max_tokens: Max tokens per chunk for boundary strategies.

    Returns:
        The appropriate ContinuationStrategy.
    """
    if isinstance(segmenter, TokenSegmenter):
        return TokenContinuationStrategy(
            backend=backend,
            token_size=segmenter.token_size,
        )
    return BoundaryContinuationStrategy(
        backend=backend,
        segmenter=segmenter,
        chunk_max_tokens=chunk_max_tokens,
    )
