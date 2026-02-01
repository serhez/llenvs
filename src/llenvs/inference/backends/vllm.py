"""vLLM backend implementation.

vLLM provides full feature support including:
- Batched generation
- Log probabilities
- Prefix continuation
- Streaming
"""

from typing import Any

from env_evals.inference.protocol import (
    ModelBackend,
    BackendCapabilities,
    SamplingParams,
    GenerationResult,
    ChatMessage,
    StopReason,
    TokenLogprob,
)


def _convert_stop_reason(reason: str | None) -> StopReason:
    """Convert vLLM finish reason to StopReason."""
    if reason is None:
        return StopReason.UNKNOWN
    reason = reason.lower()
    if reason == "stop":
        return StopReason.STOP_SEQUENCE
    elif reason == "length":
        return StopReason.MAX_TOKENS
    elif reason in ("eos", "eos_token"):
        return StopReason.END_OF_TEXT
    return StopReason.UNKNOWN


class VLLMBackend(ModelBackend):
    """vLLM-based inference backend.

    Provides high-performance local inference with full feature support.
    Requires a GPU with sufficient memory for the model.

    Attributes:
        model_path: Path or HuggingFace model ID.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.9,
        seed: int = 42,
        **vllm_kwargs: Any,
    ) -> None:
        """Initialize vLLM backend.

        Args:
            model_path: Path to model or HuggingFace model ID.
            tensor_parallel_size: Number of GPUs for parallelism.
            dtype: Data type (auto, float16, bfloat16, float32).
            max_model_len: Maximum sequence length.
            gpu_memory_utilization: Fraction of GPU memory to use.
            seed: Random seed.
            **vllm_kwargs: Additional arguments passed to vLLM LLM.

        Raises:
            ImportError: If vLLM is not installed.
        """
        try:
            from vllm import LLM, SamplingParams as VLLMSamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is required for VLLMBackend. Install with: pip install vllm"
            ) from e

        self._model_path = model_path
        self._VLLMSamplingParams = VLLMSamplingParams

        # Initialize vLLM engine
        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "seed": seed,
            **vllm_kwargs,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self._llm = LLM(**llm_kwargs)
        self._tokenizer = self._llm.get_tokenizer()

        # Determine max context length
        model_config = self._llm.llm_engine.model_config
        self._max_context_length = getattr(model_config, "max_model_len", None)

    @property
    def capabilities(self) -> BackendCapabilities:
        """vLLM supports all major features."""
        return BackendCapabilities(
            supports_logprobs=True,
            supports_prefix_continuation=True,
            supports_batching=True,
            supports_streaming=True,
            supports_chat=True,
            supports_function_calling=False,
            max_batch_size=None,  # Limited by GPU memory
            max_context_length=self._max_context_length,
        )

    @property
    def model_name(self) -> str:
        """Get the model path/identifier."""
        return self._model_path

    @property
    def tokenizer(self) -> Any:
        """Get the underlying tokenizer."""
        return self._tokenizer

    def _to_vllm_params(self, params: SamplingParams) -> Any:
        """Convert to vLLM SamplingParams."""
        kwargs: dict[str, Any] = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "n": params.n,
            "presence_penalty": params.presence_penalty,
            "frequency_penalty": params.frequency_penalty,
        }

        if params.top_k > 0:
            kwargs["top_k"] = params.top_k

        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)

        if params.logprobs:
            kwargs["logprobs"] = params.num_logprobs

        return self._VLLMSamplingParams(**kwargs)

    def _extract_logprobs(self, output: Any) -> tuple[TokenLogprob, ...] | None:
        """Extract log probabilities from vLLM output."""
        if output.logprobs is None:
            return None

        logprobs = []
        for token_logprob in output.logprobs:
            if token_logprob is None:
                continue

            # Get the chosen token's info
            for token_id, logprob_obj in token_logprob.items():
                # Build top logprobs dict
                top_logprobs = None
                if hasattr(logprob_obj, "top_logprobs") and logprob_obj.top_logprobs:
                    top_logprobs = {
                        tok: lp.logprob for tok, lp in logprob_obj.top_logprobs.items()
                    }

                logprobs.append(
                    TokenLogprob(
                        token=logprob_obj.decoded_token
                        if hasattr(logprob_obj, "decoded_token")
                        else str(token_id),
                        token_id=token_id,
                        logprob=logprob_obj.logprob,
                        top_logprobs=top_logprobs,
                    )
                )
                break  # Only first entry per position

        return tuple(logprobs) if logprobs else None

    def generate(
        self,
        prompts: list[str],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        """Generate completions for text prompts.

        Args:
            prompts: List of prompt strings.
            params: Sampling parameters.

        Returns:
            List of GenerationResults.
        """
        vllm_params = self._to_vllm_params(params)
        outputs = self._llm.generate(prompts, vllm_params, use_tqdm=False)

        results = []
        for output in outputs:
            # Take the first completion (n=1 case)
            completion = output.outputs[0]

            # Extract logprobs if available
            token_logprobs = self._extract_logprobs(completion)

            results.append(
                GenerationResult(
                    text=completion.text,
                    finish_reason=_convert_stop_reason(completion.finish_reason),
                    token_logprobs=token_logprobs,
                    prompt_tokens=len(output.prompt_token_ids),
                    completion_tokens=len(completion.token_ids),
                    metadata={
                        "request_id": output.request_id,
                    },
                )
            )

        return results

    def generate_chat(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        """Generate a response for a chat conversation.

        Args:
            messages: List of chat messages.
            params: Sampling parameters.

        Returns:
            GenerationResult for the conversation.
        """
        # Convert to dict format for chat template
        message_dicts = [m.to_dict() for m in messages]

        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True,
        )

        results = self.generate([prompt], params)
        return results[0]

    def continue_from_prefix(
        self,
        prefix: str,
        params: SamplingParams,
        num_continuations: int = 1,
    ) -> list[GenerationResult]:
        """Continue generation from a partial response.

        Uses vLLM's prefix caching for efficient continuation.

        Args:
            prefix: The partial response to continue from.
            params: Sampling parameters.
            num_continuations: Number of different continuations.

        Returns:
            List of GenerationResults.
        """
        # Generate multiple continuations with n > 1
        multi_params = SamplingParams(
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            stop_sequences=params.stop_sequences,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
            n=num_continuations,
            logprobs=params.logprobs,
            num_logprobs=params.num_logprobs,
        )

        vllm_params = self._to_vllm_params(multi_params)
        outputs = self._llm.generate([prefix], vllm_params, use_tqdm=False)

        results = []
        for completion in outputs[0].outputs:
            token_logprobs = self._extract_logprobs(completion)
            results.append(
                GenerationResult(
                    text=completion.text,
                    finish_reason=_convert_stop_reason(completion.finish_reason),
                    token_logprobs=token_logprobs,
                    prompt_tokens=len(outputs[0].prompt_token_ids),
                    completion_tokens=len(completion.token_ids),
                    metadata={"is_continuation": True},
                )
            )

        return results
