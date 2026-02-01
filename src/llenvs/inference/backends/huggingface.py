"""HuggingFace Transformers backend implementation.

A lightweight backend for local inference using the transformers library.
Provides a simpler alternative to vLLM for running HuggingFace models directly.

Features:
- Batched generation with attention masks
- Log probabilities via output_scores
- Prefix continuation
- Chat via tokenizer.apply_chat_template()
"""

from typing import Any

from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    ModelBackend,
    SamplingParams,
    StopReason,
    TokenLogprob,
)


def _convert_stop_reason(
    generated_length: int,
    max_new_tokens: int,
    eos_token_id: int | list[int] | None,
    last_token_id: int | None,
) -> StopReason:
    """Determine why generation stopped."""
    if generated_length >= max_new_tokens:
        return StopReason.MAX_TOKENS

    if eos_token_id is not None and last_token_id is not None:
        eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        if last_token_id in eos_ids:
            return StopReason.END_OF_TEXT

    return StopReason.UNKNOWN


class HuggingFaceBackend(ModelBackend):
    """HuggingFace Transformers inference backend.

    Provides local inference using the transformers library. Simpler to set up
    than vLLM but may be slower for high-throughput scenarios.

    Attributes:
        model_path: Path or HuggingFace model ID.
        device: Device to run on (cuda, mps, cpu, or auto).
        dtype: Data type for model weights.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        device_map: str | None = None,
        torch_compile: bool = False,
        **model_kwargs: Any,
    ) -> None:
        """Initialize HuggingFace backend.

        Args:
            model_path: Path to model or HuggingFace model ID.
            device: Device to run on ("cuda", "mps", "cpu", or "auto").
            dtype: Data type ("float16", "bfloat16", "float32", or "auto").
            device_map: Device map for multi-GPU or offloading (e.g., "auto").
                       If set, overrides the device parameter.
            torch_compile: Enable torch.compile() for optimization.
            **model_kwargs: Additional arguments passed to from_pretrained().

        Raises:
            ImportError: If transformers or torch is not installed.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for HuggingFaceBackend. "
                "Install with: pip install 'llenvs[transformers]'"
            ) from e

        self._model_path = model_path
        self._torch = torch

        # Resolve dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(dtype, "auto")

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = device
        self._device = resolved_device

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            **model_kwargs,
        }

        if device_map is not None:
            load_kwargs["device_map"] = device_map
        elif resolved_device != "cpu":
            load_kwargs["device_map"] = resolved_device

        self._model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        # Apply torch.compile if requested
        if torch_compile:
            self._model = torch.compile(self._model)

        # Determine max context length
        config = self._model.config
        self._max_context_length = getattr(
            config,
            "max_position_embeddings",
            getattr(config, "n_positions", None),
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        """HuggingFace backend capabilities."""
        return BackendCapabilities(
            supports_logprobs=True,
            supports_prefix_continuation=True,
            supports_batching=True,
            supports_streaming=False,
            supports_chat=True,
            supports_function_calling=False,
            max_batch_size=None,
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

    @property
    def model(self) -> Any:
        """Get the underlying model."""
        return self._model

    def _to_generate_kwargs(self, params: SamplingParams) -> dict[str, Any]:
        """Convert SamplingParams to transformers generate() kwargs."""
        kwargs: dict[str, Any] = {
            "max_new_tokens": params.max_tokens,
            "do_sample": params.temperature > 0,
            "num_return_sequences": params.n,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        if params.temperature > 0:
            kwargs["temperature"] = params.temperature
            kwargs["top_p"] = params.top_p
            if params.top_k > 0:
                kwargs["top_k"] = params.top_k

        if params.stop_sequences:
            # Convert stop sequences to token IDs
            stop_token_ids = []
            for seq in params.stop_sequences:
                tokens = self._tokenizer.encode(seq, add_special_tokens=False)
                if tokens:
                    stop_token_ids.append(tokens[0])
            if stop_token_ids:
                # Add to eos_token_id
                existing_eos = kwargs.get("eos_token_id")
                if existing_eos is None:
                    kwargs["eos_token_id"] = stop_token_ids
                elif isinstance(existing_eos, int):
                    kwargs["eos_token_id"] = [existing_eos] + stop_token_ids
                else:
                    kwargs["eos_token_id"] = list(existing_eos) + stop_token_ids

        if params.logprobs:
            kwargs["output_scores"] = True
            kwargs["return_dict_in_generate"] = True

        return kwargs

    def _extract_logprobs(
        self,
        scores: tuple[Any, ...],
        generated_ids: Any,
        num_logprobs: int,
    ) -> tuple[TokenLogprob, ...]:
        """Extract log probabilities from generation scores.

        Args:
            scores: Tuple of score tensors from generate().
            generated_ids: The generated token IDs.
            num_logprobs: Number of top logprobs to return per token.

        Returns:
            Tuple of TokenLogprob objects.
        """
        import torch.nn.functional as F

        logprobs_list = []

        for i, score in enumerate(scores):
            # score shape: (batch_size * num_return_sequences, vocab_size)
            # We take the first sequence for now
            token_scores = score[0]  # (vocab_size,)

            # Convert to log probabilities
            log_probs = F.log_softmax(token_scores, dim=-1)

            # Get the chosen token
            token_id = generated_ids[i].item()
            token_logprob = log_probs[token_id].item()
            token_str = self._tokenizer.decode([token_id])

            # Get top-k logprobs
            top_logprobs = None
            if num_logprobs > 0:
                top_values, top_indices = log_probs.topk(num_logprobs)
                top_logprobs = {
                    self._tokenizer.decode([idx.item()]): val.item()
                    for idx, val in zip(top_indices, top_values)
                }

            logprobs_list.append(
                TokenLogprob(
                    token=token_str,
                    token_id=token_id,
                    logprob=token_logprob,
                    top_logprobs=top_logprobs,
                )
            )

        return tuple(logprobs_list)

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
        # Tokenize with padding
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Move to device
        device = self._model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

        # Generate
        generate_kwargs = self._to_generate_kwargs(params)
        outputs = self._model.generate(**inputs, **generate_kwargs)

        # Handle output format based on whether we requested scores
        if params.logprobs and hasattr(outputs, "sequences"):
            sequences = outputs.sequences
            scores = outputs.scores
        else:
            sequences = outputs
            scores = None

        results = []
        for i, (seq, prompt_len) in enumerate(zip(sequences, prompt_lengths)):
            # Extract generated tokens (excluding prompt)
            generated_ids = seq[prompt_len:]
            generated_text = self._tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            # Count tokens
            completion_tokens = len(generated_ids)

            # Determine stop reason
            last_token_id = generated_ids[-1].item() if len(generated_ids) > 0 else None
            finish_reason = _convert_stop_reason(
                generated_length=completion_tokens,
                max_new_tokens=params.max_tokens,
                eos_token_id=self._tokenizer.eos_token_id,
                last_token_id=last_token_id,
            )

            # Extract logprobs if available
            token_logprobs = None
            if scores is not None:
                # For batched generation, extract scores for this sequence
                batch_scores = tuple(s[i : i + 1] for s in scores)
                token_logprobs = self._extract_logprobs(
                    batch_scores, generated_ids, params.num_logprobs
                )

            results.append(
                GenerationResult(
                    text=generated_text,
                    finish_reason=finish_reason,
                    token_logprobs=token_logprobs,
                    prompt_tokens=prompt_len,
                    completion_tokens=completion_tokens,
                    metadata={"device": str(device)},
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

        Args:
            prefix: The partial response to continue from.
            params: Sampling parameters.
            num_continuations: Number of different continuations.

        Returns:
            List of GenerationResults.
        """
        # Tokenize the prefix
        inputs = self._tokenizer(
            prefix,
            return_tensors="pt",
            truncation=True,
        )

        device = self._model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        prompt_length = inputs["input_ids"].shape[1]

        # Create params with n = num_continuations
        generate_kwargs = self._to_generate_kwargs(params)
        generate_kwargs["num_return_sequences"] = num_continuations

        # Need sampling for multiple continuations
        if num_continuations > 1 and not generate_kwargs.get("do_sample", False):
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = params.temperature if params.temperature > 0 else 0.7

        outputs = self._model.generate(**inputs, **generate_kwargs)

        if params.logprobs and hasattr(outputs, "sequences"):
            sequences = outputs.sequences
            scores = outputs.scores
        else:
            sequences = outputs
            scores = None

        results = []
        for i, seq in enumerate(sequences):
            generated_ids = seq[prompt_length:]
            generated_text = self._tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            completion_tokens = len(generated_ids)
            last_token_id = generated_ids[-1].item() if len(generated_ids) > 0 else None
            finish_reason = _convert_stop_reason(
                generated_length=completion_tokens,
                max_new_tokens=params.max_tokens,
                eos_token_id=self._tokenizer.eos_token_id,
                last_token_id=last_token_id,
            )

            token_logprobs = None
            if scores is not None:
                batch_scores = tuple(s[i : i + 1] for s in scores)
                token_logprobs = self._extract_logprobs(
                    batch_scores, generated_ids, params.num_logprobs
                )

            results.append(
                GenerationResult(
                    text=generated_text,
                    finish_reason=finish_reason,
                    token_logprobs=token_logprobs,
                    prompt_tokens=prompt_length,
                    completion_tokens=completion_tokens,
                    metadata={"is_continuation": True},
                )
            )

        return results
