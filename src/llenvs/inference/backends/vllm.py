"""vLLM backend implementation.

vLLM provides full feature support including:
- Batched generation
- Log probabilities
- Prefix continuation
- Streaming
- Vision Language Model (VLM) support
"""

from __future__ import annotations

import base64
import io
from typing import Any

from llenvs.core.state import ImageContent
from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    ModelBackend,
    SamplingParams,
    StopReason,
    TokenLogprob,
)

# Known VLM model types (checked against model_config.hf_config.model_type)
_VLM_MODEL_TYPES = frozenset(
    {
        "llava",
        "llava_next",
        "llava_next_video",
        "llava_onevision",
        "qwen2_vl",
        "paligemma",
        "paligemma2",
        "internvl_chat",
        "phi3_v",
        "fuyu",
        "chameleon",
        "minicpmv",
        "mllama",
        "pixtral",
        "idefics2",
        "idefics3",
        "molmo",
        "aria",
    }
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


def _decode_image(img: ImageContent) -> Any:
    """Decode a base64-encoded ImageContent to a PIL Image.

    Args:
        img: ImageContent with base64-encoded data.

    Returns:
        PIL.Image.Image instance.
    """
    from PIL import Image

    raw = base64.b64decode(img.data)
    return Image.open(io.BytesIO(raw))


def _extract_images(messages: list[ChatMessage]) -> list[ImageContent]:
    """Extract all images from a list of ChatMessages in order.

    Args:
        messages: List of chat messages.

    Returns:
        List of ImageContent objects in conversation order.
    """
    images: list[ImageContent] = []
    for msg in messages:
        if msg.images:
            images.extend(msg.images)
    return images


def _is_vlm_model(model_config: Any) -> bool:
    """Detect whether a vLLM model config represents a VLM.

    Args:
        model_config: vLLM's ModelConfig object.

    Returns:
        True if the model is a VLM.
    """
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is not None:
        model_type = getattr(hf_config, "model_type", "")
        if model_type in _VLM_MODEL_TYPES:
            return True
    return False


class VLLMBackend(ModelBackend):
    """vLLM-based inference backend.

    Provides high-performance local inference with full feature support.
    Requires a GPU with sufficient memory for the model. Automatically
    detects VLMs and enables multimodal input via ``multi_modal_data``.

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
        chat_template_kwargs: dict[str, Any] | None = None,
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
            chat_template_kwargs: Extra keyword arguments passed to
                ``tokenizer.apply_chat_template()`` (e.g. ``enable_thinking``).
            **vllm_kwargs: Additional arguments passed to vLLM LLM.

        Raises:
            ImportError: If vLLM is not installed.
        """
        try:
            from vllm import LLM
            from vllm import SamplingParams as VLLMSamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is required for VLLMBackend. Install with: pip install vllm"
            ) from e

        self._model_path = model_path
        self._VLLMSamplingParams = VLLMSamplingParams
        self._chat_template_kwargs = chat_template_kwargs or {}

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

        # Register V1 thinking budget processor class (no-op on vLLM <0.8)
        from llenvs.inference.thinking import make_v1_thinking_processor_class

        v1_cls = make_v1_thinking_processor_class()
        if v1_cls is not None:
            existing = llm_kwargs.get("logits_processors") or []
            llm_kwargs["logits_processors"] = list(existing) + [v1_cls]

        self._llm = LLM(**llm_kwargs)
        self._tokenizer = self._llm.get_tokenizer()

        # Determine max context length
        model_config = self._llm.llm_engine.model_config
        self._max_context_length = getattr(model_config, "max_model_len", None)

        # Detect VLM
        self._is_vlm = _is_vlm_model(model_config)

        # Detect V1 engine (default since vLLM 0.8, V0 removed in 0.10)
        engine_module = type(self._llm.llm_engine).__module__ or ""
        self._is_v1 = ".v1." in engine_module or engine_module.startswith("v1.")
        self._has_v1_thinking_processor = self._is_v1 and v1_cls is not None

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
            supports_vision=self._is_vlm,
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
        """Convert to vLLM SamplingParams.

        Maps common SamplingParams fields to vLLM arguments, then merges
        any backend-specific params from `extra`.
        """
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

        # Merge backend-specific extra params (these take precedence)
        if params.extra:
            extra = dict(params.extra)
            thinking_budget = extra.pop("thinking_budget", None)
            soft_ratio = extra.pop("thinking_budget_soft_ratio", None)
            _absent = object()
            early_stopping_text = extra.pop("thinking_early_stopping_text", _absent)
            per_block = extra.pop("thinking_budget_per_block", None)
            kwargs.update(extra)

            if thinking_budget is not None:
                if self._is_v1:
                    if not self._has_v1_thinking_processor:
                        raise ValueError(
                            "thinking_budget requires the V1 AdapterLogitsProcessor API, "
                            "which could not be imported. Ensure vLLM >=0.8 is installed."
                        )
                    extra_args = kwargs.get("extra_args") or {}
                    extra_args["thinking_budget"] = int(thinking_budget)
                    if soft_ratio is not None:
                        extra_args["thinking_budget_soft_ratio"] = float(soft_ratio)
                    if early_stopping_text is not _absent:
                        extra_args["thinking_early_stopping_text"] = early_stopping_text
                    if per_block is not None:
                        extra_args["thinking_budget_per_block"] = bool(per_block)
                    kwargs["extra_args"] = extra_args
                else:
                    from llenvs.inference.thinking import ThinkingBudgetProcessor

                    proc_kwargs: dict[str, Any] = {
                        "soft_budget_ratio": float(soft_ratio) if soft_ratio is not None else None,
                        "per_block": bool(per_block) if per_block is not None else False,
                    }
                    if early_stopping_text is not _absent:
                        proc_kwargs["early_stopping_text"] = early_stopping_text
                    processor = ThinkingBudgetProcessor(
                        self._tokenizer,
                        int(thinking_budget),
                        **proc_kwargs,
                    )
                    processors = kwargs.get("logits_processors", [])
                    kwargs["logits_processors"] = list(processors) + [processor.vllm_processor]

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
                    top_logprobs = {tok: lp.logprob for tok, lp in logprob_obj.top_logprobs.items()}

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

    def _generate_vlm(
        self,
        prompts: list[str],
        params: SamplingParams,
        per_prompt_images: list[list[Any]] | None = None,
    ) -> list[GenerationResult]:
        """Generate completions with optional multi_modal_data for VLMs.

        Args:
            prompts: List of prompt strings.
            params: Sampling parameters.
            per_prompt_images: List of PIL Image lists, one per prompt.
                None or empty lists for text-only prompts.

        Returns:
            List of GenerationResults.
        """
        vllm_params = self._to_vllm_params(params)

        if per_prompt_images:
            inputs = []
            for prompt, imgs in zip(prompts, per_prompt_images):
                if imgs:
                    inputs.append(
                        {
                            "prompt": prompt,
                            "multi_modal_data": {"image": imgs if len(imgs) > 1 else imgs[0]},
                        }
                    )
                else:
                    inputs.append(prompt)
            outputs = self._llm.generate(inputs, vllm_params, use_tqdm=False)
        else:
            outputs = self._llm.generate(prompts, vllm_params, use_tqdm=False)

        results = []
        for output in outputs:
            completion = output.outputs[0]
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

    def generate_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        """Generate responses for multiple conversations in one batched call.

        Converts each conversation to a prompt string via the chat template,
        then passes all prompts to generate() for efficient GPU batching.
        For VLMs, images are extracted and passed via multi_modal_data.
        """
        if not messages_batch:
            return []

        prompts = [
            self._tokenizer.apply_chat_template(
                [m.to_dict() for m in msgs],
                tokenize=False,
                add_generation_prompt=True,
                **self._chat_template_kwargs,
            )
            for msgs in messages_batch
        ]

        if self._is_vlm:
            per_prompt_images: list[list[Any]] = []
            has_any_images = False
            for msgs in messages_batch:
                img_contents = _extract_images(msgs)
                if img_contents:
                    has_any_images = True
                    per_prompt_images.append([_decode_image(ic) for ic in img_contents])
                else:
                    per_prompt_images.append([])

            if has_any_images:
                return self._generate_vlm(prompts, params, per_prompt_images)

        return self.generate(prompts, params)

    def generate_chat(
        self,
        messages: list[ChatMessage],
        params: SamplingParams,
    ) -> GenerationResult:
        """Generate a response for a chat conversation.

        For VLMs, images are extracted from messages and passed via
        multi_modal_data to the vLLM engine.

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
            **self._chat_template_kwargs,
        )

        # For VLMs, extract and decode images
        if self._is_vlm:
            img_contents = _extract_images(messages)
            if img_contents:
                pil_images = [_decode_image(ic) for ic in img_contents]
                results = self._generate_vlm([prompt], params, [pil_images])
                return results[0]

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
