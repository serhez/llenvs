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
import gc
import io
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

from llenvs.core.state import ImageContent
from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    ModelBackend,
    PromptTooLongError,
    SamplingParams,
    ScoringResult,
    StopReason,
    TokenLogprob,
    TokenScore,
)

# Known VLM model types (checked against model_config.hf_config.model_type).
# `gemma4` is only recognised by vllm>=0.19 running inside the Singularity
# container (see docs/guides/singularity.md); older vllm never surfaces that
# model_type string, so listing it here is safe for both versions.
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
        "gemma4",
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
        post_shutdown_delay: float = 5.0,
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
            post_shutdown_delay: Seconds to wait after engine shutdown in
                :meth:`close` for the CUDA driver to reclaim GPU memory.
                Increase this when closing one backend and immediately
                creating another on the same GPUs.
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
        self._post_shutdown_delay = post_shutdown_delay
        self._closed = False

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
            supports_full_scoring=True,
            max_batch_size=None,  # Limited by GPU memory
            max_context_length=self._max_context_length,
        )

    @property
    def model_name(self) -> str:
        """Get the model path/identifier."""
        return self._model_path

    def _shutdown_engine(self, llm: Any) -> None:
        """Best-effort shutdown for pinned vLLM engine layouts."""
        engine = getattr(llm, "llm_engine", None)
        if engine is None:
            return

        engine_core = getattr(engine, "engine_core", None)
        if engine_core is not None:
            shutdown = getattr(engine_core, "shutdown", None)
            if callable(shutdown):
                shutdown()
                return

        model_executor = getattr(engine, "model_executor", None)
        if model_executor is not None:
            shutdown = getattr(model_executor, "shutdown", None)
            if callable(shutdown):
                shutdown()

    def close(self) -> None:
        """Release vLLM engine and cached model references."""
        if getattr(self, "_closed", False):
            return

        llm = getattr(self, "_llm", None)
        if llm is not None:
            self._shutdown_engine(llm)

        self._llm = None
        self._tokenizer = None
        if hasattr(self, "_scoring_model"):
            self._scoring_model = None

        gc.collect()
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        delay = getattr(self, "_post_shutdown_delay", 0)
        if delay > 0:
            logger.info(
                "Waiting %.0fs for GPU memory reclamation after vLLM shutdown",
                delay,
            )
            time.sleep(delay)

        self._closed = True

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
            kwargs.update(params.extra)

        # Thinking budget
        if params.thinking_budget is not None:
            if self._is_v1:
                if not self._has_v1_thinking_processor:
                    raise ValueError(
                        "thinking_budget requires the V1 AdapterLogitsProcessor API, "
                        "which could not be imported. Ensure vLLM >=0.8 is installed."
                    )
                extra_args = kwargs.get("extra_args") or {}
                extra_args["thinking_budget"] = params.thinking_budget
                if params.thinking_budget_soft_ratio is not None:
                    extra_args["thinking_budget_soft_ratio"] = params.thinking_budget_soft_ratio
                if params.thinking_budget_suffix is not None:
                    extra_args["thinking_early_stopping_text"] = params.thinking_budget_suffix
                else:
                    extra_args["thinking_early_stopping_text"] = None
                if params.thinking_budget_per_block:
                    extra_args["thinking_budget_per_block"] = True
                kwargs["extra_args"] = extra_args
            else:
                from llenvs.inference.thinking import ThinkingBudgetProcessor

                processor = ThinkingBudgetProcessor(
                    self._tokenizer,
                    params.thinking_budget,
                    soft_budget_ratio=params.thinking_budget_soft_ratio,
                    early_stopping_text=params.thinking_budget_suffix,
                    per_block=params.thinking_budget_per_block,
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

        Raises:
            PromptTooLongError: If any prompt exceeds the model's max context.
        """
        vllm_params = self._to_vllm_params(params)
        try:
            outputs = self._llm.generate(prompts, vllm_params, use_tqdm=False)
        except ValueError as exc:
            if "longer than the maximum model length" in str(exc):
                # Compute per-prompt token lengths for diagnostics
                prompt_lengths = [
                    len(self._tokenizer.encode(p, add_special_tokens=False))
                    for p in prompts
                ]
                max_len = self._max_context_length or 0
                offending = [
                    i for i, length in enumerate(prompt_lengths)
                    if length > max_len
                ]
                raise PromptTooLongError(
                    str(exc),
                    model_name=self._model_path,
                    max_model_len=max_len,
                    batch_size=len(prompts),
                    prompt_token_lengths=prompt_lengths,
                    offending_indices=offending,
                    offending_prompts=[prompts[i] for i in offending],
                ) from exc
            raise

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

    # --- Chat prefix continuation ---

    def generate_chat_with_prefix(
        self,
        messages: list[ChatMessage],
        assistant_prefix: str,
        params: SamplingParams,
    ) -> GenerationResult:
        """Generate a continuation from a partial assistant response.

        Applies the chat template, appends *assistant_prefix*, then
        delegates to ``generate()`` which treats the full string as a
        prompt and returns only newly generated text.
        """
        prompt = self._tokenizer.apply_chat_template(
            [m.to_dict() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
            **self._chat_template_kwargs,
        )
        full_prefix = prompt + assistant_prefix
        results = self.continue_from_prefix(full_prefix, params, num_continuations=1)
        return results[0]

    def generate_chat_with_prefix_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        assistant_prefixes: list[str],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        """Batched chat prefix continuation via a single vLLM generate call."""
        if not messages_batch:
            return []

        prefixes = [
            self._tokenizer.apply_chat_template(
                [m.to_dict() for m in msgs],
                tokenize=False,
                add_generation_prompt=True,
                **self._chat_template_kwargs,
            )
            + prefix
            for msgs, prefix in zip(messages_batch, assistant_prefixes)
        ]

        vllm_params = self._to_vllm_params(params)
        outputs = self._llm.generate(prefixes, vllm_params, use_tqdm=False)

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
                    metadata={"is_continuation": True},
                )
            )

        return results

    # --- Full-vocabulary scoring ---

    def _ensure_scoring_model(self) -> Any:
        """Lazily load a HuggingFace model for full-vocabulary scoring.

        The vLLM engine's internal model cannot be used directly for raw
        forward passes because its attention layers require vLLM-specific
        execution context (paged attention metadata, KV cache management).
        Instead, we load a standard HuggingFace ``AutoModelForCausalLM``
        from the same model path. The model is cached after first load.

        Returns:
            A HuggingFace ``PreTrainedModel`` in eval mode.

        Raises:
            ImportError: If ``transformers`` is not installed.
            RuntimeError: If the model cannot be loaded.
        """
        if hasattr(self, "_scoring_model"):
            return self._scoring_model

        import logging

        import torch

        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "HuggingFace transformers is required for full-vocabulary scoring. "
                "Install with: pip install transformers"
            ) from e

        logger = logging.getLogger(__name__)

        # Match the dtype used by the vLLM engine
        model_config = self._llm.llm_engine.model_config
        dtype = getattr(model_config, "dtype", torch.float16)

        logger.info(
            "Loading HuggingFace model for full-vocabulary scoring: %s (dtype=%s). "
            "This model is loaded separately from the vLLM engine and uses "
            "additional memory.",
            self._model_path,
            dtype,
        )

        try:
            self._scoring_model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            logger.info("Scoring model loaded on GPU")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load scoring model on GPU. Lower gpu_memory_utilization "
                f"on the vLLM backend to free GPU memory for the scoring model: {e}"
            ) from e

        self._scoring_model.eval()
        return self._scoring_model

    def score_chat(
        self,
        messages: list[ChatMessage],
        continuation: str,
    ) -> ScoringResult:
        """Score continuation tokens with full-vocabulary log-probabilities.

        Loads a HuggingFace model (lazily, on first call) from the same
        model path as the vLLM engine, runs a single forward pass over
        the chat-templated prompt concatenated with the continuation, and
        returns per-token full log-probability distributions.

        Args:
            messages: Chat context.
            continuation: Text whose tokens are scored.

        Returns:
            ``ScoringResult`` with per-token ``TokenScore`` entries.
        """
        import torch

        if not continuation:
            return ScoringResult(token_scores=(), prompt_tokens=0, scored_tokens=0)

        model = self._ensure_scoring_model()

        # Apply chat template
        prompt_text = self._tokenizer.apply_chat_template(
            [m.to_dict() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
            **self._chat_template_kwargs,
        )

        # Tokenize prompt and full text
        full_text = prompt_text + continuation
        prompt_ids = self._tokenizer.encode(prompt_text)
        full_ids = self._tokenizer.encode(full_text)
        prompt_len = len(prompt_ids)

        if len(full_ids) <= prompt_len:
            return ScoringResult(
                token_scores=(), prompt_tokens=prompt_len, scored_tokens=0
            )

        # Forward pass
        device = next(model.parameters()).device
        input_tensor = torch.tensor([full_ids], device=device)

        with torch.no_grad():
            logits = model(input_tensor).logits  # (1, seq_len, vocab_size)

        # Logit at position t predicts token at t+1.
        # Continuation tokens occupy positions [prompt_len, total_len-1].
        # Their predicting logits are at positions [prompt_len-1, total_len-2].
        total_len = len(full_ids)
        cont_logits = logits[0, prompt_len - 1 : total_len - 1, :]
        # Use float32 for numerical stability in log_softmax
        log_probs = torch.nn.functional.log_softmax(
            cont_logits.float(), dim=-1
        ).cpu()

        # Build TokenScore tuples
        cont_ids = full_ids[prompt_len:]
        token_scores = tuple(
            TokenScore(
                token=self._tokenizer.decode([tid]),
                token_id=tid,
                logprob=log_probs[i, tid].item(),
                log_probs_all=log_probs[i],
            )
            for i, tid in enumerate(cont_ids)
        )

        return ScoringResult(
            token_scores=token_scores,
            prompt_tokens=prompt_len,
            scored_tokens=len(token_scores),
        )

    def score_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        continuations: list[str],
    ) -> list[ScoringResult]:
        """Batched scoring via sequential forward passes.

        Each (messages, continuation) pair is scored independently.
        The self-distillation method controls its own sub-batching,
        so sequential processing here is acceptable.
        """
        return [
            self.score_chat(msgs, cont)
            for msgs, cont in zip(messages_batch, continuations)
        ]
