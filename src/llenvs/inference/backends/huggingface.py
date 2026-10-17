"""HuggingFace Transformers backend implementation.

A lightweight backend for local inference using the transformers library.
Provides a simpler alternative to vLLM for running HuggingFace models directly.

Features:
- Batched generation with attention masks
- Log probabilities via output_scores
- Prefix continuation
- Chat via tokenizer.apply_chat_template()
- Vision Language Model (VLM) support via AutoProcessor
"""

from __future__ import annotations

import base64
import gc
import io
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

from llenvs.core.state import ImageContent
from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    LogprobsNotReturnedError,
    ModelBackend,
    SamplingParams,
    StopReason,
    TokenLogprob,
)
from llenvs.inference.reasoning import local_chat_messages
from llenvs.inference.sampling import sampling_metadata, validate_sampling

# Known VLM model types for HuggingFace detection
_VLM_MODEL_TYPES = frozenset(
    {
        "llava",
        "llava_next",
        "llava_onevision",
        "qwen2_vl",
        "qwen3_5",
        "qwen3_5_moe",
        "paligemma",
        "paligemma2",
        "internvl_chat",
        "phi3_v",
        "fuyu",
        "idefics2",
        "idefics3",
        "mllama",
        "pixtral",
        "molmo",
    }
)


def _is_vlm_model(config: Any) -> bool:
    """Detect whether a model config represents a VLM.

    Args:
        config: HuggingFace model config object.

    Returns:
        True if the model is a VLM.
    """
    model_type = getattr(config, "model_type", "")
    return model_type in _VLM_MODEL_TYPES


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


def _convert_stop_reason(
    generated_length: int,
    max_new_tokens: int | None,
    eos_token_id: int | list[int] | None,
    last_token_id: int | None,
) -> StopReason:
    """Determine why generation stopped."""
    if eos_token_id is not None and last_token_id is not None:
        eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        if last_token_id in eos_ids:
            return StopReason.END_OF_TEXT

    if max_new_tokens is not None and generated_length >= max_new_tokens:
        return StopReason.MAX_TOKENS

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

    _model: Any
    _tokenizer: Any
    _processor: Any
    _torch: Any

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        device_map: str | None = None,
        torch_compile: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
        reasoning_family: str | None = None,
        revision: str | None = None,
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
            chat_template_kwargs: Extra keyword arguments passed to
                ``tokenizer.apply_chat_template()`` (e.g. ``enable_thinking``).
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
        self._chat_template_kwargs = chat_template_kwargs or {}
        self._reasoning_family = reasoning_family
        self._revision = revision
        self._closed = False

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

        from transformers import AutoConfig

        revision_kwargs = {"revision": revision} if revision is not None else {}
        model_config = AutoConfig.from_pretrained(model_path, **revision_kwargs)
        resolved_revision = getattr(model_config, "_commit_hash", None)
        if isinstance(resolved_revision, str):
            self._revision = resolved_revision
            revision_kwargs = {"revision": resolved_revision}
        # All artifact loads share the configuration's resolved snapshot.
        tokenizer: Any = AutoTokenizer.from_pretrained(model_path, **revision_kwargs)
        self._tokenizer = tokenizer
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        # Load model
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            **revision_kwargs,
            **model_kwargs,
        }

        if device_map is not None:
            load_kwargs["device_map"] = device_map
        elif resolved_device != "cpu":
            load_kwargs["device_map"] = resolved_device

        # Detect VLM before loading model
        self._is_vlm = _is_vlm_model(model_config)

        if self._is_vlm:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(model_path, **revision_kwargs)
            if model_config.model_type in {"qwen3_5", "qwen3_5_moe"}:
                from transformers import AutoModelForImageTextToText

                self._model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
            else:
                from transformers import AutoModelForVision2Seq

                self._model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
        else:
            self._processor = None
            self._model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        # Apply torch.compile if requested
        if torch_compile:
            from typing import cast

            self._model = cast(Any, torch.compile(self._model))

        # Determine max context length
        config = self._model.config
        self._max_context_length = getattr(
            config,
            "max_position_embeddings",
            getattr(config, "n_positions", None),
        )
        self._load_settings = {
            "dtype": str(self._model.dtype),
            "device_map": device_map,
            "torch_compile": torch_compile,
        }

    @property
    def capabilities(self) -> BackendCapabilities:
        """HuggingFace backend capabilities."""
        return BackendCapabilities(
            supports_explicit_sampling=True,
            supports_logprobs=True,
            supports_prefix_continuation=True,
            supports_batching=True,
            supports_streaming=False,
            supports_chat=True,
            supports_function_calling=False,
            supports_vision=self._is_vlm,
            supports_after_reasoning_text=getattr(self, "_reasoning_family", None) == "qwen3.8",
            max_batch_size=None,
            max_context_length=self._max_context_length,
        )

    @property
    def model_name(self) -> str:
        """Get the model path/identifier."""
        return self._model_path

    def descriptor(self):
        from llenvs.inference.records import (
            dependency_versions,
            distribution_identity,
            hardware_identity,
            model_artifact,
            processor_identity,
            tokenizer_identity,
        )

        if not hasattr(self, "_descriptor"):
            revision = getattr(self._model.config, "_commit_hash", None) or self._revision
            model_config = self._model.config.to_dict()
            model_config.pop("_name_or_path", None)
            device = self._torch.device(self._model.device)
            device_map = getattr(self._model, "hf_device_map", {})
            indices = sorted(
                {
                    int(str(value).rsplit(":", 1)[-1])
                    for value in device_map.values()
                    if isinstance(value, int) or str(value).startswith("cuda:")
                }
            )
            self._descriptor = {
                "identity": {
                    "backend": "huggingface",
                    "protocol_version": 1,
                    **model_artifact(self._model_path, revision),
                    **tokenizer_identity(self._tokenizer),
                    **processor_identity(self._processor),
                    "hardware": hardware_identity(
                        self._torch,
                        device_type="cuda" if indices else device.type,
                        device_indices=indices or [device.index or 0],
                    ),
                    **self._load_settings,
                    "model_config": model_config,
                    "chat_template_kwargs": deepcopy(self._chat_template_kwargs),
                    "reasoning_family": self._reasoning_family,
                    "runtime": distribution_identity("transformers"),
                    "dependencies": dependency_versions("transformers", "torch", "tokenizers"),
                    "container_digest": None,
                },
                "provenance": {"model_path": self._model_path, "device": self._device},
            }
        return deepcopy(self._descriptor)

    def close(self) -> None:
        """Release model and tokenizer resources."""
        if getattr(self, "_closed", False):
            return

        self._model = None
        self._tokenizer = None
        self._processor = None
        gc.collect()

        torch = getattr(self, "_torch", None)
        if torch is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        self._closed = True

    @property
    def tokenizer(self) -> Any:
        """Get the underlying tokenizer."""
        return self._tokenizer

    @property
    def model(self) -> Any:
        """Get the underlying model."""
        return self._model

    def _to_generate_kwargs(self, params: SamplingParams) -> dict[str, Any]:
        """Convert SamplingParams to transformers generate() kwargs.

        Maps common SamplingParams fields to HuggingFace generate() arguments,
        then merges any backend-specific params from `extra`.
        """
        if params.response_schema is not None:
            raise ValueError("HuggingFaceBackend has no native response_schema guarantee")
        validate_sampling(
            params,
            unsupported={"presence_penalty": 0.0, "frequency_penalty": 0.0},
            single_result=False,
        )
        config = deepcopy(getattr(self._model, "generation_config"))
        do_sample = params.temperature > 0 if params.temperature is not None else config.do_sample
        kwargs: dict[str, Any] = {
            "generation_config": config,
            "num_return_sequences": params.n,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": config.eos_token_id
            if isinstance(config.eos_token_id, (int, list))
            else self._tokenizer.eos_token_id,
        }
        if params.max_tokens is not None:
            kwargs["max_new_tokens"] = params.max_tokens
        if params.temperature is not None:
            kwargs["do_sample"] = do_sample
        for name in ("top_p", "top_k", "min_p", "repetition_penalty"):
            value = getattr(params, name)
            if value is not None:
                setattr(config, name, value)

        if do_sample:
            for name in ("temperature", "top_p", "top_k", "min_p"):
                value = getattr(params, name)
                if value is not None:
                    kwargs[name] = value
        if params.repetition_penalty is not None:
            kwargs["repetition_penalty"] = params.repetition_penalty

        if params.stop_sequences:
            kwargs["stop_strings"] = list(params.stop_sequences)
            kwargs["tokenizer"] = self._tokenizer

        if params.logprobs:
            kwargs["output_scores"] = True
            kwargs["return_dict_in_generate"] = True

        # Owned keys were validated before merging backend-specific options.
        if params.extra:
            kwargs.update(params.extra)

        # Thinking budget
        if params.thinking_budget is not None and params.reasoning_effort != "none":
            from llenvs.inference.thinking import ThinkingBudgetProcessor

            processor = ThinkingBudgetProcessor(
                self._tokenizer,
                params.thinking_budget,
                soft_budget_ratio=params.thinking_budget_soft_ratio,
                early_stopping_text=params.thinking_budget_suffix,
                per_block=params.thinking_budget_per_block,
            )
            processors = kwargs.get("logits_processor", [])
            kwargs["logits_processor"] = list(processors) + [processor.hf_processor]

        return kwargs

    @staticmethod
    def _generation_stopping(kwargs, prompt_length):
        """Retain each row's native string-stop boundary before padding."""
        if not kwargs.get("stop_strings"):
            return kwargs, None
        from transformers import StoppingCriteriaList, StopStringCriteria

        class RecordedStop(StopStringCriteria):
            lengths = None

            def __call__(self, input_ids, scores, **options):
                matched = super().__call__(input_ids, scores, **options)
                if self.lengths is None:
                    self.lengths = [None] * len(input_ids)
                for index, stopped in enumerate(matched.tolist()):
                    if stopped and self.lengths[index] is None:
                        self.lengths[index] = input_ids.shape[-1] - prompt_length
                return matched

        runtime = dict(kwargs)
        tracker = RecordedStop(runtime.pop("tokenizer"), runtime.pop("stop_strings"))
        runtime["stopping_criteria"] = StoppingCriteriaList(
            [*runtime.get("stopping_criteria", []), tracker]
        )
        return runtime, tracker

    @staticmethod
    def _sampled_ids(ids, eos, limit=None):
        """Keep a sampled EOS, excluding only subsequent native batch padding."""
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        length = len(ids) if limit is None else limit
        for index, token in enumerate(ids[:length].tolist()):
            if token in eos:
                length = index + 1
                break
        return ids[:length]

    def _native_request(self, kwargs: dict[str, Any], params: SamplingParams) -> dict[str, Any]:
        config = kwargs["generation_config"].to_dict()
        options = dict(config) if isinstance(config, dict) else {}
        options.update({key: value for key, value in kwargs.items() if key != "generation_config"})
        options["chat_template_kwargs"] = self._chat_template_kwargs_for(params)
        return sampling_metadata(params, options, omitted=("presence_penalty", "frequency_penalty"))

    @staticmethod
    def _effective_max_tokens(kwargs: dict[str, Any], input_length: int) -> int | None:
        config = kwargs["generation_config"]
        limit = kwargs.get("max_new_tokens", config.max_new_tokens)
        if isinstance(limit, int):
            return limit
        return config.max_length - input_length if isinstance(config.max_length, int) else None

    def _generate_tokens(self, ids, params, budget, *, phase_name, stop_id, schema, observer=None):
        phase_params = replace(
            params,
            max_tokens=budget,
            after_reasoning_text=None,
            response_schema=schema,
            thinking_budget=None,
            reasoning_effort=None,
        )
        kwargs = self._to_generate_kwargs(phase_params)
        if stop_id is not None:
            eos = kwargs.get("eos_token_id")
            kwargs["eos_token_id"] = list(
                dict.fromkeys(([eos] if isinstance(eos, int) else list(eos or [])) + [stop_id])
            )
        inputs = self._torch.tensor([list(ids)], device=self._model.device)
        if observer is not None:
            observer.started(0, 0)
        with self._torch.inference_mode():
            output = self._model.generate(
                input_ids=inputs, attention_mask=self._torch.ones_like(inputs), **kwargs
            )
        sequences = output.sequences if params.logprobs else output
        generated = sequences[0, len(ids) :]
        sampled = tuple(generated.tolist())
        logprobs = (
            self._extract_logprobs(output.scores, generated, params.num_logprobs)
            if params.logprobs
            else None
        )
        reason = _convert_stop_reason(
            len(sampled), budget, kwargs.get("eos_token_id"), sampled[-1] if sampled else None
        )
        if sampled and stop_id is not None and sampled[-1] == stop_id:
            reason = StopReason.STOP_SEQUENCE
        receipt = {
            "token_ids": sampled,
            "token_logprobs": [asdict(lp) for lp in logprobs or ()],
            "logprobs_requested": params.logprobs,
            "finish_reason": reason.name,
            "prompt_tokens": len(ids),
            "metadata": {
                **self._native_request(kwargs, phase_params),
                "phase": phase_name,
                "prompt_token_ids": list(ids),
            },
        }
        if observer is not None:
            observer.received(0, receipt)
        result = self.reparse_token_receipt(receipt)
        if observer is not None:
            observer.completed(0, result)
        return result

    def reparse_token_receipt(self, receipt):
        from llenvs.inference.controlled import TokenPhase

        sampled = tuple(receipt["token_ids"])
        return TokenPhase(
            sampled,
            GenerationResult(
                text=self._tokenizer.decode(list(sampled), skip_special_tokens=False),
                finish_reason=StopReason[receipt["finish_reason"]],
                token_logprobs=tuple(TokenLogprob(**entry) for entry in receipt["token_logprobs"])
                if receipt["logprobs_requested"]
                else None,
                prompt_tokens=receipt["prompt_tokens"],
                completion_tokens=len(sampled),
                metadata=receipt["metadata"],
            ),
        )

    def _chat_template_kwargs_for(self, params: SamplingParams) -> dict[str, Any]:
        """Return chat-template kwargs for one generation call."""
        from llenvs.inference.reasoning import local_template_kwargs

        return local_template_kwargs(
            getattr(self, "_reasoning_family", None),
            params.reasoning_effort,
            self._chat_template_kwargs,
        )

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
        import torch.nn.functional as F  # noqa: N812

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

    def generate(self, prompts, params, *, observer=None):
        if observer is not None:
            raise ValueError("HuggingFace native batches do not expose per-request progress")
        if params.reasoning_effort is not None:
            raise ValueError("Local reasoning_effort requires generate_chat and its chat template")
        if params.after_reasoning_text is not None:
            raise ValueError("after_reasoning_text requires generate_chat")
        return self._generate_rendered(prompts, params, split_reasoning=False, observer=observer)

    def _generate_rendered(
        self,
        prompts: list[str],
        params: SamplingParams,
        *,
        split_reasoning=True,
        observer=None,
    ) -> list[GenerationResult]:
        """Generate completions for text prompts.

        Args:
            prompts: List of prompt strings.
            params: Sampling parameters.

        Returns:
            List of GenerationResults.
        """
        validate_sampling(params, unsupported={"presence_penalty": 0.0, "frequency_penalty": 0.0})
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
        # Padded input length — the correct slice point for extracting
        # generated tokens, since generate() output includes padding.
        input_length = inputs["input_ids"].shape[1]

        # Generate
        generate_kwargs = self._to_generate_kwargs(params)
        runtime_kwargs, string_stop = self._generation_stopping(generate_kwargs, input_length)
        outputs = self._model.generate(**inputs, **runtime_kwargs)

        # Handle output format based on whether we requested scores
        if params.logprobs and hasattr(outputs, "sequences"):
            sequences = outputs.sequences
            scores = outputs.scores
        else:
            sequences = outputs
            scores = None

        vocab_size = len(self._tokenizer)

        results = []
        for i, (seq, prompt_len) in enumerate(zip(sequences, prompt_lengths)):
            # Extract generated tokens (excluding full padded input)
            stop_length = (
                string_stop.lengths[i]
                if string_stop is not None and string_stop.lengths is not None
                else None
            )
            generated_ids = self._sampled_ids(
                seq[input_length:], generate_kwargs.get("eos_token_id"), stop_length
            )
            # Filter out invalid token IDs (e.g. negative values from MPS
            # numerical issues) before passing to the Rust tokenizer which
            # expects u32 values.
            id_list = [t for t in generated_ids.tolist() if 0 <= t < vocab_size]
            generated_text = self._tokenizer.decode(id_list, skip_special_tokens=True)

            # Count tokens
            completion_tokens = len(generated_ids)

            # Determine stop reason
            last_token_id = generated_ids[-1].item() if len(generated_ids) > 0 else None
            finish_reason = _convert_stop_reason(
                generated_length=completion_tokens,
                max_new_tokens=self._effective_max_tokens(generate_kwargs, input_length),
                eos_token_id=generate_kwargs.get("eos_token_id"),
                last_token_id=last_token_id,
            )
            if stop_length == completion_tokens:
                finish_reason = StopReason.STOP_SEQUENCE

            # Extract logprobs if available
            token_logprobs = None
            if scores is not None:
                # For batched generation, extract scores for this sequence
                batch_scores = tuple(s[i : i + 1] for s in scores[:completion_tokens])
                token_logprobs = self._extract_logprobs(
                    batch_scores, generated_ids, params.num_logprobs
                )

            if params.logprobs and not token_logprobs:
                raise LogprobsNotReturnedError(
                    "HuggingFace backend returned no token logprobs despite "
                    f"logprobs=True for model {self._model_path!r}.",
                    backend_name="HuggingFaceBackend",
                    model_name=self._model_path,
                )

            results.append(
                GenerationResult(
                    text=generated_text,
                    finish_reason=finish_reason,
                    token_logprobs=token_logprobs,
                    prompt_tokens=prompt_len,
                    completion_tokens=completion_tokens,
                    metadata={
                        "device": str(device),
                        "raw_token_ids": generated_ids.tolist(),
                        **self._native_request(generate_kwargs, params),
                    },
                )
            )

        if split_reasoning and getattr(self, "_reasoning_family", None) == "qwen3.8":
            from llenvs.inference.reasoning import extract_local_reasoning

            return [
                extract_local_reasoning(
                    result,
                    result.metadata["raw_token_ids"],
                    self._tokenizer,
                    reasoning_enabled=self._chat_template_kwargs_for(params).get(
                        "enable_thinking", True
                    ),
                )
                for result in results
            ]
        return results

    def generate_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        params: SamplingParams,
        *,
        observer=None,
    ) -> list[GenerationResult]:
        """Generate responses for multiple conversations in one batched call.

        Converts each conversation to a prompt string via the chat template
        (or fallback formatter), then passes all prompts to generate() for
        efficient batched GPU inference.
        """
        if not messages_batch:
            return []
        if params.after_reasoning_text is not None:
            from llenvs.inference.controlled import controlled_chat

            results = []
            for index, messages in enumerate(messages_batch):
                result = controlled_chat(self, messages, params, observer=observer, index=index)
                if observer is not None:
                    observer.completed(index, result)
                results.append(result)
            return results
        if observer is not None:
            raise ValueError("HuggingFace native batches do not expose per-request progress")
        if self._tokenizer.chat_template is not None:
            prompts = [
                self._tokenizer.apply_chat_template(
                    local_chat_messages(msgs, getattr(self, "_reasoning_family", None)),
                    tokenize=False,
                    add_generation_prompt=True,
                    **self._chat_template_kwargs_for(params),
                )
                for msgs in messages_batch
            ]
        else:
            prompts = [self._format_messages_fallback(msgs) for msgs in messages_batch]
        return self._generate_rendered(prompts, params)

    def _format_messages_fallback(self, messages: list[ChatMessage]) -> str:
        """Format messages for models without a chat template.

        Uses a simple format that works reasonably with base models.

        Args:
            messages: List of chat messages.

        Returns:
            Formatted prompt string.
        """
        parts = []
        for msg in messages:
            role = msg.role.capitalize()
            parts.append(f"{role}: {msg.content}")
        # Add prompt for assistant response
        parts.append("Assistant:")
        return "\n\n".join(parts)

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
        if params.after_reasoning_text is not None:
            from llenvs.inference.controlled import controlled_chat

            return controlled_chat(self, messages, params)
        # Convert to dict format for chat template
        message_dicts = local_chat_messages(messages, getattr(self, "_reasoning_family", None))

        # Apply chat template if available, otherwise use fallback
        if self._tokenizer.chat_template is not None:
            prompt = self._tokenizer.apply_chat_template(
                message_dicts,
                tokenize=False,
                add_generation_prompt=True,
                **self._chat_template_kwargs_for(params),
            )
        else:
            # Fallback for base models without chat templates (e.g., GPT-2)
            prompt = self._format_messages_fallback(messages)

        results = self._generate_rendered([prompt], params)
        return results[0]

    def generate_chat_with_prefix(
        self,
        messages: list[ChatMessage],
        assistant_prefix: str,
        params: SamplingParams,
    ) -> GenerationResult:
        """Generate a continuation from a partial assistant response."""
        validate_sampling(params)
        message_dicts = local_chat_messages(messages, getattr(self, "_reasoning_family", None))
        if self._tokenizer.chat_template is not None:
            prompt = self._tokenizer.apply_chat_template(
                message_dicts,
                tokenize=False,
                add_generation_prompt=True,
                **self._chat_template_kwargs_for(params),
            )
        else:
            prompt = self._format_messages_fallback(messages)
        full_prefix = prompt + assistant_prefix
        results = self.continue_from_prefix(full_prefix, params, num_continuations=1)
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
        validate_sampling(params)
        if type(num_continuations) is not int or num_continuations < 1:
            raise ValueError("num_continuations must be a positive integer")
        results = self._generate_rendered(
            [prefix] * num_continuations, params, split_reasoning=False
        )
        return [
            replace(result, metadata={**result.metadata, "is_continuation": True})
            for result in results
        ]
