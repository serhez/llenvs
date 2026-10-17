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
import uuid
from copy import deepcopy
from dataclasses import replace
from typing import Any

from llenvs.core.state import ImageContent
from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    LogprobsNotReturnedError,
    ModelBackend,
    PromptTooLongError,
    SamplingParams,
    ScoringResult,
    StopReason,
    TokenLogprob,
    TokenScore,
)
from llenvs.inference.reasoning import local_chat_messages
from llenvs.inference.sampling import DECODING_FIELDS, sampling_metadata, validate_sampling
from llenvs.inference.scoring_utils import build_scoring_inputs

logger = logging.getLogger(__name__)

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
        "qwen3_5",
        "qwen3_5_moe",
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

    _llm: Any
    _tokenizer: Any

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.9,
        seed: int = 42,
        chat_template_kwargs: dict[str, Any] | None = None,
        reasoning_family: str | None = None,
        request_progress: bool = False,
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
        self._reasoning_family = reasoning_family
        self._request_progress = request_progress
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
        if reasoning_family == "qwen3.8":
            from vllm.config import StructuredOutputsConfig  # ty: ignore[unresolved-import]

            if "structured_outputs_config" in vllm_kwargs:
                raise ValueError("The configured reasoning family owns structured_outputs_config")
            llm_kwargs["structured_outputs_config"] = StructuredOutputsConfig(
                backend="xgrammar", reasoning_parser="qwen3", enable_in_reasoning=False
            )
        self._engine_settings = deepcopy(llm_kwargs)

        # Register V1 thinking budget processor class (no-op on vLLM <0.8)
        from llenvs.inference.thinking import make_v1_thinking_processor_class

        v1_cls = make_v1_thinking_processor_class()
        if v1_cls is not None:
            existing = llm_kwargs.get("logits_processors") or []
            llm_kwargs["logits_processors"] = list(existing) + [v1_cls]

        if request_progress:
            from vllm.engine.arg_utils import AsyncEngineArgs  # ty: ignore[unresolved-import]
            from vllm.v1.engine.async_llm import AsyncLLM  # ty: ignore[unresolved-import]

            from llenvs.inference.backends.api import _AsyncRunner

            self._async_runner = _AsyncRunner()

            async def build_engine():
                return AsyncLLM.from_engine_args(AsyncEngineArgs(**llm_kwargs))

            self._llm = self._async_runner.run(build_engine())
        else:
            self._llm = LLM(**llm_kwargs)
        self._tokenizer = self._llm.get_tokenizer()

        # Determine max context length
        model_config = (
            self._llm.vllm_config.model_config
            if request_progress
            else self._llm.llm_engine.model_config
        )
        self._max_context_length = getattr(model_config, "max_model_len", None)

        # Detect VLM
        self._is_vlm = _is_vlm_model(model_config)

        # Detect V1 engine (default since vLLM 0.8, V0 removed in 0.10)
        engine_module = (
            "vllm.v1.engine" if request_progress else type(self._llm.llm_engine).__module__ or ""
        )
        self._is_v1 = ".v1." in engine_module or engine_module.startswith("v1.")
        self._has_v1_thinking_processor = self._is_v1 and v1_cls is not None

    @property
    def capabilities(self) -> BackendCapabilities:
        """vLLM supports all major features."""
        return BackendCapabilities(
            supports_explicit_sampling=True,
            supports_logprobs=True,
            supports_prefix_continuation=not getattr(self, "_request_progress", False),
            supports_batching=True,
            supports_streaming=True,
            supports_chat=True,
            supports_function_calling=False,
            supports_vision=self._is_vlm,
            supports_full_scoring=not getattr(self, "_request_progress", False),
            supports_request_progress=getattr(self, "_request_progress", False),
            supports_after_reasoning_text=getattr(self, "_reasoning_family", None) == "qwen3.8",
            supports_response_schema=getattr(self, "_reasoning_family", None) == "qwen3.8",
            supports_schema_after_reasoning=getattr(self, "_reasoning_family", None) == "qwen3.8",
            max_batch_size=None,  # Limited by GPU memory
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
            tokenizer_identity,
        )
        from llenvs.inference.sampling import _record_value

        if not hasattr(self, "_descriptor"):
            engine = self._llm if self._request_progress else self._llm.llm_engine
            config = engine.vllm_config
            revision = getattr(config.model_config.hf_config, "_commit_hash", None) or getattr(
                config.model_config, "revision", None
            )
            import torch

            engine_settings = {
                key: value
                for key, value in self._engine_settings.items()
                if key not in {"model", "tokenizer", "download_dir"}
            }
            device_count = (
                config.parallel_config.tensor_parallel_size
                * config.parallel_config.pipeline_parallel_size
            )
            self._descriptor = {
                "identity": {
                    "backend": "vllm",
                    "protocol_version": 1,
                    **model_artifact(self._model_path, revision),
                    **tokenizer_identity(self._tokenizer),
                    "engine": _record_value(engine_settings),
                    "hardware": hardware_identity(torch, device_indices=range(device_count)),
                    "effective_config": _record_value(
                        {
                            name: vars(getattr(config, name))
                            for name in (
                                "cache_config",
                                "parallel_config",
                                "scheduler_config",
                                "compilation_config",
                            )
                        }
                    ),
                    "dtype": str(config.model_config.dtype),
                    "quantization": config.model_config.quantization,
                    "request_progress": self._request_progress,
                    "runtime": distribution_identity("vllm"),
                    "dependencies": dependency_versions(
                        "vllm", "transformers", "torch", "tokenizers"
                    ),
                    "container_digest": None,
                },
                "provenance": {"model_path": self._model_path},
            }
        return deepcopy(self._descriptor)

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
            if getattr(self, "_request_progress", False):

                async def shutdown():
                    llm.shutdown()

                self._async_runner.run(shutdown())
                self._async_runner.close()
            else:
                self._shutdown_engine(llm)

        self._llm = None
        self._tokenizer = None
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
        validate_sampling(params, single_result=False)
        inherited = None
        kwargs: dict[str, Any] = {"n": params.n}
        if params.response_schema is not None:
            if getattr(self, "_reasoning_family", None) != "qwen3.8":
                raise ValueError("response_schema needs a configured structured-output runtime")
            from vllm.sampling_params import (  # ty: ignore[unresolved-import]
                StructuredOutputsParams,
            )

            kwargs["structured_outputs"] = StructuredOutputsParams(
                json=deepcopy(params.response_schema.schema)
            )
        for name in DECODING_FIELDS:
            value = getattr(params, name)
            if value is None:
                if inherited is None:
                    if self._llm is None:
                        raise RuntimeError("The vLLM backend is closed")
                    if getattr(self, "_request_progress", False):
                        inherited = self._VLLMSamplingParams(
                            **self._llm.vllm_config.model_config.get_diff_sampling_param()
                        )
                    else:
                        inherited = self._llm.get_default_sampling_params()
                value = getattr(inherited, name)
            kwargs[name] = value

        if params.stop_sequences:
            kwargs["stop"] = list(params.stop_sequences)

        if params.logprobs:
            kwargs["logprobs"] = params.num_logprobs

        # Detach nested options before adding any thinking processor arguments.
        if params.extra:
            kwargs.update(deepcopy(params.extra))

        # Thinking budget
        if params.thinking_budget is not None and params.reasoning_effort != "none":
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

    def _native_request(self, native: Any, params: SamplingParams) -> dict[str, Any]:
        names = getattr(type(native), "__struct_fields__", None)
        options = {name: getattr(native, name) for name in names} if names else vars(native)
        options = {**options, "chat_template_kwargs": self._chat_template_kwargs_for(params)}
        return sampling_metadata(params, options)

    def reparse_receipt(self, receipt, params):
        from types import SimpleNamespace

        from llenvs.inference.reasoning import extract_local_reasoning

        raw = receipt["output"]
        native = SimpleNamespace(
            token_ids=raw["token_ids"],
            logprobs=[
                {int(key): SimpleNamespace(**value) for key, value in position.items()}
                if position is not None
                else None
                for position in raw["logprobs"]
            ]
            if raw["logprobs"] is not None
            else None,
        )
        result = GenerationResult(
            text=raw["text"],
            finish_reason=_convert_stop_reason(raw["finish_reason"]),
            prompt_tokens=len(raw["prompt_token_ids"]),
            completion_tokens=len(raw["token_ids"]),
            token_logprobs=self._extract_logprobs(native),
            metadata={**receipt["metadata"], "request_id": raw["request_id"]},
        )
        if params.logprobs and not result.token_logprobs:
            raise LogprobsNotReturnedError("Native vLLM omitted requested token logprobs")
        if receipt.get("phase_name") is not None:
            from llenvs.inference.controlled import TokenPhase

            return TokenPhase(tuple(raw["token_ids"]), result)
        return extract_local_reasoning(
            result,
            raw["token_ids"],
            self._tokenizer,
            reasoning_enabled=receipt.get("split_reasoning", True)
            and getattr(self, "_reasoning_family", None) == "qwen3.8"
            and self._chat_template_kwargs_for(params).get("enable_thinking", True),
        )

    async def _generate_one_async(
        self, index, prompt, params, native, observer, *, phase_name=None, split_reasoning=True
    ):
        from llenvs.inference.backends.api import _notify

        await _notify(observer, "started", index, 0)
        try:
            final = None
            phase_options = {"reasoning_ended": True} if phase_name == "answer" else {}
            async for output in self._llm.generate(
                prompt, native, uuid.uuid4().hex, **phase_options
            ):
                if output.finished:
                    final = output
            if final is None:
                raise RuntimeError("Native async engine returned no final output")
            receipt = self._token_receipt(
                final, native, params, phase_name=phase_name, split_reasoning=split_reasoning
            )
            await _notify(observer, "received", index, receipt)
            result = self.reparse_receipt(receipt, params)
            await _notify(observer, "completed", index, result)
            return result
        except Exception as exc:
            await _notify(observer, "failed", index, exc)
            raise

    def _token_receipt(self, final, native, params, *, phase_name=None, split_reasoning=True):
        completion = final.outputs[0]
        return {
            "split_reasoning": split_reasoning,
            "phase_name": phase_name,
            "params": params,
            "output": {
                "text": completion.text,
                "token_ids": list(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "prompt_token_ids": list(final.prompt_token_ids),
                "logprobs": [
                    {
                        key: {
                            "logprob": value.logprob,
                            "decoded_token": value.decoded_token,
                            "rank": getattr(value, "rank", None),
                        }
                        for key, value in position.items()
                    }
                    if position is not None
                    else None
                    for position in completion.logprobs
                ]
                if completion.logprobs is not None
                else None,
                "request_id": final.request_id,
            },
            "metadata": self._native_request(native, params),
        }

    def _generate_async_batch(self, prompts, params, *, observer=None, split_reasoning=True):
        from llenvs.inference.backends.api import _run_concurrent

        validate_sampling(params)
        native = self._to_vllm_params(params)
        return _run_concurrent(
            lambda item: self._generate_one_async(
                item[0], item[1], params, native, observer, split_reasoning=split_reasoning
            ),
            list(enumerate(prompts)),
            max(1, len(prompts)),
            return_partial=True,
            runner=self._async_runner,
        )

    def _generate_tokens(self, ids, params, budget, *, phase_name, stop_id, schema, observer=None):
        extra = deepcopy(params.extra)
        extra["skip_special_tokens"] = False
        if stop_id is not None:
            extra["stop_token_ids"] = [stop_id]
        phase_params = replace(
            params,
            max_tokens=budget,
            after_reasoning_text=None,
            response_schema=schema,
            thinking_budget=None,
            reasoning_effort=None,
            extra=extra,
        )
        native = self._to_vllm_params(phase_params)
        if getattr(self, "_request_progress", False):
            return self._async_runner.run(
                self._generate_one_async(
                    0,
                    {"prompt_token_ids": list(ids)},
                    phase_params,
                    native,
                    observer,
                    phase_name=phase_name,
                )
            )
        if observer is not None:
            observer.started(0, 0)
        output = self._llm.generate([{"prompt_token_ids": list(ids)}], native, use_tqdm=False)[0]
        receipt = self._token_receipt(output, native, phase_params, phase_name=phase_name)
        if observer is not None:
            observer.received(0, receipt)
        result = self.reparse_token_receipt(receipt)
        if observer is not None:
            observer.completed(0, result)
        return result

    def reparse_token_receipt(self, receipt):
        return self.reparse_receipt(receipt, receipt["params"])

    def _chat_template_kwargs_for(self, params: SamplingParams) -> dict[str, Any]:
        """Return chat-template kwargs for one generation call."""
        from llenvs.inference.reasoning import local_template_kwargs

        return local_template_kwargs(
            getattr(self, "_reasoning_family", None),
            params.reasoning_effort,
            self._chat_template_kwargs,
        )

    def _extract_logprobs(self, output: Any) -> tuple[TokenLogprob, ...] | None:
        """Extract log probabilities from vLLM output."""
        if output.logprobs is None:
            return None

        if len(output.logprobs) != len(output.token_ids):
            raise ValueError("vLLM logprobs do not align with sampled token IDs")
        logprobs = []
        for token_id, candidates in zip(output.token_ids, output.logprobs, strict=True):
            if candidates is None or token_id not in candidates:
                raise ValueError("vLLM omitted a sampled token's log probability")
            chosen = candidates[token_id]
            logprobs.append(
                TokenLogprob(
                    token=getattr(chosen, "decoded_token", None) or str(token_id),
                    token_id=token_id,
                    logprob=chosen.logprob,
                    top_logprobs={
                        getattr(value, "decoded_token", None) or str(key): value.logprob
                        for key, value in candidates.items()
                    },
                )
            )
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
        validate_sampling(params)
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

            if params.logprobs and not token_logprobs:
                raise LogprobsNotReturnedError(
                    "vLLM VLM path returned no token logprobs despite "
                    f"logprobs=True for model {self._model_path!r}.",
                    backend_name="VLLMBackend",
                    model_name=self._model_path,
                )

            results.append(
                GenerationResult(
                    text=completion.text,
                    finish_reason=_convert_stop_reason(completion.finish_reason),
                    token_logprobs=token_logprobs,
                    prompt_tokens=len(output.prompt_token_ids),
                    completion_tokens=len(completion.token_ids),
                    metadata={
                        "request_id": output.request_id,
                        "raw_token_ids": list(completion.token_ids),
                        **self._native_request(vllm_params, params),
                    },
                )
            )

        if getattr(self, "_reasoning_family", None) == "qwen3.8":
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

    def generate(self, prompts, params, *, observer=None):
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

        Raises:
            PromptTooLongError: If any prompt exceeds the model's max context.
        """
        validate_sampling(params)
        if getattr(self, "_request_progress", False):
            return self._generate_async_batch(
                prompts, params, observer=observer, split_reasoning=split_reasoning
            )
        vllm_params = self._to_vllm_params(params)
        try:
            outputs = self._llm.generate(prompts, vllm_params, use_tqdm=False)
        except ValueError as exc:
            if "longer than the maximum model length" in str(exc):
                # Compute per-prompt token lengths for diagnostics
                prompt_lengths = [
                    len(self._tokenizer.encode(p, add_special_tokens=False)) for p in prompts
                ]
                max_len = self._max_context_length or 0
                offending = [i for i, length in enumerate(prompt_lengths) if length > max_len]
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

            if params.logprobs and not token_logprobs:
                raise LogprobsNotReturnedError(
                    "vLLM returned no token logprobs despite logprobs=True for "
                    f"model {self._model_path!r}.",
                    backend_name="VLLMBackend",
                    model_name=self._model_path,
                )

            results.append(
                GenerationResult(
                    text=completion.text,
                    finish_reason=_convert_stop_reason(completion.finish_reason),
                    token_logprobs=token_logprobs,
                    prompt_tokens=len(output.prompt_token_ids),
                    completion_tokens=len(completion.token_ids),
                    metadata={
                        "request_id": output.request_id,
                        "raw_token_ids": list(completion.token_ids),
                        **self._native_request(vllm_params, params),
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

        Converts each conversation to a prompt string via the chat template,
        then passes all prompts to generate() for efficient GPU batching.
        For VLMs, images are extracted and passed via multi_modal_data.
        """
        if not messages_batch:
            return []
        if params.after_reasoning_text is not None:
            if getattr(self, "_request_progress", False):
                import asyncio

                from llenvs.inference.backends.api import _notify, _run_concurrent
                from llenvs.inference.controlled import controlled_chat

                async def one(item):
                    index, messages = item
                    result = await asyncio.to_thread(
                        controlled_chat, self, messages, params, observer=observer, index=index
                    )
                    await _notify(observer, "completed", index, result)
                    return result

                return _run_concurrent(
                    one,
                    list(enumerate(messages_batch)),
                    max(1, len(messages_batch)),
                    return_partial=True,
                    runner=self._async_runner,
                )
            from llenvs.inference.controlled import controlled_chat

            results = []
            for index, messages in enumerate(messages_batch):
                result = controlled_chat(self, messages, params, observer=observer, index=index)
                if observer is not None:
                    observer.completed(index, result)
                results.append(result)
            return results

        prompts = [
            self._tokenizer.apply_chat_template(
                local_chat_messages(msgs, getattr(self, "_reasoning_family", None)),
                tokenize=False,
                add_generation_prompt=True,
                **self._chat_template_kwargs_for(params),
            )
            for msgs in messages_batch
        ]
        if getattr(self, "_request_progress", False):
            if any(
                message.images or message.content_blocks
                for messages in messages_batch
                for message in messages
            ):
                raise ValueError("Native request progress currently requires text-only prompts")
            return self._generate_async_batch(prompts, params, observer=observer)
        if observer is not None:
            raise ValueError("Per-request progress requires request_progress=True")

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

        return self._generate_rendered(prompts, params)

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
        if params.after_reasoning_text is not None:
            from llenvs.inference.controlled import controlled_chat

            return controlled_chat(self, messages, params)
        # Convert to dict format for chat template
        message_dicts = local_chat_messages(messages, getattr(self, "_reasoning_family", None))

        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True,
            **self._chat_template_kwargs_for(params),
        )

        # For VLMs, extract and decode images
        if self._is_vlm:
            img_contents = _extract_images(messages)
            if img_contents:
                pil_images = [_decode_image(ic) for ic in img_contents]
                results = self._generate_vlm([prompt], params, [pil_images])
                return results[0]

        results = self._generate_rendered([prompt], params)
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
        if getattr(self, "_request_progress", False):
            raise ValueError("Prefix continuation requires request_progress=False")
        multi_params = replace(params, n=num_continuations)

        vllm_params = self._to_vllm_params(multi_params)
        outputs = self._llm.generate([prefix], vllm_params, use_tqdm=False)

        results = []
        for completion in outputs[0].outputs:
            token_logprobs = self._extract_logprobs(completion)

            if multi_params.logprobs and not token_logprobs:
                raise LogprobsNotReturnedError(
                    "vLLM returned no token logprobs during prefix continuation "
                    f"despite logprobs=True for model {self._model_path!r}.",
                    backend_name="VLLMBackend",
                    model_name=self._model_path,
                )

            results.append(
                GenerationResult(
                    text=completion.text,
                    finish_reason=_convert_stop_reason(completion.finish_reason),
                    token_logprobs=token_logprobs,
                    prompt_tokens=len(outputs[0].prompt_token_ids),
                    completion_tokens=len(completion.token_ids),
                    metadata={
                        "is_continuation": True,
                        **self._native_request(vllm_params, multi_params),
                    },
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
        delegates to ``continue_from_prefix()`` which treats the full string as a
        prompt and returns only newly generated text.
        """
        validate_sampling(params)
        prompt = self._tokenizer.apply_chat_template(
            local_chat_messages(messages, getattr(self, "_reasoning_family", None)),
            tokenize=False,
            add_generation_prompt=True,
            **self._chat_template_kwargs_for(params),
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
        validate_sampling(params)
        if getattr(self, "_request_progress", False):
            raise ValueError("Prefix continuation requires request_progress=False")
        if not messages_batch:
            return []

        prefixes = [
            self._tokenizer.apply_chat_template(
                local_chat_messages(msgs, getattr(self, "_reasoning_family", None)),
                tokenize=False,
                add_generation_prompt=True,
                **self._chat_template_kwargs_for(params),
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

            if params.logprobs and not token_logprobs:
                raise LogprobsNotReturnedError(
                    "vLLM returned no token logprobs during batched chat prefix "
                    f"continuation despite logprobs=True for model {self._model_path!r}.",
                    backend_name="VLLMBackend",
                    model_name=self._model_path,
                )

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

    # --- Continuation scoring ---

    def _score_prompt_logprobs_to_result(
        self,
        *,
        prompt_len: int,
        full_ids: list[int],
        prompt_logprobs: Any,
    ) -> ScoringResult:
        """Extract continuation-token logprobs from vLLM prompt logprobs."""
        if len(full_ids) <= prompt_len:
            return ScoringResult(token_scores=(), prompt_tokens=prompt_len, scored_tokens=0)
        if prompt_logprobs is None:
            raise LogprobsNotReturnedError(
                "vLLM returned no prompt logprobs during continuation scoring "
                f"for model {self._model_path!r}.",
                backend_name="VLLMBackend",
                model_name=self._model_path,
            )

        token_scores: list[TokenScore] = []
        for pos in range(prompt_len, len(full_ids)):
            token_id = full_ids[pos]
            try:
                position_logprobs = prompt_logprobs[pos]
            except IndexError as exc:
                raise LogprobsNotReturnedError(
                    "vLLM returned too few prompt logprob positions during "
                    f"continuation scoring for model {self._model_path!r}.",
                    backend_name="VLLMBackend",
                    model_name=self._model_path,
                ) from exc

            if not position_logprobs or token_id not in position_logprobs:
                raise LogprobsNotReturnedError(
                    "vLLM prompt logprobs did not include the continuation token "
                    f"id {token_id} at position {pos} for model {self._model_path!r}.",
                    backend_name="VLLMBackend",
                    model_name=self._model_path,
                )
            logprob_obj = position_logprobs[token_id]
            token_scores.append(
                TokenScore(
                    token=(
                        logprob_obj.decoded_token
                        if getattr(logprob_obj, "decoded_token", None) is not None
                        else self._tokenizer.decode([token_id])
                    ),
                    token_id=token_id,
                    logprob=logprob_obj.logprob,
                )
            )

        return ScoringResult(
            token_scores=tuple(token_scores),
            prompt_tokens=prompt_len,
            scored_tokens=len(token_scores),
        )

    def score_chat(
        self,
        messages: list[ChatMessage],
        continuation: str,
    ) -> ScoringResult:
        """Score a continuation using the resident vLLM engine.

        The prompt plus continuation is submitted as a prompt-logprobs request.
        vLLM includes each prompt token's own logprob, so the continuation span
        can be scored without loading a separate HuggingFace model copy.
        """
        return self.score_chat_batch([messages], [continuation])[0]

    def score_chat_batch(
        self,
        messages_batch: list[list[ChatMessage]],
        continuations: list[str],
    ) -> list[ScoringResult]:
        """Batched continuation scoring through vLLM prompt logprobs."""
        if getattr(self, "_request_progress", False):
            raise ValueError("Continuation scoring requires request_progress=False")
        full_texts, prompt_lengths, full_token_ids, empty_results = build_scoring_inputs(
            self._tokenizer,
            self._chat_template_kwargs,
            messages_batch,
            continuations,
            reasoning_family=getattr(self, "_reasoning_family", None),
        )

        if not full_texts:
            return [empty_results[i] for i in range(len(messages_batch))]

        params = self._VLLMSamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            prompt_logprobs=0,
            detokenize=False,
        )
        try:
            outputs = self._llm.generate(full_texts, params, use_tqdm=False)
        except ValueError as exc:
            if "longer than the maximum model length" in str(exc):
                max_len = self._max_context_length or 0
                lengths = [len(ids) for ids in full_token_ids]
                offending = [i for i, length in enumerate(lengths) if length > max_len]
                raise PromptTooLongError(
                    str(exc),
                    model_name=self._model_path,
                    max_model_len=max_len,
                    batch_size=len(full_texts),
                    prompt_token_lengths=lengths,
                    offending_indices=offending,
                    offending_prompts=[full_texts[i] for i in offending],
                ) from exc
            raise

        scored_iter = iter(
            self._score_prompt_logprobs_to_result(
                prompt_len=prompt_len,
                full_ids=full_ids,
                prompt_logprobs=output.prompt_logprobs,
            )
            for output, prompt_len, full_ids in zip(outputs, prompt_lengths, full_token_ids)
        )

        results: list[ScoringResult] = []
        for index in range(len(messages_batch)):
            if index in empty_results:
                results.append(empty_results[index])
            else:
                results.append(next(scored_iter))
        return results
