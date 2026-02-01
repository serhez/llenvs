"""Inference layer for model backends and generation."""

from llenvs.inference.protocol import (
    ModelBackend,
    BackendCapabilities,
    SamplingParams,
    GenerationResult,
    ChatMessage,
    StopReason,
    TokenLogprob,
)
from llenvs.inference.prompting import (
    PromptTransformer,
    PromptPipeline,
    SystemPromptInjector,
    FewShotInjector,
    ChainOfThoughtWrapper,
    AnswerFormatInjector,
    MessageTrimmer,
    RoleMapper,
    ContentWrapper,
    build_standard_pipeline,
)

__all__ = [
    # Protocol
    "ModelBackend",
    "BackendCapabilities",
    "SamplingParams",
    "GenerationResult",
    "ChatMessage",
    "StopReason",
    "TokenLogprob",
    # Prompting
    "PromptTransformer",
    "PromptPipeline",
    "SystemPromptInjector",
    "FewShotInjector",
    "ChainOfThoughtWrapper",
    "AnswerFormatInjector",
    "MessageTrimmer",
    "RoleMapper",
    "ContentWrapper",
    "build_standard_pipeline",
]
