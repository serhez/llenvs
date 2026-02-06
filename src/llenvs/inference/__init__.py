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
    PromptTemplateTransformer,
    build_standard_pipeline,
)
from llenvs.inference.prompts import (
    PromptFragment,
    SystemPrompt,
    PromptTemplate,
    ModelProfile,
    compose_system_prompt,
    resolve_system_prompt,
    detect_model_profile,
    resolve_prompt_config,
    FRAGMENT_REGISTRY,
    SYSTEM_PROMPT_REGISTRY,
    TEMPLATE_REGISTRY,
    PROFILE_REGISTRY,
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
    "PromptTemplateTransformer",
    "build_standard_pipeline",
    # Prompt library
    "PromptFragment",
    "SystemPrompt",
    "PromptTemplate",
    "ModelProfile",
    "compose_system_prompt",
    "resolve_system_prompt",
    "detect_model_profile",
    "resolve_prompt_config",
    "FRAGMENT_REGISTRY",
    "SYSTEM_PROMPT_REGISTRY",
    "TEMPLATE_REGISTRY",
    "PROFILE_REGISTRY",
]
