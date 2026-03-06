"""Inference layer for model backends and generation."""

from llenvs.inference.prompting import (
    AnswerFormatInjector,
    ChainOfThoughtWrapper,
    ContentWrapper,
    FewShotInjector,
    MessageTrimmer,
    PromptPipeline,
    PromptTemplateTransformer,
    PromptTransformer,
    RoleMapper,
    SystemPromptInjector,
    build_standard_pipeline,
)
from llenvs.inference.prompts import (
    FRAGMENT_REGISTRY,
    PROFILE_REGISTRY,
    SYSTEM_PROMPT_REGISTRY,
    TEMPLATE_REGISTRY,
    ModelProfile,
    PromptFragment,
    PromptTemplate,
    SystemPrompt,
    compose_system_prompt,
    detect_model_profile,
    resolve_prompt_config,
    resolve_system_prompt,
)
from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    ModelBackend,
    SamplingParams,
    StopReason,
    TokenLogprob,
)
from llenvs.inference.thinking import DEFAULT_EARLY_STOPPING_SUFFIX

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
    # Thinking
    "DEFAULT_EARLY_STOPPING_SUFFIX",
]
