"""Prompt library - fragments, system prompts, templates, model profiles.

Provides composable building blocks for prompt engineering:
- PromptFragment: Reusable instruction snippets
- SystemPrompt: Complete system prompts composed from fragments
- PromptTemplate: Wrappers for task questions
- ModelProfile: Model-family-specific prompting adjustments
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llenvs.inference.prompting import PromptTransformer


# ============================================================================
# PromptFragment
# ============================================================================


@dataclass(frozen=True)
class PromptFragment:
    """A composable prompt building block.

    Attributes:
        name: Registry key (e.g., "think_step_by_step").
        content: The actual instruction text.
        category: Grouping: "reasoning", "format", "persona", "constraint".
    """

    name: str
    content: str
    category: str


FRAGMENT_REGISTRY: dict[str, PromptFragment] = {}


def _register_fragment(name: str, content: str, category: str) -> PromptFragment:
    frag = PromptFragment(name=name, content=content, category=category)
    FRAGMENT_REGISTRY[name] = frag
    return frag


# -- Reasoning fragments --
_register_fragment(
    "think_step_by_step",
    "Think through this step by step before giving your final answer.",
    "reasoning",
)
_register_fragment(
    "show_your_work",
    "Show your work and reasoning before providing the answer.",
    "reasoning",
)
_register_fragment(
    "reflect_then_answer",
    "Consider the problem carefully, reflect on your approach, then provide your answer.",
    "reasoning",
)
_register_fragment(
    "verify_your_answer",
    "After finding your answer, verify it by checking your work.",
    "reasoning",
)
_register_fragment(
    "direct_answer",
    "Provide only the final answer with no reasoning or explanation.",
    "reasoning",
)

# -- Format fragments --
_register_fragment(
    "xml_answer",
    "Put your final answer in <answer>...</answer> tags.",
    "format",
)
_register_fragment(
    "boxed_answer",
    r"Put your final answer in \boxed{}.",
    "format",
)
_register_fragment(
    "gsm8k_answer",
    "End your response with '#### ' followed by your numerical answer.",
    "format",
)
_register_fragment(
    "json_answer",
    "Provide your answer as a JSON object with an 'answer' field.",
    "format",
)

# -- Persona fragments --
_register_fragment(
    "math_expert",
    "You are an expert mathematician. Solve problems rigorously and precisely.",
    "persona",
)
_register_fragment(
    "coding_expert",
    "You are an expert programmer. Write clean, correct, and efficient code.",
    "persona",
)
_register_fragment(
    "general_assistant",
    "You are a helpful assistant.",
    "persona",
)

# -- Constraint fragments --
_register_fragment(
    "be_concise",
    "Be concise and avoid unnecessary verbosity.",
    "constraint",
)
_register_fragment(
    "no_apologies",
    "Do not apologize or hedge. Be direct and confident.",
    "constraint",
)
_register_fragment(
    "single_answer",
    "Provide exactly one answer. Do not offer alternatives.",
    "constraint",
)


# ============================================================================
# SystemPrompt
# ============================================================================


@dataclass(frozen=True)
class SystemPrompt:
    """A complete system prompt, optionally composed from fragments.

    Attributes:
        name: Registry key (e.g., "math_reasoning"), or None for ad-hoc.
        content: The full system prompt text.
    """

    name: str | None
    content: str


SYSTEM_PROMPT_REGISTRY: dict[str, SystemPrompt] = {}


def _register_system_prompt(name: str, *fragment_names: str) -> SystemPrompt:
    """Build a SystemPrompt by joining registered fragments."""
    parts = []
    for fn in fragment_names:
        if fn not in FRAGMENT_REGISTRY:
            raise KeyError(f"Unknown fragment: {fn}")
        parts.append(FRAGMENT_REGISTRY[fn].content)
    content = "\n\n".join(parts)
    sp = SystemPrompt(name=name, content=content)
    SYSTEM_PROMPT_REGISTRY[name] = sp
    return sp


_register_system_prompt("general_reasoning", "general_assistant", "think_step_by_step", "xml_answer")
_register_system_prompt("math_reasoning", "math_expert", "show_your_work", "verify_your_answer", "xml_answer")
_register_system_prompt("math_boxed", "math_expert", "show_your_work", "boxed_answer")
_register_system_prompt("math_gsm8k", "general_assistant", "show_your_work", "gsm8k_answer")
_register_system_prompt("coding_problem", "coding_expert", "think_step_by_step", "xml_answer")
_register_system_prompt(
    "concise_reasoning",
    "general_assistant",
    "think_step_by_step",
    "be_concise",
    "single_answer",
    "xml_answer",
)
_register_system_prompt("direct_answer", "general_assistant", "direct_answer", "xml_answer")


# ============================================================================
# Composition API
# ============================================================================


def compose_system_prompt(
    *parts: str | PromptFragment | SystemPrompt,
    separator: str = "\n\n",
) -> str:
    """Join fragments/strings/prompts into a single system prompt string.

    Args:
        *parts: Strings, PromptFragments, or SystemPrompts to join.
        separator: Separator between parts.

    Returns:
        Combined system prompt text.
    """
    texts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, PromptFragment):
            texts.append(part.content)
        elif isinstance(part, SystemPrompt):
            texts.append(part.content)
        else:
            texts.append(str(part))
    return separator.join(texts)


def resolve_system_prompt(value: str | list[str]) -> str:
    """Resolve a config value to final system prompt text.

    Resolution order for each string item:
    1. Look up in SYSTEM_PROMPT_REGISTRY
    2. Look up in FRAGMENT_REGISTRY
    3. Treat as literal text

    Args:
        value: A string (single lookup) or list of strings (resolve each, join).

    Returns:
        Resolved system prompt text.
    """
    if isinstance(value, list):
        resolved = [_resolve_single(item) for item in value]
        return "\n\n".join(resolved)
    return _resolve_single(value)


def _resolve_single(name: str) -> str:
    """Resolve a single name to text."""
    if name in SYSTEM_PROMPT_REGISTRY:
        return SYSTEM_PROMPT_REGISTRY[name].content
    if name in FRAGMENT_REGISTRY:
        return FRAGMENT_REGISTRY[name].content
    return name


# ============================================================================
# PromptTemplate
# ============================================================================


@dataclass(frozen=True)
class PromptTemplate:
    """Template for wrapping task questions.

    The template is a format string with a {question} placeholder.

    Attributes:
        template: Format string with {question} placeholder.
        name: Optional registry name.
    """

    template: str
    name: str | None = None

    def apply(self, question: str, **kwargs: str) -> str:
        """Apply the template to a question.

        Args:
            question: The question text to wrap.
            **kwargs: Additional format parameters.

        Returns:
            Formatted string.
        """
        return self.template.format(question=question, **kwargs)


TEMPLATE_REGISTRY: dict[str, PromptTemplate] = {}


def _register_template(name: str, template: str) -> PromptTemplate:
    t = PromptTemplate(template=template, name=name)
    TEMPLATE_REGISTRY[name] = t
    return t


_register_template("plain", "{question}")
_register_template("math", "Solve the following math problem.\n\n{question}")
_register_template("coding", "Solve the following programming problem.\n\n{question}")
_register_template("reasoning", "Answer the following question. Think carefully.\n\n{question}")


# ============================================================================
# ModelProfile
# ============================================================================


@dataclass(frozen=True)
class ModelProfile:
    """Model-family-specific prompting adjustments.

    Attributes:
        name: Profile identifier (e.g., "deepseek_r1").
        system_prompt_prefix: Text to prepend to system prompt.
        system_prompt_suffix: Text to append to system prompt.
        preferred_answer_format: Override answer format fragment name.
        stop_sequences: Additional stop sequences.
        role_mapping: Role name overrides (e.g., {"system": "developer"}).
    """

    name: str
    system_prompt_prefix: str | None = None
    system_prompt_suffix: str | None = None
    preferred_answer_format: str | None = None
    stop_sequences: tuple[str, ...] = ()
    role_mapping: dict[str, str] | None = None

    def build_transformers(self) -> list[PromptTransformer]:
        """Convert profile settings into a list of prompt transformers.

        Returns:
            List of transformers to apply for this model profile.
        """
        from llenvs.inference.prompting import ContentWrapper, RoleMapper

        transformers: list[PromptTransformer] = []

        if self.system_prompt_prefix is not None or self.system_prompt_suffix is not None:
            transformers.append(
                ContentWrapper(
                    prefix=self.system_prompt_prefix or "",
                    suffix=self.system_prompt_suffix or "",
                    roles=("system",),
                )
            )

        if self.role_mapping:
            transformers.append(RoleMapper(mapping=self.role_mapping))

        return transformers


PROFILE_REGISTRY: dict[str, ModelProfile] = {}


def _register_profile(name: str, **kwargs: object) -> ModelProfile:
    p = ModelProfile(name=name, **kwargs)  # type: ignore[arg-type]
    PROFILE_REGISTRY[name] = p
    return p


_register_profile(
    "deepseek_r1",
    system_prompt_suffix="\n\nUse <think>...</think> for your reasoning, then give your final answer.",
)
_register_profile(
    "o1",
    role_mapping={"system": "developer"},
)
_register_profile("llama3_instruct")
_register_profile("qwen_chat")


# Model name patterns for auto-detection
_PROFILE_PATTERNS: list[tuple[list[str], str]] = [
    (["deepseek-r1", "deepseek_r1", "DeepSeek-R1"], "deepseek_r1"),
    (["o1-preview", "o1-mini", "o1-"], "o1"),
    (["llama-3", "Llama-3", "llama3"], "llama3_instruct"),
    (["qwen", "Qwen"], "qwen_chat"),
]


def detect_model_profile(model_name: str) -> ModelProfile | None:
    """Auto-detect model profile from model name.

    Args:
        model_name: Model name or path (e.g., "deepseek-ai/DeepSeek-R1-7B").

    Returns:
        Matching ModelProfile or None.
    """
    lower = model_name.lower()
    for patterns, profile_name in _PROFILE_PATTERNS:
        for pattern in patterns:
            if pattern.lower() in lower:
                return PROFILE_REGISTRY[profile_name]
    return None


# ============================================================================
# Resolution
# ============================================================================


def resolve_prompt_config(
    eval_config: object,
    env_config: object,
    *,
    adapter: object | None = None,
    env_name: str | None = None,
    is_multi_turn: bool = False,
) -> tuple[str | None, PromptTemplate | None, ModelProfile | None]:
    """Resolve prompt configuration from eval and environment configs.

    Resolution priority for system_prompt (first non-None wins):
    1. env_config.system_prompt — user per-env override
    2. eval_config.system_prompt — user global setting
    3. adapter.get_default_system_prompt(env_name) — adapter/library default
    4. If single-turn: "general_reasoning" — llenvs library fallback
    5. None — multi-turn environments get no default

    Resolution priority for prompt_template (first non-None wins):
    1. env_config.prompt_template — user per-env override
    2. eval_config.prompt_template — user global setting
    3. adapter.get_prompt_template(env_name) — adapter default
    4. None — no template by default

    Model profile resolution is from eval_config only.

    Args:
        eval_config: EvalConfig instance.
        env_config: EnvironmentConfig instance.
        adapter: Optional adapter instance for default lookups.
        env_name: Environment name for adapter lookups.
        is_multi_turn: Whether the environment is multi-turn.

    Returns:
        Tuple of (resolved_system_prompt, resolved_template, resolved_profile).
    """
    # System prompt: env overrides eval
    raw_system_prompt = getattr(env_config, "system_prompt", None)
    if raw_system_prompt is None:
        raw_system_prompt = getattr(eval_config, "system_prompt", None)

    resolved_system: str | None = None
    if raw_system_prompt is not None:
        resolved_system = resolve_system_prompt(raw_system_prompt)
    else:
        # Try adapter default
        if adapter is not None:
            adapter_prompt = adapter.get_default_system_prompt(env_name or "")
            if adapter_prompt is not None:
                resolved_system = resolve_system_prompt(adapter_prompt)

        # Library fallback for single-turn environments
        if resolved_system is None and not is_multi_turn:
            resolved_system = resolve_system_prompt("general_reasoning")

    # Prompt template: env overrides eval
    raw_template = getattr(env_config, "prompt_template", None)
    if raw_template is None:
        raw_template = getattr(eval_config, "prompt_template", None)

    resolved_template: PromptTemplate | None = None
    if raw_template is not None:
        if raw_template in TEMPLATE_REGISTRY:
            resolved_template = TEMPLATE_REGISTRY[raw_template]
        else:
            # Treat as literal template string
            resolved_template = PromptTemplate(template=raw_template)
    elif adapter is not None:
        # Try adapter default template
        resolved_template = adapter.get_prompt_template(env_name or "")

    # Model profile
    raw_profile = getattr(eval_config, "model_profile", None)
    resolved_profile: ModelProfile | None = None
    if raw_profile is not None:
        if raw_profile == "auto":
            model_name = getattr(getattr(eval_config, "model", None), "model", "")
            resolved_profile = detect_model_profile(model_name)
        elif raw_profile in PROFILE_REGISTRY:
            resolved_profile = PROFILE_REGISTRY[raw_profile]

    return resolved_system, resolved_template, resolved_profile
