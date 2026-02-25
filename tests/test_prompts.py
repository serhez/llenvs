"""Tests for prompt library - fragments, system prompts, templates, profiles."""

import pytest

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

# ============================================================================
# PromptFragment
# ============================================================================


class TestPromptFragment:
    """Tests for PromptFragment dataclass."""

    def test_creation(self):
        frag = PromptFragment(
            name="test_frag",
            content="Think carefully.",
            category="reasoning",
        )
        assert frag.name == "test_frag"
        assert frag.content == "Think carefully."
        assert frag.category == "reasoning"

    def test_frozen(self):
        frag = PromptFragment(name="a", content="b", category="c")
        with pytest.raises(AttributeError):
            frag.name = "x"


# ============================================================================
# Fragment Registry
# ============================================================================


class TestFragmentRegistry:
    """Tests for pre-built fragment registry."""

    def test_registry_not_empty(self):
        assert len(FRAGMENT_REGISTRY) > 0

    def test_reasoning_fragments_exist(self):
        for name in [
            "think_step_by_step",
            "show_your_work",
            "reflect_then_answer",
            "verify_your_answer",
            "direct_answer",
        ]:
            assert name in FRAGMENT_REGISTRY, f"Missing reasoning fragment: {name}"
            assert FRAGMENT_REGISTRY[name].category == "reasoning"

    def test_format_fragments_exist(self):
        for name in ["xml_answer", "boxed_answer", "gsm8k_answer", "json_answer"]:
            assert name in FRAGMENT_REGISTRY, f"Missing format fragment: {name}"
            assert FRAGMENT_REGISTRY[name].category == "format"

    def test_persona_fragments_exist(self):
        for name in ["math_expert", "coding_expert", "general_assistant"]:
            assert name in FRAGMENT_REGISTRY, f"Missing persona fragment: {name}"
            assert FRAGMENT_REGISTRY[name].category == "persona"

    def test_constraint_fragments_exist(self):
        for name in ["be_concise", "no_apologies", "single_answer"]:
            assert name in FRAGMENT_REGISTRY, f"Missing constraint fragment: {name}"
            assert FRAGMENT_REGISTRY[name].category == "constraint"

    def test_all_fragments_have_content(self):
        for name, frag in FRAGMENT_REGISTRY.items():
            assert frag.content, f"Fragment {name} has empty content"
            assert frag.name == name, f"Fragment key {name} != frag.name {frag.name}"


# ============================================================================
# SystemPrompt
# ============================================================================


class TestSystemPrompt:
    """Tests for SystemPrompt dataclass."""

    def test_creation(self):
        sp = SystemPrompt(name="test", content="You are helpful.")
        assert sp.name == "test"
        assert sp.content == "You are helpful."

    def test_frozen(self):
        sp = SystemPrompt(name="a", content="b")
        with pytest.raises(AttributeError):
            sp.content = "x"


class TestSystemPromptRegistry:
    """Tests for pre-built system prompt registry."""

    def test_registry_not_empty(self):
        assert len(SYSTEM_PROMPT_REGISTRY) > 0

    def test_expected_prompts_exist(self):
        expected = [
            "general_reasoning",
            "math_reasoning",
            "math_boxed",
            "math_gsm8k",
            "coding_problem",
            "concise_reasoning",
            "direct_answer",
        ]
        for name in expected:
            assert name in SYSTEM_PROMPT_REGISTRY, f"Missing system prompt: {name}"

    def test_all_prompts_have_content(self):
        for name, sp in SYSTEM_PROMPT_REGISTRY.items():
            assert sp.content, f"SystemPrompt {name} has empty content"
            assert sp.name == name


# ============================================================================
# Composition API
# ============================================================================


class TestComposeSystemPrompt:
    """Tests for compose_system_prompt."""

    def test_compose_strings(self):
        result = compose_system_prompt("Hello.", "World.")
        assert "Hello." in result
        assert "World." in result

    def test_compose_fragments(self):
        f1 = PromptFragment(name="a", content="Think step by step.", category="reasoning")
        f2 = PromptFragment(name="b", content="Be concise.", category="constraint")
        result = compose_system_prompt(f1, f2)
        assert "Think step by step." in result
        assert "Be concise." in result

    def test_compose_system_prompts(self):
        sp = SystemPrompt(name="test", content="Full prompt here.")
        result = compose_system_prompt(sp)
        assert "Full prompt here." in result

    def test_compose_mixed(self):
        frag = PromptFragment(name="a", content="Step by step.", category="reasoning")
        result = compose_system_prompt("You are helpful.", frag, "Be clear.")
        assert "You are helpful." in result
        assert "Step by step." in result
        assert "Be clear." in result

    def test_custom_separator(self):
        result = compose_system_prompt("A", "B", separator=" | ")
        assert result == "A | B"

    def test_empty(self):
        result = compose_system_prompt()
        assert result == ""


class TestResolveSystemPrompt:
    """Tests for resolve_system_prompt."""

    def test_resolve_system_prompt_name(self):
        """Known system prompt name resolves to its content."""
        result = resolve_system_prompt("math_reasoning")
        expected = SYSTEM_PROMPT_REGISTRY["math_reasoning"].content
        assert result == expected

    def test_resolve_fragment_name(self):
        """Known fragment name resolves to its content."""
        result = resolve_system_prompt("think_step_by_step")
        expected = FRAGMENT_REGISTRY["think_step_by_step"].content
        assert result == expected

    def test_resolve_literal_string(self):
        """Unknown name treated as literal text."""
        result = resolve_system_prompt("You are a custom assistant.")
        assert result == "You are a custom assistant."

    def test_resolve_list(self):
        """List of items resolved and joined."""
        result = resolve_system_prompt(["math_expert", "think_step_by_step", "xml_answer"])
        assert FRAGMENT_REGISTRY["math_expert"].content in result
        assert FRAGMENT_REGISTRY["think_step_by_step"].content in result
        assert FRAGMENT_REGISTRY["xml_answer"].content in result

    def test_resolve_list_with_literal(self):
        """List mixing registry names and literals."""
        result = resolve_system_prompt(["math_expert", "Custom instruction here."])
        assert FRAGMENT_REGISTRY["math_expert"].content in result
        assert "Custom instruction here." in result


# ============================================================================
# PromptTemplate
# ============================================================================


class TestPromptTemplate:
    """Tests for PromptTemplate."""

    def test_basic_apply(self):
        t = PromptTemplate(template="Solve: {question}", name="math")
        result = t.apply("What is 2+2?")
        assert result == "Solve: What is 2+2?"

    def test_plain_template(self):
        t = PromptTemplate(template="{question}")
        result = t.apply("Hello")
        assert result == "Hello"

    def test_extra_kwargs(self):
        t = PromptTemplate(template="[{subject}] {question}", name="custom")
        result = t.apply("What is 2+2?", subject="Math")
        assert result == "[Math] What is 2+2?"

    def test_frozen(self):
        t = PromptTemplate(template="{question}")
        with pytest.raises(AttributeError):
            t.template = "new"


class TestTemplateRegistry:
    """Tests for pre-built template registry."""

    def test_registry_not_empty(self):
        assert len(TEMPLATE_REGISTRY) > 0

    def test_plain_exists(self):
        assert "plain" in TEMPLATE_REGISTRY
        result = TEMPLATE_REGISTRY["plain"].apply("Hello")
        assert result == "Hello"

    def test_math_exists(self):
        assert "math" in TEMPLATE_REGISTRY
        result = TEMPLATE_REGISTRY["math"].apply("What is 2+2?")
        assert "2+2" in result

    def test_coding_exists(self):
        assert "coding" in TEMPLATE_REGISTRY

    def test_reasoning_exists(self):
        assert "reasoning" in TEMPLATE_REGISTRY


# ============================================================================
# ModelProfile
# ============================================================================


class TestModelProfile:
    """Tests for ModelProfile."""

    def test_basic_creation(self):
        p = ModelProfile(name="test")
        assert p.name == "test"
        assert p.system_prompt_prefix is None
        assert p.system_prompt_suffix is None
        assert p.preferred_answer_format is None
        assert p.stop_sequences == ()
        assert p.role_mapping is None

    def test_build_transformers_empty(self):
        """Profile with no overrides produces empty transformer list."""
        p = ModelProfile(name="empty")
        transformers = p.build_transformers()
        assert transformers == []

    def test_build_transformers_with_prefix(self):
        p = ModelProfile(name="prefixed", system_prompt_prefix="PREFIX: ")
        transformers = p.build_transformers()
        assert len(transformers) >= 1

    def test_build_transformers_with_suffix(self):
        p = ModelProfile(name="suffixed", system_prompt_suffix="\nEND")
        transformers = p.build_transformers()
        assert len(transformers) >= 1

    def test_build_transformers_with_role_mapping(self):
        p = ModelProfile(name="mapped", role_mapping={"system": "developer"})
        transformers = p.build_transformers()
        assert len(transformers) >= 1

    def test_frozen(self):
        p = ModelProfile(name="test")
        with pytest.raises(AttributeError):
            p.name = "x"


class TestProfileRegistry:
    """Tests for pre-built profile registry."""

    def test_registry_not_empty(self):
        assert len(PROFILE_REGISTRY) > 0

    def test_deepseek_r1_exists(self):
        assert "deepseek_r1" in PROFILE_REGISTRY
        p = PROFILE_REGISTRY["deepseek_r1"]
        assert p.system_prompt_suffix is not None

    def test_o1_exists(self):
        assert "o1" in PROFILE_REGISTRY
        p = PROFILE_REGISTRY["o1"]
        assert p.role_mapping is not None
        assert p.role_mapping.get("system") == "developer"

    def test_llama3_instruct_exists(self):
        assert "llama3_instruct" in PROFILE_REGISTRY

    def test_qwen_chat_exists(self):
        assert "qwen_chat" in PROFILE_REGISTRY


class TestDetectModelProfile:
    """Tests for detect_model_profile."""

    def test_detect_deepseek_r1(self):
        profile = detect_model_profile("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        assert profile is not None
        assert profile.name == "deepseek_r1"

    def test_detect_o1(self):
        profile = detect_model_profile("o1-preview")
        assert profile is not None
        assert profile.name == "o1"

    def test_detect_o1_mini(self):
        profile = detect_model_profile("o1-mini")
        assert profile is not None
        assert profile.name == "o1"

    def test_detect_llama3(self):
        profile = detect_model_profile("meta-llama/Llama-3-8B-Instruct")
        assert profile is not None
        assert profile.name == "llama3_instruct"

    def test_detect_qwen(self):
        profile = detect_model_profile("Qwen/Qwen2.5-7B-Chat")
        assert profile is not None
        assert profile.name == "qwen_chat"

    def test_detect_unknown(self):
        profile = detect_model_profile("some-random-model-name")
        assert profile is None


# ============================================================================
# resolve_prompt_config
# ============================================================================


class TestResolvePromptConfig:
    """Tests for resolve_prompt_config."""

    def test_defaults_single_turn(self):
        """No config set -> library fallback for single-turn."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
        )
        env_cfg = EnvironmentConfig(name="test")

        sys_prompt, template, profile = resolve_prompt_config(eval_cfg, env_cfg)
        # Single-turn (default) gets library fallback
        assert sys_prompt == SYSTEM_PROMPT_REGISTRY["general_reasoning"].content
        assert template is None
        assert profile is None

    def test_eval_config_system_prompt(self):
        """system_prompt from eval config is resolved."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
            system_prompt="math_reasoning",
        )
        env_cfg = EnvironmentConfig(name="test")

        sys_prompt, _, _ = resolve_prompt_config(eval_cfg, env_cfg)
        assert sys_prompt is not None
        assert sys_prompt == SYSTEM_PROMPT_REGISTRY["math_reasoning"].content

    def test_env_config_overrides_eval_system_prompt(self):
        """Per-env system_prompt overrides eval-level."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
            system_prompt="general_reasoning",
        )
        env_cfg = EnvironmentConfig(name="test", system_prompt="math_reasoning")

        sys_prompt, _, _ = resolve_prompt_config(eval_cfg, env_cfg)
        assert sys_prompt == SYSTEM_PROMPT_REGISTRY["math_reasoning"].content

    def test_literal_system_prompt(self):
        """Literal string system prompt passed through."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
            system_prompt="You are a helpful assistant.",
        )
        env_cfg = EnvironmentConfig(name="test")

        sys_prompt, _, _ = resolve_prompt_config(eval_cfg, env_cfg)
        assert sys_prompt == "You are a helpful assistant."

    def test_list_system_prompt(self):
        """List of fragments resolved and joined."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
            system_prompt=["math_expert", "think_step_by_step"],
        )
        env_cfg = EnvironmentConfig(name="test")

        sys_prompt, _, _ = resolve_prompt_config(eval_cfg, env_cfg)
        assert sys_prompt is not None
        assert FRAGMENT_REGISTRY["math_expert"].content in sys_prompt
        assert FRAGMENT_REGISTRY["think_step_by_step"].content in sys_prompt

    def test_eval_prompt_template(self):
        """prompt_template from eval config resolved."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
            prompt_template="math",
        )
        env_cfg = EnvironmentConfig(name="test")

        _, template, _ = resolve_prompt_config(eval_cfg, env_cfg)
        assert template is not None
        assert template.name == "math"

    def test_env_prompt_template_overrides_eval(self):
        """Per-env prompt_template overrides eval-level."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
            prompt_template="reasoning",
        )
        env_cfg = EnvironmentConfig(name="test", prompt_template="math")

        _, template, _ = resolve_prompt_config(eval_cfg, env_cfg)
        assert template is not None
        assert template.name == "math"

    def test_model_profile_by_name(self):
        """model_profile resolved by name."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
            model_profile="deepseek_r1",
        )
        env_cfg = EnvironmentConfig(name="test")

        _, _, profile = resolve_prompt_config(eval_cfg, env_cfg)
        assert profile is not None
        assert profile.name == "deepseek_r1"

    def test_model_profile_auto(self):
        """model_profile='auto' detects from model name."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="deepseek-ai/DeepSeek-R1"),
            model_profile="auto",
        )
        env_cfg = EnvironmentConfig(name="test")

        _, _, profile = resolve_prompt_config(eval_cfg, env_cfg)
        assert profile is not None
        assert profile.name == "deepseek_r1"

    def test_model_profile_auto_no_match(self):
        """model_profile='auto' with unknown model returns None."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="some-unknown-model"),
            model_profile="auto",
        )
        env_cfg = EnvironmentConfig(name="test")

        _, _, profile = resolve_prompt_config(eval_cfg, env_cfg)
        assert profile is None

    def test_library_fallback_single_turn(self):
        """Single-turn env with no user/adapter prompt gets library fallback."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
        )
        env_cfg = EnvironmentConfig(name="test")

        sys_prompt, _, _ = resolve_prompt_config(
            eval_cfg,
            env_cfg,
            is_multi_turn=False,
        )
        assert sys_prompt is not None
        assert sys_prompt == SYSTEM_PROMPT_REGISTRY["general_reasoning"].content

    def test_no_library_fallback_multi_turn(self):
        """Multi-turn env with no user/adapter prompt gets None."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
        )
        env_cfg = EnvironmentConfig(name="test")

        sys_prompt, _, _ = resolve_prompt_config(
            eval_cfg,
            env_cfg,
            is_multi_turn=True,
        )
        assert sys_prompt is None

    def test_adapter_default_system_prompt(self):
        """Adapter default system prompt used when no user config."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        class MockAdapter:
            name = "mock"

            def get_default_system_prompt(self, name):
                return "Adapter default prompt."

            def get_prompt_template(self, name):
                return None

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
        )
        env_cfg = EnvironmentConfig(name="test")

        sys_prompt, _, _ = resolve_prompt_config(
            eval_cfg, env_cfg, adapter=MockAdapter(), env_name="test"
        )
        assert sys_prompt == "Adapter default prompt."

    def test_user_config_overrides_adapter_default(self):
        """User system_prompt overrides adapter default."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        class MockAdapter:
            name = "mock"

            def get_default_system_prompt(self, name):
                return "Adapter default."

            def get_prompt_template(self, name):
                return None

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
            system_prompt="User prompt.",
        )
        env_cfg = EnvironmentConfig(name="test")

        sys_prompt, _, _ = resolve_prompt_config(
            eval_cfg, env_cfg, adapter=MockAdapter(), env_name="test"
        )
        assert sys_prompt == "User prompt."

    def test_env_config_overrides_adapter_default(self):
        """Per-env system_prompt overrides adapter default."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        class MockAdapter:
            name = "mock"

            def get_default_system_prompt(self, name):
                return "Adapter default."

            def get_prompt_template(self, name):
                return None

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
        )
        env_cfg = EnvironmentConfig(name="test", system_prompt="Env-level prompt.")

        sys_prompt, _, _ = resolve_prompt_config(
            eval_cfg, env_cfg, adapter=MockAdapter(), env_name="test"
        )
        assert sys_prompt == "Env-level prompt."

    def test_adapter_default_overrides_library_fallback(self):
        """Adapter default takes priority over library fallback."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        class MockAdapter:
            name = "mock"

            def get_default_system_prompt(self, name):
                return "Adapter default."

            def get_prompt_template(self, name):
                return None

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
        )
        env_cfg = EnvironmentConfig(name="test")

        sys_prompt, _, _ = resolve_prompt_config(
            eval_cfg,
            env_cfg,
            adapter=MockAdapter(),
            env_name="test",
            is_multi_turn=False,
        )
        # Adapter default wins over library fallback
        assert sys_prompt == "Adapter default."

    def test_adapter_none_falls_to_library_fallback(self):
        """Adapter returning None for system prompt falls through to library fallback."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        class MockAdapter:
            name = "mock"

            def get_default_system_prompt(self, name):
                return None

            def get_prompt_template(self, name):
                return None

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
        )
        env_cfg = EnvironmentConfig(name="test")

        sys_prompt, _, _ = resolve_prompt_config(
            eval_cfg,
            env_cfg,
            adapter=MockAdapter(),
            env_name="test",
            is_multi_turn=False,
        )
        # Falls to library fallback
        assert sys_prompt == SYSTEM_PROMPT_REGISTRY["general_reasoning"].content

    def test_adapter_prompt_template_used(self):
        """Adapter prompt template used when no user config."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        class MockAdapter:
            name = "mock"

            def get_default_system_prompt(self, name):
                return None

            def get_prompt_template(self, name):
                return PromptTemplate(template="Adapter: {question}", name="adapter_tpl")

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
        )
        env_cfg = EnvironmentConfig(name="test")

        _, template, _ = resolve_prompt_config(
            eval_cfg, env_cfg, adapter=MockAdapter(), env_name="test"
        )
        assert template is not None
        assert template.name == "adapter_tpl"

    def test_user_template_overrides_adapter(self):
        """User prompt_template overrides adapter default."""
        from llenvs.core.config import EnvironmentConfig, EvalConfig, ModelConfig

        class MockAdapter:
            name = "mock"

            def get_default_system_prompt(self, name):
                return None

            def get_prompt_template(self, name):
                return PromptTemplate(template="Adapter: {question}", name="adapter_tpl")

        eval_cfg = EvalConfig(
            environments=[],
            model=ModelConfig(model="test"),
            prompt_template="math",
        )
        env_cfg = EnvironmentConfig(name="test")

        _, template, _ = resolve_prompt_config(
            eval_cfg, env_cfg, adapter=MockAdapter(), env_name="test"
        )
        assert template is not None
        assert template.name == "math"
