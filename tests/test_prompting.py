"""Tests for prompt transformers and pipelines."""

import pytest
from llenvs.inference.protocol import ChatMessage
from llenvs.inference.prompting import (
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


@pytest.fixture
def user_message() -> list[ChatMessage]:
    """Single user message."""
    return [ChatMessage(role="user", content="What is 2+2?")]


@pytest.fixture
def conversation() -> list[ChatMessage]:
    """Multi-turn conversation."""
    return [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there!"),
        ChatMessage(role="user", content="What is 2+2?"),
    ]


class TestSystemPromptInjector:
    """Tests for SystemPromptInjector."""

    def test_inject_when_no_system(self, user_message):
        """Test injection when no system message exists."""
        injector = SystemPromptInjector(content="Be concise.")
        result = injector.transform(user_message)

        assert len(result) == 2
        assert result[0].role == "system"
        assert result[0].content == "Be concise."
        assert result[1].role == "user"

    def test_replace_existing_system(self, conversation):
        """Test replacing existing system message."""
        injector = SystemPromptInjector(content="New system prompt.", replace_existing=True)
        result = injector.transform(conversation)

        system_msgs = [m for m in result if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "New system prompt."

    def test_prepend_to_existing(self, conversation):
        """Test prepending to existing system message."""
        injector = SystemPromptInjector(content="Important:", replace_existing=False)
        result = injector.transform(conversation)

        system_msgs = [m for m in result if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content.startswith("Important:")
        assert "You are helpful." in system_msgs[0].content

    def test_callable(self, user_message):
        """Test that transformer is callable."""
        injector = SystemPromptInjector(content="Test")
        result = injector(user_message)
        assert len(result) == 2


class TestFewShotInjector:
    """Tests for FewShotInjector."""

    def test_inject_examples_after_system(self, conversation):
        """Test examples injected after system message."""
        injector = FewShotInjector(
            examples=[
                ("Example Q1", "Example A1"),
                ("Example Q2", "Example A2"),
            ]
        )
        result = injector.transform(conversation)

        # System, example1_user, example1_assistant, example2_user, example2_assistant, ...
        assert result[0].role == "system"
        assert result[1].role == "user"
        assert result[1].content == "Example Q1"
        assert result[2].role == "assistant"
        assert result[2].content == "Example A1"

    def test_inject_examples_no_system(self, user_message):
        """Test examples injected before first user message when no system."""
        injector = FewShotInjector(examples=[("Q", "A")])
        result = injector.transform(user_message)

        assert result[0].role == "user"
        assert result[0].content == "Q"
        assert result[1].role == "assistant"
        assert result[-1].content == "What is 2+2?"

    def test_empty_examples(self, user_message):
        """Test with no examples."""
        injector = FewShotInjector(examples=[])
        result = injector.transform(user_message)
        assert result == user_message


class TestChainOfThoughtWrapper:
    """Tests for ChainOfThoughtWrapper."""

    def test_think_step_by_step(self, user_message):
        """Test default CoT prompt."""
        wrapper = ChainOfThoughtWrapper(style="think_step_by_step")
        result = wrapper.transform(user_message)

        assert "step by step" in result[-1].content.lower()
        assert "What is 2+2?" in result[-1].content

    def test_show_work(self, user_message):
        """Test show_work style."""
        wrapper = ChainOfThoughtWrapper(style="show_work")
        result = wrapper.transform(user_message)

        assert "show your work" in result[-1].content.lower()

    def test_custom_prompt(self, user_message):
        """Test custom CoT prompt."""
        wrapper = ChainOfThoughtWrapper(style="Think carefully before answering.")
        result = wrapper.transform(user_message)

        assert "Think carefully" in result[-1].content

    def test_modifies_last_user_message(self, conversation):
        """Test that only last user message is modified."""
        wrapper = ChainOfThoughtWrapper()
        result = wrapper.transform(conversation)

        # First user message unchanged
        assert result[1].content == "Hello"
        # Last user message modified
        assert "step by step" in result[-1].content.lower()


class TestAnswerFormatInjector:
    """Tests for AnswerFormatInjector."""

    def test_xml_answer_format(self, user_message):
        """Test XML answer format injection."""
        injector = AnswerFormatInjector(format_type="xml_answer", tag_name="answer")
        result = injector.transform(user_message)

        assert "<answer>" in result[-1].content
        assert "</answer>" in result[-1].content

    def test_custom_tag_name(self, user_message):
        """Test custom tag name."""
        injector = AnswerFormatInjector(format_type="xml_answer", tag_name="solution")
        result = injector.transform(user_message)

        assert "<solution>" in result[-1].content

    def test_gsm8k_format(self, user_message):
        """Test GSM8K format."""
        injector = AnswerFormatInjector(format_type="gsm8k")
        result = injector.transform(user_message)

        assert "####" in result[-1].content

    def test_adds_to_system_if_present(self, conversation):
        """Test that format added to system message when present."""
        injector = AnswerFormatInjector(format_type="xml_answer")
        result = injector.transform(conversation)

        # Should be in system message
        system = [m for m in result if m.role == "system"][0]
        assert "<answer>" in system.content


class TestMessageTrimmer:
    """Tests for MessageTrimmer."""

    def test_no_trim_needed(self, conversation):
        """Test when messages are under limit."""
        trimmer = MessageTrimmer(max_messages=10)
        result = trimmer.transform(conversation)
        assert len(result) == len(conversation)

    def test_trim_keeping_system(self, conversation):
        """Test trimming while keeping system message."""
        trimmer = MessageTrimmer(max_messages=2, keep_system=True)
        result = trimmer.transform(conversation)

        assert len(result) == 2
        assert result[0].role == "system"

    def test_trim_without_keeping_system(self, conversation):
        """Test trimming without keeping system."""
        trimmer = MessageTrimmer(max_messages=2, keep_system=False)
        result = trimmer.transform(conversation)

        assert len(result) == 2
        # Should be the last 2 messages
        assert result[-1].content == "What is 2+2?"

    def test_keeps_most_recent(self):
        """Test that most recent messages are kept."""
        messages = [ChatMessage(role="user", content=f"msg_{i}") for i in range(10)]
        trimmer = MessageTrimmer(max_messages=3)
        result = trimmer.transform(messages)

        assert len(result) == 3
        assert result[-1].content == "msg_9"
        assert result[-2].content == "msg_8"


class TestRoleMapper:
    """Tests for RoleMapper."""

    def test_role_mapping(self, conversation):
        """Test basic role mapping."""
        mapper = RoleMapper(mapping={"system": "developer"})
        result = mapper.transform(conversation)

        assert result[0].role == "developer"
        assert result[1].role == "user"  # Unchanged

    def test_no_mapping(self, conversation):
        """Test with empty mapping."""
        mapper = RoleMapper(mapping={})
        result = mapper.transform(conversation)
        assert result == conversation


class TestContentWrapper:
    """Tests for ContentWrapper."""

    def test_wrap_all_messages(self, user_message):
        """Test wrapping all messages."""
        wrapper = ContentWrapper(prefix="[START]", suffix="[END]")
        result = wrapper.transform(user_message)

        assert result[0].content.startswith("[START]")
        assert result[0].content.endswith("[END]")

    def test_wrap_specific_roles(self, conversation):
        """Test wrapping only specific roles."""
        wrapper = ContentWrapper(prefix=">>", suffix="<<", roles=("user",))
        result = wrapper.transform(conversation)

        for msg in result:
            if msg.role == "user":
                assert msg.content.startswith(">>")
            else:
                assert not msg.content.startswith(">>")


class TestPromptPipeline:
    """Tests for PromptPipeline."""

    def test_compose_with_rshift(self, user_message):
        """Test composing transformers with >> operator."""
        pipeline = SystemPromptInjector("System.") >> AnswerFormatInjector("xml_answer")
        result = pipeline.transform(user_message)

        assert result[0].role == "system"
        assert "<answer>" in result[0].content

    def test_multi_stage_pipeline(self, user_message):
        """Test multi-stage pipeline."""
        pipeline = (
            SystemPromptInjector("Be helpful.")
            >> FewShotInjector([("Q", "A")])
            >> ChainOfThoughtWrapper()
            >> AnswerFormatInjector("xml_answer")
        )
        result = pipeline.transform(user_message)

        # Check all transformations applied
        assert any(m.role == "system" for m in result)
        assert "step by step" in result[-1].content.lower()

    def test_callable(self, user_message):
        """Test that pipeline is callable."""
        pipeline = PromptPipeline(transformers=[SystemPromptInjector("Test")])
        result = pipeline(user_message)
        assert result[0].role == "system"

    def test_extend_pipeline(self, user_message):
        """Test extending an existing pipeline."""
        pipeline1 = SystemPromptInjector("A") >> FewShotInjector([("Q", "A")])
        pipeline2 = pipeline1 >> AnswerFormatInjector("xml_answer")

        result = pipeline2.transform(user_message)
        # Original pipeline unchanged
        assert len(pipeline1.transformers) == 2
        # Extended pipeline has all three
        assert len(pipeline2.transformers) == 3


class TestBuildStandardPipeline:
    """Tests for build_standard_pipeline helper."""

    def test_minimal_pipeline(self, user_message):
        """Test pipeline with minimal options."""
        pipeline = build_standard_pipeline()
        result = pipeline.transform(user_message)

        # Should at least have answer format
        assert "<answer>" in result[-1].content

    def test_full_pipeline(self, user_message):
        """Test pipeline with all options."""
        pipeline = build_standard_pipeline(
            system_prompt="You are a math tutor.",
            examples=[("2+2?", "4")],
            use_cot=True,
            answer_format="xml_answer",
            tag_name="solution",
        )
        result = pipeline.transform(user_message)

        # Check system prompt
        system = [m for m in result if m.role == "system"][0]
        assert "math tutor" in system.content

        # Check CoT
        assert "step by step" in result[-1].content.lower()

        # Check format
        assert "<solution>" in system.content or "<solution>" in result[-1].content


class TestPromptTemplateTransformer:
    """Tests for PromptTemplateTransformer."""

    def test_applies_to_last_user_message(self, user_message):
        """Test that template wraps the last user message."""
        from llenvs.inference.prompts import PromptTemplate

        template = PromptTemplate(template="Solve: {question}", name="math")
        transformer = PromptTemplateTransformer(template=template)
        result = transformer.transform(user_message)

        assert len(result) == 1
        assert result[0].content == "Solve: What is 2+2?"

    def test_applies_to_last_user_in_conversation(self, conversation):
        """Test that only the last user message is wrapped."""
        from llenvs.inference.prompts import PromptTemplate

        template = PromptTemplate(template="[Q] {question}", name="test")
        transformer = PromptTemplateTransformer(template=template)
        result = transformer.transform(conversation)

        # First user message unchanged
        assert result[1].content == "Hello"
        # Last user message wrapped
        assert result[-1].content == "[Q] What is 2+2?"

    def test_empty_messages(self):
        """Test with empty message list."""
        from llenvs.inference.prompts import PromptTemplate

        template = PromptTemplate(template="Solve: {question}")
        transformer = PromptTemplateTransformer(template=template)
        result = transformer.transform([])
        assert result == []

    def test_no_user_messages(self):
        """Test with no user messages."""
        from llenvs.inference.prompts import PromptTemplate

        template = PromptTemplate(template="Solve: {question}")
        transformer = PromptTemplateTransformer(template=template)
        messages = [ChatMessage(role="system", content="Hello")]
        result = transformer.transform(messages)
        assert result == messages

    def test_composable_with_pipeline(self, user_message):
        """Test that it composes with other transformers."""
        from llenvs.inference.prompts import PromptTemplate

        template = PromptTemplate(template="Problem: {question}")
        pipeline = SystemPromptInjector("You are helpful.") >> PromptTemplateTransformer(
            template=template
        )
        result = pipeline.transform(user_message)

        assert result[0].role == "system"
        assert result[1].content == "Problem: What is 2+2?"
