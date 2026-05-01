"""Tests for multimodal observation support (ImageContent, image in messages)."""

from unittest.mock import MagicMock

import pytest

from llenvs.core.state import ImageContent, Observation, ObservationContent, State, StateMetadata
from llenvs.inference.protocol import BackendCapabilities, ChatMessage

# ---------------------------------------------------------------------------
# ImageContent
# ---------------------------------------------------------------------------


class TestImageContent:
    def test_construction(self):
        img = ImageContent(data="iVBOR...", media_type="image/png")
        assert img.data == "iVBOR..."
        assert img.media_type == "image/png"

    def test_default_media_type(self):
        img = ImageContent(data="abc123")
        assert img.media_type == "image/png"

    def test_frozen(self):
        img = ImageContent(data="abc")
        with pytest.raises(AttributeError):
            img.data = "xyz"  # type: ignore[misc]

    def test_jpeg_media_type(self):
        img = ImageContent(data="abc", media_type="image/jpeg")
        assert img.media_type == "image/jpeg"


# ---------------------------------------------------------------------------
# Observation with images
# ---------------------------------------------------------------------------


class TestObservationImages:
    def test_default_empty(self):
        obs = Observation(prompt="Hello")
        assert obs.get_images().all == ()

    def test_with_images(self):
        img1 = ImageContent(data="img1")
        img2 = ImageContent(data="img2")
        obs = Observation(prompt="Hello", state=ObservationContent(text="", images=(img1, img2)))
        assert len(obs.state.images) == 2
        assert obs.state.images[0].data == "img1"

    def test_frozen_with_images(self):
        obs = Observation(prompt="Hello", state=ObservationContent(text="", images=(ImageContent(data="a"),)))
        with pytest.raises(AttributeError):
            obs.state = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ChatMessage.to_dict() with images — OpenAI format
# ---------------------------------------------------------------------------


class TestChatMessageOpenAIFormat:
    def test_no_images_unchanged(self):
        msg = ChatMessage(role="user", content="Hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_empty_images_unchanged(self):
        msg = ChatMessage(role="user", content="Hello", images=())
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_with_images_content_array(self):
        img = ImageContent(data="iVBOR...", media_type="image/png")
        msg = ChatMessage(role="user", content="What is this?", images=(img,))
        d = msg.to_dict()
        assert d["role"] == "user"
        assert isinstance(d["content"], list)
        assert len(d["content"]) == 2
        # Text block
        assert d["content"][0] == {"type": "text", "text": "What is this?"}
        # Image block
        assert d["content"][1]["type"] == "image_url"
        assert d["content"][1]["image_url"]["url"] == "data:image/png;base64,iVBOR..."

    def test_with_multiple_images(self):
        img1 = ImageContent(data="a", media_type="image/png")
        img2 = ImageContent(data="b", media_type="image/jpeg")
        msg = ChatMessage(role="user", content="Describe", images=(img1, img2))
        d = msg.to_dict()
        assert isinstance(d["content"], list)
        assert len(d["content"]) == 3  # text + 2 images
        assert d["content"][1]["image_url"]["url"] == "data:image/png;base64,a"
        assert d["content"][2]["image_url"]["url"] == "data:image/jpeg;base64,b"

    def test_images_with_none_content(self):
        img = ImageContent(data="a")
        msg = ChatMessage(role="user", content=None, images=(img,))
        d = msg.to_dict()
        assert isinstance(d["content"], list)
        # Should only have image block, no text block
        assert len(d["content"]) == 1
        assert d["content"][0]["type"] == "image_url"


# ---------------------------------------------------------------------------
# ChatMessage.to_anthropic_dict() with images — Anthropic format
# ---------------------------------------------------------------------------


class TestChatMessageAnthropicFormat:
    def test_no_images(self):
        msg = ChatMessage(role="user", content="Hello")
        d = msg.to_anthropic_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_with_images_content_array(self):
        img = ImageContent(data="iVBOR...", media_type="image/png")
        msg = ChatMessage(role="user", content="What is this?", images=(img,))
        d = msg.to_anthropic_dict()
        assert d["role"] == "user"
        assert isinstance(d["content"], list)
        assert len(d["content"]) == 2
        # Image block first (Anthropic convention)
        assert d["content"][0]["type"] == "image"
        assert d["content"][0]["source"]["type"] == "base64"
        assert d["content"][0]["source"]["media_type"] == "image/png"
        assert d["content"][0]["source"]["data"] == "iVBOR..."
        # Text block second
        assert d["content"][1] == {"type": "text", "text": "What is this?"}

    def test_images_with_none_content(self):
        img = ImageContent(data="a")
        msg = ChatMessage(role="user", content=None, images=(img,))
        d = msg.to_anthropic_dict()
        assert isinstance(d["content"], list)
        assert len(d["content"]) == 1
        assert d["content"][0]["type"] == "image"

    def test_tool_calls_preserved(self):
        """to_anthropic_dict should preserve tool calls like to_dict."""
        from llenvs.core.tools import ToolCall

        tc = ToolCall(id="tc1", name="search", arguments={"q": "hello"})
        msg = ChatMessage(role="assistant", content="Let me search", tool_calls=(tc,))
        d = msg.to_anthropic_dict()
        assert "tool_calls" in d
        assert d["tool_calls"][0]["id"] == "tc1"


# ---------------------------------------------------------------------------
# BackendCapabilities.supports_vision
# ---------------------------------------------------------------------------


class TestBackendCapabilitiesVision:
    def test_default_false(self):
        cap = BackendCapabilities()
        assert cap.supports_vision is False

    def test_explicit_true(self):
        cap = BackendCapabilities(supports_vision=True)
        assert cap.supports_vision is True


# ---------------------------------------------------------------------------
# Runner _build_messages propagation
# ---------------------------------------------------------------------------


class TestRunnerImagePropagation:
    def test_initial_message_carries_images(self):
        """Runner should propagate images from Observation to ChatMessage."""
        from llenvs.core.environment import EnvironmentSpec
        from llenvs.evaluation.runner import TrajectoryRunner

        img = ImageContent(data="abc")
        obs = Observation(prompt="What is in this image?", state=ObservationContent(text="", images=(img,)))
        state = State(
            observation=obs,
            hidden=None,
            metadata=StateMetadata(step=0, episode_id="test"),
        )

        mock_env = MagicMock()
        mock_env.spec = EnvironmentSpec(name="test")
        mock_backend = MagicMock()

        runner = TrajectoryRunner(environment=mock_env, backend=mock_backend)
        messages = runner._build_messages(state)

        # First non-system message should be user with images
        user_msgs = [m for m in messages if m.role == "user"]
        assert len(user_msgs) == 1
        assert len(user_msgs[0].images) == 1
        assert user_msgs[0].images[0].data == img.data
        assert user_msgs[0].images[0].source == "state"
        assert user_msgs[0].content == "What is in this image?"

    def test_history_user_messages_carry_images(self):
        """User messages in history should carry images if present."""
        from llenvs.core.environment import EnvironmentSpec
        from llenvs.evaluation.runner import TrajectoryRunner

        obs = Observation(
            prompt="Initial prompt",
            state=ObservationContent(text="", images=(ImageContent(data="init_img"),)),
            messages=(
                {"role": "assistant", "content": "I see the image"},
                {
                    "role": "user",
                    "content": "Step 1",
                    "images": [{"data": "step1_img", "media_type": "image/png"}],
                },
            ),
        )
        state = State(
            observation=obs,
            hidden=None,
            metadata=StateMetadata(step=1, episode_id="test"),
        )

        mock_env = MagicMock()
        mock_env.spec = EnvironmentSpec(name="test")
        mock_backend = MagicMock()

        runner = TrajectoryRunner(environment=mock_env, backend=mock_backend)
        messages = runner._build_messages(state)

        # First user message should have initial images
        assert messages[0].role == "user"
        assert len(messages[0].images) == 1
        assert messages[0].images[0].data == "init_img"


# ---------------------------------------------------------------------------
# Prompt transformer image preservation
# ---------------------------------------------------------------------------


class TestTransformerImagePreservation:
    def _make_msg(self, role="user", content="Hello", images=()):
        return ChatMessage(role=role, content=content, images=images)

    def test_chain_of_thought_preserves_images(self):
        from llenvs.inference.prompting import ChainOfThoughtWrapper

        img = ImageContent(data="abc")
        msgs = [self._make_msg(images=(img,))]
        result = ChainOfThoughtWrapper().transform(msgs)
        assert result[0].images == (img,)

    def test_answer_format_preserves_images_in_user(self):
        from llenvs.inference.prompting import AnswerFormatInjector

        img = ImageContent(data="abc")
        msgs = [self._make_msg(images=(img,))]
        result = AnswerFormatInjector().transform(msgs)
        assert result[0].images == (img,)

    def test_answer_format_preserves_images_in_system(self):
        from llenvs.inference.prompting import AnswerFormatInjector

        img = ImageContent(data="abc")
        msgs = [
            self._make_msg(role="system", content="System"),
            self._make_msg(images=(img,)),
        ]
        result = AnswerFormatInjector().transform(msgs)
        # System message modified, user message untouched
        user_msgs = [m for m in result if m.role == "user"]
        assert user_msgs[0].images == (img,)

    def test_content_wrapper_preserves_images(self):
        from llenvs.inference.prompting import ContentWrapper

        img = ImageContent(data="abc")
        msgs = [self._make_msg(images=(img,))]
        result = ContentWrapper(prefix="<p>", suffix="</p>").transform(msgs)
        assert result[0].images == (img,)

    def test_role_mapper_preserves_images(self):
        from llenvs.inference.prompting import RoleMapper

        img = ImageContent(data="abc")
        msgs = [self._make_msg(images=(img,))]
        result = RoleMapper(mapping={"user": "human"}).transform(msgs)
        assert result[0].images == (img,)
        assert result[0].role == "human"

    def test_prompt_template_preserves_images(self):
        from llenvs.inference.prompting import PromptTemplateTransformer
        from llenvs.inference.prompts import PromptTemplate

        img = ImageContent(data="abc")
        msgs = [self._make_msg(images=(img,))]
        tmpl = PromptTemplate(name="test", template="Q: {question}\nA:")
        result = PromptTemplateTransformer(template=tmpl).transform(msgs)
        assert result[0].images == (img,)
        assert "Q: Hello" in result[0].content


# ---------------------------------------------------------------------------
# Container serialization
# ---------------------------------------------------------------------------


class TestContainerImageSerialization:
    def test_serialize_observation_with_images(self):
        from llenvs.container.serialization import serialize_observation

        img = ImageContent(data="abc", media_type="image/jpeg")
        obs = Observation(prompt="Hi", state=ObservationContent(text="", images=(img,)))
        d = serialize_observation(obs)
        assert "state" in d
        assert len(d["state"]["images"]) == 1
        assert d["state"]["images"][0]["data"] == "abc"
        assert d["state"]["images"][0]["media_type"] == "image/jpeg"

    def test_serialize_observation_empty_images(self):
        from llenvs.container.serialization import serialize_observation

        obs = Observation(prompt="Hi")
        d = serialize_observation(obs)
        assert "state" not in d

    def test_deserialize_observation_with_images(self):
        from llenvs.container.serialization import deserialize_observation

        data = {
            "prompt": "Hi",
            "state": {"text": "", "images": [{"data": "abc", "media_type": "image/jpeg"}]},
        }
        obs = deserialize_observation(data)
        assert obs.state is not None
        assert len(obs.state.images) == 1
        assert obs.state.images[0].data == "abc"
        assert obs.state.images[0].media_type == "image/jpeg"

    def test_deserialize_observation_no_images_key(self):
        from llenvs.container.serialization import deserialize_observation

        data = {"prompt": "Hi"}
        obs = deserialize_observation(data)
        assert obs.get_images().all == ()

    def test_round_trip(self):
        from llenvs.container.serialization import (
            deserialize_observation,
            serialize_observation,
        )

        img1 = ImageContent(data="img1", media_type="image/png")
        img2 = ImageContent(data="img2", media_type="image/jpeg")
        obs = Observation(prompt="Test", state=ObservationContent(text="", images=(img1, img2)))
        restored = deserialize_observation(serialize_observation(obs))
        assert restored.prompt == "Test"
        assert restored.state is not None
        assert len(restored.state.images) == 2
        assert restored.state.images[0].data == "img1"
        assert restored.state.images[1].media_type == "image/jpeg"


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_core_exports_image_content(self):
        from llenvs.core import ImageContent as CoreImageContent

        assert CoreImageContent is ImageContent

    def test_top_level_exports_image_content(self):
        from llenvs import ImageContent as TopImageContent

        assert TopImageContent is ImageContent
