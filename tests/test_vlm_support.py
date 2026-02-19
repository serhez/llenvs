"""Tests for full VLM (Vision Language Model) support.

Covers:
1. vLLM VLM detection + image handling
2. HuggingFace VLM detection + image handling
3. Memory management (image truncation, trajectory strip, log strip)
4. Gymnasium ImageObservationMapper
5. DatasetProvider image support
"""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llenvs.core.reward import SignalBundle
from llenvs.core.state import Action, ImageContent, Observation, State, StateMetadata
from llenvs.core.trajectory import Trajectory, Transition
from llenvs.inference.protocol import (
    ChatMessage,
    SamplingParams,
)

# ===========================================================================
# 1. vLLM VLM Support
# ===========================================================================


class TestVLLMImageDecoding:
    """Test image decoding utility for vLLM backend."""

    def test_decode_png_image(self):
        from llenvs.inference.backends.vllm import _decode_image

        # Create a tiny valid PNG as base64
        # (1x1 red pixel PNG)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        b64 = base64.b64encode(png_bytes).decode()
        img = ImageContent(data=b64, media_type="image/png")

        pil_img = _decode_image(img)
        assert pil_img.size == (1, 1)

    def test_decode_returns_pil_image(self):
        from PIL import Image

        from llenvs.inference.backends.vllm import _decode_image

        # Minimal valid image
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        b64 = base64.b64encode(png_bytes).decode()
        img = ImageContent(data=b64)
        result = _decode_image(img)
        assert isinstance(result, Image.Image)


class TestVLLMImageExtraction:
    """Test extracting images from ChatMessage lists for vLLM."""

    def test_extract_no_images(self):
        from llenvs.inference.backends.vllm import _extract_images

        msgs = [ChatMessage(role="user", content="Hello")]
        images = _extract_images(msgs)
        assert images == []

    def test_extract_single_image(self):
        from llenvs.inference.backends.vllm import _extract_images

        img = ImageContent(data="abc")
        msgs = [ChatMessage(role="user", content="Hi", images=(img,))]
        images = _extract_images(msgs)
        assert len(images) == 1
        assert images[0] is img

    def test_extract_multiple_images_across_messages(self):
        from llenvs.inference.backends.vllm import _extract_images

        img1 = ImageContent(data="a")
        img2 = ImageContent(data="b")
        img3 = ImageContent(data="c")
        msgs = [
            ChatMessage(role="user", content="First", images=(img1, img2)),
            ChatMessage(role="assistant", content="I see"),
            ChatMessage(role="user", content="Second", images=(img3,)),
        ]
        images = _extract_images(msgs)
        assert len(images) == 3
        assert [i.data for i in images] == ["a", "b", "c"]

    def test_extract_preserves_order(self):
        from llenvs.inference.backends.vllm import _extract_images

        imgs = [ImageContent(data=f"img{i}") for i in range(5)]
        msgs = [
            ChatMessage(role="user", content="A", images=tuple(imgs[:2])),
            ChatMessage(role="user", content="B", images=tuple(imgs[2:])),
        ]
        images = _extract_images(msgs)
        assert [i.data for i in images] == [f"img{i}" for i in range(5)]


class TestVLLMVisionDetection:
    """Test VLM detection in vLLM backend."""

    def test_is_vlm_model_llava(self):
        from llenvs.inference.backends.vllm import _is_vlm_model

        config = MagicMock()
        config.hf_config = MagicMock()
        config.hf_config.model_type = "llava"
        assert _is_vlm_model(config) is True

    def test_is_vlm_model_llama(self):
        from llenvs.inference.backends.vllm import _is_vlm_model

        config = MagicMock()
        config.hf_config = MagicMock()
        config.hf_config.model_type = "llama"
        assert _is_vlm_model(config) is False

    def test_non_vlm_capabilities(self):
        """Non-VLM model should have supports_vision=False."""
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)
        backend._model_path = "test-model"
        backend._llm = MagicMock()
        backend._tokenizer = MagicMock()
        backend._max_context_length = 4096
        backend._chat_template_kwargs = {}
        backend._is_vlm = False
        backend._VLLMSamplingParams = MagicMock()

        assert backend.capabilities.supports_vision is False

    def test_vlm_capabilities(self):
        """VLM model should have supports_vision=True."""
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)
        backend._model_path = "llava-hf/llava-1.5-7b-hf"
        backend._llm = MagicMock()
        backend._tokenizer = MagicMock()
        backend._max_context_length = 4096
        backend._chat_template_kwargs = {}
        backend._is_vlm = True
        backend._VLLMSamplingParams = MagicMock()

        assert backend.capabilities.supports_vision is True


class TestVLLMVisionGeneration:
    """Test that vLLM passes multi_modal_data when images present."""

    def test_generate_chat_with_images_calls_generate_vlm(self):
        """generate_chat with images should use _generate_vlm path."""
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)
        backend._model_path = "llava-hf/llava-1.5-7b-hf"
        backend._is_vlm = True
        backend._chat_template_kwargs = {}

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<image> What is this?"
        backend._tokenizer = mock_tokenizer
        backend._VLLMSamplingParams = MagicMock(return_value=MagicMock())
        backend._max_context_length = 4096

        mock_llm = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [
            MagicMock(
                text="A cat",
                finish_reason="stop",
                logprobs=None,
                token_ids=[1, 2],
            )
        ]
        mock_output.prompt_token_ids = [1, 2, 3]
        mock_output.request_id = "req1"
        mock_llm.generate.return_value = [mock_output]
        backend._llm = mock_llm

        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        b64 = base64.b64encode(png_bytes).decode()
        img = ImageContent(data=b64)
        msgs = [ChatMessage(role="user", content="What is this?", images=(img,))]

        backend.generate_chat(msgs, SamplingParams())

        # Verify generate was called with dict inputs containing multi_modal_data
        call_args = mock_llm.generate.call_args
        inputs = call_args[0][0]  # First positional arg (list of inputs)
        assert isinstance(inputs[0], dict)
        assert "multi_modal_data" in inputs[0]

    def test_generate_chat_no_images_no_multi_modal_data(self):
        """generate_chat without images should not pass multi_modal_data."""
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)
        backend._model_path = "llava-hf/llava-1.5-7b-hf"
        backend._is_vlm = True
        backend._chat_template_kwargs = {}

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "What is 2+2?"
        backend._tokenizer = mock_tokenizer

        mock_llm = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [
            MagicMock(
                text="4",
                finish_reason="stop",
                logprobs=None,
                token_ids=[1],
            )
        ]
        mock_output.prompt_token_ids = [1, 2]
        mock_output.request_id = "req2"
        mock_llm.generate.return_value = [mock_output]
        backend._llm = mock_llm
        backend._VLLMSamplingParams = MagicMock(return_value=MagicMock())
        backend._max_context_length = 4096

        msgs = [ChatMessage(role="user", content="What is 2+2?")]
        backend.generate_chat(msgs, SamplingParams())

        # Should call generate with plain string prompt
        call_args = mock_llm.generate.call_args
        inputs = call_args[0][0]  # First positional arg
        assert isinstance(inputs[0], str)


class TestVLLMBatchVision:
    """Test batch generation with vision models."""

    def test_batch_with_mixed_images(self):
        """Batch should handle mix of text-only and image messages."""
        from llenvs.inference.backends.vllm import VLLMBackend

        backend = VLLMBackend.__new__(VLLMBackend)
        backend._model_path = "llava-model"
        backend._is_vlm = True
        backend._chat_template_kwargs = {}

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = [
            "<image> Describe",
            "Hello world",
        ]
        backend._tokenizer = mock_tokenizer

        mock_llm = MagicMock()
        mock_outputs = []
        for i in range(2):
            out = MagicMock()
            out.outputs = [
                MagicMock(
                    text=f"Response {i}",
                    finish_reason="stop",
                    logprobs=None,
                    token_ids=[1],
                )
            ]
            out.prompt_token_ids = [1, 2]
            out.request_id = f"req{i}"
            mock_outputs.append(out)
        mock_llm.generate.return_value = mock_outputs
        backend._llm = mock_llm
        backend._VLLMSamplingParams = MagicMock(return_value=MagicMock())
        backend._max_context_length = 4096

        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        b64 = base64.b64encode(png_bytes).decode()

        batch = [
            [ChatMessage(role="user", content="Describe", images=(ImageContent(data=b64),))],
            [ChatMessage(role="user", content="Hello")],
        ]

        results = backend.generate_chat_batch(batch, SamplingParams())
        assert len(results) == 2

        # Verify dict-based inputs were used (due to images in first conv)
        call_args = mock_llm.generate.call_args
        inputs = call_args[0][0]
        # First input should be dict (has images), second can be string
        assert isinstance(inputs[0], dict)
        assert "multi_modal_data" in inputs[0]


# ===========================================================================
# 2. HuggingFace VLM Support
# ===========================================================================


class TestHFVisionDetection:
    """Test VLM detection in HuggingFace backend."""

    def test_detect_vlm_model_type(self):
        from llenvs.inference.backends.huggingface import _is_vlm_model

        mock_config = MagicMock()
        mock_config.model_type = "llava"
        assert _is_vlm_model(mock_config) is True

    def test_detect_non_vlm_model_type(self):
        from llenvs.inference.backends.huggingface import _is_vlm_model

        mock_config = MagicMock()
        mock_config.model_type = "llama"
        assert _is_vlm_model(mock_config) is False

    def test_detect_qwen_vl(self):
        from llenvs.inference.backends.huggingface import _is_vlm_model

        mock_config = MagicMock()
        mock_config.model_type = "qwen2_vl"
        assert _is_vlm_model(mock_config) is True

    def test_detect_paligemma(self):
        from llenvs.inference.backends.huggingface import _is_vlm_model

        mock_config = MagicMock()
        mock_config.model_type = "paligemma"
        assert _is_vlm_model(mock_config) is True

    def test_detect_phi3_vision(self):
        from llenvs.inference.backends.huggingface import _is_vlm_model

        mock_config = MagicMock()
        mock_config.model_type = "phi3_v"
        assert _is_vlm_model(mock_config) is True

    def test_detect_internvl(self):
        from llenvs.inference.backends.huggingface import _is_vlm_model

        mock_config = MagicMock()
        mock_config.model_type = "internvl_chat"
        assert _is_vlm_model(mock_config) is True


class TestHFImageDecoding:
    """Test image decoding for HuggingFace backend."""

    def test_decode_image(self):
        from llenvs.inference.backends.huggingface import _decode_image

        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        b64 = base64.b64encode(png_bytes).decode()
        img = ImageContent(data=b64)
        pil_img = _decode_image(img)
        assert pil_img.size == (1, 1)


class TestHFImageExtraction:
    """Test extracting images from ChatMessage lists for HF backend."""

    def test_extract_no_images(self):
        from llenvs.inference.backends.huggingface import _extract_images

        msgs = [ChatMessage(role="user", content="Hello")]
        assert _extract_images(msgs) == []

    def test_extract_images(self):
        from llenvs.inference.backends.huggingface import _extract_images

        img = ImageContent(data="abc")
        msgs = [ChatMessage(role="user", content="Hi", images=(img,))]
        assert len(_extract_images(msgs)) == 1


class TestHFVisionCapabilities:
    """Test that HF backend reports vision capability for VLMs."""

    def test_vlm_supports_vision(self):
        from llenvs.inference.backends.huggingface import HuggingFaceBackend

        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend._model_path = "llava-model"
        backend._is_vlm = True
        backend._max_context_length = 4096
        backend._tokenizer = MagicMock()
        backend._model = MagicMock()
        backend._torch = MagicMock()
        backend._chat_template_kwargs = {}

        assert backend.capabilities.supports_vision is True

    def test_non_vlm_no_vision(self):
        from llenvs.inference.backends.huggingface import HuggingFaceBackend

        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
        backend._model_path = "llama-model"
        backend._is_vlm = False
        backend._max_context_length = 4096
        backend._tokenizer = MagicMock()
        backend._model = MagicMock()
        backend._torch = MagicMock()
        backend._chat_template_kwargs = {}

        assert backend.capabilities.supports_vision is False


# ===========================================================================
# 3. Memory Management
# ===========================================================================


class TestImageHistoryTruncation:
    """Test max_image_history on TrajectoryRunner."""

    def _make_image_state(self, step: int, num_images: int) -> State[None]:
        """Create a state with images."""
        images = tuple(ImageContent(data=f"img_step{step}_{i}") for i in range(num_images))
        msgs = ()
        if step > 0:
            # Build message history with images at each step
            msg_list = []
            for s in range(step):
                msg_list.append({"role": "assistant", "content": f"Response {s}"})
                step_imgs = [
                    {"data": f"img_step{s + 1}_{i}", "media_type": "image/png"}
                    for i in range(num_images)
                ]
                msg_list.append({"role": "user", "content": f"Step {s + 1}", "images": step_imgs})
            msgs = tuple(msg_list)

        return State(
            observation=Observation(prompt="Initial", images=images, messages=msgs),
            hidden=None,
            metadata=StateMetadata(step=step, episode_id="test"),
        )

    def test_no_truncation_by_default(self):
        """Without max_image_history, all images preserved."""
        from llenvs.core.environment import EnvironmentSpec
        from llenvs.evaluation.runner import TrajectoryRunner

        mock_env = MagicMock()
        mock_env.spec = EnvironmentSpec(name="test")

        runner = TrajectoryRunner(environment=mock_env, backend=MagicMock())
        assert runner.max_image_history is None

    def test_truncation_removes_old_images(self):
        """With max_image_history=1, only most recent image is kept."""
        from llenvs.core.environment import EnvironmentSpec
        from llenvs.evaluation.runner import TrajectoryRunner

        mock_env = MagicMock()
        mock_env.spec = EnvironmentSpec(name="test")

        runner = TrajectoryRunner(
            environment=mock_env,
            backend=MagicMock(),
            max_image_history=1,
        )

        state = self._make_image_state(step=3, num_images=1)
        messages = runner._build_messages(state)

        # Count images in messages
        img_count = sum(len(m.images) for m in messages)
        assert img_count == 1  # Only most recent

    def test_truncation_preserves_text(self):
        """Image truncation replaces images with text placeholder but keeps message."""
        from llenvs.core.environment import EnvironmentSpec
        from llenvs.evaluation.runner import TrajectoryRunner

        mock_env = MagicMock()
        mock_env.spec = EnvironmentSpec(name="test")

        runner = TrajectoryRunner(
            environment=mock_env,
            backend=MagicMock(),
            max_image_history=0,
        )

        img = ImageContent(data="abc")
        state = State(
            observation=Observation(prompt="What is this?", images=(img,)),
            hidden=None,
            metadata=StateMetadata(step=0, episode_id="test"),
        )
        messages = runner._build_messages(state)

        # All images should be stripped
        img_count = sum(len(m.images) for m in messages)
        assert img_count == 0
        # Text content should still be there
        user_msgs = [m for m in messages if m.role == "user"]
        assert user_msgs[0].content == "What is this?"


class TestTrajectoryStripImages:
    """Test Trajectory.strip_images() method."""

    def _make_transition(self, step: int, with_image: bool = True) -> Transition[None]:
        """Create a transition with optional images."""
        images = (ImageContent(data=f"img_{step}"),) if with_image else ()

        state = State(
            observation=Observation(prompt=f"Step {step}", images=images),
            hidden=None,
            metadata=StateMetadata(step=step, episode_id="test"),
        )
        next_state = State(
            observation=Observation(prompt=f"Step {step + 1}", images=images),
            hidden=None,
            metadata=StateMetadata(step=step + 1, episode_id="test"),
        )
        return Transition(
            state=state,
            action=Action(text=f"action_{step}"),
            next_state=next_state,
            rewards=SignalBundle(signals=()),
        )

    def test_strip_images_returns_new_trajectory(self):
        state = State(
            observation=Observation(prompt="Start", images=(ImageContent(data="init"),)),
            hidden=None,
            metadata=StateMetadata(step=0, episode_id="test"),
        )
        traj = Trajectory.create(state)
        traj.add_transition(self._make_transition(0))

        stripped = traj.strip_images()
        assert stripped is not traj
        assert stripped.episode_id == traj.episode_id

    def test_strip_images_removes_all_images(self):
        state = State(
            observation=Observation(prompt="Start", images=(ImageContent(data="init"),)),
            hidden=None,
            metadata=StateMetadata(step=0, episode_id="test"),
        )
        traj = Trajectory.create(state)
        traj.add_transition(self._make_transition(0))
        traj.add_transition(self._make_transition(1))

        stripped = traj.strip_images()

        # Initial state should have no images
        assert stripped.initial_state.observation.images == ()

        # All transition states should have no images
        for t in stripped.transitions:
            assert t.state.observation.images == ()
            assert t.next_state.observation.images == ()

    def test_strip_images_preserves_text(self):
        state = State(
            observation=Observation(prompt="Start", images=(ImageContent(data="init"),)),
            hidden=None,
            metadata=StateMetadata(step=0, episode_id="test"),
        )
        traj = Trajectory.create(state)
        traj.add_transition(self._make_transition(0))

        stripped = traj.strip_images()
        assert stripped.initial_state.observation.prompt == "Start"
        assert stripped.transitions[0].state.observation.prompt == "Step 0"

    def test_strip_images_no_op_when_no_images(self):
        state = State(
            observation=Observation(prompt="Start"),
            hidden=None,
            metadata=StateMetadata(step=0, episode_id="test"),
        )
        traj = Trajectory.create(state)
        traj.add_transition(self._make_transition(0, with_image=False))

        stripped = traj.strip_images()
        assert len(stripped.transitions) == len(traj.transitions)


class TestLogConfigStripImages:
    """Test strip_images field on LogConfig."""

    def test_default_false(self):
        from llenvs.evaluation.logging import LogConfig

        config = LogConfig()
        assert config.strip_images is False

    def test_explicit_true(self):
        from llenvs.evaluation.logging import LogConfig

        config = LogConfig(strip_images=True)
        assert config.strip_images is True


# ===========================================================================
# 4. Gymnasium ImageObservationMapper
# ===========================================================================


class TestMultimodalObservationMapper:
    """Test MultimodalObservationMapper protocol."""

    def test_protocol_check(self):
        from llenvs.adapters.gymnasium import MultimodalObservationMapper

        class MyMapper:
            _multimodal = True

            def map(self, obs: Any, info: dict[str, Any]) -> tuple[str, tuple[ImageContent, ...]]:
                return ("text", ())

            def describe(self) -> str:
                return "desc"

        assert isinstance(MyMapper(), MultimodalObservationMapper)

    def test_old_mapper_not_multimodal(self):
        from llenvs.adapters.gymnasium import MultimodalObservationMapper, ObservationMapper

        class OldMapper:
            def map(self, obs: Any, info: dict[str, Any]) -> str:
                return "text"

            def describe(self) -> str:
                return "desc"

        mapper = OldMapper()
        assert isinstance(mapper, ObservationMapper)
        # Old mappers without _multimodal attribute are not MultimodalObservationMapper
        assert not isinstance(mapper, MultimodalObservationMapper)

    def test_image_observation_mapper_is_multimodal(self):
        from llenvs.adapters.gymnasium import ImageObservationMapper, MultimodalObservationMapper

        mapper = ImageObservationMapper()
        assert isinstance(mapper, MultimodalObservationMapper)


class TestImageObservationMapper:
    """Test ImageObservationMapper for pixel observations."""

    def test_map_returns_tuple(self):
        from llenvs.adapters.gymnasium import ImageObservationMapper

        mapper = ImageObservationMapper()
        obs = np.zeros((64, 64, 3), dtype=np.uint8)
        text, images = mapper.map(obs, {})
        assert isinstance(text, str)
        assert isinstance(images, tuple)
        assert len(images) == 1

    def test_map_produces_valid_image_content(self):
        from llenvs.adapters.gymnasium import ImageObservationMapper

        mapper = ImageObservationMapper()
        obs = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        text, images = mapper.map(obs, {})
        img = images[0]
        assert isinstance(img, ImageContent)
        assert img.media_type == "image/png"
        # Should be valid base64
        decoded = base64.b64decode(img.data)
        assert decoded[:4] == b"\x89PNG"

    def test_jpeg_format(self):
        from llenvs.adapters.gymnasium import ImageObservationMapper

        mapper = ImageObservationMapper(format="jpeg")
        obs = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        text, images = mapper.map(obs, {})
        assert images[0].media_type == "image/jpeg"

    def test_custom_text_description(self):
        from llenvs.adapters.gymnasium import ImageObservationMapper

        mapper = ImageObservationMapper(text_description="Game frame:")
        obs = np.zeros((10, 10, 3), dtype=np.uint8)
        text, images = mapper.map(obs, {})
        assert "Game frame:" in text

    def test_default_text(self):
        from llenvs.adapters.gymnasium import ImageObservationMapper

        mapper = ImageObservationMapper()
        obs = np.zeros((10, 10, 3), dtype=np.uint8)
        text, images = mapper.map(obs, {})
        assert text  # Should have some default text

    def test_describe(self):
        from llenvs.adapters.gymnasium import ImageObservationMapper

        mapper = ImageObservationMapper()
        desc = mapper.describe()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_grayscale_observation(self):
        """Should handle 2D (H, W) grayscale arrays."""
        from llenvs.adapters.gymnasium import ImageObservationMapper

        mapper = ImageObservationMapper()
        obs = np.zeros((64, 64), dtype=np.uint8)
        text, images = mapper.map(obs, {})
        assert len(images) == 1

    def test_rgba_observation(self):
        """Should handle (H, W, 4) RGBA arrays."""
        from llenvs.adapters.gymnasium import ImageObservationMapper

        mapper = ImageObservationMapper()
        obs = np.zeros((32, 32, 4), dtype=np.uint8)
        text, images = mapper.map(obs, {})
        assert len(images) == 1


try:
    import gymnasium  # noqa: F401

    _HAS_GYMNASIUM = True
except ImportError:
    _HAS_GYMNASIUM = False


@pytest.mark.skipif(not _HAS_GYMNASIUM, reason="gymnasium not installed")
class TestGymnasiumMultimodalIntegration:
    """Test GymnasiumEnvironment with multimodal observation mappers."""

    def test_multimodal_mapper_produces_images_in_observation(self):
        """When using MultimodalObservationMapper, Observation.images should be populated."""
        import gymnasium.spaces as spaces

        from llenvs.adapters.gymnasium import (
            AutoActionMapper,
            GymnasiumEnvironment,
            ImageObservationMapper,
        )

        mock_gym = MagicMock()
        mock_gym.observation_space = MagicMock()
        mock_gym.action_space = spaces.Discrete(2)

        obs_array = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_gym.reset.return_value = (obs_array, {})

        env = GymnasiumEnvironment(
            gym_env=mock_gym,
            observation_mapper=ImageObservationMapper(),
            action_mapper=AutoActionMapper(
                mock_gym.action_space, action_names={0: "left", 1: "right"}
            ),
            num_tasks=10,
        )

        state, info = env.reset(options={"task_index": 0})
        assert len(state.observation.images) == 1
        assert isinstance(state.observation.images[0], ImageContent)

    def test_step_produces_images(self):
        """Step should also produce images in observation."""
        import gymnasium.spaces as spaces

        from llenvs.adapters.gymnasium import (
            AutoActionMapper,
            GymnasiumEnvironment,
            ImageObservationMapper,
        )

        mock_gym = MagicMock()
        obs_array = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_gym.reset.return_value = (obs_array, {})
        mock_gym.step.return_value = (obs_array, 1.0, False, False, {})
        mock_gym.action_space = spaces.Discrete(2)

        env = GymnasiumEnvironment(
            gym_env=mock_gym,
            observation_mapper=ImageObservationMapper(),
            action_mapper=AutoActionMapper(
                mock_gym.action_space, action_names={0: "left", 1: "right"}
            ),
            num_tasks=10,
        )

        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="left"))
        assert len(result.next_state.observation.images) == 1

    def test_step_images_in_history(self):
        """Step should include images in message history."""
        import gymnasium.spaces as spaces

        from llenvs.adapters.gymnasium import (
            AutoActionMapper,
            GymnasiumEnvironment,
            ImageObservationMapper,
        )

        mock_gym = MagicMock()
        obs_array = np.zeros((32, 32, 3), dtype=np.uint8)
        mock_gym.reset.return_value = (obs_array, {})
        mock_gym.step.return_value = (obs_array, 0.5, False, False, {})
        mock_gym.action_space = spaces.Discrete(2)

        env = GymnasiumEnvironment(
            gym_env=mock_gym,
            observation_mapper=ImageObservationMapper(),
            action_mapper=AutoActionMapper(
                mock_gym.action_space, action_names={0: "left", 1: "right"}
            ),
            num_tasks=10,
        )

        state, _ = env.reset(options={"task_index": 0})
        result = env.step(state, Action(text="left"))

        # Check that the user message in history includes images
        user_msgs = [m for m in result.next_state.observation.messages if m.get("role") == "user"]
        assert len(user_msgs) > 0
        assert "images" in user_msgs[-1]


# ===========================================================================
# 5. DatasetProvider Images
# ===========================================================================


class TestTaskItemImages:
    """Test TaskItem with images field."""

    def test_default_no_images(self):
        from llenvs.integrations.dataset_provider import TaskItem

        item = TaskItem(
            task_index=0,
            prompt="Hello",
            messages=(),
            ground_truth="world",
            metadata={},
        )
        assert item.images == ()

    def test_with_images(self):
        from llenvs.integrations.dataset_provider import TaskItem

        imgs = (ImageContent(data="a"), ImageContent(data="b"))
        item = TaskItem(
            task_index=0,
            prompt="Describe",
            messages=(),
            ground_truth=None,
            metadata={},
            images=imgs,
        )
        assert len(item.images) == 2
        assert item.images[0].data == "a"

    def test_frozen(self):
        from llenvs.integrations.dataset_provider import TaskItem

        item = TaskItem(
            task_index=0,
            prompt="Hello",
            messages=(),
            ground_truth=None,
            metadata={},
        )
        with pytest.raises(AttributeError):
            item.images = ()  # type: ignore[misc]


class TestDatasetProviderImages:
    """Test DatasetProvider propagates images."""

    def _make_mock_env(self, with_images: bool = False) -> MagicMock:
        from llenvs.core.environment import EnvironmentSpec

        mock_env = MagicMock()
        mock_env.spec = EnvironmentSpec(
            name="test",
            supports_task_index=True,
            supports_len=True,
        )
        mock_env.__len__ = MagicMock(return_value=5)

        images = (ImageContent(data="test_img"),) if with_images else ()
        obs = Observation(prompt="Test prompt", images=images)
        state = State(
            observation=obs,
            hidden=MagicMock(expected_answer="42"),
            metadata=StateMetadata(step=0, episode_id="ep1"),
        )
        mock_env.reset.return_value = (state, {"task_index": 0})
        return mock_env

    def test_getitem_includes_images(self):
        from llenvs.integrations.dataset_provider import DatasetProvider

        env = self._make_mock_env(with_images=True)
        provider = DatasetProvider(env)
        item = provider[0]
        assert len(item.images) == 1
        assert item.images[0].data == "test_img"

    def test_getitem_no_images(self):
        from llenvs.integrations.dataset_provider import DatasetProvider

        env = self._make_mock_env(with_images=False)
        provider = DatasetProvider(env)
        item = provider[0]
        assert item.images == ()

    def test_to_hf_dataset_includes_images(self):
        """to_hf_dataset should include image data."""
        from llenvs.integrations.dataset_provider import DatasetProvider

        env = self._make_mock_env(with_images=True)
        provider = DatasetProvider(env)

        mock_dataset_cls = MagicMock()
        mock_dataset_cls.from_dict.return_value = MagicMock()

        with patch.dict("sys.modules", {"datasets": MagicMock(Dataset=mock_dataset_cls)}):
            provider.to_hf_dataset(indices=[0])

            call_args = mock_dataset_cls.from_dict.call_args[0][0]
            assert "images" in call_args
