# Multimodal Observations

LLEnvs supports image observations alongside text, enabling evaluation of vision-language models (VLMs) on visual environments.

## ImageContent

Images are represented as base64-encoded data:

```python
from llenvs.core import ImageContent

img = ImageContent(data="iVBORw0KGgo...", media_type="image/png")
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data` | `str` | required | Base64-encoded image data |
| `media_type` | `str` | `"image/png"` | MIME type (`image/png`, `image/jpeg`) |

## Observation Images

Environments produce image observations via the `images` field on `Observation`:

```python
from llenvs.core.state import Observation, ImageContent

obs = Observation(
    prompt="What do you see in this image?",
    images=(ImageContent(data="iVBOR..."),),
)
```

Text-only environments leave `images` as an empty tuple (the default).

## Pipeline

Images flow through the full pipeline:

```
Environment.reset() / .step()
    → Observation(prompt="...", images=(...))
        → TrajectoryRunner._build_messages()
            → ChatMessage(role="user", content="...", images=(...))
                → .to_dict()        # OpenAI format
                → .to_anthropic_dict()  # Anthropic format
```

### OpenAI Format

`ChatMessage.to_dict()` produces the OpenAI vision format when images are present:

```python
msg = ChatMessage(role="user", content="What is this?", images=(img,))
msg.to_dict()
# {
#     "role": "user",
#     "content": [
#         {"type": "text", "text": "What is this?"},
#         {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
#     ]
# }
```

Without images, `to_dict()` produces standard format (`"content": "text"`).

### Anthropic Format

`ChatMessage.to_anthropic_dict()` produces Anthropic's image format:

```python
msg.to_anthropic_dict()
# {
#     "role": "user",
#     "content": [
#         {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR..."}},
#         {"type": "text", "text": "What is this?"},
#     ]
# }
```

## Backend Support

| Backend | Vision Support | Notes |
|---------|---------------|-------|
| OpenAI | Yes | Via `image_url` content blocks |
| Anthropic | Yes | Via `base64` source blocks |
| OpenRouter | Yes | Via OpenAI format |
| vLLM | Auto-detected | VLMs (LLaVA, Qwen-VL, PaliGemma, etc.) via `multi_modal_data` |
| HuggingFace | Auto-detected | VLMs via `AutoProcessor` + `AutoModelForVision2Seq` |

Check at runtime:

```python
backend.capabilities.supports_vision  # True/False
```

### Local VLM Support (vLLM / HuggingFace)

Both local backends auto-detect vision-language models by checking `model_type` against a known list (llava, qwen2_vl, paligemma, internvl_chat, phi3_v, etc.). When a VLM is detected:

- `capabilities.supports_vision` returns `True`
- Images from `ChatMessage.images` are decoded from base64 to PIL Images
- **vLLM**: Images are passed via `multi_modal_data={"image": [pil_img, ...]}` alongside the text prompt
- **HuggingFace**: Uses `AutoProcessor` (combined tokenizer + image processor) and `AutoModelForVision2Seq` instead of `AutoModelForCausalLM`

No configuration is needed — just use a VLM model path:

```python
from llenvs.inference.backends import VLLMBackend

backend = VLLMBackend(model_path="llava-hf/llava-1.5-7b-hf")
print(backend.capabilities.supports_vision)  # True
```

Batch generation with images works automatically — each conversation in a batch can have different images.

## Creating Visual Environments

Environments produce images by including `ImageContent` in their observations:

```python
from llenvs.core.state import ImageContent, Observation

# In your environment's reset() or step():
img = ImageContent(data=base64_encoded_png, media_type="image/png")
obs = Observation(
    prompt="Describe what you see and choose an action.",
    images=(img,),
)
```

### Multi-Turn Image History

For multi-turn environments, images in history messages are preserved. The observation's `messages` list can include image data:

```python
obs = Observation(
    prompt="Initial prompt",
    images=(current_frame,),
    messages=(
        {"role": "assistant", "content": "I see a forest."},
        {
            "role": "user",
            "content": "Now look again.",
            "images": [{"data": "iVBOR...", "media_type": "image/png"}],
        },
    ),
)
```

## Memory Management

Multi-turn trajectories with images at every step can consume large amounts of memory. Several mechanisms help control this.

### Image History Truncation

Limit how many images are kept in the message history during evaluation:

```python
from llenvs.evaluation import TrajectoryRunner

runner = TrajectoryRunner(
    environment=env,
    backend=backend,
    max_image_history=5,  # Keep only 5 most recent images
)
```

When `max_image_history` is set, older images are stripped from messages while preserving the text content. `None` (default) keeps all images.

### Trajectory Image Stripping

Remove all images from a completed trajectory (e.g., before serialization or logging):

```python
stripped = trajectory.strip_images()
# Returns a new Trajectory with all Observation.images set to ()
```

### Log Image Stripping

Strip images from evaluation logs to reduce JSONL/wandb payload size:

```python
from llenvs.evaluation.logging import LogConfig

log = LogConfig(
    targets=["file"],
    strip_images=True,  # Images stripped before logging
)
```

## Adapters with Image Support

| Adapter | Image Mode |
|---------|-----------|
| [Craftax](../adapters/craftax.md) | `observation_mode="pixels"` — renders game frames as PNG |
| [Gymnasium](../adapters/gymnasium.md) | `ImageObservationMapper` for pixel observations (Atari presets included) |
| [AlfWorld](../adapters/alfworld.md) | `visual=True` — AI2-THOR ego-centric RGB frames |

## Container Serialization

Image observations serialize to JSON naturally (base64 strings). The container protocol handles `images` in `Observation` round-trip:

```python
from llenvs.container.serialization import serialize_observation, deserialize_observation

serialized = serialize_observation(obs)
# {"prompt": "...", "images": [{"data": "...", "media_type": "image/png"}]}

restored = deserialize_observation(serialized)
assert restored.images[0].data == obs.images[0].data
```

## Prompt Transformers

All prompt transformers preserve images when modifying messages:

- `ChainOfThoughtWrapper`
- `AnswerFormatInjector`
- `ContentWrapper`
- `RoleMapper`
- `PromptTemplateTransformer`
