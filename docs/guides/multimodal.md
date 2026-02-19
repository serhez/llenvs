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

| Backend | Vision Support |
|---------|---------------|
| OpenAI | Yes |
| Anthropic | Yes |
| OpenRouter | Yes |
| vLLM | No |
| HuggingFace | No |

Check at runtime:

```python
backend.capabilities.supports_vision  # True/False
```

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

## Adapters with Image Support

| Adapter | Image Mode |
|---------|-----------|
| [Craftax](../adapters/craftax.md) | `observation_mode="pixels"` — renders game frames as PNG |
| [Gymnasium](../adapters/gymnasium.md) | Custom `ObservationMapper` needed for 2D+ observations |

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
