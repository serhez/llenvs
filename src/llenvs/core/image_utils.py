"""Utilities for encoding pixel arrays to base64 ImageContent."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

from llenvs.core.state import ImageContent

if TYPE_CHECKING:
    import numpy as np


def pixels_to_image_content(obs: np.ndarray) -> ImageContent:
    """Convert a pixel array (HxWx3 uint8) to base64 PNG ImageContent.

    Uses PIL if available, falls back to minimal PNG encoding.
    """
    try:
        from PIL import Image
    except ImportError:
        return _numpy_to_png_content(obs)

    import numpy as np

    img = Image.fromarray(np.asarray(obs).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return ImageContent(data=data, media_type="image/png")


def _numpy_to_png_content(obs: np.ndarray) -> ImageContent:
    """Minimal PNG encoding without PIL."""
    import struct
    import zlib

    import numpy as np

    arr = np.asarray(obs).astype(np.uint8)
    h, w = arr.shape[:2]
    channels = arr.shape[2] if arr.ndim == 3 else 1

    raw = b""
    for row in range(h):
        raw += b"\x00"  # no filter
        raw += arr[row].tobytes()

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    color_type = 2 if channels == 3 else (6 if channels == 4 else 0)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr)
    png += _chunk(b"IDAT", zlib.compress(raw))
    png += _chunk(b"IEND", b"")

    data = base64.b64encode(png).decode("ascii")
    return ImageContent(data=data, media_type="image/png")
