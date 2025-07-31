"""Utility helpers for data processing."""

import base64
from io import BytesIO
from typing import Any, Dict

from PIL import Image


def _pil_to_base64(image: Image.Image) -> str:
    """Encode a PIL image to a base64 string.

    This allows storing the image inside JSON without relying on the
    filesystem.
    """

    buffer = BytesIO()
    # Use PNG to avoid issues with different JPEG encoders.
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def to_funsd_format(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a record to FUNSD-style format.

    The original dataset contains a PIL ``Image`` object which is not JSON
    serialisable.  We encode the image to a base64 string so that the entire
    record can be written to JSON without errors.
    """

    converted = dict(record)
    img = converted.get("image")
    if isinstance(img, Image.Image):
        converted["image"] = _pil_to_base64(img)
    return converted