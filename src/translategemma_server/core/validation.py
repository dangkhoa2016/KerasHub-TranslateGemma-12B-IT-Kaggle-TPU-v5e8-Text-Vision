from __future__ import annotations

import base64
import binascii
import io

from PIL import Image

from .errors import ValidationError


def _language(data: dict, key: str, legacy_key: str) -> str:
    value = data.get(key, data.get(legacy_key))
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"Field '{key}' must be a non-empty string")
    return value.strip()


def _max_tokens(data: dict, config) -> int:
    value = data.get(
        "max_new_tokens",
        data.get("max_tokens", config.default_output_tokens),
    )
    if isinstance(value, bool):
        raise ValidationError("max_new_tokens must be an integer")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("max_new_tokens must be an integer") from exc

    if not 1 <= value <= config.max_output_tokens:
        raise ValidationError(
            f"max_new_tokens must be between 1 and {config.max_output_tokens}"
        )
    return value


def parse_translation_payload(data: dict, config) -> dict:
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValidationError("Field 'text' must be a non-empty string")
    if len(text) > config.max_input_chars:
        raise ValidationError("Text input is too large")

    return {
        "text": text.strip(),
        "src": _language(data, "source_lang", "src"),
        "tgt": _language(data, "target_lang", "tgt"),
        "src_code": data.get("source_lang_code"),
        "tgt_code": data.get("target_lang_code"),
        "max_tokens": _max_tokens(data, config),
    }


def parse_image_translation_binary(binary: bytes, data: dict, config) -> dict:
    """Validate decoded image bytes and build the common vision job payload."""
    if not config.vision_enabled:
        raise ValidationError("Vision translation is disabled")
    if not isinstance(binary, (bytes, bytearray)) or not binary:
        raise ValidationError("Image file must not be empty")
    if len(binary) > config.max_image_bytes:
        raise ValidationError("Image is too large")

    try:
        with Image.open(io.BytesIO(binary)) as source:
            if source.width * source.height > config.max_image_pixels:
                raise ValidationError("Image has too many pixels")
            image = source.convert("RGB").copy()
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("Decoded image is not a supported image") from exc

    return {
        "text": "",
        "image": image,
        "src": _language(data, "source_lang", "src"),
        "tgt": _language(data, "target_lang", "tgt"),
        "src_code": data.get("source_lang_code"),
        "tgt_code": data.get("target_lang_code"),
        "max_tokens": _max_tokens(data, config),
    }


def parse_image_translation_payload(data: dict, config) -> dict:
    if not config.vision_enabled:
        raise ValidationError("Vision translation is disabled")

    raw = data.get("image_base64", data.get("image"))
    if not isinstance(raw, str) or not raw.strip():
        raise ValidationError(
            "Field 'image_base64' must be a non-empty base64 string"
        )

    value = raw.strip()
    if value.startswith("data:"):
        try:
            value = value.split(",", 1)[1]
        except IndexError as exc:
            raise ValidationError("Invalid data URL") from exc

    try:
        binary = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError("image_base64 is not valid base64") from exc

    return parse_image_translation_binary(binary, data, config)
