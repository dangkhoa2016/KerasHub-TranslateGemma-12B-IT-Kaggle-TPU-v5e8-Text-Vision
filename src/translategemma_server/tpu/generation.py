from __future__ import annotations

import ast
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

VISION_PROMPT_PROFILE = "translategemma_structured_v6_translation_output_only_frozen"


@dataclass(frozen=True)
class GenerationPlan:
    prompt_tokens: int
    max_new_tokens: int
    max_length: int
    bucketed: bool

    def as_dict(self) -> dict:
        return asdict(self)


# This serving subset keeps only the stable language/prompt helpers used by the
# v10 engine. Benchmark-only lexical-audit/strategy helpers are intentionally
# absent from the REST project.
_LANGUAGE_CODES = {
    "english": "en",
    "vietnamese": "vi",
    "viet nam": "vi",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "simplified chinese": "zh",
    "traditional chinese": "zh",
    "thai": "th",
    "indonesian": "id",
    "malay": "ms",
    "russian": "ru",
    "arabic": "ar",
    "hindi": "hi",
}


def language_code(language: str, explicit_code: str | None = None) -> str:
    if explicit_code and explicit_code.strip():
        return explicit_code.strip()

    value = language.strip()
    lowered = value.casefold()
    if lowered in _LANGUAGE_CODES:
        return _LANGUAGE_CODES[lowered]
    if re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", value):
        return value
    raise ValueError(
        f"No language code is known for {language!r}. Pass an explicit code."
    )


def plan_generation(
    prompt_tokens: int,
    max_new_tokens: int,
    *,
    buckets: Iterable[int],
    bucket_step: int,
    bucketing: bool,
    minimum_length: int = 0,
) -> GenerationPlan:
    required = max(
        1,
        int(prompt_tokens) + int(max_new_tokens),
        int(minimum_length),
    )
    if not bucketing:
        max_length = required
    else:
        choices = sorted(set(int(value) for value in buckets if int(value) > 0))
        max_length = next((value for value in choices if value >= required), 0)
        if not max_length:
            if bucket_step <= 0:
                raise ValueError("bucket_step must be positive")
            max_length = int(math.ceil(required / bucket_step) * bucket_step)

    return GenerationPlan(
        prompt_tokens=int(prompt_tokens),
        max_new_tokens=int(max_new_tokens),
        max_length=int(max_length),
        bucketed=bool(bucketing),
    )


def scalar_text(value: Any) -> str:
    """Normalize Keras/JAX/tokenizer outputs to one plain Python string."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, str):
        text = value.strip()
        if len(text) >= 2 and text[0] in "[(" and text[-1] in ")]":
            try:
                return scalar_text(ast.literal_eval(text))
            except (ValueError, SyntaxError):
                return text
        return text
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return scalar_text(value[0])
        return "\n".join(
            part
            for part in (scalar_text(item) for item in value)
            if part
        ).strip()
    return str(value).strip()


def _wrap_gemma_user_turn(body: str) -> str:
    return f"<start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n"


def translation_prompt(
    text: str,
    source_lang: str,
    target_lang: str,
    *,
    source_lang_code: str | None = None,
    target_lang_code: str | None = None,
) -> str:
    """Build the same stable structured text prompt used by the v10 engine."""
    src_code = language_code(source_lang, source_lang_code)
    tgt_code = language_code(target_lang, target_lang_code)
    body = (
        f"You are a professional {source_lang} ({src_code}) to {target_lang} ({tgt_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {source_lang} text "
        f"while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities. "
        f"Produce only the {target_lang} translation, without any additional explanations or commentary. "
        f"Please translate the following {source_lang} text into {target_lang}:\n\n\n{text.strip()}"
    )
    return _wrap_gemma_user_turn(body)


def vision_translation_prompt(
    source_lang: str,
    target_lang: str,
    *,
    source_lang_code: str | None = None,
    target_lang_code: str | None = None,
) -> str:
    """Build the frozen v6/v10 image-translation prompt contract."""
    src_code = language_code(source_lang, source_lang_code)
    tgt_code = language_code(target_lang, target_lang_code)
    body = (
        f"You are a professional {source_lang} ({src_code}) to {target_lang} ({tgt_code}) translator. "
        f"The image displays {source_lang} text. Provide the {target_lang} translation of that text. "
        f"Output only the translation, nothing else. "
        f"Do not output the {source_lang} text and do not comment on the image.\n\n"
        "<start_of_image>"
    )
    return _wrap_gemma_user_turn(body)
