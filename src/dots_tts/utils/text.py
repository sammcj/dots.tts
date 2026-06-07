from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from langcodes import Language as LangcodesLanguage

# `lingua` (language autodetect) and `tn`/WeTextProcessing (text normalisation)
# are imported lazily so this module loads without them. WeTextProcessing pulls
# pynini, which has no macOS arm64 wheel and needs OpenFST to compile from
# source - that breaks installing a relocatable env on a clean Mac. When the
# caller passes an explicit `language` and leaves normalisation off (Cloney's
# path), neither package is reached at runtime; the fallbacks below keep the
# autodetect/normalise paths degrading gracefully instead of crashing on import.

TextLanguage = Literal["zh", "en", "unknown"]

_WHITESPACE_PATTERN = re.compile(r"\s+")


class _IdentityNormalizer:
    """Fallback when WeTextProcessing (tn) isn't installed: text passes through
    unchanged (number/abbreviation expansion is skipped)."""

    def normalize(self, text: str) -> str:
        return text


@lru_cache(maxsize=1)
def get_chinese_text_normalizer():
    try:
        from tn.chinese.normalizer import Normalizer as ZhNormalizer
    except Exception:
        return _IdentityNormalizer()
    return ZhNormalizer()


@lru_cache(maxsize=1)
def get_english_text_normalizer():
    try:
        from tn.english.normalizer import Normalizer as EnNormalizer
    except Exception:
        return _IdentityNormalizer()
    return EnNormalizer()


@lru_cache(maxsize=1)
def get_language_detector():
    try:
        from lingua import Language, LanguageDetectorBuilder
    except Exception:
        return None
    supported_languages = tuple(
        sorted(Language.all(), key=lambda language: language.name)
    )
    return LanguageDetectorBuilder.from_languages(*supported_languages).build()


def _lingua_language_to_code(language) -> str | None:
    if language is None:
        return None
    iso_code_639_1 = getattr(language.iso_code_639_1, "name", None)
    if iso_code_639_1:
        return iso_code_639_1.lower()
    iso_code_639_3 = getattr(language.iso_code_639_3, "name", None)
    if iso_code_639_3:
        return iso_code_639_3.lower()
    return language.name.lower()


def detect(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    detector = get_language_detector()
    if detector is None:
        return None
    language = detector.detect_language_of(stripped)
    return _lingua_language_to_code(language)


def normalize_language_code(language: str | None) -> str | None:
    if language is None:
        return None

    stripped = language.strip()
    if not stripped or stripped.lower() in {"none", "unknown"}:
        return None
    if stripped.startswith("口音:"):
        return stripped

    for resolver in (LangcodesLanguage.get, LangcodesLanguage.find):
        try:
            normalized_language = resolver(stripped).prefer_macrolanguage()
        except Exception:
            continue

        language_code = (normalized_language.language or "").strip().upper()
        if language_code and language_code != "UND":
            return language_code
    return None


def attach_language_tag(text: str, language: str | None) -> str:
    if not text:
        return text

    language_code = normalize_language_code(language)
    if language_code is None:
        return text

    if language_code == "YUE":
        language_code = "口音:粤语"

    language_tag = f"[{language_code}]"
    if text.startswith(language_tag):
        return text
    return f"{language_tag}{text}"


def detect_text_language(text: str) -> TextLanguage:
    language_code = detect(text)
    if language_code == "zh":
        return "zh"
    if language_code == "en":
        return "en"
    return "unknown"


def _normalize_with(normalizer, text: str) -> str:
    normalized = normalizer.normalize(text)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalize_chinese_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return _normalize_with(get_chinese_text_normalizer(), stripped)


def normalize_english_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return _normalize_with(get_english_text_normalizer(), stripped)


def normalize_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    language = detect_text_language(stripped)
    if language == "zh":
        return _normalize_with(get_chinese_text_normalizer(), stripped)
    if language == "en":
        return _normalize_with(get_english_text_normalizer(), stripped)
    return stripped
