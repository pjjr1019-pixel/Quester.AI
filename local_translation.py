"""Lightweight local translation helpers for optional multilingual features."""

from __future__ import annotations

import importlib.util
import re
from typing import Any

from data_structures import TextTranslationResult

try:  # pragma: no cover - optional dependency
    from argostranslate import translate as argos_translate
except Exception:  # pragma: no cover - optional dependency
    argos_translate = None


_LANGUAGE_ALIASES = {
    "english": "en",
    "en": "en",
    "spanish": "es",
    "es": "es",
    "french": "fr",
    "fr": "fr",
    "german": "de",
    "de": "de",
}

_EN_TO_ES = {
    "hello": "hola",
    "what": "que",
    "is": "es",
    "the": "el",
    "answer": "respuesta",
    "how": "como",
    "long": "largo",
    "can": "puede",
    "you": "tu",
    "think": "pensar",
    "local": "local",
    "models": "modelos",
    "runtime": "tiempo de ejecucion",
    "question": "pregunta",
    "file": "archivo",
    "summary": "resumen",
}
_EN_TO_FR = {
    "hello": "bonjour",
    "what": "quoi",
    "is": "est",
    "the": "le",
    "answer": "reponse",
    "how": "comment",
    "long": "long",
    "can": "peut",
    "you": "vous",
    "think": "penser",
    "local": "local",
    "models": "modeles",
    "runtime": "execution",
    "question": "question",
    "file": "fichier",
    "summary": "resume",
}


def argos_translate_available() -> bool:
    """Return True when Argos Translate is importable."""
    return importlib.util.find_spec("argostranslate") is not None and argos_translate is not None


def translate_with_stub(
    text: str,
    *,
    source_language: str,
    target_language: str,
    translation_model: str,
    max_chars: int,
    source_scope: str = "free_text",
) -> TextTranslationResult:
    """Translate text with a tiny deterministic lexicon and bounded fallback."""
    normalized_source = _normalize_language(source_language)
    normalized_target = _normalize_language(target_language)
    clipped_text = str(text).strip()[: max(1, int(max_chars))]
    warnings: list[str] = []
    if len(str(text).strip()) > len(clipped_text):
        warnings.append("text_clipped")
    if not clipped_text:
        return TextTranslationResult(
            status="blocked",
            source_text="",
            translated_text="",
            source_language=normalized_source,
            target_language=normalized_target,
            translation_backend="stub_translation",
            translation_model=translation_model,
            source_scope=source_scope,
            warnings=("empty_input",),
        )
    if normalized_source == normalized_target:
        warnings.append("same_language_pair")
        translated_text = clipped_text
    else:
        translated_text, approximation = _stub_translate_pair(
            clipped_text,
            source_language=normalized_source,
            target_language=normalized_target,
        )
        if approximation:
            warnings.append("stub_pair_approximation")
    return TextTranslationResult(
        status="translated",
        source_text=clipped_text,
        translated_text=translated_text,
        source_language=normalized_source,
        target_language=normalized_target,
        translation_backend="stub_translation",
        translation_model=translation_model,
        source_scope=source_scope,
        warnings=tuple(warnings),
    )


def translate_with_argos(
    text: str,
    *,
    source_language: str,
    target_language: str,
    translation_model: str,
    max_chars: int,
    source_scope: str = "free_text",
) -> TextTranslationResult:
    """Translate text with Argos Translate when it is installed locally."""
    normalized_source = _normalize_language(source_language)
    normalized_target = _normalize_language(target_language)
    clipped_text = str(text).strip()[: max(1, int(max_chars))]
    warnings: list[str] = []
    if len(str(text).strip()) > len(clipped_text):
        warnings.append("text_clipped")
    if not clipped_text:
        return TextTranslationResult(
            status="blocked",
            source_text="",
            translated_text="",
            source_language=normalized_source,
            target_language=normalized_target,
            translation_backend="argos_translate",
            translation_model=translation_model,
            source_scope=source_scope,
            warnings=("empty_input",),
        )
    if not argos_translate_available():
        return TextTranslationResult(
            status="blocked",
            source_text=clipped_text,
            translated_text="",
            source_language=normalized_source,
            target_language=normalized_target,
            translation_backend="argos_translate",
            translation_model=translation_model,
            source_scope=source_scope,
            warnings=("argostranslate_missing",),
        )
    try:  # pragma: no cover - optional dependency runtime
        translated_text = argos_translate.translate(
            clipped_text,
            from_code=normalized_source,
            to_code=normalized_target,
        )
    except Exception as exc:  # pragma: no cover - optional dependency runtime
        return TextTranslationResult(
            status="blocked",
            source_text=clipped_text,
            translated_text="",
            source_language=normalized_source,
            target_language=normalized_target,
            translation_backend="argos_translate",
            translation_model=translation_model,
            source_scope=source_scope,
            warnings=(f"argostranslate_failed:{exc}",),
        )
    return TextTranslationResult(
        status="translated",
        source_text=clipped_text,
        translated_text=str(translated_text).strip(),
        source_language=normalized_source,
        target_language=normalized_target,
        translation_backend="argos_translate",
        translation_model=translation_model,
        source_scope=source_scope,
        warnings=tuple(warnings),
    )


def _normalize_language(value: str) -> str:
    normalized = str(value).strip().lower()
    return _LANGUAGE_ALIASES.get(normalized, normalized or "auto")


def _stub_translate_pair(
    text: str,
    *,
    source_language: str,
    target_language: str,
) -> tuple[str, bool]:
    if source_language == "en" and target_language == "es":
        return _translate_tokens(text, _EN_TO_ES), False
    if source_language == "es" and target_language == "en":
        return _translate_tokens(text, {value: key for key, value in _EN_TO_ES.items()}), False
    if source_language == "en" and target_language == "fr":
        return _translate_tokens(text, _EN_TO_FR), False
    if source_language == "fr" and target_language == "en":
        return _translate_tokens(text, {value: key for key, value in _EN_TO_FR.items()}), False
    return f"[{source_language}->{target_language}] {text}", True


def _translate_tokens(text: str, mapping: dict[str, str]) -> str:
    parts = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)
    translated: list[str] = []
    for part in parts:
        key = part.lower()
        if key in mapping:
            replacement = mapping[key]
            if part[:1].isupper():
                replacement = replacement[:1].upper() + replacement[1:]
            translated.append(replacement)
        else:
            translated.append(part)
    return "".join(translated)

