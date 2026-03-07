"""Helpers for schema-constrained model output with bounded repair."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, TypeVar

from config import APP_CONFIG, AppConfig
from model_manager import ModelManager

DecodedType = TypeVar("DecodedType")


@dataclass(slots=True, frozen=True)
class StructuredDecodeResult(Generic[DecodedType]):
    """Result of a bounded structured-generation attempt."""

    value: DecodedType
    raw_text: str
    repaired_text: str | None
    used_repair: bool
    used_fallback: bool
    error_message: str | None = None


class StructuredGenerationService:
    """Generate and validate schema-constrained JSON with one bounded repair."""

    def __init__(self, model_manager: ModelManager, config: AppConfig = APP_CONFIG):
        self.model_manager = model_manager
        self.config = config

    async def decode_json_output(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        parser: Callable[[Mapping[str, Any]], DecodedType],
        fallback_factory: Callable[[str | None], DecodedType],
        max_tokens: int | None = None,
    ) -> StructuredDecodeResult[DecodedType]:
        request_prompt = self._build_structured_prompt(prompt=prompt, schema=schema)
        raw_text = await self.model_manager.generate(request_prompt, max_tokens=max_tokens)
        parsed, error_message = self._try_parse_payload(raw_text, parser)
        if parsed is not None:
            return StructuredDecodeResult(
                value=parsed,
                raw_text=raw_text,
                repaired_text=None,
                used_repair=False,
                used_fallback=False,
            )

        repair_prompt = self._build_repair_prompt(
            schema=schema,
            previous_response=raw_text,
            error_message=error_message or "unknown_parse_failure",
        )
        repaired_text = await self.model_manager.generate(repair_prompt, max_tokens=max_tokens)
        repaired, repaired_error = self._try_parse_payload(repaired_text, parser)
        if repaired is not None:
            return StructuredDecodeResult(
                value=repaired,
                raw_text=raw_text,
                repaired_text=repaired_text,
                used_repair=True,
                used_fallback=False,
            )

        return StructuredDecodeResult(
            value=fallback_factory(repaired_error or error_message),
            raw_text=raw_text,
            repaired_text=repaired_text,
            used_repair=True,
            used_fallback=True,
            error_message=repaired_error or error_message,
        )

    def _build_structured_prompt(self, *, prompt: str, schema: Mapping[str, Any]) -> str:
        return (
            f"{prompt}\n"
            "Return JSON only. Do not include markdown, commentary, or prose.\n"
            f"Schema:\n{json.dumps(schema, sort_keys=True)}"
        )

    def _build_repair_prompt(
        self,
        *,
        schema: Mapping[str, Any],
        previous_response: str,
        error_message: str,
    ) -> str:
        return (
            "Repair the previous response into valid JSON only.\n"
            f"Schema:\n{json.dumps(schema, sort_keys=True)}\n"
            f"ValidationError: {error_message}\n"
            f"PreviousResponse:\n{previous_response}"
        )

    def _try_parse_payload(
        self,
        text: str,
        parser: Callable[[Mapping[str, Any]], DecodedType],
    ) -> tuple[DecodedType | None, str | None]:
        try:
            payload = self._extract_json_payload(text)
        except ValueError as exc:
            return None, str(exc)
        if not isinstance(payload, dict):
            return None, "structured output must decode to a JSON object"
        try:
            return parser(payload), None
        except Exception as exc:  # pragma: no cover - validation path
            return None, str(exc)

    def _extract_json_payload(self, text: str) -> Any:
        trimmed = text.strip()
        if not trimmed:
            raise ValueError("model output was empty")
        try:
            return json.loads(trimmed)
        except json.JSONDecodeError:
            start_candidates = [index for index in (trimmed.find("{"), trimmed.find("[")) if index >= 0]
            if not start_candidates:
                raise ValueError("model output did not contain JSON")
            start = min(start_candidates)
            opening = trimmed[start]
            closing = "}" if opening == "{" else "]"
            end = trimmed.rfind(closing)
            if end <= start:
                raise ValueError("model output did not contain a complete JSON payload")
            candidate = trimmed[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON payload: {exc}") from exc
