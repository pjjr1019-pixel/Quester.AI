"""Helpers for schema-constrained model output with bounded repair."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, TypeVar

from config import APP_CONFIG, AppConfig
from model_manager import ModelManager

try:  # pragma: no cover - optional dependency path
    from jsonschema import ValidationError as JsonSchemaValidationError
    from jsonschema import validate as jsonschema_validate
except Exception:  # pragma: no cover - optional dependency path
    JsonSchemaValidationError = None
    jsonschema_validate = None

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
        """Store the shared model manager and structured-output tuning config."""
        self.model_manager = model_manager
        self.config = config

    async def decode_json_output(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        parser: Callable[[Any], DecodedType],
        fallback_factory: Callable[[str | None], DecodedType],
        max_tokens: int | None = None,
    ) -> StructuredDecodeResult[DecodedType]:
        """Return one typed decode result after at most one repair attempt.

        Inputs:
        - `prompt`: the task-specific structured-generation instruction.
        - `schema`: the JSON schema the model output must satisfy.
        - `parser`: converts the validated payload into a typed domain object.
        - `fallback_factory`: builds the deterministic fallback value when both
          decode attempts fail.

        Output:
        - A `StructuredDecodeResult` carrying the typed value plus repair and
          fallback metadata.

        Failure behavior:
        - The method never loops indefinitely. It performs one initial decode,
          one repair attempt, and then returns the fallback value with the parse
          error captured in the result metadata.
        """
        request_prompt = self._build_structured_prompt(prompt=prompt, schema=schema)
        raw_text = await self.model_manager.generate(request_prompt, max_tokens=max_tokens)
        parsed, error_message = self._try_parse_payload(raw_text, schema, parser)
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
        repaired, repaired_error = self._try_parse_payload(repaired_text, schema, parser)
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
        schema: Mapping[str, Any],
        parser: Callable[[Any], DecodedType],
    ) -> tuple[DecodedType | None, str | None]:
        try:
            payload = self._extract_json_payload(text)
        except ValueError as exc:
            return None, str(exc)
        try:
            self._validate_schema(payload, schema)
        except ValueError as exc:
            return None, str(exc)
        try:
            return parser(payload), None
        except Exception as exc:  # pragma: no cover - validation path
            return None, str(exc)

    def _validate_schema(
        self,
        payload: Any,
        schema: Mapping[str, Any],
        *,
        path: str = "$",
    ) -> None:
        if jsonschema_validate is not None and JsonSchemaValidationError is not None:
            try:
                jsonschema_validate(instance=payload, schema=schema)
            except JsonSchemaValidationError as exc:
                raise ValueError(self._format_jsonschema_error(exc)) from exc
            return

        schema_type = schema.get("type")
        if schema_type == "object":
            if not isinstance(payload, dict):
                raise ValueError(f"{path} must be an object")
            required = tuple(str(item) for item in schema.get("required", ()))
            properties = schema.get("properties", {})
            additional_properties = bool(schema.get("additionalProperties", True))
            for field_name in required:
                if field_name not in payload:
                    raise ValueError(f"{path}.{field_name} is required")
            if not additional_properties:
                unknown_fields = sorted(set(payload) - set(properties))
                if unknown_fields:
                    raise ValueError(f"{path} contains unsupported fields: {', '.join(unknown_fields)}")
            for field_name, field_value in payload.items():
                field_schema = properties.get(field_name)
                if isinstance(field_schema, Mapping):
                    self._validate_schema(field_value, field_schema, path=f"{path}.{field_name}")
            return

        if schema_type == "array":
            if not isinstance(payload, list):
                raise ValueError(f"{path} must be an array")
            item_schema = schema.get("items")
            if isinstance(item_schema, Mapping):
                for index, item in enumerate(payload):
                    self._validate_schema(item, item_schema, path=f"{path}[{index}]")
            return

        if schema_type == "string":
            if not isinstance(payload, str):
                raise ValueError(f"{path} must be a string")
            return

        if schema_type == "number":
            if isinstance(payload, bool) or not isinstance(payload, (int, float)):
                raise ValueError(f"{path} must be a number")
            return

        if schema_type == "integer":
            if isinstance(payload, bool) or not isinstance(payload, int):
                raise ValueError(f"{path} must be an integer")
            return

        if schema_type == "boolean":
            if not isinstance(payload, bool):
                raise ValueError(f"{path} must be a boolean")
            return

        if schema_type == "null":
            if payload is not None:
                raise ValueError(f"{path} must be null")

    def _format_jsonschema_error(self, exc: Exception) -> str:
        if JsonSchemaValidationError is None or not isinstance(exc, JsonSchemaValidationError):
            return str(exc)
        path = "$"
        for part in exc.absolute_path:
            if isinstance(part, int):
                path += f"[{part}]"
            else:
                path += f".{part}"
        message = exc.message or "schema validation failed"
        return f"{path}: {message}"

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
