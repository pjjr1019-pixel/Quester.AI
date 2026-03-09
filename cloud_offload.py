"""Provider-agnostic auxiliary cloud offload manager."""

from __future__ import annotations

import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from data_structures import (
    CloudFallbackBehavior,
    CloudJobContract,
    CloudOffloadCapability,
    CloudOffloadOutcome,
    CloudOffloadRecord,
    utc_now,
)
from retrieval import stable_hash


def _payload_bytes(payload: Mapping[str, Any]) -> int:
    return len(json.dumps(dict(payload), sort_keys=True, default=str).encode("utf-8"))


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class _StubCloudAdapter:
    """Deterministic built-in adapter for bounded auxiliary cloud flows."""

    provider_family = "provider_agnostic"

    def is_available(self) -> bool:
        return True

    def supports(self, capability: CloudOffloadCapability) -> bool:
        return True

    async def dispatch(self, contract: CloudJobContract, payload: Mapping[str, Any]) -> dict[str, Any]:
        payload_keys = tuple(sorted(str(key) for key in payload.keys()))
        return {
            "status": "accepted",
            "response_ref": f"cloud://stub_cloud/{contract.job_id}",
            "capability": contract.capability.value,
            "payload_keys": payload_keys[:8],
        }


class CloudOffloadManager:
    """Owns provider-agnostic auxiliary cloud adapters and bounded dispatch behavior."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("quester.cloud_offload")
        self._adapters: dict[str, Any] = {}
        self.register_adapter("stub_cloud", _StubCloudAdapter())

    def register_adapter(self, provider_name: str, adapter: Any) -> None:
        """Register or replace one provider adapter by name."""
        resolved_name = str(provider_name).strip()
        if not resolved_name:
            raise ValueError("provider_name must not be empty.")
        self._adapters[resolved_name] = adapter

    def provider_names(self) -> tuple[str, ...]:
        """Return registered provider names in stable sorted order."""
        return tuple(sorted(self._adapters))

    def provider_available(
        self,
        provider_name: str,
        *,
        capability: CloudOffloadCapability | None = None,
    ) -> bool:
        """Report whether one provider can currently accept the requested capability."""
        adapter = self._adapters.get(str(provider_name).strip())
        if adapter is None:
            return False
        if capability is not None:
            supports = getattr(adapter, "supports", None)
            if callable(supports) and not bool(supports(capability)):
                return False
        is_available = getattr(adapter, "is_available", None)
        if callable(is_available):
            return bool(is_available())
        return True

    def provider_family(self, provider_name: str, *, fallback: str = "provider_agnostic") -> str:
        """Resolve the adapter family string for readiness and audit records."""
        adapter = self._adapters.get(str(provider_name).strip())
        family = getattr(adapter, "provider_family", "") if adapter is not None else ""
        resolved = str(family or fallback).strip()
        return resolved or "provider_agnostic"

    async def dispatch(
        self,
        *,
        provider_name: str,
        contract: CloudJobContract,
        payload: Mapping[str, Any],
        local_fallback: Callable[[], Awaitable[Any] | Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CloudOffloadRecord:
        """Dispatch one bounded auxiliary cloud job and return a typed audit record."""
        resolved_provider = str(provider_name).strip()
        if not resolved_provider:
            raise ValueError("provider_name must not be empty.")

        created_at = utc_now()
        dispatch_id = f"cloud_{stable_hash(f'{resolved_provider}:{contract.job_id}:{created_at.isoformat()}')[:16]}"
        payload_size = _payload_bytes(payload)
        record_metadata = dict(metadata or {})
        record_metadata.setdefault("dispatch_mode", contract.dispatch_mode.value)
        record_metadata.setdefault("payload_keys", tuple(sorted(str(key) for key in payload.keys()))[:8])

        async def _run_local_fallback(
            reason: str,
            *,
            detail: str,
            retry_count: int = 0,
            latency_ms: int = 0,
        ) -> CloudOffloadRecord:
            if local_fallback is None:
                return CloudOffloadRecord(
                    dispatch_id=dispatch_id,
                    job_id=contract.job_id,
                    capability=contract.capability,
                    provider_name=resolved_provider,
                    provider_family=self.provider_family(resolved_provider, fallback=contract.provider_family),
                    payload_class=contract.payload_class,
                    privacy_class=contract.privacy_class,
                    outcome=CloudOffloadOutcome.FAILED,
                    summary="Auxiliary cloud dispatch failed with no local fallback.",
                    detail=detail,
                    fallback_behavior=contract.fallback_behavior,
                    bytes_sent=payload_size,
                    latency_ms=latency_ms,
                    retry_count=retry_count,
                    local_fallback_used=False,
                    fallback_reason=reason,
                    metadata=record_metadata,
                    created_at=created_at,
                )
            await _maybe_await(local_fallback())
            return CloudOffloadRecord(
                dispatch_id=dispatch_id,
                job_id=contract.job_id,
                capability=contract.capability,
                provider_name=resolved_provider,
                provider_family=self.provider_family(resolved_provider, fallback=contract.provider_family),
                payload_class=contract.payload_class,
                privacy_class=contract.privacy_class,
                outcome=CloudOffloadOutcome.LOCAL_FALLBACK,
                summary="Auxiliary cloud dispatch fell back to local execution.",
                detail=detail,
                fallback_behavior=contract.fallback_behavior,
                bytes_sent=payload_size,
                latency_ms=latency_ms,
                retry_count=retry_count,
                local_fallback_used=True,
                fallback_reason=reason,
                metadata=record_metadata,
                created_at=created_at,
            )

        if payload_size > contract.max_payload_bytes:
            return await _run_local_fallback(
                "cloud_payload_too_large",
                detail=(
                    f"Payload size {payload_size} exceeded the configured cloud limit of "
                    f"{contract.max_payload_bytes} bytes."
                ),
            )

        adapter = self._adapters.get(resolved_provider)
        if adapter is None:
            return await _run_local_fallback(
                "cloud_provider_missing",
                detail=f"Configured provider '{resolved_provider}' is not registered.",
            )
        if not self.provider_available(resolved_provider, capability=contract.capability):
            return await _run_local_fallback(
                "cloud_provider_unavailable",
                detail=f"Configured provider '{resolved_provider}' is unavailable for {contract.capability.value}.",
            )

        max_attempts = 1 if contract.fallback_behavior == CloudFallbackBehavior.LOCAL_ONLY else 1 + contract.max_retries
        last_error = ""
        start = time.perf_counter()
        for attempt in range(1, max_attempts + 1):
            try:
                response = await _maybe_await(adapter.dispatch(contract, payload))
                latency_ms = max(0, int((time.perf_counter() - start) * 1000))
                response_ref = ""
                if isinstance(response, Mapping):
                    response_ref = str(response.get("response_ref", "")).strip()
                    if response:
                        record_metadata["provider_response"] = dict(response)
                return CloudOffloadRecord(
                    dispatch_id=dispatch_id,
                    job_id=contract.job_id,
                    capability=contract.capability,
                    provider_name=resolved_provider,
                    provider_family=self.provider_family(resolved_provider, fallback=contract.provider_family),
                    payload_class=contract.payload_class,
                    privacy_class=contract.privacy_class,
                    outcome=CloudOffloadOutcome.SUCCEEDED,
                    summary="Auxiliary cloud dispatch succeeded.",
                    detail=(
                        f"Provider '{resolved_provider}' accepted {contract.capability.value} "
                        f"after {attempt} attempt(s)."
                    ),
                    fallback_behavior=contract.fallback_behavior,
                    bytes_sent=payload_size,
                    latency_ms=latency_ms,
                    retry_count=max(0, attempt - 1),
                    local_fallback_used=False,
                    response_ref=response_ref or f"cloud://{resolved_provider}/{contract.job_id}",
                    metadata=record_metadata,
                    created_at=created_at,
                )
            except Exception as exc:  # pragma: no cover - exercised in integration tests
                last_error = f"{type(exc).__name__}: {exc}"
                self.logger.warning(
                    "Auxiliary cloud dispatch failed (provider=%s, capability=%s, attempt=%s/%s): %s",
                    resolved_provider,
                    contract.capability.value,
                    attempt,
                    max_attempts,
                    last_error,
                )

        latency_ms = max(0, int((time.perf_counter() - start) * 1000))
        return await _run_local_fallback(
            "cloud_dispatch_failed",
            detail=(
                f"Provider '{resolved_provider}' failed for {contract.capability.value} after "
                f"{max_attempts} attempt(s): {last_error or 'unknown_error'}"
            ),
            retry_count=max(0, max_attempts - 1),
            latency_ms=latency_ms,
        )
