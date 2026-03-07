"""Structured runtime errors for model backends and resource handling."""

from __future__ import annotations


class ModelRuntimeError(RuntimeError):
    """Base runtime error for model-manager and backend failures."""


class BackendStartupError(ModelRuntimeError):
    """Raised when a backend cannot be initialized."""


class BackendUnavailableError(ModelRuntimeError):
    """Raised when a configured backend or model is unavailable."""


class ModelTimeoutError(ModelRuntimeError):
    """Raised when a backend request exceeds the configured timeout."""


class ResourcePressureError(ModelRuntimeError):
    """Raised when runtime telemetry indicates unsafe memory pressure."""


class BackendFallbackError(ModelRuntimeError):
    """Raised when fallback was required but no usable fallback exists."""


class WebLookupError(RuntimeError):
    """Base runtime error for bounded web lookup failures."""


class WebLookupTimeoutError(WebLookupError):
    """Raised when a web lookup request exceeds the configured timeout."""
