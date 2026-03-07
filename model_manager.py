"""Shared model runtime manager for generation and embedding operations."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from config import APP_CONFIG, AppConfig
from model_backends import (
    BackendHealth,
    EmbeddingBackendAdapter,
    GenerationBackendAdapter,
    LlamaCppGenerationBackend,
    OllamaEmbeddingBackend,
    OllamaGenerationBackend,
    SentenceTransformersEmbeddingBackend,
    StubEmbeddingBackend,
    StubGenerationBackend,
)
from runtime_errors import (
    BackendFallbackError,
    BackendStartupError,
    BackendUnavailableError,
    ModelTimeoutError,
    ResourcePressureError,
)
from utils import cancel_task, utc_now_iso

try:  # pragma: no cover - optional dependency
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


@dataclass(slots=True)
class ModelHealthSnapshot:
    """Current runtime health and activity counters."""

    started: bool
    generation_backend: str
    embedding_backend: str
    active_generation_jobs: int
    active_embedding_jobs: int
    last_used_at: str | None
    fallback_active: bool
    fallback_reason: str | None
    available_ram_gb: float | None
    total_ram_gb: float | None
    generation_backend_vram_gb: float | None
    embedding_backend_vram_gb: float | None
    telemetry_enabled: bool
    last_error: str | None


class ModelManager:
    """Centralized owner of model runtime lifecycle and concurrency limits."""

    def __init__(
        self,
        config: AppConfig = APP_CONFIG,
        logger: logging.Logger | None = None,
        generation_backend: GenerationBackendAdapter | None = None,
        embedding_backend: EmbeddingBackendAdapter | None = None,
        generation_fallback_backend: GenerationBackendAdapter | None = None,
        embedding_fallback_backend: EmbeddingBackendAdapter | None = None,
    ):
        self.config = config
        self.logger = logger or logging.getLogger("quester.model_manager")
        self._generation_sem = asyncio.Semaphore(config.concurrency.generation_slots)
        self._embedding_sem = asyncio.Semaphore(config.concurrency.embedding_slots)
        self._started = False
        self._active_generation_jobs = 0
        self._active_embedding_jobs = 0
        self._last_used_at: str | None = None
        self._last_used_monotonic: float | None = None
        self._last_error: str | None = None
        self._fallback_active = False
        self._fallback_reason: str | None = None
        self._stop_event = asyncio.Event()
        self._maintenance_task: asyncio.Task[None] | None = None

        self._primary_generation_backend = generation_backend
        self._primary_embedding_backend = embedding_backend
        self._fallback_generation_backend = generation_fallback_backend
        self._fallback_embedding_backend = embedding_fallback_backend
        self._active_generation_backend: GenerationBackendAdapter | None = None
        self._active_embedding_backend: EmbeddingBackendAdapter | None = None

        self._generation_health = BackendHealth(
            backend_name="uninitialized",
            model_name="uninitialized",
            started=False,
            available=False,
            mode="unknown",
        )
        self._embedding_health = BackendHealth(
            backend_name="uninitialized",
            model_name="uninitialized",
            started=False,
            available=False,
            mode="unknown",
        )

    async def start(self) -> None:
        """Start model services, backends, and idle-maintenance loop."""
        if self._started:
            return
        self.config.validate()
        self._build_backends_if_needed()
        await self._activate_primary_backends()
        self._started = True
        self._stop_event.clear()
        self._maintenance_task = asyncio.create_task(
            self._maintenance_loop(),
            name="model-manager-maintenance",
        )
        self.logger.info(
            "ModelManager started (gen=%s, embed=%s, stub_mode=%s)",
            self._active_generation_backend.backend_name,
            self._active_embedding_backend.backend_name,
            self.config.preflight.flags.stub_mode,
        )

    async def stop(self) -> None:
        """Stop model services cleanly."""
        if not self._started:
            return
        self._stop_event.set()
        await cancel_task(self._maintenance_task, timeout_s=self.config.preflight.flags.shutdown_timeout_s)
        self._maintenance_task = None
        self._started = False

        seen_ids: set[int] = set()
        for backend in (
            self._active_generation_backend,
            self._active_embedding_backend,
            self._primary_generation_backend,
            self._primary_embedding_backend,
            self._fallback_generation_backend,
            self._fallback_embedding_backend,
        ):
            if backend is None or id(backend) in seen_ids:
                continue
            seen_ids.add(id(backend))
            try:
                await backend.stop()
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                self.logger.warning("Backend stop failed for %s: %s", backend.backend_name, exc)
        self.logger.info("ModelManager stopped.")

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Run a generation request through the active bounded backend."""
        self._require_started()
        max_tokens = max_tokens or self.config.model_tuning.default_max_tokens
        async with self._generation_sem:
            self._active_generation_jobs += 1
            try:
                await self._maybe_activate_low_memory_fallback(kind="generation")
                backend = await self._ensure_generation_backend_ready()
                try:
                    response = await asyncio.wait_for(
                        backend.generate(
                            prompt,
                            max_tokens=max_tokens,
                            temperature=self.config.model_tuning.default_temperature,
                        ),
                        timeout=self.config.backend_runtime.request_timeout_s,
                    )
                except asyncio.TimeoutError as exc:
                    await self._handle_generation_failure(ModelTimeoutError("Generation request timed out."))
                    raise ModelTimeoutError("Generation request timed out.") from exc
                except (BackendUnavailableError, BackendStartupError) as exc:
                    response = await self._handle_generation_failure(exc, prompt=prompt, max_tokens=max_tokens)
                self._mark_used()
                await self._refresh_health()
                return response
            finally:
                self._active_generation_jobs -= 1

    async def embed(self, text: str) -> list[float]:
        """Run an embedding request through the active bounded backend."""
        return await self._embed_with_role(text, role="generic")

    async def embed_query(self, text: str) -> list[float]:
        """Run a retrieval-query embedding request through the active backend."""
        return await self._embed_with_role(text, role="query")

    async def embed_document(self, text: str) -> list[float]:
        """Run a retrieval-document embedding request through the active backend."""
        return await self._embed_with_role(text, role="document")

    async def _embed_with_role(self, text: str, *, role: str) -> list[float]:
        """Run an embedding request through the active bounded backend."""
        self._require_started()
        async with self._embedding_sem:
            self._active_embedding_jobs += 1
            try:
                await self._maybe_activate_low_memory_fallback(kind="embedding")
                backend = await self._ensure_embedding_backend_ready()
                try:
                    vector = await asyncio.wait_for(
                        self._dispatch_embedding_request(backend, text=text, role=role),
                        timeout=self.config.backend_runtime.request_timeout_s,
                    )
                except asyncio.TimeoutError as exc:
                    await self._handle_embedding_failure(ModelTimeoutError("Embedding request timed out."))
                    raise ModelTimeoutError("Embedding request timed out.") from exc
                except (BackendUnavailableError, BackendStartupError) as exc:
                    vector = await self._handle_embedding_failure(exc, text=text, role=role)
                self._mark_used()
                await self._refresh_health()
                return vector
            finally:
                self._active_embedding_jobs -= 1

    def health_snapshot(self) -> ModelHealthSnapshot:
        """Return current health info for logs/UI."""
        total_ram_gb, available_ram_gb = self._read_ram_telemetry()
        telemetry_enabled = bool(
            self.config.backend_runtime.telemetry_enable_psutil
            or self.config.backend_runtime.telemetry_enable_backend_stats
        )
        active_generation_name = (
            self._active_generation_backend.backend_name
            if self._active_generation_backend is not None
            else self._generation_health.backend_name
        )
        active_embedding_name = (
            self._active_embedding_backend.backend_name
            if self._active_embedding_backend is not None
            else self._embedding_health.backend_name
        )
        return ModelHealthSnapshot(
            started=self._started,
            generation_backend=active_generation_name,
            embedding_backend=active_embedding_name,
            active_generation_jobs=self._active_generation_jobs,
            active_embedding_jobs=self._active_embedding_jobs,
            last_used_at=self._last_used_at,
            fallback_active=self._fallback_active,
            fallback_reason=self._fallback_reason,
            available_ram_gb=available_ram_gb,
            total_ram_gb=total_ram_gb,
            generation_backend_vram_gb=self._generation_health.estimated_vram_gb,
            embedding_backend_vram_gb=self._embedding_health.estimated_vram_gb,
            telemetry_enabled=telemetry_enabled,
            last_error=self._last_error,
        )

    async def _activate_primary_backends(self) -> None:
        self._active_generation_backend = self._primary_generation_backend
        self._active_embedding_backend = self._primary_embedding_backend
        assert self._active_generation_backend is not None
        assert self._active_embedding_backend is not None

        await self._start_backend_with_fallback(kind="generation")
        await self._start_backend_with_fallback(kind="embedding")
        await self._refresh_health()

    async def _start_backend_with_fallback(self, kind: str) -> None:
        if kind == "generation":
            active = self._active_generation_backend
            fallback = self._fallback_generation_backend
        else:
            active = self._active_embedding_backend
            fallback = self._fallback_embedding_backend

        assert active is not None
        try:
            await active.start()
            return
        except BackendStartupError as exc:
            self._last_error = str(exc)
            if not self.config.backend_runtime.enable_fallback_on_error or fallback is None:
                raise
            await fallback.start()
            if kind == "generation":
                self._active_generation_backend = fallback
            else:
                self._active_embedding_backend = fallback
            self._fallback_active = True
            self._fallback_reason = f"{kind} startup fallback: {exc}"
            self.logger.warning("Activated %s fallback backend after startup error: %s", kind, exc)

    async def _ensure_generation_backend_ready(self) -> GenerationBackendAdapter:
        backend = self._active_generation_backend
        if backend is None:
            raise BackendUnavailableError("No active generation backend is configured.")
        health = await backend.health()
        if not health.started or not health.available:
            await backend.start()
            health = await backend.health()
        self._generation_health = health
        return backend

    async def _ensure_embedding_backend_ready(self) -> EmbeddingBackendAdapter:
        backend = self._active_embedding_backend
        if backend is None:
            raise BackendUnavailableError("No active embedding backend is configured.")
        health = await backend.health()
        if not health.started or not health.available:
            await backend.start()
            health = await backend.health()
        self._embedding_health = health
        return backend

    async def _handle_generation_failure(
        self,
        error: Exception,
        *,
        prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self._last_error = str(error)
        if (
            not self.config.backend_runtime.enable_fallback_on_error
            or self._fallback_generation_backend is None
            or self._active_generation_backend is self._fallback_generation_backend
        ):
            raise error
        await self._activate_backend_fallback(kind="generation", reason=str(error))
        if prompt is None or max_tokens is None:
            raise error
        backend = await self._ensure_generation_backend_ready()
        return await asyncio.wait_for(
            backend.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=self.config.model_tuning.default_temperature,
            ),
            timeout=self.config.backend_runtime.request_timeout_s,
        )

    async def _handle_embedding_failure(
        self,
        error: Exception,
        *,
        text: str | None = None,
        role: str = "generic",
    ) -> list[float]:
        self._last_error = str(error)
        if (
            not self.config.backend_runtime.enable_fallback_on_error
            or self._fallback_embedding_backend is None
            or self._active_embedding_backend is self._fallback_embedding_backend
        ):
            raise error
        await self._activate_backend_fallback(kind="embedding", reason=str(error))
        if text is None:
            raise error
        backend = await self._ensure_embedding_backend_ready()
        return await asyncio.wait_for(
            self._dispatch_embedding_request(backend, text=text, role=role),
            timeout=self.config.backend_runtime.request_timeout_s,
        )

    async def _activate_backend_fallback(self, kind: str, reason: str) -> None:
        if kind == "generation":
            fallback = self._fallback_generation_backend
            if fallback is None:
                raise BackendFallbackError("No generation fallback backend is configured.")
            try:
                await fallback.start()
            except BackendStartupError as exc:
                raise BackendFallbackError(
                    f"Generation fallback backend could not start: {exc}"
                ) from exc
            self._active_generation_backend = fallback
        else:
            fallback = self._fallback_embedding_backend
            if fallback is None:
                raise BackendFallbackError("No embedding fallback backend is configured.")
            try:
                await fallback.start()
            except BackendStartupError as exc:
                raise BackendFallbackError(
                    f"Embedding fallback backend could not start: {exc}"
                ) from exc
            self._active_embedding_backend = fallback
        self._fallback_active = True
        self._fallback_reason = f"{kind} fallback activated: {reason}"
        self.logger.warning("%s", self._fallback_reason)
        await self._refresh_health()

    async def _maybe_activate_low_memory_fallback(self, kind: str) -> None:
        if self.config.preflight.flags.stub_mode:
            return
        if not self.config.backend_runtime.enable_fallback_on_low_memory:
            return
        low_memory_reason = self._detect_memory_pressure()
        if low_memory_reason is None:
            return
        try:
            await self._activate_backend_fallback(kind=kind, reason=low_memory_reason)
        except BackendFallbackError:
            raise ResourcePressureError(low_memory_reason) from None

    async def _maintenance_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._refresh_health()
                await self._maybe_unload_idle_backends()
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                self._last_error = str(exc)
                self.logger.warning("ModelManager maintenance loop error: %s", exc)
            await asyncio.sleep(self.config.backend_runtime.idle_check_interval_s)

    async def _maybe_unload_idle_backends(self) -> None:
        if self._last_used_monotonic is None:
            return
        if self._active_generation_jobs or self._active_embedding_jobs:
            return
        idle_seconds = time.monotonic() - self._last_used_monotonic
        if idle_seconds < self.config.backend_runtime.idle_unload_after_s:
            return
        unloaded = False
        for backend in (self._active_generation_backend, self._active_embedding_backend):
            if backend is None:
                continue
            unloaded = await backend.unload() or unloaded
        if unloaded:
            self.logger.info("Unloaded idle model backend state after %.2f seconds.", idle_seconds)
            await self._refresh_health()

    async def _refresh_health(self) -> None:
        if self._active_generation_backend is not None:
            self._generation_health = await self._active_generation_backend.health()
        if self._active_embedding_backend is not None:
            self._embedding_health = await self._active_embedding_backend.health()

    def _mark_used(self) -> None:
        self._last_used_at = utc_now_iso()
        self._last_used_monotonic = time.monotonic()

    def _build_backends_if_needed(self) -> None:
        if self._primary_generation_backend is None:
            self._primary_generation_backend = self._make_generation_backend(
                backend_name=self.config.preflight.backends.generation_backend,
                model_name=self.config.preflight.backends.generation_model,
            )
        if self._primary_embedding_backend is None:
            self._primary_embedding_backend = self._make_embedding_backend(
                backend_name=self.config.preflight.backends.embedding_backend,
                model_name=self.config.preflight.backends.embedding_model,
            )
        if self._fallback_generation_backend is None:
            self._fallback_generation_backend = self._make_generation_backend(
                backend_name=self.config.preflight.backends.generation_fallback_backend,
                model_name=self.config.preflight.backends.generation_fallback_model,
            )
        if self._fallback_embedding_backend is None:
            self._fallback_embedding_backend = self._make_embedding_backend(
                backend_name=self.config.preflight.backends.embedding_fallback_backend,
                model_name=self.config.preflight.backends.embedding_fallback_model,
            )
        if self.config.preflight.flags.stub_mode:
            self._primary_generation_backend = StubGenerationBackend()
            self._primary_embedding_backend = StubEmbeddingBackend(
                dimensions=self.config.model_tuning.embedding_dimensions,
                query_prefix=self.config.retrieval.query_prefix,
                document_prefix=self.config.retrieval.document_prefix,
            )
            self._fallback_generation_backend = None
            self._fallback_embedding_backend = None

    def _make_generation_backend(
        self,
        *,
        backend_name: str,
        model_name: str,
    ) -> GenerationBackendAdapter:
        if backend_name == "ollama":
            return OllamaGenerationBackend(
                base_url=self.config.backend_runtime.ollama_base_url,
                model_name=model_name,
                timeout_s=self.config.backend_runtime.request_timeout_s,
            )
        if backend_name == "llama_cpp":
            return LlamaCppGenerationBackend(
                model_name=model_name,
                models_dir=self.config.backend_runtime.models_dir,
                context_window=self.config.backend_runtime.llama_cpp_context_window,
                gpu_layers=self.config.backend_runtime.llama_cpp_gpu_layers,
                timeout_s=self.config.backend_runtime.request_timeout_s,
            )
        raise BackendUnavailableError(f"Unsupported generation backend: {backend_name}")

    def _make_embedding_backend(
        self,
        *,
        backend_name: str,
        model_name: str,
    ) -> EmbeddingBackendAdapter:
        if backend_name == "sentence_transformers":
            return SentenceTransformersEmbeddingBackend(
                model_name=model_name,
                timeout_s=self.config.backend_runtime.request_timeout_s,
                prefer_asymmetric_embeddings=self.config.retrieval.prefer_asymmetric_embeddings,
                query_prefix=self.config.retrieval.query_prefix,
                document_prefix=self.config.retrieval.document_prefix,
            )
        if backend_name == "ollama_embeddings":
            return OllamaEmbeddingBackend(
                base_url=self.config.backend_runtime.ollama_base_url,
                model_name=model_name,
                timeout_s=self.config.backend_runtime.request_timeout_s,
                query_prefix=self.config.retrieval.query_prefix,
                document_prefix=self.config.retrieval.document_prefix,
            )
        raise BackendUnavailableError(f"Unsupported embedding backend: {backend_name}")

    async def _dispatch_embedding_request(
        self,
        backend: EmbeddingBackendAdapter,
        *,
        text: str,
        role: str,
    ) -> list[float]:
        if role == "query":
            return await backend.embed_query(text)
        if role == "document":
            return await backend.embed_document(text)
        return await backend.embed(text)

    def _read_ram_telemetry(self) -> tuple[float | None, float | None]:
        if not self.config.backend_runtime.telemetry_enable_psutil or psutil is None:
            return (None, None)
        memory = psutil.virtual_memory()
        return (memory.total / (1024**3), memory.available / (1024**3))

    def _detect_memory_pressure(self) -> str | None:
        total_ram_gb, available_ram_gb = self._read_ram_telemetry()
        reasons: list[str] = []
        if available_ram_gb is not None and available_ram_gb <= self.config.backend_runtime.low_ram_headroom_gb:
            reasons.append(
                f"available RAM {available_ram_gb:.2f}GB is below headroom "
                f"{self.config.backend_runtime.low_ram_headroom_gb:.2f}GB"
            )
        if (
            self._generation_health.estimated_vram_gb is not None
            and self._generation_health.estimated_vram_gb
            >= self.config.preflight.hardware.max_vram_gb - self.config.backend_runtime.low_vram_headroom_gb
        ):
            reasons.append(
                f"generation backend VRAM {self._generation_health.estimated_vram_gb:.2f}GB is near "
                f"the {self.config.preflight.hardware.max_vram_gb:.2f}GB target"
            )
        if (
            self._embedding_health.estimated_vram_gb is not None
            and self._embedding_health.estimated_vram_gb
            >= self.config.preflight.hardware.max_vram_gb - self.config.backend_runtime.low_vram_headroom_gb
        ):
            reasons.append(
                f"embedding backend VRAM {self._embedding_health.estimated_vram_gb:.2f}GB is near "
                f"the {self.config.preflight.hardware.max_vram_gb:.2f}GB target"
            )
        _ = total_ram_gb
        if not reasons:
            return None
        return "; ".join(reasons)

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("ModelManager must be started before use.")
