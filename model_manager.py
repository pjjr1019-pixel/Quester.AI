"""Shared model runtime manager for generation, embedding, and specialist roles."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from bounded_cache import BoundedCacheManager
from config import APP_CONFIG, AppConfig
from data_structures import (
    AudioSynthesisResult,
    AudioTranscriptionResult,
    CodeSpecialistResult,
    CompressionInsightSummary,
    ModelLoadPolicy,
    ModelRegistration,
    ModelRegistryView,
    ModelResourceClass,
    ModelRole,
    ModelRouteDecision,
    OptimizerSuggestionKind,
    OptimizerSuggestionRecord,
    TextTranslationResult,
    VisionInspectionResult,
    VoiceActivityReport,
)
from local_audio import (
    analyze_voice_activity,
    synthesize_with_stub,
    synthesize_with_system_speech,
    system_speech_available,
    transcribe_with_stub,
    transcribe_with_system_speech,
)
from local_code_specialist import analyze_code_file_with_stub, analyze_code_with_stub
from local_translation import argos_translate_available, translate_with_argos, translate_with_stub
from local_vision import inspect_image_with_stub
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
from retrieval import SearchResult, rerank_search_results
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

if TYPE_CHECKING:
    from data_structures import UserSettingsProfile


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
    active_heavy_roles: tuple[str, ...] = field(default_factory=tuple)
    heavy_slot_limit: int = 2
    governor_active: bool = False
    governor_pressure_reasons: tuple[str, ...] = field(default_factory=tuple)
    governor_degraded_features: tuple[str, ...] = field(default_factory=tuple)
    queue_pressure: bool = False
    backend_health_degraded: bool = False
    allow_continuous_capture: bool = True
    allow_ocr_on_step: bool = True
    allow_vision_on_step: bool = True
    allow_optional_heavy_residency: bool = True
    allow_background_work: bool = True
    governor_summary: str = ""
    telemetry_enabled: bool = False
    last_error: str | None = None


@dataclass(slots=True, frozen=True)
class HardwareGovernorState:
    """Live pressure state used to degrade optional observation/model behavior before core reasoning."""

    active: bool = False
    pressure_reasons: tuple[str, ...] = ()
    degraded_features: tuple[str, ...] = ()
    queue_pressure: bool = False
    backend_health_degraded: bool = False
    allow_continuous_capture: bool = True
    allow_ocr_on_step: bool = True
    allow_vision_on_step: bool = True
    allow_optional_heavy_residency: bool = True
    allow_background_work: bool = True
    summary: str = ""


@dataclass(slots=True, frozen=True)
class _GovernorAdvisoryInput:
    """Clamped optimizer input that may inform, but never override, live governor decisions."""

    suggestion_id: str
    kind: OptimizerSuggestionKind
    optional_heavy_roles: tuple[str, ...] = ()
    cache_namespaces: tuple[str, ...] = ()
    warm_key_count: int = 0
    expires_at_monotonic: float = 0.0

    def summary_fragment(self, *, now: float) -> str:
        ttl_seconds = max(0, int(self.expires_at_monotonic - now))
        if self.kind == OptimizerSuggestionKind.MODEL_LOADING:
            roles = ",".join(self.optional_heavy_roles) if self.optional_heavy_roles else "optional_heavy"
            return f"retain({roles})/{ttl_seconds}s"
        namespaces = ",".join(self.cache_namespaces) if self.cache_namespaces else "background_cache"
        return f"prefetch({namespaces}:{self.warm_key_count})/{ttl_seconds}s"


class ModelManager:
    """Centralized owner of model runtime lifecycle and concurrency limits."""

    _MANDATORY_ROLES = frozenset({ModelRole.GENERATION, ModelRole.EMBEDDING})
    _OPTIONAL_SETTINGS_ROLES = (
        ModelRole.RERANKER,
        ModelRole.SPEECH_TO_TEXT,
        ModelRole.VAD,
        ModelRole.TEXT_TO_SPEECH,
        ModelRole.TRANSLATION,
        ModelRole.CODE_SPECIALIST,
        ModelRole.VISION,
        ModelRole.SPECIALIST_PERCEPTION,
    )
    _RECOMMENDED_SPECIALIST_DEFAULTS: dict[ModelRole, str] = {
        ModelRole.RERANKER: "jinaai/jina-reranker-v1-tiny-en",
        ModelRole.SPEECH_TO_TEXT: "openai/whisper-tiny or whisper.cpp",
        ModelRole.TEXT_TO_SPEECH: "Piper",
        ModelRole.VAD: "Silero VAD",
        ModelRole.TRANSLATION: "Argos Translate",
        ModelRole.CODE_SPECIALIST: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        ModelRole.VISION: "HuggingFaceTB/SmolVLM-256M-Instruct",
        ModelRole.SPECIALIST_PERCEPTION: "PaddleOCR",
    }
    _GOVERNOR_ADVISORY_KINDS = frozenset(
        {
            OptimizerSuggestionKind.MODEL_LOADING,
            OptimizerSuggestionKind.CACHE_PREFETCH,
        }
    )
    _GOVERNOR_ADVISORY_CACHE_NAMESPACES = frozenset(
        {"retrieval_candidates", "runtime_subsets", "strategy_artifacts", "compression_artifacts"}
    )
    _MAX_GOVERNOR_ADVISORIES = 4
    _MAX_GOVERNOR_ADVISORY_ROLES = 2
    _MAX_GOVERNOR_ADVISORY_WARM_KEYS = 4
    _MAX_GOVERNOR_ADVISORY_RETENTION_S = 120.0

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
        self._registrations: dict[str, ModelRegistration] = {}
        self._preferred_models: dict[str, str] = {}
        self._route_history: list[ModelRouteDecision] = []
        self._active_heavy_roles: list[str] = []
        self._loaded_optional_heavy_roles: dict[str, float] = {}
        self._governor_advisory_inputs: dict[str, _GovernorAdvisoryInput] = {}
        self._suspended_default_heavy_roles: set[str] = set()
        self._heavy_slot_limit = 2
        self._cache_manager = BoundedCacheManager(
            {
                "embeddings": 128,
                "retrieval_candidates": 64,
                "runtime_subsets": 64,
                "strategy_artifacts": 64,
                "compression_artifacts": 32,
            }
        )
        self._initialize_registry()

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
        self._active_heavy_roles = []
        self._loaded_optional_heavy_roles = {}
        self._suspended_default_heavy_roles = set()
        self.logger.info("ModelManager stopped.")

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Run a generation request through the active bounded backend."""
        self._require_started()
        decision = self.route_role(ModelRole.GENERATION, capability="generate")
        if not decision.allowed:
            raise BackendUnavailableError(decision.fallback_reason or "generation routing blocked")
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
        decision = self.route_role(ModelRole.EMBEDDING, capability=f"embed:{role}")
        if not decision.allowed:
            raise BackendUnavailableError(decision.fallback_reason or "embedding routing blocked")
        cache_key = f"{role}:{text}"
        cached_vector = self._cache_manager.get("embeddings", cache_key)
        if cached_vector is not None:
            return list(cached_vector)
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
                self._cache_manager.put("embeddings", cache_key, tuple(vector))
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
        self._sync_active_heavy_roles()
        governor = self._hardware_governor_state(
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
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
            active_heavy_roles=tuple(self._active_heavy_roles),
            heavy_slot_limit=self._heavy_slot_limit,
            governor_active=governor.active,
            governor_pressure_reasons=governor.pressure_reasons,
            governor_degraded_features=governor.degraded_features,
            queue_pressure=governor.queue_pressure,
            backend_health_degraded=governor.backend_health_degraded,
            allow_continuous_capture=governor.allow_continuous_capture,
            allow_ocr_on_step=governor.allow_ocr_on_step,
            allow_vision_on_step=governor.allow_vision_on_step,
            allow_optional_heavy_residency=governor.allow_optional_heavy_residency,
            allow_background_work=governor.allow_background_work,
            governor_summary=governor.summary,
            telemetry_enabled=telemetry_enabled,
            last_error=self._last_error,
        )

    def register_model(self, registration: ModelRegistration) -> None:
        """Register one model additively without replacing existing wrappers."""
        self._registrations[registration.registration_id] = registration
        preferred_id = self._preferred_models.get(registration.role.value, "")
        preferred = self._registrations.get(preferred_id)
        if (
            not preferred_id
            or preferred is None
            or (not preferred.enabled and registration.enabled)
        ):
            self._preferred_models[registration.role.value] = registration.registration_id

    def list_registered_models(self, *, role: ModelRole | str | None = None) -> tuple[ModelRegistration, ...]:
        """Return the current typed model registrations."""
        if role is None:
            return tuple(self._registrations.values())
        resolved_role = role if isinstance(role, ModelRole) else ModelRole(str(role))
        return tuple(item for item in self._registrations.values() if item.role == resolved_role)

    def apply_user_settings_profile(self, profile: "UserSettingsProfile") -> None:
        """Apply persisted per-role preferences without weakening the base runtime."""
        preferred_by_role = {
            str(key): str(value)
            for key, value in dict(profile.models.get("preferred_by_role", {})).items()
            if str(key).strip() and str(value).strip()
        }
        enabled_roles = {
            str(role_name).strip()
            for role_name in profile.models.get("enabled_roles", ())
            if str(role_name).strip()
        }
        enabled_roles.update(role.value for role in self._MANDATORY_ROLES)

        for role in ModelRole:
            preferred_reference = preferred_by_role.get(role.value, "")
            if preferred_reference:
                resolved = self._resolve_registration_reference(role, preferred_reference)
                if resolved is not None:
                    self._preferred_models[role.value] = resolved.registration_id
            if role in self._OPTIONAL_SETTINGS_ROLES:
                self._set_optional_role_enabled(role, role.value in enabled_roles)

        self._sync_active_heavy_roles()

    def set_role_enabled(self, role: ModelRole | str, enabled: bool) -> bool:
        """Enable or disable one optional specialist role without weakening the base runtime."""
        resolved_role = role if isinstance(role, ModelRole) else ModelRole(str(role))
        if resolved_role in self._MANDATORY_ROLES:
            return bool(enabled)
        if resolved_role not in self._OPTIONAL_SETTINGS_ROLES:
            return False
        self._set_optional_role_enabled(resolved_role, bool(enabled))
        self._sync_active_heavy_roles()
        return True

    def install_guidance_for_role(self, role: ModelRole | str) -> tuple[str, ...]:
        """Return bounded install or enable guidance for one local-AI role."""
        resolved_role = role if isinstance(role, ModelRole) else ModelRole(str(role))
        registrations = self.list_registered_models(role=resolved_role)
        concrete_registrations = tuple(
            registration for registration in registrations if not self._is_placeholder_registration(registration)
        )
        guidance: list[str] = []
        recommended = self._recommended_default_for_role(resolved_role)
        preferred = self._preferred_registration_for_role(resolved_role)
        if preferred is not None and not self._is_placeholder_registration(preferred):
            guidance.append(
                f"Preferred route: {preferred.backend} / {preferred.model_identifier}."
            )
        if concrete_registrations:
            ready_registrations = tuple(
                registration for registration in concrete_registrations if not registration.missing_dependencies
            )
            blocked_registrations = tuple(
                registration for registration in concrete_registrations if registration.missing_dependencies
            )
            if ready_registrations:
                guidance.append(
                    "Available now: "
                    + ", ".join(
                        f"{registration.backend} / {registration.model_identifier}"
                        for registration in ready_registrations[:2]
                    )
                    + "."
                )
                guidance.append("Enable the role in settings or the control plane, then use Warm or Test Ping.")
                if recommended:
                    guidance.append(f"Recommended lightweight default: {recommended}.")
            for registration in blocked_registrations[:2]:
                guidance.append(
                    f"Install {', '.join(registration.missing_dependencies)} to use "
                    f"{registration.backend} / {registration.model_identifier}."
                )
        else:
            guidance.append(f"No concrete local backend is registered for {resolved_role.value}.")
            if recommended:
                guidance.append(f"Recommended lightweight default: {recommended}.")
            guidance.append("Keep the role disabled until one local backend is installed and registered.")
        if resolved_role in self._MANDATORY_ROLES:
            guidance.append("This role is part of the baseline runtime and should stay enabled.")
        return tuple(dict.fromkeys(item for item in guidance if item.strip()))

    def recent_route_decisions_for_role(
        self,
        role: ModelRole | str,
        *,
        limit: int = 4,
    ) -> tuple[ModelRouteDecision, ...]:
        """Return recent typed route decisions for one role."""
        resolved_role = role if isinstance(role, ModelRole) else ModelRole(str(role))
        decisions = [decision for decision in self._route_history if decision.requested_role == resolved_role]
        return tuple(decisions[-max(1, limit) :])

    def recent_fallback_reasons_for_role(
        self,
        role: ModelRole | str,
        *,
        limit: int = 4,
    ) -> tuple[str, ...]:
        """Return recent distinct fallback reasons for one role."""
        resolved_role = role if isinstance(role, ModelRole) else ModelRole(str(role))
        reasons: list[str] = []
        if resolved_role.value in {ModelRole.GENERATION.value, ModelRole.EMBEDDING.value} and self._fallback_reason:
            reasons.append(self._fallback_reason)
        for decision in reversed(self._route_history):
            if decision.requested_role != resolved_role or not decision.fallback_reason:
                continue
            if decision.fallback_reason not in reasons:
                reasons.append(decision.fallback_reason)
            if len(reasons) >= max(1, limit):
                break
        return tuple(reasons[: max(1, limit)])

    async def warm_role(self, role: ModelRole | str) -> ModelRouteDecision:
        """Warm one role with a bounded readiness check."""
        self._require_started()
        resolved_role = role if isinstance(role, ModelRole) else ModelRole(str(role))
        if resolved_role == ModelRole.GENERATION:
            await self._ensure_generation_backend_ready()
            self._mark_used()
            await self._refresh_health()
            return self.route_role(resolved_role, capability="warm")
        if resolved_role == ModelRole.EMBEDDING:
            await self._ensure_embedding_backend_ready()
            self._mark_used()
            await self._refresh_health()
            return self.route_role(resolved_role, capability="warm")
        decision = self.route_role(resolved_role, capability="warm")
        if decision.allowed:
            self._mark_used()
            await self._refresh_health()
        return decision

    async def test_role_ping(
        self,
        role: ModelRole | str,
    ) -> tuple[ModelRouteDecision, str]:
        """Run one bounded test ping for a local-AI role."""
        self._require_started()
        resolved_role = role if isinstance(role, ModelRole) else ModelRole(str(role))
        if resolved_role == ModelRole.GENERATION:
            preview = await self.generate("Local AI control plane ping.", max_tokens=24)
            return self.route_role(resolved_role, capability="test_ping"), (
                f"Generated a bounded ping response: {preview[:80]}"
            )
        if resolved_role == ModelRole.EMBEDDING:
            vector = await self.embed_query("Local AI control plane ping.")
            return self.route_role(resolved_role, capability="test_ping"), (
                f"Produced an embedding preview with {len(vector)} dimensions."
            )
        if resolved_role == ModelRole.TEXT_TO_SPEECH:
            output_path = self.config.storage.logs_dir / "dashboard_tts_test.wav"
            result = await self.synthesize_text("Local AI control plane ping.", output_path=output_path)
            return self.route_role(resolved_role, capability="test_ping"), (
                f"TTS wrote {result.target_path or '(none)'} with status {result.status}."
            )
        if resolved_role == ModelRole.TRANSLATION:
            result = await self.translate_text(
                "Hello local models.",
                source_language="en",
                target_language="es",
                source_scope="test_ping",
            )
            return self.route_role(resolved_role, capability="test_ping"), (
                f"Translation returned '{result.translated_text[:80] or '(none)'}' with status {result.status}."
            )
        if resolved_role == ModelRole.CODE_SPECIALIST:
            result = await self.analyze_code(
                text="def add(a, b):\n    return a + b\n",
                request_text="Summarize this helper.",
                source_scope="test_ping",
            )
            return self.route_role(resolved_role, capability="test_ping"), (
                f"Code specialist returned '{result.summary[:80] or '(none)'}' with status {result.status}."
            )
        decision = self.route_role(resolved_role, capability="test_ping")
        if not decision.allowed:
            return decision, "Role routing is currently blocked."
        self._mark_used()
        await self._refresh_health()
        capability_messages = {
            ModelRole.RERANKER: "Reranker route is ready for bounded local-result reranking.",
            ModelRole.SPEECH_TO_TEXT: "Speech-to-text route is ready. Use the Audio tab for a WAV transcription.",
            ModelRole.VAD: "VAD route is ready. Use the Audio tab to analyze one WAV clip.",
            ModelRole.TEXT_TO_SPEECH: "Text-to-speech route is enabled but no synthesis backend is registered yet.",
            ModelRole.TRANSLATION: "Translation route is ready. Use the Translation tab for bounded local translation.",
            ModelRole.CODE_SPECIALIST: "Code-specialist route is ready. Use the Code tab for bounded local review.",
            ModelRole.VISION: "Vision route is ready for bounded on-step screenshot inspection.",
            ModelRole.SPECIALIST_PERCEPTION: (
                "Specialist-perception route is ready; keep it disabled unless CPU OCR plus vision are insufficient."
            ),
        }
        return decision, capability_messages.get(
            resolved_role,
            "Role route is ready for bounded requests.",
        )

    async def rerank_local_results(
        self,
        *,
        query_text: str,
        results: tuple[SearchResult, ...],
        top_k: int,
    ) -> tuple[tuple[SearchResult, ...], ModelRouteDecision]:
        """Apply the optional reranker role to a bounded local-result set."""
        self._require_started()
        decision = self.route_role(ModelRole.RERANKER, capability="rerank")
        if not decision.allowed or len(results) <= 1:
            return tuple(results), decision

        reranked = rerank_search_results(
            query_text,
            tuple(results),
            max_candidates=min(max(1, top_k), self.config.retrieval.max_rerank_candidates),
            combined_weight=self.config.retrieval.rerank_combined_weight,
            lexical_weight=self.config.retrieval.rerank_lexical_weight,
            order_weight=self.config.retrieval.rerank_order_weight,
            exact_phrase_weight=self.config.retrieval.rerank_exact_phrase_weight,
            title_weight=self.config.retrieval.rerank_title_weight,
        )
        self._mark_used()
        return tuple(reranked[: max(1, top_k)]), decision

    def unload_optional_role(self, role: ModelRole | str) -> bool:
        """Unload one optional role without affecting the base generation or embedding pair."""
        resolved_role = role if isinstance(role, ModelRole) else ModelRole(str(role))
        if resolved_role in self._MANDATORY_ROLES:
            return False
        removed = self._loaded_optional_heavy_roles.pop(resolved_role.value, None) is not None
        self._sync_active_heavy_roles()
        return removed or bool(self.list_registered_models(role=resolved_role))

    async def detect_voice_activity(self, audio_path: str | Path) -> VoiceActivityReport:
        """Run the optional VAD role over one bounded local WAV file."""
        self._require_started()
        decision = self.route_role(ModelRole.VAD, capability="voice_activity_detection")
        source_path = Path(audio_path)
        if not decision.allowed:
            return VoiceActivityReport(
                source_path=str(source_path),
                analyzer_backend=decision.selected_backend,
                warnings=(decision.fallback_reason or "vad_routing_blocked",),
            )
        report = await asyncio.to_thread(
            analyze_voice_activity,
            source_path,
            max_duration_s=self.config.audio.max_input_duration_s,
            frame_ms=self.config.audio.vad_frame_ms,
            min_speech_ms=self.config.audio.vad_min_speech_ms,
            merge_silence_ms=self.config.audio.vad_merge_silence_ms,
            energy_threshold=self.config.audio.vad_energy_threshold,
            analyzer_backend=decision.selected_backend or "energy_vad",
        )
        self._mark_used()
        await self._refresh_health()
        return report

    async def transcribe_audio(self, audio_path: str | Path) -> AudioTranscriptionResult:
        """Run the optional speech-to-text role over one bounded local WAV file."""
        self._require_started()
        decision = self.route_role(ModelRole.SPEECH_TO_TEXT, capability="transcribe")
        source_path = Path(audio_path)
        if not decision.allowed:
            return AudioTranscriptionResult(
                source_path=str(source_path),
                status="blocked",
                transcription_backend=decision.selected_backend,
                transcription_model=decision.selected_model_identifier,
                warnings=(decision.fallback_reason or "speech_to_text_routing_blocked",),
            )
        vad_decision = self.route_role(ModelRole.VAD, capability="voice_activity_detection")
        used_vad = vad_decision.allowed
        if used_vad:
            vad_report = await self.detect_voice_activity(source_path)
        else:
            vad_report = await asyncio.to_thread(
                analyze_voice_activity,
                source_path,
                max_duration_s=self.config.audio.max_input_duration_s,
                frame_ms=self.config.audio.vad_frame_ms,
                min_speech_ms=self.config.audio.vad_min_speech_ms,
                merge_silence_ms=self.config.audio.vad_merge_silence_ms,
                energy_threshold=self.config.audio.vad_energy_threshold,
                analyzer_backend="passive_audio_scan",
            )
            if vad_decision.fallback_reason:
                vad_report = replace(
                    vad_report,
                    warnings=tuple(vad_report.warnings) + (f"vad_{vad_decision.fallback_reason}",),
                )
        backend_name = decision.selected_backend or "stub_speech_to_text"
        if backend_name == "windows_system_speech":
            result = await asyncio.to_thread(
                transcribe_with_system_speech,
                source_path,
                transcription_model=decision.selected_model_identifier or "System.Speech",
                vad_report=vad_report,
                used_vad=used_vad,
                max_duration_s=self.config.audio.max_input_duration_s,
                timeout_s=self.config.audio.system_speech_timeout_s,
                max_transcript_chars=self.config.audio.max_transcript_chars,
            )
        else:
            result = await asyncio.to_thread(
                transcribe_with_stub,
                source_path,
                transcription_model=decision.selected_model_identifier or "stub-whisper-tiny",
                vad_report=vad_report,
                used_vad=used_vad,
                max_transcript_chars=self.config.audio.max_transcript_chars,
            )
        self._mark_used()
        await self._refresh_health()
        return result

    async def synthesize_text(
        self,
        text: str,
        *,
        output_path: str | Path,
    ) -> AudioSynthesisResult:
        """Run the optional text-to-speech role over one bounded local text request."""
        self._require_started()
        decision = self.route_role(ModelRole.TEXT_TO_SPEECH, capability="synthesize")
        target_path = Path(output_path)
        if not decision.allowed:
            return AudioSynthesisResult(
                target_path=str(target_path),
                status="blocked",
                source_text=str(text),
                clipped_text="",
                synthesis_backend=decision.selected_backend,
                synthesis_model=decision.selected_model_identifier,
                warnings=(decision.fallback_reason or "text_to_speech_routing_blocked",),
            )
        backend_name = decision.selected_backend or "stub_text_to_speech"
        if backend_name == "windows_system_speech":
            result = await asyncio.to_thread(
                synthesize_with_system_speech,
                text,
                output_path=target_path,
                synthesis_model=decision.selected_model_identifier or "System.Speech",
                max_chars=self.config.audio.max_tts_chars,
                timeout_s=self.config.audio.system_speech_synthesis_timeout_s,
            )
        else:
            result = await asyncio.to_thread(
                synthesize_with_stub,
                text,
                output_path=target_path,
                synthesis_model=decision.selected_model_identifier or "stub-piper",
                max_chars=self.config.audio.max_tts_chars,
                sample_rate_hz=self.config.audio.tts_sample_rate_hz,
            )
        self._mark_used()
        await self._refresh_health()
        return result

    async def translate_text(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
        source_scope: str = "free_text",
    ) -> TextTranslationResult:
        """Run the optional translation role over one bounded local text request."""
        self._require_started()
        decision = self.route_role(ModelRole.TRANSLATION, capability="translate")
        if not decision.allowed:
            return TextTranslationResult(
                status="blocked",
                source_text=str(text).strip(),
                translated_text="",
                source_language=str(source_language).strip(),
                target_language=str(target_language).strip(),
                translation_backend=decision.selected_backend,
                translation_model=decision.selected_model_identifier,
                source_scope=source_scope,
                warnings=(decision.fallback_reason or "translation_routing_blocked",),
            )
        backend_name = decision.selected_backend or "stub_translation"
        if backend_name == "argos_translate":
            result = await asyncio.to_thread(
                translate_with_argos,
                text,
                source_language=source_language,
                target_language=target_language,
                translation_model=decision.selected_model_identifier or "Argos Translate",
                max_chars=self.config.translation.max_input_chars,
                source_scope=source_scope,
            )
        else:
            result = await asyncio.to_thread(
                translate_with_stub,
                text,
                source_language=source_language,
                target_language=target_language,
                translation_model=decision.selected_model_identifier or "stub-argos",
                max_chars=self.config.translation.max_input_chars,
                source_scope=source_scope,
            )
        self._mark_used()
        await self._refresh_health()
        return result

    async def analyze_code(
        self,
        *,
        text: str,
        request_text: str,
        source_scope: str = "snippet",
    ) -> CodeSpecialistResult:
        """Run the optional code-specialist role over bounded local code text."""
        self._require_started()
        decision = self.route_role(ModelRole.CODE_SPECIALIST, capability="code_assist")
        if not decision.allowed:
            return CodeSpecialistResult(
                status="blocked",
                source_scope=source_scope,
                request_text=request_text,
                code_backend=decision.selected_backend,
                code_model=decision.selected_model_identifier,
                warnings=(decision.fallback_reason or "code_specialist_routing_blocked",),
            )
        result = await asyncio.to_thread(
            analyze_code_with_stub,
            text,
            request_text=request_text,
            source_scope=source_scope,
            code_model=decision.selected_model_identifier or "stub-qwen-coder-1.5b",
            max_chars=self.config.code_specialist.max_input_chars,
            max_lines=self.config.code_specialist.max_input_lines,
        )
        self._mark_used()
        await self._refresh_health()
        return result

    async def analyze_code_file(
        self,
        source_path: str | Path,
        *,
        request_text: str,
    ) -> CodeSpecialistResult:
        """Run the optional code-specialist role over one bounded local file."""
        self._require_started()
        decision = self.route_role(ModelRole.CODE_SPECIALIST, capability="code_assist")
        if not decision.allowed:
            return CodeSpecialistResult(
                status="blocked",
                source_scope="file",
                source_path=str(source_path),
                request_text=request_text,
                code_backend=decision.selected_backend,
                code_model=decision.selected_model_identifier,
                warnings=(decision.fallback_reason or "code_specialist_routing_blocked",),
            )
        result = await asyncio.to_thread(
            analyze_code_file_with_stub,
            source_path,
            request_text=request_text,
            code_model=decision.selected_model_identifier or "stub-qwen-coder-1.5b",
            max_chars=self.config.code_specialist.max_input_chars,
            max_lines=self.config.code_specialist.max_input_lines,
        )
        self._mark_used()
        await self._refresh_health()
        return result

    async def inspect_image(
        self,
        image_path: str | Path,
        *,
        request_text: str,
        extracted_text: str = "",
        role: ModelRole | str = ModelRole.VISION,
    ) -> VisionInspectionResult:
        """Run one bounded optional visual-role inspection request."""
        self._require_started()
        resolved_role = role if isinstance(role, ModelRole) else ModelRole(str(role))
        if resolved_role not in {ModelRole.VISION, ModelRole.SPECIALIST_PERCEPTION}:
            raise ValueError("inspect_image only supports vision or specialist_perception roles.")
        capability = "vision_inference" if resolved_role == ModelRole.VISION else "specialist_perception"
        decision = self.route_role(resolved_role, capability=capability)
        source_path = Path(image_path)
        if not decision.allowed:
            return VisionInspectionResult(
                status="blocked",
                source_path=str(source_path),
                request_text=request_text,
                role=resolved_role,
                inspection_backend=decision.selected_backend,
                inspection_model=decision.selected_model_identifier,
                warnings=(decision.fallback_reason or f"{resolved_role.value}_routing_blocked",),
                degraded_reason=decision.fallback_reason or f"{resolved_role.value}_routing_blocked",
            )
        backend_name = decision.selected_backend or f"stub_{resolved_role.value}"
        if backend_name.startswith("stub_"):
            result = await asyncio.to_thread(
                inspect_image_with_stub,
                source_path,
                request_text=request_text,
                extracted_text=extracted_text,
                role=resolved_role,
                vision_model=decision.selected_model_identifier or f"stub-{resolved_role.value}",
            )
        else:
            result = VisionInspectionResult(
                status="blocked",
                source_path=str(source_path),
                request_text=request_text,
                role=resolved_role,
                inspection_backend=backend_name,
                inspection_model=decision.selected_model_identifier,
                warnings=(f"{resolved_role.value}_backend_unimplemented",),
                degraded_reason=f"{resolved_role.value}_backend_unimplemented",
            )
        self._mark_used()
        await self._refresh_health()
        return result

    def route_role(self, role: ModelRole | str, *, capability: str = "") -> ModelRouteDecision:
        """Route one requested role through the typed registry without changing wrappers."""
        resolved_role = role if isinstance(role, ModelRole) else ModelRole(str(role))
        current = self._current_registration_for_role(resolved_role)
        selected = current or self._preferred_registration_for_role(resolved_role)
        if selected is None:
            decision = ModelRouteDecision(
                requested_role=resolved_role,
                capability=capability,
                allowed=False,
                fallback_reason="no_registration_for_role",
                active_heavy_roles=tuple(self._active_heavy_roles),
                heavy_slot_limit=self._heavy_slot_limit,
            )
            self._record_route_decision(decision)
            return decision

        if not selected.enabled:
            decision = ModelRouteDecision(
                requested_role=resolved_role,
                selected_registration_id=selected.registration_id,
                selected_backend=selected.backend,
                selected_model_identifier=selected.model_identifier,
                resource_class=selected.resource_class,
                capability=capability,
                allowed=False,
                fallback_reason="role_disabled",
                active_heavy_roles=tuple(self._active_heavy_roles),
                heavy_slot_limit=self._heavy_slot_limit,
            )
            self._record_route_decision(decision)
            return decision

        if selected.missing_dependencies:
            decision = ModelRouteDecision(
                requested_role=resolved_role,
                selected_registration_id=selected.registration_id,
                selected_backend=selected.backend,
                selected_model_identifier=selected.model_identifier,
                resource_class=selected.resource_class,
                capability=capability,
                allowed=False,
                fallback_reason="missing_dependencies",
                active_heavy_roles=tuple(self._active_heavy_roles),
                heavy_slot_limit=self._heavy_slot_limit,
                metadata={"missing_dependencies": selected.missing_dependencies},
            )
            self._record_route_decision(decision)
            return decision

        swapped_out_roles: tuple[str, ...] = ()
        reactivated_roles: tuple[str, ...] = ()
        if selected.resource_class == ModelResourceClass.HEAVY:
            governor = self._hardware_governor_state()
            if (
                resolved_role not in self._MANDATORY_ROLES
                and not governor.allow_optional_heavy_residency
            ):
                decision = ModelRouteDecision(
                    requested_role=resolved_role,
                    selected_registration_id=selected.registration_id,
                    selected_backend=selected.backend,
                    selected_model_identifier=selected.model_identifier,
                    resource_class=selected.resource_class,
                    capability=capability,
                    allowed=False,
                    fallback_reason="hardware_governor_optional_heavy_residency_disabled",
                    active_heavy_roles=tuple(self._active_heavy_roles),
                    heavy_slot_limit=self._heavy_slot_limit,
                    metadata={
                        "pressure_reasons": governor.pressure_reasons,
                        "degraded_features": governor.degraded_features,
                    },
                )
                self._record_route_decision(decision)
                return decision
            if resolved_role in self._MANDATORY_ROLES:
                reactivated_roles = self._ensure_mandatory_heavy_role_active(resolved_role)
            elif resolved_role.value not in self._active_heavy_roles:
                if len(self._active_heavy_roles) >= self._heavy_slot_limit:
                    swap_result = self._swap_in_optional_heavy_role(resolved_role)
                    if swap_result is None:
                        decision = ModelRouteDecision(
                            requested_role=resolved_role,
                            selected_registration_id=selected.registration_id,
                            selected_backend=selected.backend,
                            selected_model_identifier=selected.model_identifier,
                            resource_class=selected.resource_class,
                            capability=capability,
                            allowed=False,
                            fallback_reason="heavy_slot_cap_reached",
                            active_heavy_roles=tuple(self._active_heavy_roles),
                            heavy_slot_limit=self._heavy_slot_limit,
                        )
                        self._record_route_decision(decision)
                        return decision
                    swapped_out_roles = swap_result
                else:
                    self._touch_optional_heavy_role(resolved_role)
            else:
                self._touch_optional_heavy_role(resolved_role)

        decision = ModelRouteDecision(
            requested_role=resolved_role,
            selected_registration_id=selected.registration_id,
            selected_backend=selected.backend,
            selected_model_identifier=selected.model_identifier,
            resource_class=selected.resource_class,
            capability=capability,
            allowed=True,
            used_fallback=(
                self._fallback_active
                and resolved_role in {ModelRole.GENERATION, ModelRole.EMBEDDING}
                and "fallback" in selected.registration_id
            ),
            fallback_reason=(self._fallback_reason or "") if self._fallback_active else "",
            active_heavy_roles=tuple(self._active_heavy_roles),
            heavy_slot_limit=self._heavy_slot_limit,
            metadata={
                "swapped_out_roles": swapped_out_roles,
                "reactivated_roles": reactivated_roles,
                "load_policy": selected.load_policy.value,
            },
        )
        self._record_route_decision(decision)
        return decision

    def registry_view(
        self,
        *,
        advisory_available: bool = False,
        optimizer_subscriptions: tuple[str, ...] = (),
        recent_optimizer_suggestions: tuple[OptimizerSuggestionRecord, ...] = (),
    ) -> ModelRegistryView:
        """Return the persisted control-plane view for all registered local models."""
        fallback_reasons: dict[str, str] = {}
        if self._fallback_reason:
            fallback_reasons["generation"] = self._fallback_reason
            fallback_reasons["embedding"] = self._fallback_reason
        governor = self._hardware_governor_state()
        return ModelRegistryView(
            registrations=tuple(self._registrations.values()),
            preferred_models=dict(self._preferred_models),
            active_heavy_roles=tuple(self._active_heavy_roles),
            heavy_slot_limit=self._heavy_slot_limit,
            fallback_reasons=fallback_reasons,
            governor_active=governor.active,
            governor_pressure_reasons=governor.pressure_reasons,
            governor_degraded_features=governor.degraded_features,
            governor_summary=governor.summary,
            last_route_decisions=tuple(self._route_history[-8:]),
            advisory_available=advisory_available,
            optimizer_subscriptions=optimizer_subscriptions,
            recent_optimizer_suggestions=recent_optimizer_suggestions,
            cache_snapshots=self._cache_manager.snapshots(),
            compression_insights=self.compression_insight_summaries(),
        )

    def cache_snapshots(self):
        """Return read-only bounded-cache summaries for the local-AI control plane."""
        return self._cache_manager.snapshots()

    def lookup_cache(self, namespace: str, key: str):
        """Return one bounded cache entry for cross-component lightweight scoring."""
        return self._cache_manager.get(namespace, key)

    def warm_cache(self, namespace: str, key: str, value) -> None:
        """Warm a bounded cache namespace without creating unbounded background work."""
        self._cache_manager.put(namespace, key, value)

    def apply_governor_advisory_inputs(self, suggestions: Sequence[OptimizerSuggestionRecord]) -> None:
        """Clamp optimizer hints into short-lived governor inputs without creating a second control path."""
        now = time.monotonic()
        self._prune_governor_advisory_inputs(now=now)
        accepted: dict[str, _GovernorAdvisoryInput] = {}
        for suggestion in suggestions[: self._MAX_GOVERNOR_ADVISORIES]:
            parsed = self._parse_governor_advisory_input(suggestion, now=now)
            if parsed is None:
                continue
            accepted[parsed.suggestion_id] = parsed
        if not accepted:
            return
        merged = {**self._governor_advisory_inputs, **accepted}
        ordered = sorted(
            merged.values(),
            key=lambda item: (item.expires_at_monotonic, item.kind.value, item.suggestion_id),
            reverse=True,
        )[: self._MAX_GOVERNOR_ADVISORIES]
        self._governor_advisory_inputs = {item.suggestion_id: item for item in ordered}

    def compression_insight_summaries(self, *, limit: int = 4) -> tuple[CompressionInsightSummary, ...]:
        """Return bounded typed compression summaries for the app shell."""
        summaries: list[CompressionInsightSummary] = []
        for _key, payload in self._cache_manager.recent_items("compression_artifacts", limit=limit):
            if not isinstance(payload, dict):
                continue
            source = str(payload.get("source", "")).strip()
            evidence_basis = str(payload.get("evidence_basis", "")).strip()
            if not evidence_basis:
                if source == "compression_service":
                    evidence_basis = "deterministic_analysis"
                elif source == "self_optimizer":
                    evidence_basis = "replay_evidence"
                else:
                    evidence_basis = "advisor_rerank" if "advisor" in source else "deterministic_analysis"
            validation_state = str(payload.get("validation_state", "")).strip()
            blocked_reason = str(payload.get("blocked_reason", "")).strip()
            accepted = bool(payload.get("accepted", False))
            if not validation_state:
                validation_state = "validated" if accepted else ("blocked" if blocked_reason else "deferred")
            try:
                estimated_gain = float(payload.get("compression_gain", payload.get("estimated_gain", 0.0)) or 0.0)
            except (TypeError, ValueError):
                estimated_gain = 0.0
            try:
                validation_pass_rate = float(payload.get("validation_pass_rate", 0.0) or 0.0)
            except (TypeError, ValueError):
                validation_pass_rate = 0.0
            proposal_id = str(payload.get("proposal_id", "")).strip()
            if not proposal_id:
                continue
            summaries.append(
                CompressionInsightSummary(
                    proposal_id=proposal_id,
                    macro_name=str(payload.get("macro_name", "")),
                    proof_fingerprint=str(payload.get("proof_fingerprint", "")),
                    estimated_gain=max(0.0, min(1.0, estimated_gain)),
                    validation_pass_rate=max(0.0, min(1.0, validation_pass_rate)),
                    validation_state=validation_state,
                    blocked_reason=blocked_reason,
                    evidence_basis=evidence_basis,
                    origin_component=source,
                    accepted=accepted,
                )
            )
        return tuple(summaries)

    async def _activate_primary_backends(self) -> None:
        self._active_generation_backend = self._primary_generation_backend
        self._active_embedding_backend = self._primary_embedding_backend
        assert self._active_generation_backend is not None
        assert self._active_embedding_backend is not None

        await self._start_backend_with_fallback(kind="generation")
        await self._start_backend_with_fallback(kind="embedding")
        await self._refresh_health()

    def _initialize_registry(self) -> None:
        self._preferred_models = {
            ModelRole.GENERATION.value: f"generation_primary:{self.config.preflight.backends.generation_backend}:{self.config.preflight.backends.generation_model}",
            ModelRole.EMBEDDING.value: f"embedding_primary:{self.config.preflight.backends.embedding_backend}:{self.config.preflight.backends.embedding_model}",
        }
        self._registrations = {}
        self._register_default_roles()
        self._sync_active_heavy_roles()

    def _register_default_roles(self) -> None:
        backends = self.config.preflight.backends
        self.register_model(
            ModelRegistration(
                registration_id=f"generation_primary:{backends.generation_backend}:{backends.generation_model}",
                role=ModelRole.GENERATION,
                backend=backends.generation_backend,
                model_identifier=backends.generation_model,
                resource_class=ModelResourceClass.HEAVY,
                preferred_device="auto",
                load_policy=ModelLoadPolicy.ALWAYS_ON,
                supported_capabilities=("generate",),
                missing_dependencies=self._missing_dependencies_for_backend(backends.generation_backend),
                metadata={"fallback": False},
            )
        )
        self.register_model(
            ModelRegistration(
                registration_id=(
                    f"generation_fallback:{backends.generation_fallback_backend}:{backends.generation_fallback_model}"
                ),
                role=ModelRole.GENERATION,
                backend=backends.generation_fallback_backend,
                model_identifier=backends.generation_fallback_model,
                resource_class=ModelResourceClass.HEAVY,
                preferred_device="auto",
                load_policy=ModelLoadPolicy.PREFER_IDLE_UNLOAD,
                supported_capabilities=("generate",),
                missing_dependencies=self._missing_dependencies_for_backend(backends.generation_fallback_backend),
                metadata={"fallback": True},
            )
        )
        self.register_model(
            ModelRegistration(
                registration_id=f"embedding_primary:{backends.embedding_backend}:{backends.embedding_model}",
                role=ModelRole.EMBEDDING,
                backend=backends.embedding_backend,
                model_identifier=backends.embedding_model,
                resource_class=ModelResourceClass.HEAVY,
                preferred_device="auto",
                load_policy=ModelLoadPolicy.ALWAYS_ON,
                supported_capabilities=("embed", "embed_query", "embed_document"),
                missing_dependencies=self._missing_dependencies_for_backend(backends.embedding_backend),
                metadata={"fallback": False},
            )
        )
        self.register_model(
            ModelRegistration(
                registration_id=(
                    f"embedding_fallback:{backends.embedding_fallback_backend}:{backends.embedding_fallback_model}"
                ),
                role=ModelRole.EMBEDDING,
                backend=backends.embedding_fallback_backend,
                model_identifier=backends.embedding_fallback_model,
                resource_class=ModelResourceClass.HEAVY,
                preferred_device="auto",
                load_policy=ModelLoadPolicy.PREFER_IDLE_UNLOAD,
                supported_capabilities=("embed", "embed_query", "embed_document"),
                missing_dependencies=self._missing_dependencies_for_backend(backends.embedding_fallback_backend),
                metadata={"fallback": True},
            )
        )
        if self.config.preflight.flags.stub_mode:
            self.register_model(
                ModelRegistration(
                    registration_id="generation_stub:stub_generation:stub-generation",
                    role=ModelRole.GENERATION,
                    backend="stub_generation",
                    model_identifier="stub-generation",
                    resource_class=ModelResourceClass.HEAVY,
                    preferred_device="cpu",
                    load_policy=ModelLoadPolicy.ALWAYS_ON,
                    supported_capabilities=("generate",),
                )
            )
            self.register_model(
                ModelRegistration(
                    registration_id="embedding_stub:stub_embedding:stub-embedding",
                    role=ModelRole.EMBEDDING,
                    backend="stub_embedding",
                    model_identifier="stub-embedding",
                    resource_class=ModelResourceClass.HEAVY,
                    preferred_device="cpu",
                    load_policy=ModelLoadPolicy.ALWAYS_ON,
                    supported_capabilities=("embed", "embed_query", "embed_document"),
                )
            )
        if self._windows_system_speech_backend_ready():
            self.register_model(
                ModelRegistration(
                    registration_id="speech_to_text_windows:windows_system_speech:System.Speech",
                    role=ModelRole.SPEECH_TO_TEXT,
                    backend="windows_system_speech",
                    model_identifier="System.Speech",
                    resource_class=ModelResourceClass.SIDECAR,
                    enabled=False,
                    preferred_device="cpu",
                    load_policy=ModelLoadPolicy.ON_DEMAND,
                    supported_capabilities=("transcribe",),
                    metadata={"recommended_default": "System.Speech"},
                )
            )
            self.register_model(
                ModelRegistration(
                    registration_id="text_to_speech_windows:windows_system_speech:System.Speech",
                    role=ModelRole.TEXT_TO_SPEECH,
                    backend="windows_system_speech",
                    model_identifier="System.Speech",
                    resource_class=ModelResourceClass.SIDECAR,
                    enabled=False,
                    preferred_device="cpu",
                    load_policy=ModelLoadPolicy.ON_DEMAND,
                    supported_capabilities=("synthesize",),
                    metadata={"recommended_default": "System.Speech"},
                )
            )
        if argos_translate_available():
            self.register_model(
                ModelRegistration(
                    registration_id="translation_argos:argos_translate:Argos Translate",
                    role=ModelRole.TRANSLATION,
                    backend="argos_translate",
                    model_identifier="Argos Translate",
                    resource_class=ModelResourceClass.SIDECAR,
                    enabled=False,
                    preferred_device="cpu",
                    load_policy=ModelLoadPolicy.ON_DEMAND,
                    supported_capabilities=("translate",),
                    metadata={"recommended_default": "Argos Translate"},
                )
            )
        self.register_model(
            ModelRegistration(
                registration_id="vad_energy:energy_vad:frame-energy-v1",
                role=ModelRole.VAD,
                backend="energy_vad",
                model_identifier="frame-energy-v1",
                resource_class=ModelResourceClass.SIDECAR,
                enabled=False,
                preferred_device="cpu",
                load_policy=ModelLoadPolicy.ON_DEMAND,
                supported_capabilities=("voice_activity_detection",),
                metadata={"recommended_default": "frame-energy-v1"},
            )
        )
        if self.config.preflight.flags.stub_mode:
            self.register_model(
                ModelRegistration(
                    registration_id="reranker_stub:stub_reranker:stub-reranker",
                    role=ModelRole.RERANKER,
                    backend="stub_reranker",
                    model_identifier="stub-reranker",
                    resource_class=ModelResourceClass.SIDECAR,
                    enabled=False,
                    preferred_device="cpu",
                    load_policy=ModelLoadPolicy.ON_DEMAND,
                    supported_capabilities=("rerank",),
                    metadata={"stub": True, "recommended_default": "jinaai/jina-reranker-v1-tiny-en"},
                )
            )
            self.register_model(
                ModelRegistration(
                    registration_id="speech_to_text_stub:stub_speech_to_text:stub-whisper-tiny",
                    role=ModelRole.SPEECH_TO_TEXT,
                    backend="stub_speech_to_text",
                    model_identifier="stub-whisper-tiny",
                    resource_class=ModelResourceClass.SIDECAR,
                    enabled=False,
                    preferred_device="cpu",
                    load_policy=ModelLoadPolicy.ON_DEMAND,
                    supported_capabilities=("transcribe",),
                    metadata={"stub": True, "recommended_default": "openai/whisper-tiny"},
                )
            )
            self.register_model(
                ModelRegistration(
                    registration_id="vad_stub:stub_vad:stub-silero-vad",
                    role=ModelRole.VAD,
                    backend="stub_vad",
                    model_identifier="stub-silero-vad",
                    resource_class=ModelResourceClass.SIDECAR,
                    enabled=False,
                    preferred_device="cpu",
                    load_policy=ModelLoadPolicy.ON_DEMAND,
                    supported_capabilities=("voice_activity_detection",),
                    metadata={"stub": True, "recommended_default": "Silero VAD"},
                )
            )
            self.register_model(
                ModelRegistration(
                    registration_id="text_to_speech_stub:stub_text_to_speech:stub-piper",
                    role=ModelRole.TEXT_TO_SPEECH,
                    backend="stub_text_to_speech",
                    model_identifier="stub-piper",
                    resource_class=ModelResourceClass.SIDECAR,
                    enabled=False,
                    preferred_device="cpu",
                    load_policy=ModelLoadPolicy.ON_DEMAND,
                    supported_capabilities=("synthesize",),
                    metadata={"stub": True, "recommended_default": "Piper"},
                )
            )
            self.register_model(
                ModelRegistration(
                    registration_id="translation_stub:stub_translation:stub-argos",
                    role=ModelRole.TRANSLATION,
                    backend="stub_translation",
                    model_identifier="stub-argos",
                    resource_class=ModelResourceClass.SIDECAR,
                    enabled=False,
                    preferred_device="cpu",
                    load_policy=ModelLoadPolicy.ON_DEMAND,
                    supported_capabilities=("translate",),
                    metadata={"stub": True, "recommended_default": "Argos Translate"},
                )
            )
            self.register_model(
                ModelRegistration(
                    registration_id="code_specialist_stub:stub_code_specialist:stub-qwen-coder-1.5b",
                    role=ModelRole.CODE_SPECIALIST,
                    backend="stub_code_specialist",
                    model_identifier="stub-qwen-coder-1.5b",
                    resource_class=ModelResourceClass.SIDECAR,
                    enabled=False,
                    preferred_device="cpu",
                    load_policy=ModelLoadPolicy.ON_DEMAND,
                    supported_capabilities=("code_assist",),
                    metadata={"stub": True, "recommended_default": "Qwen/Qwen2.5-Coder-1.5B-Instruct"},
                )
            )
            self.register_model(
                ModelRegistration(
                    registration_id="vision_stub:stub_vision:stub-smolvlm-256m",
                    role=ModelRole.VISION,
                    backend="stub_vision",
                    model_identifier="stub-smolvlm-256m",
                    resource_class=ModelResourceClass.HEAVY,
                    enabled=False,
                    preferred_device="cpu",
                    load_policy=ModelLoadPolicy.ON_DEMAND,
                    supported_capabilities=("vision_inference",),
                    metadata={"stub": True, "recommended_default": "HuggingFaceTB/SmolVLM-256M-Instruct"},
                )
            )
            self.register_model(
                ModelRegistration(
                    registration_id="specialist_perception_stub:stub_specialist_perception:stub-paddleocr",
                    role=ModelRole.SPECIALIST_PERCEPTION,
                    backend="stub_specialist_perception",
                    model_identifier="stub-paddleocr",
                    resource_class=ModelResourceClass.HEAVY,
                    enabled=False,
                    preferred_device="cpu",
                    load_policy=ModelLoadPolicy.ON_DEMAND,
                    supported_capabilities=("specialist_perception",),
                    metadata={
                        "stub": True,
                        "recommended_default": "PaddleOCR",
                        "upgrade_only": True,
                    },
                )
            )
        for role, resource_class, capabilities in (
            (ModelRole.RERANKER, ModelResourceClass.SIDECAR, ("rerank",)),
            (ModelRole.SPEECH_TO_TEXT, ModelResourceClass.SIDECAR, ("transcribe",)),
            (ModelRole.TEXT_TO_SPEECH, ModelResourceClass.SIDECAR, ("synthesize",)),
            (ModelRole.VAD, ModelResourceClass.SIDECAR, ("voice_activity_detection",)),
            (ModelRole.TRANSLATION, ModelResourceClass.SIDECAR, ("translate",)),
            (ModelRole.CODE_SPECIALIST, ModelResourceClass.SIDECAR, ("code_assist",)),
            (ModelRole.VISION, ModelResourceClass.HEAVY, ("vision_inference",)),
            (ModelRole.SPECIALIST_PERCEPTION, ModelResourceClass.HEAVY, ("specialist_perception",)),
        ):
            self.register_model(
                ModelRegistration(
                    registration_id=f"{role.value}:unconfigured:placeholder",
                    role=role,
                    backend="unconfigured",
                    model_identifier="placeholder",
                    resource_class=resource_class,
                    enabled=False,
                    preferred_device="cpu" if resource_class == ModelResourceClass.SIDECAR else "auto",
                    load_policy=ModelLoadPolicy.ON_DEMAND,
                    supported_capabilities=capabilities,
                    missing_dependencies=("not_installed",),
                    metadata={
                        "placeholder": True,
                        "recommended_default": self._RECOMMENDED_SPECIALIST_DEFAULTS.get(role, ""),
                    },
                )
            )

    def _missing_dependencies_for_backend(self, backend_name: str) -> tuple[str, ...]:
        backend_name = str(backend_name)
        if backend_name in {"stub_generation", "stub_embedding"}:
            return ()
        if backend_name == "sentence_transformers" and importlib.util.find_spec("sentence_transformers") is None:
            return ("sentence_transformers",)
        if backend_name == "llama_cpp" and importlib.util.find_spec("llama_cpp") is None:
            return ("llama_cpp",)
        if backend_name == "argos_translate" and not argos_translate_available():
            return ("argostranslate",)
        return ()

    def _windows_system_speech_backend_ready(self) -> bool:
        return os.name == "nt" and system_speech_available() and shutil.which("powershell") is not None

    def _current_registration_for_role(self, role: ModelRole) -> ModelRegistration | None:
        if role == ModelRole.GENERATION and self._active_generation_backend is not None:
            backend_name = self._active_generation_backend.backend_name
            model_name = getattr(self._active_generation_backend, "model_name", "")
            for registration in self.list_registered_models(role=role):
                if registration.backend == backend_name and registration.model_identifier == model_name:
                    return registration
            return self._register_runtime_backend(
                role=role,
                backend_name=backend_name,
                model_name=model_name,
                supported_capabilities=("generate",),
            )
        if role == ModelRole.EMBEDDING and self._active_embedding_backend is not None:
            backend_name = self._active_embedding_backend.backend_name
            model_name = getattr(self._active_embedding_backend, "model_name", "")
            for registration in self.list_registered_models(role=role):
                if registration.backend == backend_name and registration.model_identifier == model_name:
                    return registration
            return self._register_runtime_backend(
                role=role,
                backend_name=backend_name,
                model_name=model_name,
                supported_capabilities=("embed", "embed_query", "embed_document"),
            )
        if self.config.preflight.flags.stub_mode:
            preferred = self._preferred_registration_for_role(role)
            if preferred is not None and preferred.enabled and not self._is_placeholder_registration(preferred):
                return preferred
            for registration in self.list_registered_models(role=role):
                if registration.backend.startswith("stub_") and registration.enabled:
                    return registration
        return None

    def _preferred_registration_for_role(self, role: ModelRole) -> ModelRegistration | None:
        preferred_id = self._preferred_models.get(role.value, "")
        preferred = self._registrations.get(preferred_id)
        if preferred is not None and preferred.role == role and preferred.enabled:
            return preferred
        for registration in self.list_registered_models(role=role):
            if registration.enabled:
                return registration
        if preferred is not None and preferred.role == role:
            return preferred
        return None

    def _record_route_decision(self, decision: ModelRouteDecision) -> None:
        self._route_history.append(decision)
        del self._route_history[:-16]
        if decision.selected_registration_id:
            self._preferred_models.setdefault(decision.requested_role.value, decision.selected_registration_id)

    def _register_runtime_backend(
        self,
        *,
        role: ModelRole,
        backend_name: str,
        model_name: str,
        supported_capabilities: tuple[str, ...],
    ) -> ModelRegistration:
        registration_id = f"{role.value}_runtime:{backend_name}:{model_name}"
        registration = self._registrations.get(registration_id)
        if registration is not None:
            return registration
        registration = ModelRegistration(
            registration_id=registration_id,
            role=role,
            backend=backend_name,
            model_identifier=model_name,
            resource_class=ModelResourceClass.HEAVY,
            preferred_device="auto",
            load_policy=ModelLoadPolicy.ALWAYS_ON,
            supported_capabilities=supported_capabilities,
        )
        self.register_model(registration)
        return registration

    def _resolve_registration_reference(
        self,
        role: ModelRole,
        reference: str,
    ) -> ModelRegistration | None:
        normalized = str(reference).strip()
        if not normalized:
            return None
        direct = self._registrations.get(normalized)
        if direct is not None and direct.role == role:
            return direct
        role_registrations = self.list_registered_models(role=role)
        if ":" in normalized:
            backend_name, model_identifier = normalized.split(":", 1)
            for registration in role_registrations:
                if (
                    registration.backend == backend_name
                    and registration.model_identifier == model_identifier
                ):
                    return registration
        for registration in role_registrations:
            if registration.model_identifier == normalized:
                return registration
        return None

    def _set_optional_role_enabled(self, role: ModelRole, enabled: bool) -> None:
        registrations = self.list_registered_models(role=role)
        if not registrations:
            return
        concrete_registrations = tuple(
            registration
            for registration in registrations
            if not self._is_placeholder_registration(registration)
        )
        if concrete_registrations:
            for registration in concrete_registrations:
                self._registrations[registration.registration_id] = replace(
                    registration,
                    enabled=enabled,
                )
            for registration in registrations:
                if self._is_placeholder_registration(registration):
                    self._registrations[registration.registration_id] = replace(
                        registration,
                        enabled=False,
                    )
            preferred = self._resolve_registration_reference(
                role,
                self._preferred_models.get(role.value, ""),
            )
            if preferred is None or self._is_placeholder_registration(preferred):
                preferred = self._choose_default_registration(role)
            if preferred is not None:
                self._preferred_models[role.value] = preferred.registration_id
        else:
            for registration in registrations:
                self._registrations[registration.registration_id] = replace(
                    registration,
                    enabled=enabled,
                )
        if not enabled:
            self._loaded_optional_heavy_roles.pop(role.value, None)

    def _choose_default_registration(self, role: ModelRole) -> ModelRegistration | None:
        registrations = self.list_registered_models(role=role)
        concrete_registrations = tuple(
            registration
            for registration in registrations
            if not self._is_placeholder_registration(registration)
        )
        for registration in concrete_registrations:
            if not registration.missing_dependencies:
                return registration
        if concrete_registrations:
            return concrete_registrations[0]
        return registrations[0] if registrations else None

    def _recommended_default_for_role(self, role: ModelRole) -> str:
        registrations = self.list_registered_models(role=role)
        for registration in registrations:
            recommended = str(registration.metadata.get("recommended_default", "")).strip()
            if recommended:
                return recommended
        return self._RECOMMENDED_SPECIALIST_DEFAULTS.get(role, "")

    @staticmethod
    def _is_placeholder_registration(registration: ModelRegistration) -> bool:
        return (
            registration.backend == "unconfigured"
            or registration.model_identifier == "placeholder"
            or bool(registration.metadata.get("placeholder"))
        )

    def _ensure_mandatory_heavy_role_active(self, role: ModelRole) -> tuple[str, ...]:
        removed_roles: list[str] = []
        if role.value not in self._suspended_default_heavy_roles:
            return ()
        if len(self._active_heavy_roles) >= self._heavy_slot_limit:
            removed_role = self._select_optional_heavy_role_to_unload(exclude=role.value)
            if removed_role is not None:
                self._loaded_optional_heavy_roles.pop(removed_role, None)
                removed_roles.append(removed_role)
        self._suspended_default_heavy_roles.discard(role.value)
        self._sync_active_heavy_roles()
        return tuple(removed_roles)

    def _swap_in_optional_heavy_role(self, role: ModelRole) -> tuple[str, ...] | None:
        removed_roles: list[str] = []
        if role in self._MANDATORY_ROLES:
            return ()
        if role.value in self._active_heavy_roles:
            self._touch_optional_heavy_role(role)
            return ()
        if len(self._active_heavy_roles) < self._heavy_slot_limit:
            self._touch_optional_heavy_role(role)
            return ()
        removed_role = self._select_optional_heavy_role_to_unload(exclude=role.value)
        if removed_role is not None:
            self._loaded_optional_heavy_roles.pop(removed_role, None)
            removed_roles.append(removed_role)
        elif ModelRole.EMBEDDING.value in self._active_heavy_roles and role != ModelRole.EMBEDDING:
            self._suspended_default_heavy_roles.add(ModelRole.EMBEDDING.value)
            removed_roles.append(ModelRole.EMBEDDING.value)
        elif ModelRole.GENERATION.value in self._active_heavy_roles and role != ModelRole.GENERATION:
            self._suspended_default_heavy_roles.add(ModelRole.GENERATION.value)
            removed_roles.append(ModelRole.GENERATION.value)
        else:
            return None
        self._touch_optional_heavy_role(role)
        return tuple(removed_roles)

    def _sync_active_heavy_roles(self) -> None:
        active_roles: list[str] = []
        optional_roles = [
            role
            for role, _timestamp in sorted(
                self._loaded_optional_heavy_roles.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        if not optional_roles:
            self._suspended_default_heavy_roles.clear()
        if (
            (self._generation_health.started or self._generation_health.available)
            and ModelRole.GENERATION.value not in self._suspended_default_heavy_roles
        ):
            active_roles.append(ModelRole.GENERATION.value)
        if (
            (self._embedding_health.started or self._embedding_health.available)
            and ModelRole.EMBEDDING.value not in self._suspended_default_heavy_roles
        ):
            active_roles.append(ModelRole.EMBEDDING.value)
        remaining_slots = max(0, self._heavy_slot_limit - len(active_roles))
        kept_optional_roles = [
            role
            for role in optional_roles
            if role not in active_roles
        ][:remaining_slots]
        self._loaded_optional_heavy_roles = {
            role: self._loaded_optional_heavy_roles[role]
            for role in kept_optional_roles
        }
        active_roles.extend(kept_optional_roles)
        self._active_heavy_roles = active_roles[: self._heavy_slot_limit]

    def _touch_optional_heavy_role(self, role: ModelRole) -> None:
        self._loaded_optional_heavy_roles[role.value] = time.monotonic()
        self._mark_used()
        self._sync_active_heavy_roles()

    def _parse_governor_advisory_input(
        self,
        suggestion: OptimizerSuggestionRecord,
        *,
        now: float,
    ) -> _GovernorAdvisoryInput | None:
        if suggestion.kind not in self._GOVERNOR_ADVISORY_KINDS:
            return None
        retention_seconds = self._clamped_governor_advisory_retention_s(suggestion.metadata)
        if suggestion.kind == OptimizerSuggestionKind.MODEL_LOADING:
            roles = self._bounded_optional_heavy_roles_from_metadata(suggestion.metadata)
            if not roles:
                return None
            return _GovernorAdvisoryInput(
                suggestion_id=suggestion.suggestion_id,
                kind=suggestion.kind,
                optional_heavy_roles=roles,
                expires_at_monotonic=now + retention_seconds,
            )
        namespaces, warm_key_count = self._bounded_cache_prefetch_from_metadata(suggestion.metadata)
        if not namespaces:
            return None
        return _GovernorAdvisoryInput(
            suggestion_id=suggestion.suggestion_id,
            kind=suggestion.kind,
            cache_namespaces=namespaces,
            warm_key_count=warm_key_count,
            expires_at_monotonic=now + retention_seconds,
        )

    def _clamped_governor_advisory_retention_s(self, metadata: dict[str, object]) -> float:
        raw_retention = metadata.get("retention_seconds", 30.0)
        try:
            retention_seconds = float(raw_retention)
        except (TypeError, ValueError):
            retention_seconds = 30.0
        return max(5.0, min(retention_seconds, self._MAX_GOVERNOR_ADVISORY_RETENTION_S))

    def _bounded_optional_heavy_roles_from_metadata(self, metadata: dict[str, object]) -> tuple[str, ...]:
        raw_roles = metadata.get("roles", ())
        if not raw_roles and "role" in metadata:
            raw_roles = (metadata.get("role"),)
        if isinstance(raw_roles, str):
            raw_roles = (raw_roles,)
        accepted: list[str] = []
        role_limit = min(self._MAX_GOVERNOR_ADVISORY_ROLES, self._heavy_slot_limit)
        for item in raw_roles:
            role_value = str(item).strip()
            if not role_value:
                continue
            try:
                resolved_role = ModelRole(role_value)
            except ValueError:
                continue
            if resolved_role in self._MANDATORY_ROLES:
                continue
            registrations = self.list_registered_models(role=resolved_role)
            if not any(
                registration.resource_class == ModelResourceClass.HEAVY and registration.enabled
                for registration in registrations
            ):
                continue
            if resolved_role.value in accepted:
                continue
            accepted.append(resolved_role.value)
            if len(accepted) >= role_limit:
                break
        return tuple(accepted)

    def _bounded_cache_prefetch_from_metadata(self, metadata: dict[str, object]) -> tuple[tuple[str, ...], int]:
        raw_namespaces = metadata.get("cache_namespaces", ())
        if not raw_namespaces and "cache_namespace" in metadata:
            raw_namespaces = (metadata.get("cache_namespace"),)
        if isinstance(raw_namespaces, str):
            raw_namespaces = (raw_namespaces,)
        namespaces: list[str] = []
        for item in raw_namespaces:
            namespace = str(item).strip()
            if not namespace or namespace not in self._GOVERNOR_ADVISORY_CACHE_NAMESPACES:
                continue
            if namespace in namespaces:
                continue
            namespaces.append(namespace)
            if len(namespaces) >= 2:
                break
        raw_warm_keys = metadata.get("warm_keys", ())
        if isinstance(raw_warm_keys, str):
            raw_warm_keys = (raw_warm_keys,)
        warm_key_count = 0
        for item in raw_warm_keys:
            if not str(item).strip():
                continue
            warm_key_count += 1
            if warm_key_count >= self._MAX_GOVERNOR_ADVISORY_WARM_KEYS:
                break
        return (tuple(namespaces), warm_key_count)

    def _prune_governor_advisory_inputs(self, *, now: float | None = None) -> None:
        current_time = time.monotonic() if now is None else now
        self._governor_advisory_inputs = {
            suggestion_id: payload
            for suggestion_id, payload in self._governor_advisory_inputs.items()
            if payload.expires_at_monotonic > current_time
        }

    def _active_governor_advisory_inputs(self, *, now: float | None = None) -> tuple[_GovernorAdvisoryInput, ...]:
        current_time = time.monotonic() if now is None else now
        self._prune_governor_advisory_inputs(now=current_time)
        return tuple(
            sorted(
                self._governor_advisory_inputs.values(),
                key=lambda item: (item.expires_at_monotonic, item.kind.value, item.suggestion_id),
                reverse=True,
            )
        )

    def _advised_optional_heavy_roles(self, *, now: float | None = None) -> set[str]:
        roles: set[str] = set()
        for payload in self._active_governor_advisory_inputs(now=now):
            if payload.kind == OptimizerSuggestionKind.MODEL_LOADING:
                roles.update(payload.optional_heavy_roles)
        return roles

    def _select_optional_heavy_role_to_unload(
        self,
        *,
        exclude: str | None = None,
        now: float | None = None,
    ) -> str | None:
        current_time = time.monotonic() if now is None else now
        advised_roles = self._advised_optional_heavy_roles(now=current_time)
        candidates = [
            item
            for item in self._loaded_optional_heavy_roles.items()
            if exclude is None or item[0] != exclude
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0] in advised_roles, item[1]))
        return candidates[0][0]

    def _governor_advisory_summary(self, *, now: float | None = None) -> str:
        current_time = time.monotonic() if now is None else now
        advisory_inputs = self._active_governor_advisory_inputs(now=current_time)
        if not advisory_inputs:
            return ""
        fragments = [payload.summary_fragment(now=current_time) for payload in advisory_inputs]
        return "advisory=" + "; ".join(fragments)

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
        """
        Hardware governor/model scheduler:
        - Monitors RAM/VRAM/queue pressure
        - Enforces heavy slot cap
        - Degrades/disables features in order when under pressure
        - Surfaces degrade/disable reasons to dashboard/orchestrator
        """
        while not self._stop_event.is_set():
            try:
                await self._refresh_health()
                await self._maybe_unload_idle_backends()
                governor = self._hardware_governor_state()
                log_messages: list[str] = []
                if len(self._active_heavy_roles) > self._heavy_slot_limit:
                    log_messages.append(
                        f"heavy slot cap exceeded: {len(self._active_heavy_roles)}/{self._heavy_slot_limit}"
                    )
                    while len(self._active_heavy_roles) > self._heavy_slot_limit:
                        unload_role = self._select_optional_heavy_role_to_unload()
                        if unload_role is None:
                            break
                        self._loaded_optional_heavy_roles.pop(unload_role, None)
                        log_messages.append(f"unloaded optional heavy role: {unload_role}")
                        self._sync_active_heavy_roles()
                if not governor.allow_optional_heavy_residency and self._loaded_optional_heavy_roles:
                    unload_role = self._select_optional_heavy_role_to_unload()
                    if unload_role is None:
                        await asyncio.sleep(self.config.backend_runtime.idle_check_interval_s)
                        continue
                    self._loaded_optional_heavy_roles.pop(unload_role, None)
                    self._sync_active_heavy_roles()
                    log_messages.append(f"hardware governor unloaded optional heavy role: {unload_role}")
                if governor.active or log_messages:
                    details = list(log_messages)
                    if governor.summary:
                        details.insert(0, governor.summary)
                    self.logger.warning("Hardware governor: %s", "; ".join(details))
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                self._last_error = str(exc)
                self.logger.warning("ModelManager maintenance loop error: %s", exc)
            await asyncio.sleep(self.config.backend_runtime.idle_check_interval_s)

    async def _maybe_unload_idle_backends(self) -> None:
        if self._last_used_monotonic is None:
            return
        if self._active_generation_jobs or self._active_embedding_jobs:
            return
        current_time = time.monotonic()
        idle_seconds = current_time - self._last_used_monotonic
        if idle_seconds < self.config.backend_runtime.idle_unload_after_s:
            return
        advised_roles = self._advised_optional_heavy_roles(now=current_time)
        retained_optional_roles = {
            role: timestamp
            for role, timestamp in self._loaded_optional_heavy_roles.items()
            if role in advised_roles
        }
        optional_unloaded = len(retained_optional_roles) != len(self._loaded_optional_heavy_roles)
        self._loaded_optional_heavy_roles = retained_optional_roles
        self._sync_active_heavy_roles()
        unloaded = False
        for backend in (self._active_generation_backend, self._active_embedding_backend):
            if backend is None:
                continue
            unloaded = await backend.unload() or unloaded
        if unloaded or optional_unloaded:
            self.logger.info("Unloaded idle model backend state after %.2f seconds.", idle_seconds)
            await self._refresh_health()

    async def _refresh_health(self) -> None:
        if self._active_generation_backend is not None:
            self._generation_health = await self._active_generation_backend.health()
        if self._active_embedding_backend is not None:
            self._embedding_health = await self._active_embedding_backend.health()
        self._sync_active_heavy_roles()

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
        self._initialize_registry()

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

    def _low_ram_threshold_gb(self, total_ram_gb: float | None) -> float:
        """Return the effective low-RAM threshold used by governor and fallback checks.

        The configured headroom remains the upper bound, but the live threshold is
        reduced on larger-memory hosts so background system pressure does not
        disable optional features too aggressively during otherwise healthy runs.
        """
        configured = max(0.1, float(self.config.backend_runtime.low_ram_headroom_gb))
        if total_ram_gb is None:
            return configured
        return min(configured, max(0.25, float(total_ram_gb) * 0.05))

    def _detect_memory_pressure(self) -> str | None:
        total_ram_gb, available_ram_gb = self._read_ram_telemetry()
        reasons: list[str] = []
        low_ram_threshold_gb = self._low_ram_threshold_gb(total_ram_gb)
        if available_ram_gb is not None and available_ram_gb <= low_ram_threshold_gb:
            reasons.append(
                f"available RAM {available_ram_gb:.2f}GB is below headroom "
                f"{low_ram_threshold_gb:.2f}GB"
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

    def _hardware_governor_state(
        self,
        *,
        total_ram_gb: float | None = None,
        available_ram_gb: float | None = None,
    ) -> HardwareGovernorState:
        current_time = time.monotonic()
        if total_ram_gb is None or available_ram_gb is None:
            total_ram_gb, available_ram_gb = self._read_ram_telemetry()
        pressure_reasons: list[str] = []
        queue_pressure = False
        backend_health_degraded = False
        if self._fallback_active:
            pressure_reasons.append("model_fallback_active")
            backend_health_degraded = True
        low_ram_threshold_gb = self._low_ram_threshold_gb(total_ram_gb)
        if available_ram_gb is not None and available_ram_gb <= low_ram_threshold_gb:
            pressure_reasons.append("low_available_ram")
        max_vram_gb = self.config.preflight.hardware.max_vram_gb
        low_vram_headroom = self.config.backend_runtime.low_vram_headroom_gb
        if (
            self._generation_health.estimated_vram_gb is not None
            and self._generation_health.estimated_vram_gb >= max_vram_gb - low_vram_headroom
        ):
            pressure_reasons.append("generation_vram_pressure")
        if (
            self._embedding_health.estimated_vram_gb is not None
            and self._embedding_health.estimated_vram_gb >= max_vram_gb - low_vram_headroom
        ):
            pressure_reasons.append("embedding_vram_pressure")
        if self._active_generation_jobs >= self.config.concurrency.generation_slots:
            pressure_reasons.append("generation_queue_pressure")
            queue_pressure = True
        if self._active_embedding_jobs >= self.config.concurrency.embedding_slots:
            pressure_reasons.append("embedding_queue_pressure")
            queue_pressure = True
        if self._started and (
            bool(self._generation_health.last_error)
            or (
                self._active_generation_backend is not None
                and self._generation_health.started
                and not self._generation_health.available
            )
        ):
            pressure_reasons.append("generation_backend_unhealthy")
            backend_health_degraded = True
        if self._started and (
            bool(self._embedding_health.last_error)
            or (
                self._active_embedding_backend is not None
                and self._embedding_health.started
                and not self._embedding_health.available
            )
        ):
            pressure_reasons.append("embedding_backend_unhealthy")
            backend_health_degraded = True
        if len(self._active_heavy_roles) > self._heavy_slot_limit:
            pressure_reasons.append("heavy_slot_cap_exceeded")
        unique_reasons = tuple(dict.fromkeys(pressure_reasons))
        severe_pressure = any(
            reason in {
                "model_fallback_active",
                "low_available_ram",
                "generation_vram_pressure",
                "embedding_vram_pressure",
                "generation_backend_unhealthy",
                "embedding_backend_unhealthy",
                "heavy_slot_cap_exceeded",
            }
            for reason in unique_reasons
        )
        allow_continuous_capture = not bool(unique_reasons)
        allow_ocr_on_step = not severe_pressure
        allow_vision_on_step = not bool(unique_reasons)
        allow_optional_heavy_residency = not bool(unique_reasons)
        allow_background_work = not bool(unique_reasons)
        degraded_features: list[str] = []
        if not allow_continuous_capture:
            degraded_features.append("continuous_capture")
        if not allow_ocr_on_step:
            degraded_features.append("ocr_on_step")
        if not allow_vision_on_step:
            degraded_features.append("vision_on_step")
        if not allow_optional_heavy_residency:
            degraded_features.append("optional_heavy_residency")
        if not allow_background_work:
            degraded_features.append("background_work")
        summary = ""
        advisory_summary = self._governor_advisory_summary(now=current_time)
        if unique_reasons:
            summary = (
                "pressure="
                + ", ".join(unique_reasons)
                + " | degraded="
                + (", ".join(degraded_features) if degraded_features else "(none)")
            )
            if advisory_summary:
                summary += " | " + advisory_summary
        elif advisory_summary:
            summary = advisory_summary
        _ = total_ram_gb
        return HardwareGovernorState(
            active=bool(unique_reasons),
            pressure_reasons=unique_reasons,
            degraded_features=tuple(degraded_features),
            queue_pressure=queue_pressure,
            backend_health_degraded=backend_health_degraded,
            allow_continuous_capture=allow_continuous_capture,
            allow_ocr_on_step=allow_ocr_on_step,
            allow_vision_on_step=allow_vision_on_step,
            allow_optional_heavy_residency=allow_optional_heavy_residency,
            allow_background_work=allow_background_work,
            summary=summary,
        )

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("ModelManager must be started before use.")
