"""Async orchestrator that wires agents and services into one pipeline."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import threading
import time
import traceback
from concurrent.futures import CancelledError as ConcurrentCancelledError
from concurrent.futures import Future as ConcurrentFuture
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

from acceptance_thresholds import PHASE12_ACCEPTANCE_THRESHOLDS
from capability_runtime import CapabilityExecutor, CapabilityPolicyEngine
from cloud_offload import CloudOffloadManager
from coding_mode import CodingModeService
from compressor import CompressorAgent
from config import APP_CONFIG, AppConfig, BudgetPolicy
from critic import CriticAgent
from data_structures import (
    AgentState,
    AgentStatus,
    AudioSynthesisResult,
    AudioTranscriptionResult,
    CapabilityAuditEventType,
    CapabilityAuditRecord,
    CapabilityAvailabilityStatus,
    CapabilityExecutionResult,
    CapabilityExecutionStatus,
    CapabilityPolicyDecision,
    CapabilityPolicyOutcome,
    CapabilityRegistration,
    CapabilityRegistryView,
    CapabilityRequest,
    CapabilityType,
    CandidateTrace,
    CloudFallbackBehavior,
    CloudJobContract,
    CloudJobPayloadClass,
    CloudJobPrivacyClass,
    CloudOffloadCapability,
    CloudOffloadMode,
    CloudOffloadOutcome,
    CloudOffloadRecord,
    CodeSpecialistResult,
    CodingPattern,
    CodingTaskRequest,
    CodingTaskResult,
    CompressedTrace,
    CritiqueReport,
    CritiqueResult,
    DashboardCapabilityAvailability,
    DashboardLocalTaskSessionState,
    DashboardKnowledgeSource,
    ModelRegistryView,
    ModelRoleActionReport,
    DashboardReadinessCheck,
    DashboardReadinessReport,
    DashboardTaskHistoryEntry,
    DashboardTaskInspector,
    DecodeHint,
    LocalTaskPendingApproval,
    LocalTaskSession,
    LocalTaskSessionState,
    LongHorizonCandidateSnapshot,
    LongHorizonCheckpoint,
    LongHorizonExportBundle,
    LongHorizonSession,
    LongHorizonSessionState,
    OperationStep,
    OptimizerSuggestionKind,
    OptimizerSuggestionDisposition,
    OptimizerSuggestionRecord,
    OptimizerSuggestionUsageRecord,
    PackagedLaunchReport,
    PackagedSupportBundle,
    PerformanceMetric,
    PracticeSessionResult,
    ModelRole,
    ReasonerCriticHandoff,
    ResearchReasonerHandoff,
    ReasoningLog,
    ResourceBudget,
    RuntimeCondition,
    RuntimeEvent,
    SeverityLevel,
    TaskResult,
    TextTranslationResult,
    UserSettingsProfile,
    utc_now,
)
from dashboard import DashboardService
from model_manager import ModelHealthSnapshot, ModelManager
from planner import PlannerAgent
from phase11_content import Phase11ContentLoader
from reasoner import ReasonerAgent
from researcher import ResearcherAgent
from retrieval import stable_hash
from runtime_errors import (
    BackendUnavailableError,
    ModelTimeoutError,
    ResourcePressureError,
    WebLookupTimeoutError,
)
from self_optimizer import SelfOptimizer
from storage import StorageManager
from translation_service import TranslationService
from utils import configure_logging
from verification_tools import (
    evaluate_arithmetic_question,
    evaluate_python_code_question,
    evaluate_python_expression_question,
    evaluate_python_unit_test_question,
    expected_evidence_count,
)

_LOCAL_MODEL_SETUP_GUIDE = Path(__file__).resolve().with_name("LOCAL_MODEL_SETUP.md")


@dataclass(slots=True, frozen=True)
class _ReadinessState:
    """Internal snapshot reused by dashboard and packaged launch checks."""

    profile: UserSettingsProfile
    requested_stub_mode: bool
    snapshot: ModelHealthSnapshot
    sentence_transformers_available: bool
    chromadb_available: bool
    llama_cpp_available: bool
    primary_generation_backend: str
    primary_embedding_backend: str
    primary_uses_ollama: bool
    any_uses_ollama: bool
    ollama_service_ready: bool
    ollama_status: str
    ollama_detail: str
    llama_cpp_model_ready: bool
    llama_cpp_model_detail: str
    llama_cpp_model_required: bool
    llama_cpp_model_blocking: bool
    llama_cpp_model_status: str
    primary_generation_ready: bool
    primary_embedding_ready: bool


@dataclass(slots=True, frozen=True)
class _LongHorizonThrottleDecision:
    """One bounded cycle-budget adjustment decision for a long-horizon session."""

    budget: ResourceBudget
    throttled: bool
    reason: str
    metadata: dict[str, Any]


@dataclass(slots=True, frozen=True)
class _LongHorizonAdvisoryPlan:
    """Bounded advisory bundle prepared between long-horizon cycles."""

    suggestions: tuple[OptimizerSuggestionRecord, ...]
    usage_records: tuple[OptimizerSuggestionUsageRecord, ...]
    next_budget: ResourceBudget


@dataclass(slots=True, frozen=True)
class _PackagedStartupPlan:
    """Internal packaged-startup plan that preserves requested vs effective runtime mode."""

    requested_profile: UserSettingsProfile
    effective_profile: UserSettingsProfile
    launch_report: PackagedLaunchReport
    runtime_config: AppConfig
    first_run: bool
    persist_effective_profile: bool
    startup_notice: str
    startup_notice_severity: str


_CLOUD_OFFLOAD_CAPABILITY_DETAILS: dict[
    CloudOffloadCapability,
    tuple[str, CloudJobPayloadClass, CloudJobPrivacyClass, bool],
] = {
    CloudOffloadCapability.OFFLINE_REPLAY: (
        "Offline Replay",
        CloudJobPayloadClass.EXPORT_BUNDLE,
        CloudJobPrivacyClass.APPROVED_CONTENT,
        True,
    ),
    CloudOffloadCapability.EXPORT: (
        "Export Jobs",
        CloudJobPayloadClass.EXPORT_BUNDLE,
        CloudJobPrivacyClass.APPROVED_CONTENT,
        True,
    ),
    CloudOffloadCapability.BROWSER_HELPER: (
        "Browser Helper",
        CloudJobPayloadClass.TEXT_SNIPPET,
        CloudJobPrivacyClass.APPROVED_CONTENT,
        True,
    ),
    CloudOffloadCapability.OCR_HELPER: (
        "OCR Helper",
        CloudJobPayloadClass.IMAGE_REGION,
        CloudJobPrivacyClass.APPROVED_CONTENT,
        True,
    ),
    CloudOffloadCapability.VISION_HELPER: (
        "Vision Helper",
        CloudJobPayloadClass.IMAGE_REGION,
        CloudJobPrivacyClass.APPROVED_CONTENT,
        True,
    ),
    CloudOffloadCapability.EMBEDDING_HELPER: (
        "Embedding Helper",
        CloudJobPayloadClass.EMBEDDING_BATCH,
        CloudJobPrivacyClass.APPROVED_CONTENT,
        True,
    ),
    CloudOffloadCapability.BACKGROUND_MAINTENANCE: (
        "Background Maintenance",
        CloudJobPayloadClass.METADATA_ONLY,
        CloudJobPrivacyClass.METADATA_ONLY,
        False,
    ),
}


@dataclass(slots=True, frozen=True)
class _ContinuousCapturePlan:
    """Bounded continuous-capture plan derived from profile settings and hard caps."""

    session_id: str
    capture_directory: Path
    fps: float
    interval_s: float
    max_width: int
    max_height: int
    frame_history: int
    diff_threshold: float
    region_of_interest: str
    warnings: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class _ObservationStepPlan:
    """Bounded per-step observation plan derived from explicit profile settings."""

    session_id: str
    tier: str
    capture_directory: Path
    max_width: int
    max_height: int
    region_of_interest: str
    warnings: tuple[str, ...]


class Orchestrator:
    """Single owner of startup, pipeline execution, and clean shutdown."""

    COMPRESSION_HISTORY_SCAN_LIMIT = PHASE12_ACCEPTANCE_THRESHOLDS.compression.max_recent_reasoning_logs
    LOCAL_TASK_MAX_REPEATED_REQUESTS = 3
    OBSERVATION_TEXT_PREVIEW_CHARS = 280

    def __init__(
        self,
        config: AppConfig = APP_CONFIG,
        *,
        storage: StorageManager | None = None,
        startup_profile_override: UserSettingsProfile | None = None,
        persist_startup_profile_override: bool = False,
        model_manager: ModelManager | None = None,
        dashboard: DashboardService | None = None,
        planner: PlannerAgent | None = None,
        researcher: ResearcherAgent | None = None,
        reasoner: ReasonerAgent | None = None,
        critic: CriticAgent | None = None,
        compressor: CompressorAgent | None = None,
        self_optimizer: SelfOptimizer | None = None,
        translation: TranslationService | None = None,
        phase11_content: Phase11ContentLoader | None = None,
    ):
        self.config = config
        self.config.validate()
        self.logger = configure_logging(config.logging)

        self.storage = storage or StorageManager(config=config, logger=self.logger.getChild("storage"))
        self.model_manager = model_manager or ModelManager(config=config, logger=self.logger.getChild("model"))
        self.dashboard = dashboard or DashboardService(config=config)

        self.planner = planner or PlannerAgent(model_manager=self.model_manager, config=config)
        self.researcher = researcher or ResearcherAgent(
            model_manager=self.model_manager,
            storage=self.storage,
            config=config,
        )
        self.reasoner = reasoner or ReasonerAgent(
            model_manager=self.model_manager,
            storage=self.storage,
            config=config,
        )
        self.critic = critic or CriticAgent(
            model_manager=self.model_manager,
            storage=self.storage,
            config=config,
        )
        self.compressor = compressor or CompressorAgent(model_manager=self.model_manager, config=config)
        self.self_optimizer = self_optimizer or SelfOptimizer(
            storage=self.storage,
            config=config,
            cache_lookup=self.model_manager.lookup_cache,
            cache_warm=self.model_manager.warm_cache,
        )
        self.coding_mode = CodingModeService(
            model_manager=self.model_manager,
            storage=self.storage,
            config=config,
        )
        self.translation = translation or TranslationService()
        self.phase11_content = phase11_content or Phase11ContentLoader(config=config)
        self.capability_policy = CapabilityPolicyEngine(config=config, workspace_root=Path.cwd())
        self.capability_executor = CapabilityExecutor(config=config, workspace_root=Path.cwd())
        self.cloud_offload = CloudOffloadManager()
        self._startup_profile_override = startup_profile_override
        self._persist_startup_profile_override = bool(persist_startup_profile_override)
        self._started = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dashboard_futures: set[ConcurrentFuture[Any]] = set()
        self._active_long_horizon_tasks: dict[str, asyncio.Task[TaskResult]] = {}
        self._shutdown_requested = False
        self._local_task_emergency_stop = threading.Event()
        self._continuous_capture_task: asyncio.Task[None] | None = None
        self._continuous_capture_session_id = ""
        self.storage.add_runtime_event_listener(self._forward_runtime_event_to_dashboard)
        self.storage.add_agent_status_listener(self._forward_agent_status_to_dashboard)

    async def start(self) -> None:
        """Start all services and agents in dependency-safe order."""
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        started_stoppers: list[callable] = []
        persisted_startup_profile: UserSettingsProfile | None = None
        try:
            if self._startup_profile_override is None:
                if isinstance(self.storage, StorageManager):
                    persisted_startup_profile = await _load_startup_settings_profile(self.config)
                else:
                    persisted_startup_profile = await self.storage.load_user_settings_profile("default")
            settings_profile = self._startup_profile_override or persisted_startup_profile
            if settings_profile is None:
                settings_profile = self._default_user_settings_profile()
            self._apply_runtime_config(self._config_for_user_settings_profile(settings_profile))
            await self.storage.start()
            started_stoppers.append(self.storage.stop)
            should_persist_startup_profile = (
                self._startup_profile_override is None or self._persist_startup_profile_override
            )
            if should_persist_startup_profile and await self.storage.load_user_settings_profile("default") is None:
                await self.storage.save_user_settings_profile(settings_profile)
            await self._ensure_local_ollama_service()
            await self.model_manager.start()
            started_stoppers.append(self.model_manager.stop)
            await self.planner.start()
            started_stoppers.append(self.planner.stop)
            await self.researcher.start()
            started_stoppers.append(self.researcher.stop)
            await self.reasoner.start()
            started_stoppers.append(self.reasoner.stop)
            await self.critic.start()
            started_stoppers.append(self.critic.stop)
            await self.compressor.start()
            started_stoppers.append(self.compressor.stop)
            await self.dashboard.start()
            started_stoppers.append(self.dashboard.stop)
            await self.self_optimizer.start()
            started_stoppers.append(self.self_optimizer.stop)
            await self._apply_runtime_settings_profile(settings_profile)
            self.dashboard.attach_controller(
                submit_task=self._submit_dashboard_task_request,
                save_settings=self._save_dashboard_settings_request,
                perform_action=self._handle_dashboard_action_request,
            )
            self._started = True
            await self._emit_event(
                "orchestrator.started",
                {"stub_mode": self.config.preflight.flags.stub_mode},
            )
            startup_snapshot = self.model_manager.health_snapshot()
            await self._emit_health_snapshot(snapshot=startup_snapshot)
            if startup_snapshot.fallback_active:
                await self._emit_runtime_condition_event(
                    "runtime.fallback_activated",
                    category="fallback",
                    component="model_manager",
                    reason=startup_snapshot.fallback_reason or "startup_fallback_active",
                    severity=SeverityLevel.MEDIUM,
                    metadata={
                        "generation_backend": startup_snapshot.generation_backend,
                        "embedding_backend": startup_snapshot.embedding_backend,
                    },
                )
            await self._record_status(
                "orchestrator",
                AgentState.IDLE,
                message="orchestrator started",
            )
            await self._publish_dashboard_settings_profiles(active_profile_name=settings_profile.profile_name)
            await self._publish_dashboard_task_history()
            await self._publish_dashboard_knowledge_library()
            await self._publish_dashboard_coding_patterns()
            await self._publish_dashboard_recent_coding_activity()
            await self._publish_dashboard_readiness_report()
            await self._publish_dashboard_capability_registry_view(active_profile=settings_profile)
            await self._recover_local_task_session()
            await self._publish_dashboard_examples()
        except Exception:
            await self._cleanup_failed_start(started_stoppers)
            self._started = False
            self._loop = None
            self._shutdown_requested = False
            raise

    async def stop(self) -> None:
        """Stop all services and agents in reverse dependency order."""
        if not self._started:
            return
        self._shutdown_requested = True
        await self._record_status(
            "orchestrator",
            AgentState.STOPPING,
            message="orchestrator stopping",
        )
        await self._emit_event("orchestrator.stopping", {})
        self.dashboard.attach_controller()
        await self._request_pause_for_active_long_horizon_sessions(reason="shutdown_requested")
        await self._pause_active_local_task_session(reason="shutdown_requested")
        await self._stop_continuous_capture_task(reason="orchestrator_shutdown")
        self._cancel_active_long_horizon_tasks()
        await self._cancel_pending_dashboard_futures()
        await self.self_optimizer.stop()
        await self.dashboard.stop()
        await self.compressor.stop()
        await self.critic.stop()
        await self.reasoner.stop()
        await self.researcher.stop()
        await self.planner.stop()
        await self.model_manager.stop()
        await self.storage.stop()
        self._started = False
        self._loop = None
        self._shutdown_requested = False

    async def _cleanup_failed_start(self, started_stoppers: list[Callable[[], Awaitable[Any] | Any]]) -> None:
        """Best-effort cleanup for partial startup failures before the app is marked started."""
        self.dashboard.attach_controller()
        for stopper in reversed(started_stoppers):
            try:
                await stopper()
            except Exception:  # pragma: no cover - cleanup should not mask original failures
                self.logger.exception("Error while cleaning up a partial startup failure.")

    @staticmethod
    def _coerce_bool_setting(value: Any, fallback: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(fallback)

    def _config_for_user_settings_profile(self, profile: UserSettingsProfile | None) -> AppConfig:
        """Project one persisted settings profile onto the runtime config surface."""
        if profile is None:
            return self.config
        runtime = dict(profile.runtime)
        retrieval = dict(profile.retrieval)
        preferred_by_role = {
            str(key): str(value)
            for key, value in dict(profile.models.get("preferred_by_role", {})).items()
            if str(key).strip() and str(value).strip()
        }
        flags = self.config.preflight.flags
        backends = self.config.preflight.backends

        generation_backend = str(runtime.get("generation_backend", backends.generation_backend))
        if generation_backend not in {"ollama", "llama_cpp"}:
            generation_backend = backends.generation_backend
        generation_model = backends.generation_model

        embedding_backend = str(runtime.get("embedding_backend", backends.embedding_backend))
        if embedding_backend not in {"sentence_transformers", "ollama_embeddings"}:
            embedding_backend = backends.embedding_backend
        embedding_model = backends.embedding_model

        generation_preference = str(preferred_by_role.get("generation", "")).strip()
        if ":" in generation_preference:
            preferred_backend, preferred_model = generation_preference.split(":", 1)
            if preferred_backend in {"ollama", "llama_cpp"} and preferred_model.strip():
                generation_backend = preferred_backend
                generation_model = preferred_model.strip()

        embedding_preference = str(preferred_by_role.get("embedding", "")).strip()
        if ":" in embedding_preference:
            preferred_backend, preferred_model = embedding_preference.split(":", 1)
            if preferred_backend in {"sentence_transformers", "ollama_embeddings"} and preferred_model.strip():
                embedding_backend = preferred_backend
                embedding_model = preferred_model.strip()

        vector_store_backend = str(runtime.get("vector_store_backend", backends.vector_store_backend))
        if vector_store_backend not in {"chromadb", "simple_inmemory"}:
            vector_store_backend = backends.vector_store_backend

        provider = str(retrieval.get("provider", self.config.web.provider))
        if provider not in {"wikipedia", "stub"}:
            provider = self.config.web.provider

        requested_stub_mode = self._coerce_bool_setting(runtime.get("stub_mode", flags.stub_mode), flags.stub_mode)
        allow_web_fallback = self._coerce_bool_setting(
            retrieval.get("allow_web_fallback", runtime.get("allow_web_fallback", flags.allow_web_fallback)),
            flags.allow_web_fallback,
        )
        enable_self_optimizer = self._coerce_bool_setting(
            runtime.get("enable_self_optimizer", flags.enable_self_optimizer),
            flags.enable_self_optimizer,
        )
        enable_reranking = self._coerce_bool_setting(
            retrieval.get("reranking", self.config.retrieval.enable_reranking),
            self.config.retrieval.enable_reranking,
        )

        resolved_flags = replace(
            flags,
            stub_mode=requested_stub_mode,
            allow_web_fallback=allow_web_fallback,
            enable_self_optimizer=enable_self_optimizer,
        )
        resolved_backends = replace(
            backends,
            generation_backend=generation_backend,
            generation_model=generation_model,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            vector_store_backend=vector_store_backend,
        )
        return replace(
            self.config,
            preflight=replace(self.config.preflight, flags=resolved_flags, backends=resolved_backends),
            retrieval=replace(self.config.retrieval, enable_reranking=enable_reranking),
            web=replace(self.config.web, provider=provider),
        )

    def _apply_runtime_config(self, runtime_config: AppConfig) -> None:
        """Rebind the shared runtime config before dependent services start."""
        self.config = runtime_config
        for component in (
            self.storage,
            self.model_manager,
            self.dashboard,
            self.planner,
            self.researcher,
            self.reasoner,
            self.critic,
            self.compressor,
            self.self_optimizer,
            self.phase11_content,
        ):
            if hasattr(component, "config"):
                setattr(component, "config", runtime_config)
        self.capability_policy = CapabilityPolicyEngine(config=runtime_config, workspace_root=Path.cwd())
        self.capability_executor = CapabilityExecutor(config=runtime_config, workspace_root=Path.cwd())

    @staticmethod
    def _config_uses_ollama(config: AppConfig) -> bool:
        backends = config.preflight.backends
        return any(
            backend_name in {"ollama", "ollama_embeddings"}
            for backend_name in (
                backends.generation_backend,
                backends.embedding_backend,
                backends.generation_fallback_backend,
                backends.embedding_fallback_backend,
            )
        )

    def _discover_ollama_binary(self) -> Path | None:
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        candidates: list[Path] = []
        if local_appdata:
            candidates.append(Path(local_appdata) / "Programs" / "OllamaPortable" / "ollama.exe")
        candidates.extend(
            (
                Path.home() / "AppData" / "Local" / "Programs" / "OllamaPortable" / "ollama.exe",
                Path(r"C:\Program Files\Ollama\ollama.exe"),
            )
        )
        resolved_on_path = shutil.which("ollama")
        if resolved_on_path:
            candidates.append(Path(resolved_on_path))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    async def _ensure_local_ollama_service(
        self,
        *,
        config: AppConfig | None = None,
    ) -> tuple[bool, str]:
        """Best-effort user-space Ollama startup for real-mode launches."""
        runtime_config = config or self.config
        if runtime_config.preflight.flags.stub_mode or not self._config_uses_ollama(runtime_config):
            return (False, "ollama_not_required")
        ready, detail = self._probe_ollama_service(config=runtime_config)
        if ready:
            return (True, detail)
        binary = self._discover_ollama_binary()
        if binary is None:
            return (False, detail)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            subprocess.Popen(
                [str(binary), "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except Exception as exc:
            return (False, f"Tried to auto-start Ollama using {binary}, but launch failed: {exc}")
        deadline = time.monotonic() + min(
            10.0,
            max(2.0, float(runtime_config.preflight.flags.startup_timeout_s)),
        )
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            ready, detail = self._probe_ollama_service(config=runtime_config)
            if ready:
                self.logger.info("Auto-started Ollama service using %s.", binary)
                return (True, f"Auto-started Ollama service using {binary}.")
        return (False, f"Tried to auto-start Ollama using {binary}, but the service is still unavailable.")

    async def run_task(self, question: str, thinking_minutes: int) -> TaskResult:
        """Execute the foreground workflow from Planner to Compressor."""
        self._require_started()
        budget = BudgetPolicy.from_minutes(
            thinking_minutes,
            calibration=self.config.budget_calibration,
        )
        await self.storage.enter_foreground_task()
        try:
            if self._should_use_long_horizon(budget):
                return await self._run_long_horizon_task(
                    question=question,
                    thinking_minutes=thinking_minutes,
                    budget=budget,
                )
            return await self._run_bounded_task(
                question=question,
                thinking_minutes=thinking_minutes,
                budget=budget,
            )
        finally:
            await self.storage.exit_foreground_task()

    async def _run_bounded_task(
        self,
        *,
        question: str,
        thinking_minutes: int,
        budget,
        persist_task_result: bool = True,
        publish_history: bool = True,
        emit_completion_event: bool = True,
    ) -> TaskResult:
        """Execute one bounded pipeline cycle using the provided budget."""
        started_at = time.perf_counter()
        await self._emit_event(
            "pipeline.received",
            {
                "question": question,
                "thinking_minutes": thinking_minutes,
                "budget": budget.to_dict(),
            },
        )
        await self._record_status(
            "orchestrator",
            AgentState.RUNNING,
            message="pipeline running",
        )

        plan = None
        task_id: str | None = None
        try:
            plan = await self._run_component(
                "planner",
                task_id=None,
                start_stage="pipeline.planner_started",
                done_stage="pipeline.planner_done",
                start_payload={"question": question},
                run=lambda: self.planner.plan(question, budget),
            )
            task_id = plan.task_id
            await self._emit_event("pipeline.planner_done", {"task_id": task_id})
            await self._record_status(
                "planner",
                AgentState.IDLE,
                task_id=task_id,
                message="planning completed",
            )

            evidence = await self._run_component(
                "researcher",
                task_id=task_id,
                start_stage="pipeline.researcher_started",
                done_stage="pipeline.researcher_done",
                start_payload={"task_id": task_id},
                run=lambda: self.researcher.research(plan, budget),
            )
            await self._emit_event(
                "pipeline.researcher_done",
                self._research_event_payload(task_id=task_id, evidence=evidence),
            )
            await self._record_status(
                "researcher",
                AgentState.IDLE,
                task_id=task_id,
                message=self._research_status_message(evidence),
                severity=self._research_status_severity(evidence),
            )
            self.model_manager.warm_cache(
                "retrieval_candidates",
                task_id,
                tuple(
                    item.id
                    for item in (evidence.local_results + evidence.web_results)[:8]
                ),
            )
            if evidence.used_web_fallback:
                await self._emit_runtime_condition_event(
                    "runtime.fallback_used",
                    category="fallback",
                    component="researcher",
                    reason=(
                        "web_fallback_returned_no_results"
                        if not evidence.web_results
                        else "bounded_web_fallback_used"
                    ),
                    severity=(
                        SeverityLevel.MEDIUM if not evidence.web_results else SeverityLevel.LOW
                    ),
                    task_id=task_id,
                    metadata={
                        "web_result_count": len(evidence.web_results),
                        "local_result_count": len(evidence.local_results),
                    },
                )
            if evidence.used_web_fallback and not evidence.web_results:
                await self._emit_runtime_condition_event(
                    "runtime.degraded",
                    category="degraded",
                    component="researcher",
                    reason="web_fallback_returned_no_results",
                    severity=SeverityLevel.MEDIUM,
                    task_id=task_id,
                    metadata={
                        "local_result_count": len(evidence.local_results),
                        "web_result_count": 0,
                    },
                )
            reasoner_handoff = ResearchReasonerHandoff.from_inputs(
                plan=plan,
                evidence=evidence,
                budget=budget,
                reasoning_mode=self._select_reasoning_mode(budget),
                final_text_policy=self.reasoner.service.final_text_policy,
                implementation_mode=self.reasoner.service.implementation_mode,
            )

            reasoning = await self._run_component(
                "reasoner",
                task_id=task_id,
                start_stage="pipeline.reasoner_started",
                done_stage="pipeline.reasoner_done",
                start_payload={
                    "task_id": task_id,
                    "output_contract": reasoner_handoff.output_contract,
                    "reasoning_mode": reasoner_handoff.reasoning_mode,
                    "final_text_policy": reasoner_handoff.final_text_policy,
                },
                run=lambda: self.reasoner.reason_from_handoff(reasoner_handoff),
            )
            await self._emit_event(
                "pipeline.reasoner_done",
                self._reasoning_event_payload(task_id=task_id, reasoning=reasoning),
            )
            await self._record_status(
                "reasoner",
                AgentState.IDLE,
                task_id=task_id,
                message="reasoning completed",
            )
            reasoning_log = ReasoningLog(
                task_id=task_id,
                compressed_chain=reasoning.tokens,
                macros_used=reasoning.macros_used,
            )
            self.model_manager.warm_cache(
                "runtime_subsets",
                task_id,
                {
                    "macros": tuple(reasoning.macros_used),
                    "symbol_refs": tuple(reasoning.symbol_table_refs),
                },
            )
            await self.storage.record_reasoning_trace(reasoning)
            await self.storage.record_reasoning_log(reasoning_log)
            critic_handoff = ReasonerCriticHandoff.from_inputs(
                plan=plan,
                evidence=evidence,
                trace=reasoning,
                budget=budget,
                final_text_policy=self.critic.service.final_text_policy,
                implementation_mode=self.critic.service.implementation_mode,
            )

            critique = await self._run_component(
                "critic",
                task_id=task_id,
                start_stage="pipeline.critic_started",
                done_stage="pipeline.critic_done",
                start_payload={
                    "task_id": task_id,
                    "output_contract": critic_handoff.output_contract,
                    "final_text_policy": critic_handoff.final_text_policy,
                    "proof_hash": critic_handoff.proof_hash,
                },
                run=lambda: self.critic.review_from_handoff(critic_handoff),
            )
            await self._emit_event(
                "pipeline.critic_done",
                self._critique_event_payload(task_id=task_id, critique=critique),
            )
            await self._record_status(
                "critic",
                AgentState.IDLE if critique.is_valid else AgentState.ERROR,
                task_id=task_id,
                message="critique completed" if critique.is_valid else "critique reported issues",
                severity=SeverityLevel.LOW if critique.is_valid else SeverityLevel.HIGH,
            )
            reasoning, critique, repair_history = await self._run_repair_loop(
                task_id=task_id,
                plan=plan,
                evidence=evidence,
                budget=budget,
                reasoner_handoff=reasoner_handoff,
                reasoning=reasoning,
                critique=critique,
            )
            await self._record_status(
                "critic",
                self._critique_status_state(critique),
                task_id=task_id,
                message=self._critique_status_message(critique, repair_history=repair_history),
                severity=self._critique_status_severity(critique),
            )
            if critique.result == CritiqueResult.DEGRADED or critique.degraded_reason:
                await self._emit_runtime_condition_event(
                    "runtime.degraded",
                    category="degraded",
                    component="critic",
                    reason=critique.degraded_reason or critique.result.value,
                    severity=self._critique_status_severity(critique),
                    task_id=task_id,
                    metadata={
                        "failure_categories": list(critique.failure_categories),
                        "repair_actions": list(critique.repair_actions),
                        "critique_result": critique.result.value,
                    },
                )
            recent_reasoning_logs = await self._load_recent_reasoning_logs(
                limit=self.COMPRESSION_HISTORY_SCAN_LIMIT
            )

            compression = await self._run_component(
                "compressor",
                task_id=task_id,
                start_stage="pipeline.compressor_started",
                done_stage="pipeline.compressor_done",
                start_payload={"task_id": task_id},
                run=lambda: self.compressor.propose(reasoning, logs=recent_reasoning_logs),
            )
            await self._emit_event("pipeline.compressor_done", {"task_id": task_id})
            await self._record_status(
                "compressor",
                AgentState.IDLE,
                task_id=task_id,
                message="compression proposal generation completed",
            )

            warnings = self._build_warnings(evidence=evidence, critique=critique)
            if repair_history:
                warnings.extend(self._build_repair_warnings(repair_history))
            metric = self._build_performance_metric(
                task_id=task_id,
                started_at=started_at,
                budget=budget,
            )
            await self.storage.record_performance_metric(metric)

            model_snapshot = self.model_manager.health_snapshot()
            if model_snapshot.fallback_active:
                await self._record_status(
                    "model_manager",
                    AgentState.RUNNING,
                    task_id=task_id,
                    message=model_snapshot.fallback_reason or "model fallback active",
                    severity=SeverityLevel.MEDIUM,
                )
            await self._emit_health_snapshot(snapshot=model_snapshot, task_id=task_id)

            answer_text = self._build_answer_text(
                evidence=evidence,
                reasoning=reasoning,
                critique=critique,
            )
            result = TaskResult(
                task_id=task_id,
                plan=plan,
                evidence=evidence,
                reasoning=reasoning,
                critique=critique,
                compression=tuple(compression),
                answer_text=answer_text,
                warnings=tuple(warnings),
                metrics=(metric,),
            )
            if persist_task_result:
                await self.storage.record_task_result(result)
            if publish_history:
                await self._publish_dashboard_task_history()
            if self.dashboard.dropped_events > 0:
                await self._emit_runtime_condition_event(
                    "runtime.backpressure_detected",
                    category="backpressure",
                    component="dashboard",
                    reason="dashboard_queue_overflow",
                    severity=SeverityLevel.MEDIUM,
                    task_id=task_id,
                    metadata={"dropped_events": self.dashboard.dropped_events},
                )
            if emit_completion_event:
                await self._emit_event(
                    "pipeline.completed",
                    self._completion_event_payload(
                        task_id=task_id,
                        evidence=evidence,
                        reasoning=reasoning,
                        critique=critique,
                        answer_text=answer_text,
                        warning_count=len(result.warnings),
                    ),
                )
            await self._record_status(
                "orchestrator",
                AgentState.IDLE,
                task_id=task_id,
                message="pipeline completed",
            )
            return result
        except asyncio.CancelledError:
            await self._emit_event(
                "pipeline.cancelled",
                {
                    "task_id": task_id,
                    "question": question,
                },
            )
            await self._record_status(
                "orchestrator",
                AgentState.ERROR,
                task_id=task_id,
                message="pipeline cancelled",
                severity=SeverityLevel.MEDIUM,
            )
            await self._emit_health_snapshot(task_id=task_id)
            raise
        except Exception as exc:
            await self._emit_event(
                "pipeline.failed",
                {
                    "task_id": task_id,
                    "question": question,
                    "error": str(exc),
                },
            )
            await self._record_status(
                "orchestrator",
                AgentState.ERROR,
                task_id=task_id,
                message=str(exc),
                severity=SeverityLevel.CRITICAL,
            )
            await self._emit_health_snapshot(task_id=task_id)
            raise

    async def run_pipeline(self, question: str) -> TaskResult:
        """Run the compatibility pipeline entrypoint with the minimum valid budget.

        Input:
        - `question`: the user task text.

        Output:
        - The same `TaskResult` shape returned by `run_task(...)`.

        Failure behavior:
        - Raises the same startup or runtime exceptions as `run_task(...)`.
        """
        return await self.run_task(question, thinking_minutes=1)

    async def _run_long_horizon_task(
        self,
        *,
        question: str,
        thinking_minutes: int,
        budget,
    ) -> TaskResult:
        """Run repeated bounded pipeline cycles with checkpoint persistence between bursts."""
        session_id = self._long_horizon_session_id(question=question, thinking_minutes=thinking_minutes)
        session = LongHorizonSession(
            session_id=session_id,
            question=question,
            requested_minutes=thinking_minutes,
            budget=budget,
            status=LongHorizonSessionState.RUNNING,
            total_cycles=budget.planned_cycles,
            updated_at=utc_now(),
        )
        await self.storage.save_long_horizon_session(session)
        await self._emit_event(
            "pipeline.long_horizon_started",
            {
                "session_id": session_id,
                "question": question,
                "thinking_minutes": thinking_minutes,
                "planned_cycles": budget.planned_cycles,
                **self._long_horizon_dashboard_payload(
                    session=session,
                    checkpoints=(),
                    latest_result=None,
                    current_phase="initializing",
                ),
            },
        )
        return await self._continue_long_horizon_session(session=session, start_cycle=1)

    async def _resume_long_horizon_session(self, session_id: str) -> TaskResult:
        """Resume a paused or checkpointed long-horizon session from its next cycle."""
        session = await self.storage.load_long_horizon_session(session_id)
        if session is None:
            raise RuntimeError(f"Unknown long-horizon session '{session_id}'.")
        if session.session_id in self._active_long_horizon_tasks:
            raise RuntimeError(f"Long-horizon session '{session_id}' is already running.")
        if session.status not in {
            LongHorizonSessionState.CHECKPOINTED,
            LongHorizonSessionState.PAUSED,
        }:
            raise RuntimeError(
                f"Long-horizon session '{session_id}' is not resumable from status '{session.status.value}'."
            )
        if session.completed_cycles >= session.total_cycles:
            raise RuntimeError(f"Long-horizon session '{session_id}' is already complete.")
        if session.resume_count >= session.budget.max_resume_count:
            raise RuntimeError(
                f"Long-horizon session '{session_id}' exhausted its resume budget ({session.budget.max_resume_count})."
            )

        resumed = replace(
            session,
            status=LongHorizonSessionState.RUNNING,
            resume_count=session.resume_count + 1,
            pause_requested=False,
            cancel_requested=False,
            last_control_reason="resume_requested",
            last_error="",
            updated_at=utc_now(),
        )
        await self.storage.save_long_horizon_session(resumed)
        checkpoints = await self.storage.list_long_horizon_checkpoints(resumed.session_id)
        await self._emit_event(
            "pipeline.long_horizon_resumed",
            {
                "session_id": resumed.session_id,
                "completed_cycles": resumed.completed_cycles,
                "total_cycles": resumed.total_cycles,
                "resume_count": resumed.resume_count,
                "next_cycle_index": resumed.completed_cycles + 1,
                **self._long_horizon_dashboard_payload(
                    session=resumed,
                    checkpoints=checkpoints,
                    latest_result=None,
                    current_phase="resumed",
                ),
            },
        )
        return await self._continue_long_horizon_session(
            session=resumed,
            start_cycle=resumed.completed_cycles + 1,
        )

    async def _continue_long_horizon_session(
        self,
        *,
        session: LongHorizonSession,
        start_cycle: int,
    ) -> TaskResult:
        """Continue a checkpointed long-horizon session from the requested cycle."""
        historical_checkpoints = await self.storage.list_long_horizon_checkpoints(session.session_id)
        last_result: TaskResult | None = None
        current_session = session
        pending_budget = self._next_cycle_budget_from_checkpoints(
            checkpoints=historical_checkpoints,
            default_budget=current_session.budget,
        )
        self._register_active_long_horizon_task(session.session_id)
        try:
            for cycle_index in range(start_cycle, current_session.budget.planned_cycles + 1):
                current_session = await self._refresh_long_horizon_session(current_session)
                if current_session.cancel_requested:
                    raise asyncio.CancelledError()

                cycle_budget = pending_budget
                pending_budget = current_session.budget
                throttle = self._long_horizon_throttle_decision(cycle_budget)
                current_session = await self._publish_long_horizon_throttle_state(
                    session=current_session,
                    cycle_index=cycle_index,
                    throttle=throttle,
                )
                await self._emit_event(
                    "pipeline.long_horizon_cycle_started",
                    {
                        "session_id": current_session.session_id,
                        "cycle_index": cycle_index,
                        "total_cycles": current_session.total_cycles,
                        "resume_count": current_session.resume_count,
                        "throttled": throttle.throttled,
                        "throttle_reason": throttle.reason,
                        **self._long_horizon_dashboard_payload(
                            session=current_session,
                            checkpoints=historical_checkpoints,
                            latest_result=None,
                            current_phase="running_bounded_cycle",
                            effective_budget=throttle.budget,
                        ),
                    },
                )
                result = await self._run_bounded_task(
                    question=current_session.question,
                    thinking_minutes=throttle.budget.cycle_budget_minutes,
                    budget=throttle.budget,
                    persist_task_result=False,
                    publish_history=False,
                    emit_completion_event=False,
                )
                last_result = result
                advisory_plan = _LongHorizonAdvisoryPlan(
                    suggestions=(),
                    usage_records=(),
                    next_budget=throttle.budget,
                )
                if cycle_index < current_session.total_cycles:
                    advisory_plan = await self._plan_long_horizon_advisory(
                        session=current_session,
                        cycle_index=cycle_index,
                        latest_result=result,
                        checkpoints=historical_checkpoints,
                        budget=throttle.budget,
                    )
                    if advisory_plan.suggestions:
                        await self.storage.record_optimizer_suggestion_records(advisory_plan.suggestions)
                        self.model_manager.apply_governor_advisory_inputs(advisory_plan.suggestions)
                        for suggestion in advisory_plan.suggestions:
                            self.model_manager.warm_cache(
                                "strategy_artifacts",
                                suggestion.suggestion_id,
                                suggestion.summary,
                            )
                            if suggestion.kind == OptimizerSuggestionKind.MACRO_ADVICE:
                                self.model_manager.warm_cache(
                                    "compression_artifacts",
                                    suggestion.suggestion_id,
                                    suggestion.summary,
                                )
                    if advisory_plan.usage_records:
                        await self.storage.record_optimizer_suggestion_usage_records(advisory_plan.usage_records)
                    if advisory_plan.suggestions:
                        await self._emit_event(
                            "pipeline.long_horizon_advisory_planned",
                            {
                                "session_id": current_session.session_id,
                                "cycle_index": cycle_index,
                                "task_id": result.task_id,
                                "suggestion_ids": [record.suggestion_id for record in advisory_plan.suggestions],
                                "suggestion_kinds": [record.kind.value for record in advisory_plan.suggestions],
                                "usage": [record.to_dict() for record in advisory_plan.usage_records],
                                **self._long_horizon_dashboard_payload(
                                    session=current_session,
                                    checkpoints=historical_checkpoints,
                                    latest_result=result,
                                    current_phase="advisory_planned",
                                    effective_budget=advisory_plan.next_budget,
                                ),
                            },
                        )
                    pending_budget = advisory_plan.next_budget
                checkpoint = self._build_long_horizon_checkpoint(
                    session_id=current_session.session_id,
                    question=current_session.question,
                    budget=throttle.budget,
                    result=result,
                    cycle_index=cycle_index,
                    total_cycles=current_session.total_cycles,
                    resume_count=current_session.resume_count,
                    throttled=throttle.throttled,
                    throttle_reason=throttle.reason,
                    advisory_plan=advisory_plan if cycle_index < current_session.total_cycles else None,
                )
                await self.storage.save_long_horizon_checkpoint(checkpoint)
                historical_checkpoints = tuple((*historical_checkpoints, checkpoint))
                # Preserve any pause or cancel requests that arrived while the cycle was running.
                current_session = await self._refresh_long_horizon_session(current_session)
                current_session = replace(
                    current_session,
                    status=(
                        LongHorizonSessionState.COMPLETED
                        if cycle_index == current_session.total_cycles
                        else LongHorizonSessionState.CHECKPOINTED
                    ),
                    completed_cycles=cycle_index,
                    last_task_id=result.task_id,
                    last_checkpoint_cycle=cycle_index,
                    throttled=throttle.throttled,
                    throttle_reason=throttle.reason,
                    latest_answer_preview=self._cycle_answer_preview(result.answer_text),
                    updated_at=checkpoint.created_at,
                )
                await self.storage.save_long_horizon_session(current_session)
                early_stop_reason = ""
                if cycle_index < current_session.total_cycles:
                    early_stop_reason = self._long_horizon_early_stop_reason(
                        session=current_session,
                        checkpoints=historical_checkpoints,
                    )
                    if early_stop_reason:
                        final_checkpoint = replace(
                            historical_checkpoints[-1],
                            metadata={
                                **historical_checkpoints[-1].metadata,
                                "early_stop_reason": early_stop_reason,
                            },
                        )
                        await self.storage.save_long_horizon_checkpoint(final_checkpoint)
                        historical_checkpoints = tuple((*historical_checkpoints[:-1], final_checkpoint))
                await self._emit_event(
                    "pipeline.long_horizon_cycle_completed",
                    {
                        "session_id": current_session.session_id,
                        "cycle_index": cycle_index,
                        "total_cycles": current_session.total_cycles,
                        "resume_count": current_session.resume_count,
                        "task_id": result.task_id,
                        "critique_result": result.critique.result.value,
                        "candidate_trace_count": len(result.reasoning.candidate_traces),
                        "throttled": throttle.throttled,
                        "throttle_reason": throttle.reason,
                        **self._long_horizon_dashboard_payload(
                            session=current_session,
                            checkpoints=historical_checkpoints,
                            latest_result=result,
                            current_phase=(
                                "early_stopped"
                                if early_stop_reason
                                else "finalizing"
                                if cycle_index == current_session.total_cycles
                                else "cooldown"
                            ),
                            effective_budget=throttle.budget,
                        ),
                    },
                )

                current_session = await self._refresh_long_horizon_session(current_session)
                if current_session.cancel_requested:
                    raise asyncio.CancelledError()
                if current_session.pause_requested:
                    paused = replace(
                        current_session,
                        status=LongHorizonSessionState.PAUSED,
                        pause_requested=False,
                        cancel_requested=False,
                        last_control_reason=current_session.last_control_reason or "pause_requested",
                        updated_at=utc_now(),
                    )
                    return await self._build_paused_long_horizon_result(
                        session=paused,
                        latest_result=result,
                        checkpoints=historical_checkpoints,
                    )
                if cycle_index < current_session.total_cycles:
                    if early_stop_reason:
                        return await self._finalize_long_horizon_completion(
                            session=replace(
                                current_session,
                                status=LongHorizonSessionState.COMPLETED,
                                pause_requested=False,
                                cancel_requested=False,
                                last_control_reason="no_measurable_improvement",
                                updated_at=last_result.completed_at if last_result is not None else utc_now(),
                            ),
                            latest_result=last_result,
                            checkpoints=historical_checkpoints,
                            early_stop_reason=early_stop_reason,
                        )
                    current_session = await self._wait_long_horizon_cooldown(
                        session=current_session,
                        cooldown_seconds=throttle.budget.cooldown_seconds,
                    )
                    if current_session.cancel_requested:
                        raise asyncio.CancelledError()
                    if current_session.pause_requested:
                        paused = replace(
                            current_session,
                            status=LongHorizonSessionState.PAUSED,
                            pause_requested=False,
                            cancel_requested=False,
                            last_control_reason=current_session.last_control_reason or "pause_requested",
                            updated_at=utc_now(),
                        )
                        return await self._build_paused_long_horizon_result(
                            session=paused,
                            latest_result=result,
                            checkpoints=historical_checkpoints,
                        )

            return await self._finalize_long_horizon_completion(
                session=replace(
                    current_session,
                    status=LongHorizonSessionState.COMPLETED,
                    pause_requested=False,
                    cancel_requested=False,
                    updated_at=last_result.completed_at if last_result is not None else utc_now(),
                ),
                latest_result=last_result,
                checkpoints=historical_checkpoints,
            )
        except asyncio.CancelledError:
            await self._handle_long_horizon_cancellation(current_session)
            raise
        except Exception as exc:
            failed = replace(
                current_session,
                status=LongHorizonSessionState.FAILED,
                last_error=str(exc),
                updated_at=utc_now(),
            )
            await self.storage.save_long_horizon_session(failed)
            await self._emit_event(
                "pipeline.long_horizon_failed",
                {
                    "session_id": failed.session_id,
                    "question": failed.question,
                    "completed_cycles": failed.completed_cycles,
                    "error": str(exc),
                    **self._long_horizon_dashboard_payload(
                        session=failed,
                        checkpoints=historical_checkpoints,
                        latest_result=last_result,
                        current_phase="failed",
                    ),
                },
            )
            raise
        finally:
            self._unregister_active_long_horizon_task(session.session_id)

    def _should_use_long_horizon(self, budget) -> bool:
        return int(getattr(budget, "planned_cycles", 1)) > 1

    def _long_horizon_session_id(self, *, question: str, thinking_minutes: int) -> str:
        return f"lh_{stable_hash(f'{question}|{thinking_minutes}|{time.time_ns()}')[:16]}"

    def _cycle_answer_preview(self, answer_text: str, *, limit: int = 240) -> str:
        preview = " ".join(answer_text.split())
        return preview[:limit]

    def _long_horizon_selected_candidate_score(
        self,
        checkpoint: LongHorizonCheckpoint | None,
    ) -> float:
        if checkpoint is None:
            return 0.0
        if checkpoint.selected_candidate_id:
            for candidate in checkpoint.candidate_summaries:
                if candidate.candidate_id == checkpoint.selected_candidate_id:
                    return float(candidate.total_score)
        if checkpoint.candidate_summaries:
            return float(max(candidate.total_score for candidate in checkpoint.candidate_summaries))
        return 0.0

    def _long_horizon_dashboard_payload(
        self,
        *,
        session: LongHorizonSession,
        checkpoints: tuple[LongHorizonCheckpoint, ...],
        latest_result: TaskResult | None,
        current_phase: str,
        effective_budget: ResourceBudget | None = None,
    ) -> dict[str, Any]:
        first_checkpoint = checkpoints[0] if checkpoints else None
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        budget = effective_budget or (latest_checkpoint.budget if latest_checkpoint is not None else session.budget)
        candidate_count = (
            len(latest_checkpoint.candidate_summaries)
            if latest_checkpoint is not None
            else len(latest_result.reasoning.candidate_traces)
            if latest_result is not None
            else 0
        )
        selected_candidate_id = (
            latest_checkpoint.selected_candidate_id
            if latest_checkpoint is not None
            else self._selected_candidate_id(latest_result.reasoning)
            if latest_result is not None
            else ""
        )
        candidate_score = self._long_horizon_selected_candidate_score(latest_checkpoint)
        if candidate_score <= 0.0 and latest_result is not None:
            candidate_score = float(latest_result.critique.candidate_score)
        supporting_evidence_ids = (
            latest_checkpoint.supporting_evidence_ids
            if latest_checkpoint is not None
            else tuple(item.id for item in (latest_result.evidence.local_results + latest_result.evidence.web_results))
            if latest_result is not None
            else ()
        )
        critique_result = (
            latest_checkpoint.critique_result
            if latest_checkpoint is not None
            else latest_result.critique.result.value
            if latest_result is not None
            else ""
        )
        repair_actions = (
            latest_checkpoint.repair_actions
            if latest_checkpoint is not None
            else latest_result.critique.repair_actions
            if latest_result is not None
            else ()
        )
        initial_candidate_count = (
            len(first_checkpoint.candidate_summaries)
            if first_checkpoint is not None
            else candidate_count
        )
        peak_candidate_count = max(
            (len(checkpoint.candidate_summaries) for checkpoint in checkpoints),
            default=candidate_count,
        )
        initial_supporting_evidence_count = (
            len(first_checkpoint.supporting_evidence_ids)
            if first_checkpoint is not None
            else len(supporting_evidence_ids)
        )
        first_candidate_score = self._long_horizon_selected_candidate_score(first_checkpoint)
        if first_candidate_score <= 0.0:
            first_candidate_score = candidate_score
        first_critique_result = (
            first_checkpoint.critique_result
            if first_checkpoint is not None
            else critique_result
        )
        elapsed_seconds = round(
            max(0.0, (utc_now() - session.created_at).total_seconds()),
            3,
        )
        remaining_cycles = max(0, session.total_cycles - session.completed_cycles)
        if current_phase in {"completed", "cancelled", "failed", "early_stopped"}:
            eta_seconds = 0.0
        else:
            eta_seconds = round(
                max(0.0, (session.requested_minutes * 60.0) * remaining_cycles / max(1, session.total_cycles)),
                3,
            )

        payload: dict[str, Any] = {
            "execution_mode": "long_horizon",
            "requested_minutes": session.requested_minutes,
            "cycle_budget_minutes": budget.cycle_budget_minutes,
            "checkpoint_interval_minutes": budget.checkpoint_interval_minutes,
            "duty_cycle_ratio": budget.duty_cycle_ratio,
            "cooldown_seconds": budget.cooldown_seconds,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
            "current_phase": current_phase,
            "candidate_trace_count": candidate_count,
            "selected_candidate_id": selected_candidate_id,
            "candidate_score": round(candidate_score, 3),
            "critique_result": critique_result,
            "repair_actions": tuple(repair_actions),
            "repair_action_count": len(repair_actions),
            "supporting_evidence_ids": tuple(supporting_evidence_ids),
            "supporting_evidence_count": len(supporting_evidence_ids),
            "initial_candidate_count": initial_candidate_count,
            "peak_candidate_count": peak_candidate_count,
            "additional_candidate_count": max(0, peak_candidate_count - initial_candidate_count),
            "initial_supporting_evidence_count": initial_supporting_evidence_count,
            "additional_supporting_evidence_count": max(
                0,
                len(supporting_evidence_ids) - initial_supporting_evidence_count,
            ),
            "total_verification_passes": sum(checkpoint.budget.critic_passes for checkpoint in checkpoints),
            "total_repairs": sum(len(checkpoint.repair_actions) for checkpoint in checkpoints),
            "first_candidate_score": round(first_candidate_score, 3),
            "confidence_gain": round(candidate_score - first_candidate_score, 3),
            "first_critique_result": first_critique_result,
            "validity_improved": (
                bool(first_critique_result)
                and first_critique_result != CritiqueResult.VALID.value
                and critique_result == CritiqueResult.VALID.value
            ),
            "advisory_requested_count": 0,
            "advisory_accepted_count": 0,
            "advisory_rejected_count": 0,
            "advisory_deferred_count": 0,
            "advisory_entries": (),
            "early_stop_reason": "",
            "latest_answer_preview": (
                latest_checkpoint.answer_preview
                if latest_checkpoint is not None
                else session.latest_answer_preview
            ),
        }
        if latest_result is not None:
            payload.update(
                {
                    "local_result_count": len(latest_result.evidence.local_results),
                    "web_result_count": len(latest_result.evidence.web_results),
                    "used_web_fallback": latest_result.evidence.used_web_fallback,
                }
            )
        advisory_entries: list[str] = []
        advisory_requested_count = 0
        advisory_accepted_count = 0
        advisory_rejected_count = 0
        advisory_deferred_count = 0
        early_stop_reason = ""
        for checkpoint in checkpoints:
            early_stop_reason = str(checkpoint.metadata.get("early_stop_reason", early_stop_reason))
            for raw_record in checkpoint.metadata.get("advisory_usage_records", ()):
                try:
                    record = OptimizerSuggestionUsageRecord.from_dict(raw_record)
                except (KeyError, TypeError, ValueError):
                    continue
                if record.disposition == OptimizerSuggestionDisposition.REQUESTED:
                    advisory_requested_count += 1
                elif record.disposition == OptimizerSuggestionDisposition.ACCEPTED:
                    advisory_accepted_count += 1
                elif record.disposition == OptimizerSuggestionDisposition.REJECTED:
                    advisory_rejected_count += 1
                elif record.disposition == OptimizerSuggestionDisposition.DEFERRED:
                    advisory_deferred_count += 1
            suggestion_map: dict[str, OptimizerSuggestionRecord] = {}
            for raw_record in checkpoint.metadata.get("advisory_suggestions", ()):
                try:
                    record = OptimizerSuggestionRecord.from_dict(raw_record)
                except (KeyError, TypeError, ValueError):
                    continue
                suggestion_map[record.suggestion_id] = record
            for raw_record in checkpoint.metadata.get("advisory_usage_records", ()):
                try:
                    record = OptimizerSuggestionUsageRecord.from_dict(raw_record)
                except (KeyError, TypeError, ValueError):
                    continue
                suggestion = suggestion_map.get(record.suggestion_id)
                summary = suggestion.summary if suggestion is not None else record.suggestion_id
                entry = (
                    f"cycle {record.cycle_index}: {record.disposition.value} "
                    f"{summary} ({record.reason or 'no_reason'})"
                )
                if entry not in advisory_entries:
                    advisory_entries.append(entry)
        payload.update(
            {
                "advisory_requested_count": advisory_requested_count,
                "advisory_accepted_count": advisory_accepted_count,
                "advisory_rejected_count": advisory_rejected_count,
                "advisory_deferred_count": advisory_deferred_count,
                "advisory_entries": tuple(advisory_entries[-8:]),
                "early_stop_reason": early_stop_reason,
            }
        )
        return payload

    def _next_cycle_budget_from_checkpoints(
        self,
        *,
        checkpoints: tuple[LongHorizonCheckpoint, ...],
        default_budget: ResourceBudget,
    ) -> ResourceBudget:
        if not checkpoints:
            return default_budget
        payload = checkpoints[-1].metadata.get("next_cycle_budget")
        if not isinstance(payload, dict):
            return default_budget
        try:
            return ResourceBudget.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            return default_budget

    def _apply_optimizer_suggestions_to_budget(
        self,
        *,
        session_id: str,
        cycle_index: int,
        task_id: str,
        budget: ResourceBudget,
        suggestions: tuple[OptimizerSuggestionRecord, ...],
    ) -> tuple[ResourceBudget, tuple[OptimizerSuggestionUsageRecord, ...]]:
        requested_records: list[OptimizerSuggestionUsageRecord] = []
        disposition_records: list[OptimizerSuggestionUsageRecord] = []
        adjusted_budget = budget
        max_budget = self.config.budget_calibration

        for suggestion in suggestions:
            requested_records.append(
                OptimizerSuggestionUsageRecord(
                    usage_id=f"usage:{stable_hash(f'{suggestion.suggestion_id}:requested')[:16]}",
                    suggestion_id=suggestion.suggestion_id,
                    session_id=session_id,
                    disposition=OptimizerSuggestionDisposition.REQUESTED,
                    cycle_index=cycle_index,
                    task_id=task_id,
                )
            )
            delta = dict(suggestion.metadata.get("budget_delta", {}))
            if suggestion.kind == OptimizerSuggestionKind.MACRO_ADVICE:
                disposition = OptimizerSuggestionDisposition.DEFERRED
                reason = str(suggestion.metadata.get("advisory_only_reason", "proposal_only_policy"))
            elif not delta:
                disposition = OptimizerSuggestionDisposition.DEFERRED
                reason = "no_budget_change_requested"
            else:
                candidate_budget = ResourceBudget(
                    retrieval_top_k=max(1, adjusted_budget.retrieval_top_k + int(delta.get("retrieval_top_k", 0) or 0)),
                    max_web_queries=max(0, adjusted_budget.max_web_queries + int(delta.get("max_web_queries", 0) or 0)),
                    reasoner_passes=max(1, adjusted_budget.reasoner_passes + int(delta.get("reasoner_passes", 0) or 0)),
                    critic_passes=max(1, adjusted_budget.critic_passes + int(delta.get("critic_passes", 0) or 0)),
                    macro_depth=max(1, adjusted_budget.macro_depth + int(delta.get("macro_depth", 0) or 0)),
                    wall_clock_minutes=adjusted_budget.wall_clock_minutes,
                    cycle_budget_minutes=adjusted_budget.cycle_budget_minutes,
                    checkpoint_interval_minutes=min(
                        adjusted_budget.checkpoint_interval_minutes,
                        adjusted_budget.cycle_budget_minutes,
                    ),
                    duty_cycle_ratio=adjusted_budget.duty_cycle_ratio,
                    cooldown_seconds=adjusted_budget.cooldown_seconds,
                    max_resume_count=adjusted_budget.max_resume_count,
                    planned_cycles=adjusted_budget.planned_cycles,
                )
                clamped_budget = max_budget.clamp_budget(candidate_budget)
                if clamped_budget == adjusted_budget:
                    disposition = OptimizerSuggestionDisposition.REJECTED
                    reason = "budget_already_at_cap_or_floor"
                else:
                    adjusted_budget = clamped_budget
                    disposition = OptimizerSuggestionDisposition.ACCEPTED
                    reason = "bounded_budget_adjustment_applied"
            disposition_records.append(
                OptimizerSuggestionUsageRecord(
                    usage_id=f"usage:{stable_hash(f'{suggestion.suggestion_id}:{disposition.value}')[:16]}",
                    suggestion_id=suggestion.suggestion_id,
                    session_id=session_id,
                    disposition=disposition,
                    cycle_index=cycle_index,
                    task_id=task_id,
                    reason=reason,
                )
            )
        return adjusted_budget, tuple((*requested_records, *disposition_records))

    async def _plan_long_horizon_advisory(
        self,
        *,
        session: LongHorizonSession,
        cycle_index: int,
        latest_result: TaskResult,
        checkpoints: tuple[LongHorizonCheckpoint, ...],
        budget: ResourceBudget,
    ) -> _LongHorizonAdvisoryPlan:
        suggestions = await self.self_optimizer.suggest_for_long_horizon(
            session=session,
            cycle_index=cycle_index,
            latest_result=latest_result,
            checkpoints=checkpoints,
            budget=budget,
        )
        next_budget, usage_records = self._apply_optimizer_suggestions_to_budget(
            session_id=session.session_id,
            cycle_index=cycle_index,
            task_id=latest_result.task_id,
            budget=budget,
            suggestions=suggestions,
        )
        return _LongHorizonAdvisoryPlan(
            suggestions=suggestions,
            usage_records=usage_records,
            next_budget=next_budget,
        )

    def _long_horizon_early_stop_reason(
        self,
        *,
        session: LongHorizonSession,
        checkpoints: tuple[LongHorizonCheckpoint, ...],
    ) -> str:
        if len(checkpoints) < 2 or session.completed_cycles >= session.total_cycles:
            return ""
        latest = checkpoints[-1]
        previous = checkpoints[-2]
        latest_score = self._long_horizon_selected_candidate_score(latest)
        previous_score = self._long_horizon_selected_candidate_score(previous)
        score_gain = latest_score - previous_score
        evidence_gain = len(latest.supporting_evidence_ids) - len(previous.supporting_evidence_ids)
        candidate_gain = len(latest.candidate_summaries) - len(previous.candidate_summaries)
        critique_improved = (
            previous.critique_result != CritiqueResult.VALID.value
            and latest.critique_result == CritiqueResult.VALID.value
        )
        if critique_improved or score_gain >= 0.02 or evidence_gain > 0 or candidate_gain > 0:
            return ""
        return (
            "No measurable improvement after multiple bounded cycles: candidate score, evidence coverage, and "
            "critique state all stayed effectively flat."
        )

    def _selected_candidate_id(self, trace: CompressedTrace) -> str:
        if trace.context_frames:
            candidate_id = str(trace.context_frames[0].metadata.get("cid", "")).strip()
            if candidate_id:
                return candidate_id
        if trace.candidate_traces:
            return trace.candidate_traces[0].candidate_id
        return ""

    def _build_long_horizon_checkpoint(
        self,
        *,
        session_id: str,
        question: str,
        budget,
        result: TaskResult,
        cycle_index: int,
        total_cycles: int,
        resume_count: int,
        throttled: bool,
        throttle_reason: str,
        advisory_plan: _LongHorizonAdvisoryPlan | None = None,
        early_stop_reason: str = "",
    ) -> LongHorizonCheckpoint:
        critique_summary = tuple(result.critique.issues[:4]) or tuple(result.critique.failure_categories[:4])
        return LongHorizonCheckpoint(
            session_id=session_id,
            cycle_index=cycle_index,
            total_cycles=total_cycles,
            task_id=result.task_id,
            question=question,
            budget=budget,
            candidate_summaries=tuple(
                LongHorizonCandidateSnapshot(
                    candidate_id=candidate.candidate_id,
                    strategy=candidate.strategy,
                    verifier_type=candidate.verifier_type,
                    verified=candidate.verified,
                    total_score=candidate.total_score,
                    degraded_reason=candidate.degraded_reason,
                    supporting_evidence_ids=candidate.supporting_evidence_ids,
                )
                for candidate in result.reasoning.candidate_traces
            ),
            selected_candidate_id=self._selected_candidate_id(result.reasoning),
            supporting_evidence_ids=tuple(
                item.id for item in (result.evidence.local_results + result.evidence.web_results)
            ),
            refreshed_web_source_refs=tuple(item.source_ref for item in result.evidence.web_results),
            critique_result=result.critique.result.value,
            critique_summary=critique_summary,
            repair_actions=result.critique.repair_actions,
            answer_preview=self._cycle_answer_preview(result.answer_text),
            resume_count=resume_count,
            throttled=throttled,
            throttle_reason=throttle_reason,
            metadata={
                "used_web_fallback": result.evidence.used_web_fallback,
                "warning_count": len(result.warnings),
                "metrics": [metric.to_dict() for metric in result.metrics],
                "advisory_suggestions": (
                    [record.to_dict() for record in advisory_plan.suggestions]
                    if advisory_plan is not None
                    else []
                ),
                "advisory_usage_records": (
                    [record.to_dict() for record in advisory_plan.usage_records]
                    if advisory_plan is not None
                    else []
                ),
                "next_cycle_budget": (
                    advisory_plan.next_budget.to_dict()
                    if advisory_plan is not None
                    else budget.to_dict()
                ),
                "early_stop_reason": early_stop_reason,
            },
            created_at=result.completed_at,
        )

    def _register_active_long_horizon_task(self, session_id: str) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._active_long_horizon_tasks[session_id] = task

    def _unregister_active_long_horizon_task(self, session_id: str) -> None:
        self._active_long_horizon_tasks.pop(session_id, None)

    async def _refresh_long_horizon_session(self, session: LongHorizonSession) -> LongHorizonSession:
        latest = await self.storage.load_long_horizon_session(session.session_id)
        return latest or session

    async def _request_pause_for_active_long_horizon_sessions(self, *, reason: str) -> None:
        for session_id in tuple(self._active_long_horizon_tasks):
            await self._request_long_horizon_pause(session_id, reason=reason, emit_paused=False)

    async def _request_long_horizon_pause(
        self,
        session_id: str,
        *,
        reason: str,
        emit_paused: bool = True,
    ) -> bool:
        session = await self.storage.load_long_horizon_session(session_id)
        if session is None or session.status in {
            LongHorizonSessionState.COMPLETED,
            LongHorizonSessionState.CANCELLED,
            LongHorizonSessionState.FAILED,
        }:
            return False
        if session_id not in self._active_long_horizon_tasks:
            paused = replace(
                session,
                status=LongHorizonSessionState.PAUSED,
                pause_requested=False,
                cancel_requested=False,
                last_control_reason=reason,
                updated_at=utc_now(),
            )
            await self.storage.save_long_horizon_session(paused)
            if emit_paused:
                await self._emit_event(
                    "pipeline.long_horizon_paused",
                    {
                        "session_id": paused.session_id,
                        "task_id": paused.last_task_id,
                        "completed_cycles": paused.completed_cycles,
                        "total_cycles": paused.total_cycles,
                        "resume_count": paused.resume_count,
                        "reason": reason,
                    },
                )
            return True

        pending = replace(
            session,
            pause_requested=True,
            cancel_requested=False,
            last_control_reason=reason,
            updated_at=utc_now(),
        )
        await self.storage.save_long_horizon_session(pending)
        await self._emit_event(
            "pipeline.long_horizon_pause_requested",
            {
                "session_id": pending.session_id,
                "completed_cycles": pending.completed_cycles,
                "total_cycles": pending.total_cycles,
                "reason": reason,
            },
        )
        return True

    async def _request_long_horizon_cancel(self, session_id: str, *, reason: str) -> bool:
        session = await self.storage.load_long_horizon_session(session_id)
        if session is None or session.status in {
            LongHorizonSessionState.COMPLETED,
            LongHorizonSessionState.CANCELLED,
            LongHorizonSessionState.FAILED,
        }:
            return False
        if session_id not in self._active_long_horizon_tasks:
            cancelled = replace(
                session,
                status=LongHorizonSessionState.CANCELLED,
                pause_requested=False,
                cancel_requested=False,
                last_control_reason=reason,
                updated_at=utc_now(),
            )
            await self.storage.save_long_horizon_session(cancelled)
            await self._emit_event(
                "pipeline.long_horizon_cancelled",
                {
                    "session_id": cancelled.session_id,
                    "question": cancelled.question,
                    "completed_cycles": cancelled.completed_cycles,
                    "total_cycles": cancelled.total_cycles,
                    "reason": reason,
                },
            )
            return True

        pending = replace(
            session,
            pause_requested=False,
            cancel_requested=True,
            last_control_reason=reason,
            updated_at=utc_now(),
        )
        await self.storage.save_long_horizon_session(pending)
        await self._emit_event(
            "pipeline.long_horizon_cancel_requested",
            {
                "session_id": pending.session_id,
                "completed_cycles": pending.completed_cycles,
                "total_cycles": pending.total_cycles,
                "reason": reason,
            },
        )
        task = self._active_long_horizon_tasks.get(session_id)
        if task is not None:
            task.cancel()
        return True

    def _cancel_active_long_horizon_tasks(self) -> None:
        for task in tuple(self._active_long_horizon_tasks.values()):
            task.cancel()

    def _long_horizon_throttle_decision(self, budget: ResourceBudget) -> _LongHorizonThrottleDecision:
        snapshot = self.model_manager.health_snapshot()
        reasons: list[str] = []
        metadata: dict[str, Any] = {
            "generation_backend": snapshot.generation_backend,
            "embedding_backend": snapshot.embedding_backend,
            "available_ram_gb": snapshot.available_ram_gb,
            "total_ram_gb": snapshot.total_ram_gb,
            "generation_backend_vram_gb": snapshot.generation_backend_vram_gb,
            "embedding_backend_vram_gb": snapshot.embedding_backend_vram_gb,
            "fallback_active": snapshot.fallback_active,
            "dashboard_dropped_events": self.dashboard.dropped_events,
        }
        total_ram = snapshot.total_ram_gb or 0.0
        available_ram = snapshot.available_ram_gb
        if snapshot.fallback_active:
            reasons.append("model_fallback_active")
        if not self.config.preflight.flags.stub_mode:
            if available_ram is not None and total_ram > 0 and (available_ram / total_ram) <= 0.25:
                reasons.append("low_available_ram")
            baseline_vram = float(PHASE12_ACCEPTANCE_THRESHOLDS.resources.baseline_vram_gb)
            if (
                snapshot.generation_backend_vram_gb is not None
                and snapshot.generation_backend_vram_gb >= baseline_vram * 0.8
            ):
                reasons.append("generation_vram_pressure")
            if (
                snapshot.embedding_backend_vram_gb is not None
                and snapshot.embedding_backend_vram_gb >= baseline_vram * 0.8
            ):
                reasons.append("embedding_vram_pressure")
        if self.dashboard.dropped_events > 0:
            reasons.append("dashboard_backpressure")
        if snapshot.last_error:
            reasons.append("model_runtime_error")
        if not reasons:
            return _LongHorizonThrottleDecision(
                budget=budget,
                throttled=False,
                reason="",
                metadata=metadata,
            )

        throttled_budget = ResourceBudget(
            retrieval_top_k=max(1, budget.retrieval_top_k - 2),
            max_web_queries=max(0, budget.max_web_queries - 1),
            reasoner_passes=max(1, budget.reasoner_passes - 1),
            critic_passes=max(1, budget.critic_passes - 1),
            macro_depth=max(1, budget.macro_depth - 1),
            wall_clock_minutes=budget.wall_clock_minutes,
            cycle_budget_minutes=budget.cycle_budget_minutes,
            checkpoint_interval_minutes=min(budget.checkpoint_interval_minutes, budget.cycle_budget_minutes),
            duty_cycle_ratio=max(
                self.config.budget_calibration.min_duty_cycle_ratio,
                min(budget.duty_cycle_ratio, 0.5),
            ),
            cooldown_seconds=min(
                self.config.budget_calibration.max_cooldown_seconds,
                max(budget.cooldown_seconds + 0.2, budget.cooldown_seconds),
            ),
            max_resume_count=budget.max_resume_count,
            planned_cycles=budget.planned_cycles,
        )
        return _LongHorizonThrottleDecision(
            budget=self.config.budget_calibration.clamp_budget(throttled_budget),
            throttled=True,
            reason=",".join(dict.fromkeys(reasons)),
            metadata=metadata,
        )

    async def _publish_long_horizon_throttle_state(
        self,
        *,
        session: LongHorizonSession,
        cycle_index: int,
        throttle: _LongHorizonThrottleDecision,
    ) -> LongHorizonSession:
        if not throttle.throttled:
            if session.throttled or session.throttle_reason:
                session = replace(
                    session,
                    throttled=False,
                    throttle_reason="",
                    updated_at=utc_now(),
                )
                await self.storage.save_long_horizon_session(session)
            return session

        updated = replace(
            session,
            throttled=True,
            throttle_reason=throttle.reason,
            updated_at=utc_now(),
        )
        await self.storage.save_long_horizon_session(updated)
        await self._emit_event(
            "pipeline.long_horizon_throttled",
            {
                "session_id": updated.session_id,
                "cycle_index": cycle_index,
                "total_cycles": updated.total_cycles,
                "reason": throttle.reason,
                "retrieval_top_k": throttle.budget.retrieval_top_k,
                "max_web_queries": throttle.budget.max_web_queries,
                "reasoner_passes": throttle.budget.reasoner_passes,
                "critic_passes": throttle.budget.critic_passes,
                "macro_depth": throttle.budget.macro_depth,
                "duty_cycle_ratio": throttle.budget.duty_cycle_ratio,
                "cooldown_seconds": throttle.budget.cooldown_seconds,
            },
        )
        await self._emit_runtime_condition_event(
            "runtime.long_horizon_throttled",
            category="resource_pressure",
            component="orchestrator",
            reason=throttle.reason,
            severity=SeverityLevel.MEDIUM,
            task_id=updated.last_task_id or None,
            metadata=throttle.metadata,
        )
        return updated

    async def _wait_long_horizon_cooldown(
        self,
        *,
        session: LongHorizonSession,
        cooldown_seconds: float,
    ) -> LongHorizonSession:
        remaining = max(0.0, float(cooldown_seconds))
        current = session
        while remaining > 0.0:
            sleep_s = min(0.05, remaining)
            await asyncio.sleep(sleep_s)
            remaining = max(0.0, remaining - sleep_s)
            current = await self._refresh_long_horizon_session(current)
            if current.pause_requested or current.cancel_requested or self._shutdown_requested:
                break
        return current

    def _checkpoint_metrics(
        self,
        checkpoints: tuple[LongHorizonCheckpoint, ...],
    ) -> tuple[PerformanceMetric, ...]:
        metrics: list[PerformanceMetric] = []
        for checkpoint in checkpoints:
            for payload in checkpoint.metadata.get("metrics", ()):
                try:
                    metrics.append(PerformanceMetric.from_dict(payload))
                except (KeyError, TypeError, ValueError):
                    continue
        return tuple(metrics)

    async def _build_paused_long_horizon_result(
        self,
        *,
        session: LongHorizonSession,
        latest_result: TaskResult,
        checkpoints: tuple[LongHorizonCheckpoint, ...],
    ) -> TaskResult:
        await self.storage.save_long_horizon_session(session)
        warnings = list(latest_result.warnings)
        warnings.extend(
            (
                "long_horizon_paused",
                "long_horizon_resume_available",
                f"long_horizon_cycles_completed:{session.completed_cycles}",
                f"long_horizon_session:{session.session_id}",
            )
        )
        if session.resume_count > 0:
            warnings.append(f"long_horizon_resume_count:{session.resume_count}")
        if session.throttled and session.throttle_reason:
            warnings.append(f"long_horizon_throttled:{session.throttle_reason}")
        paused_result = replace(
            latest_result,
            warnings=tuple(dict.fromkeys(warnings)),
            metrics=self._checkpoint_metrics(checkpoints),
        )
        await self._emit_event(
            "pipeline.long_horizon_paused",
            {
                "session_id": session.session_id,
                "task_id": latest_result.task_id,
                "completed_cycles": session.completed_cycles,
                "total_cycles": session.total_cycles,
                "resume_count": session.resume_count,
                "reason": session.last_control_reason or "pause_requested",
                "throttled": session.throttled,
                "throttle_reason": session.throttle_reason,
                **self._long_horizon_dashboard_payload(
                    session=session,
                    checkpoints=checkpoints,
                    latest_result=latest_result,
                    current_phase="paused",
                ),
            },
        )
        return paused_result

    async def _finalize_long_horizon_completion(
        self,
        *,
        session: LongHorizonSession,
        latest_result: TaskResult | None,
        checkpoints: tuple[LongHorizonCheckpoint, ...],
        early_stop_reason: str = "",
    ) -> TaskResult:
        if latest_result is None:
            raise RuntimeError(f"Long-horizon session '{session.session_id}' completed without a final result.")
        final_session = replace(
            session,
            status=LongHorizonSessionState.COMPLETED,
            pause_requested=False,
            cancel_requested=False,
            updated_at=latest_result.completed_at,
        )
        await self.storage.save_long_horizon_session(final_session)
        aggregated_warnings = list(latest_result.warnings)
        aggregated_warnings.append("long_horizon_checkpointed_run")
        aggregated_warnings.append(f"long_horizon_cycles_completed:{final_session.completed_cycles}")
        aggregated_warnings.append(f"long_horizon_session:{final_session.session_id}")
        if final_session.resume_count > 0:
            aggregated_warnings.append(f"long_horizon_resume_count:{final_session.resume_count}")
        if final_session.throttled and final_session.throttle_reason:
            aggregated_warnings.append(f"long_horizon_throttled:{final_session.throttle_reason}")
        if early_stop_reason:
            aggregated_warnings.append("long_horizon_early_stop")
            aggregated_warnings.append(f"long_horizon_early_stop_reason:{early_stop_reason}")
        final_result = replace(
            latest_result,
            warnings=tuple(dict.fromkeys(aggregated_warnings)),
            metrics=self._checkpoint_metrics(checkpoints),
        )
        await self.storage.record_task_result(final_result)
        await self._publish_dashboard_task_history()
        if early_stop_reason:
            await self._emit_event(
                "pipeline.long_horizon_early_stopped",
                {
                    "session_id": final_session.session_id,
                    "task_id": final_result.task_id,
                    "completed_cycles": final_session.completed_cycles,
                    "total_cycles": final_session.total_cycles,
                    "reason": early_stop_reason,
                    **self._long_horizon_dashboard_payload(
                        session=final_session,
                        checkpoints=checkpoints,
                        latest_result=final_result,
                        current_phase="early_stopped",
                    ),
                },
            )
        await self._emit_event(
            "pipeline.completed",
            self._completion_event_payload(
                task_id=final_result.task_id,
                evidence=final_result.evidence,
                reasoning=final_result.reasoning,
                critique=final_result.critique,
                answer_text=final_result.answer_text,
                warning_count=len(final_result.warnings),
            ),
        )
        await self._emit_event(
            "pipeline.long_horizon_completed",
            {
                "session_id": final_session.session_id,
                "task_id": final_result.task_id,
                "completed_cycles": final_session.completed_cycles,
                "total_cycles": final_session.total_cycles,
                "resume_count": final_session.resume_count,
                "warning_count": len(final_result.warnings),
                "throttled": final_session.throttled,
                "throttle_reason": final_session.throttle_reason,
                **self._long_horizon_dashboard_payload(
                    session=final_session,
                    checkpoints=checkpoints,
                    latest_result=final_result,
                    current_phase="early_stopped" if early_stop_reason else "completed",
                ),
            },
        )
        return final_result

    async def _handle_long_horizon_cancellation(self, session: LongHorizonSession) -> None:
        current = await self._refresh_long_horizon_session(session)
        checkpoints = await self.storage.list_long_horizon_checkpoints(current.session_id)
        if current.cancel_requested:
            cancelled = replace(
                current,
                status=LongHorizonSessionState.CANCELLED,
                pause_requested=False,
                cancel_requested=False,
                last_control_reason=current.last_control_reason or "cancel_requested",
                updated_at=utc_now(),
            )
            await self.storage.save_long_horizon_session(cancelled)
            await self._emit_event(
                "pipeline.long_horizon_cancelled",
                {
                    "session_id": cancelled.session_id,
                    "question": cancelled.question,
                    "completed_cycles": cancelled.completed_cycles,
                    "total_cycles": cancelled.total_cycles,
                    "reason": cancelled.last_control_reason,
                    **self._long_horizon_dashboard_payload(
                        session=cancelled,
                        checkpoints=checkpoints,
                        latest_result=None,
                        current_phase="cancelled",
                    ),
                },
            )
            return
        if current.pause_requested or self._shutdown_requested:
            paused = replace(
                current,
                status=LongHorizonSessionState.PAUSED,
                pause_requested=False,
                cancel_requested=False,
                last_control_reason=current.last_control_reason or (
                    "shutdown_requested" if self._shutdown_requested else "pause_requested"
                ),
                updated_at=utc_now(),
            )
            await self.storage.save_long_horizon_session(paused)
            await self._emit_event(
                "pipeline.long_horizon_paused",
                {
                    "session_id": paused.session_id,
                    "task_id": paused.last_task_id,
                    "completed_cycles": paused.completed_cycles,
                    "total_cycles": paused.total_cycles,
                    "resume_count": paused.resume_count,
                    "reason": paused.last_control_reason,
                    **self._long_horizon_dashboard_payload(
                        session=paused,
                        checkpoints=checkpoints,
                        latest_result=None,
                        current_phase="paused",
                    ),
                },
            )
            return

        failed = replace(
            current,
            status=LongHorizonSessionState.FAILED,
            last_error="cancelled",
            updated_at=utc_now(),
        )
        await self.storage.save_long_horizon_session(failed)
        await self._emit_event(
            "pipeline.long_horizon_failed",
            {
                "session_id": failed.session_id,
                "question": failed.question,
                "completed_cycles": failed.completed_cycles,
                "error": "cancelled",
                **self._long_horizon_dashboard_payload(
                    session=failed,
                    checkpoints=checkpoints,
                    latest_result=None,
                    current_phase="failed",
                ),
            },
        )

    async def _emit_event(self, stage: str, payload: dict[str, Any]) -> None:
        event = RuntimeEvent(
            stage=stage,
            payload=dict(payload),
        )
        await self.storage.record_runtime_event(event)

    async def _emit_runtime_condition_event(
        self,
        stage: str,
        *,
        category: str,
        component: str,
        reason: str,
        severity: SeverityLevel,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        condition = RuntimeCondition(
            stage=stage,
            category=category,
            component=component,
            reason=reason,
            severity=severity,
            task_id=task_id,
            metadata=dict(metadata or {}),
        )
        await self._emit_event(stage, condition.to_event_payload())

    async def _emit_health_snapshot(
        self,
        *,
        snapshot: ModelHealthSnapshot | None = None,
        task_id: str | None = None,
    ) -> None:
        snapshot = snapshot or self.model_manager.health_snapshot()
        await self._emit_event(
            "runtime.health_snapshot",
            self._model_health_payload(snapshot=snapshot, task_id=task_id),
        )
        await self._publish_dashboard_model_registry_view()

    def _forward_runtime_event_to_dashboard(self, event: RuntimeEvent) -> None:
        self.dashboard.publish_event(
            {
                "stage": event.stage,
                **event.payload,
                "timestamp": event.timestamp.isoformat(),
            }
        )

    def _forward_agent_status_to_dashboard(self, status: AgentStatus) -> None:
        self.dashboard.publish_event(
            {
                "stage": "status.updated",
                **status.to_dict(),
            }
        )

    def _submit_dashboard_task_request(self, question: str, thinking_minutes: int) -> None:
        """Submit one dashboard task request, expanding to the saved long-horizon wall-clock budget when enabled."""
        if self._loop is None:
            raise RuntimeError("Orchestrator event loop is not available for dashboard task submission.")
        app_state = self.dashboard.app_state_snapshot()
        effective_minutes = int(thinking_minutes)
        if app_state.user_settings.long_horizon.get("enabled", False):
            effective_minutes = max(
                effective_minutes,
                int(app_state.user_settings.long_horizon.get("wall_clock_minutes", effective_minutes) or effective_minutes),
            )
        self._submit_dashboard_coroutine(
            self.run_task(question, effective_minutes),
        )

    def _local_task_session_id(self, *, label: str) -> str:
        """Build a stable short identifier for one explicit local task session."""
        return f"lts-{stable_hash(f'{label}:{utc_now().isoformat()}')[:12]}"

    @staticmethod
    def _terminal_local_task_session_states() -> frozenset[LocalTaskSessionState]:
        return frozenset(
            {
                LocalTaskSessionState.STOPPED,
                LocalTaskSessionState.KILLED,
                LocalTaskSessionState.COMPLETED,
                LocalTaskSessionState.FAILED,
            }
        )

    @staticmethod
    def _dashboard_local_task_session_state(
        session: LocalTaskSession | None,
    ) -> DashboardLocalTaskSessionState:
        if session is None:
            return DashboardLocalTaskSessionState()
        return DashboardLocalTaskSessionState(
            session_id=session.session_id,
            label=session.label,
            profile_name=session.profile_name,
            status=session.status.value,
            control_mode=session.control_mode,
            current_target=session.current_target,
            last_action_summary=session.last_action_summary,
            last_request_id=session.last_request_id,
            continuous_capture_active=session.continuous_capture_active,
            continuous_capture_directory=session.continuous_capture_directory,
            continuous_capture_frame_count=session.continuous_capture_frame_count,
            continuous_capture_retained_frame_count=session.continuous_capture_retained_frame_count,
            continuous_capture_last_frame_path=session.continuous_capture_last_frame_path,
            continuous_capture_region=session.continuous_capture_region,
            continuous_capture_fps=session.continuous_capture_fps,
            continuous_capture_max_width=session.continuous_capture_max_width,
            continuous_capture_max_height=session.continuous_capture_max_height,
            continuous_capture_last_diff_ratio=session.continuous_capture_last_diff_ratio,
            continuous_capture_warnings=session.continuous_capture_warnings,
            continuous_capture_last_capture_at=session.continuous_capture_last_capture_at,
            requested_observation_tier=session.requested_observation_tier,
            effective_observation_tier=session.effective_observation_tier,
            observation_degraded_reason=session.observation_degraded_reason,
            observation_degraded_features=session.observation_degraded_features,
            last_observation_tier=session.last_observation_tier,
            last_observation_status=session.last_observation_status,
            last_observation_summary=session.last_observation_summary,
            last_observation_output_ref=session.last_observation_output_ref,
            last_observation_text_preview=session.last_observation_text_preview,
            last_observation_backend=session.last_observation_backend,
            last_observation_warnings=session.last_observation_warnings,
            last_observation_at=session.last_observation_at,
            pending_approval_summaries=tuple(item.summary for item in session.pending_approvals),
            pause_requested=session.pause_requested,
            stop_requested=session.stop_requested,
            kill_switch_engaged=session.kill_switch_engaged,
            last_control_reason=session.last_control_reason,
            last_error=session.last_error,
            created_at=session.created_at,
            updated_at=session.updated_at,
            ended_at=session.ended_at,
        )

    async def _publish_dashboard_local_task_session(
        self,
        session: LocalTaskSession | None = None,
    ) -> None:
        resolved_session = session if session is not None else await self.storage.load_active_local_task_session()
        dashboard_state = self._dashboard_local_task_session_state(resolved_session)
        self.dashboard.publish_event(
            {
                "stage": "dashboard.local_task_session_loaded",
                "local_task_session": dashboard_state.to_dict(),
            }
        )

    def _session_continuous_capture_directory(self, session_id: str) -> Path:
        return self.config.storage.logs_dir / self.config.observation_runtime.capture_directory_name / session_id

    def _session_observation_step_directory(self, session_id: str) -> Path:
        return self.config.storage.logs_dir / "observation_steps" / session_id

    @staticmethod
    def _requested_observation_tier(profile: UserSettingsProfile) -> str:
        return str(profile.observation.get("tier", "screenshot_on_demand") or "screenshot_on_demand")

    @staticmethod
    def _effective_observation_tier(requested_tier: str, snapshot: ModelHealthSnapshot) -> str:
        if requested_tier == "continuous_capture":
            return "continuous_capture" if snapshot.allow_continuous_capture else "screenshot_on_demand"
        if requested_tier == "vision_on_step":
            if snapshot.allow_vision_on_step:
                return "vision_on_step"
            if snapshot.allow_ocr_on_step:
                return "ocr_on_step"
            return "screenshot_on_demand"
        if requested_tier == "ocr_on_step":
            return "ocr_on_step" if snapshot.allow_ocr_on_step else "screenshot_on_demand"
        return "screenshot_on_demand"

    def _session_with_observation_governor(
        self,
        session: LocalTaskSession,
        *,
        profile: UserSettingsProfile,
        snapshot: ModelHealthSnapshot | None = None,
        updated_at: Any | None = None,
    ) -> LocalTaskSession:
        snapshot = snapshot or self.model_manager.health_snapshot()
        requested_tier = self._requested_observation_tier(profile)
        effective_tier = self._effective_observation_tier(requested_tier, snapshot)
        degraded_reason = ""
        if effective_tier != requested_tier and snapshot.governor_pressure_reasons:
            degraded_reason = ",".join(snapshot.governor_pressure_reasons)
        return replace(
            session,
            requested_observation_tier=requested_tier,
            effective_observation_tier=effective_tier,
            observation_degraded_reason=degraded_reason,
            observation_degraded_features=snapshot.governor_degraded_features,
            updated_at=updated_at if updated_at is not None else session.updated_at,
        )

    def _observation_step_plan_for_profile(
        self,
        session_id: str,
        profile: UserSettingsProfile,
    ) -> _ObservationStepPlan | None:
        observation = dict(profile.observation)
        tier = str(observation.get("tier", "screenshot_on_demand"))
        if tier not in {"ocr_on_step", "vision_on_step"}:
            return None
        if not bool(profile.desktop.get("enabled", False)):
            return None
        if tier == "ocr_on_step" and not bool(observation.get("ocr_on_step", False)):
            return None
        if tier == "vision_on_step" and not bool(observation.get("vision_on_step", False)):
            return None
        caps = self.config.observation_runtime
        warnings: list[str] = []
        requested_width = int(observation.get("capture_max_width", caps.default_capture_width) or caps.default_capture_width)
        max_width = max(64, min(requested_width, caps.max_capture_width))
        if max_width != requested_width:
            warnings.append("observation_step_width_capped")
        requested_height = int(
            observation.get("capture_max_height", caps.default_capture_height) or caps.default_capture_height
        )
        max_height = max(64, min(requested_height, caps.max_capture_height))
        if max_height != requested_height:
            warnings.append("observation_step_height_capped")
        region_of_interest, region_warnings = self._observation_step_region(
            str(observation.get("region_of_interest", "full_screen")),
            max_width=max_width,
            max_height=max_height,
        )
        warnings.extend(region_warnings)
        return _ObservationStepPlan(
            session_id=session_id,
            tier=tier,
            capture_directory=self._session_observation_step_directory(session_id),
            max_width=max_width,
            max_height=max_height,
            region_of_interest=region_of_interest,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _observation_step_region(
        raw_region: str,
        *,
        max_width: int,
        max_height: int,
    ) -> tuple[str, tuple[str, ...]]:
        normalized = raw_region.strip() or "full_screen"
        if normalized.lower() == "full_screen":
            return "full_screen", ()
        parts = [part.strip() for part in normalized.split(",")]
        if len(parts) != 4:
            return "full_screen", ("observation_step_region_reset",)
        try:
            left, top, width, height = (int(part) for part in parts)
        except ValueError:
            return "full_screen", ("observation_step_region_reset",)
        warnings: list[str] = []
        if left < 0 or top < 0:
            left = max(0, left)
            top = max(0, top)
            warnings.append("observation_step_region_clamped")
        if width <= 0 or height <= 0:
            return "full_screen", ("observation_step_region_reset",)
        bounded_width = min(width, max_width)
        bounded_height = min(height, max_height)
        if bounded_width != width or bounded_height != height:
            warnings.append("observation_step_region_clamped")
        return f"{left},{top},{bounded_width},{bounded_height}", tuple(dict.fromkeys(warnings))

    def _continuous_capture_plan_for_profile(
        self,
        session_id: str,
        profile: UserSettingsProfile,
    ) -> _ContinuousCapturePlan | None:
        observation = dict(profile.observation)
        if str(observation.get("tier", "screenshot_on_demand")) != "continuous_capture":
            return None
        desktop_enabled = bool(profile.desktop.get("enabled", False))
        enabled_capabilities = {
            str(item)
            for item in profile.desktop.get("enabled_capabilities", ())
            if str(item).strip()
        }
        if not desktop_enabled or "screenshot" not in enabled_capabilities:
            return None
        caps = self.config.observation_runtime
        warnings: list[str] = []

        requested_fps = float(observation.get("capture_fps", caps.default_capture_fps) or caps.default_capture_fps)
        fps = max(0.05, min(requested_fps, caps.max_capture_fps))
        if fps != requested_fps:
            warnings.append("continuous_capture_fps_capped")

        requested_width = int(observation.get("capture_max_width", caps.default_capture_width) or caps.default_capture_width)
        max_width = max(64, min(requested_width, caps.max_capture_width))
        if max_width != requested_width:
            warnings.append("continuous_capture_width_capped")

        requested_height = int(
            observation.get("capture_max_height", caps.default_capture_height) or caps.default_capture_height
        )
        max_height = max(64, min(requested_height, caps.max_capture_height))
        if max_height != requested_height:
            warnings.append("continuous_capture_height_capped")

        requested_history = int(
            observation.get("capture_frame_history", caps.default_frame_history) or caps.default_frame_history
        )
        frame_history = max(1, min(requested_history, caps.max_frame_history))
        if frame_history != requested_history:
            warnings.append("continuous_capture_history_capped")

        requested_diff_threshold = float(
            observation.get("capture_diff_threshold", caps.default_diff_threshold) or caps.default_diff_threshold
        )
        diff_threshold = max(caps.min_diff_threshold, min(requested_diff_threshold, caps.max_diff_threshold))
        if diff_threshold != requested_diff_threshold:
            warnings.append("continuous_capture_diff_threshold_capped")

        region_of_interest, region_warnings = self._continuous_capture_region(
            str(observation.get("region_of_interest", "full_screen")),
            max_width=max_width,
            max_height=max_height,
        )
        warnings.extend(region_warnings)

        return _ContinuousCapturePlan(
            session_id=session_id,
            capture_directory=self._session_continuous_capture_directory(session_id),
            fps=fps,
            interval_s=max(0.05, 1.0 / fps),
            max_width=max_width,
            max_height=max_height,
            frame_history=frame_history,
            diff_threshold=diff_threshold,
            region_of_interest=region_of_interest,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _continuous_capture_region(
        raw_region: str,
        *,
        max_width: int,
        max_height: int,
    ) -> tuple[str, tuple[str, ...]]:
        normalized = raw_region.strip() or "full_screen"
        if normalized.lower() == "full_screen":
            return "full_screen", ()
        parts = [part.strip() for part in normalized.split(",")]
        if len(parts) != 4:
            return "full_screen", ("continuous_capture_region_reset",)
        try:
            left, top, width, height = (int(part) for part in parts)
        except ValueError:
            return "full_screen", ("continuous_capture_region_reset",)
        warnings: list[str] = []
        if left < 0 or top < 0:
            left = max(0, left)
            top = max(0, top)
            warnings.append("continuous_capture_region_clamped")
        if width <= 0 or height <= 0:
            return "full_screen", ("continuous_capture_region_reset",)
        bounded_width = min(width, max_width)
        bounded_height = min(height, max_height)
        if bounded_width != width or bounded_height != height:
            warnings.append("continuous_capture_region_clamped")
        return f"{left},{top},{bounded_width},{bounded_height}", tuple(dict.fromkeys(warnings))

    @staticmethod
    def _byte_diff_ratio(previous: bytes, current: bytes) -> float:
        if not previous and not current:
            return 0.0
        compared = max(len(previous), len(current))
        if compared <= 0:
            return 0.0
        mismatches = abs(len(previous) - len(current))
        mismatches += sum(1 for prior, next_byte in zip(previous, current) if prior != next_byte)
        return mismatches / compared

    @staticmethod
    def _should_run_observation_step(
        request: CapabilityRequest,
        result: CapabilityExecutionResult,
    ) -> bool:
        if result.status != CapabilityExecutionStatus.SUCCEEDED:
            return False
        if request.capability_type == CapabilityType.APP_WINDOW_FOCUS:
            return True
        if request.capability_type == CapabilityType.DESKTOP_INPUT:
            return True
        if request.capability_type == CapabilityType.BROWSER_ACTION and request.browser_action is not None:
            return request.browser_action.action.strip().lower() == "navigate"
        return False

    def _observation_step_gate_reasons(self, tier: str) -> tuple[str, ...]:
        snapshot = self.model_manager.health_snapshot()
        if tier == "vision_on_step" and not snapshot.allow_vision_on_step:
            return snapshot.governor_pressure_reasons or ("hardware_governor_degraded",)
        if tier == "ocr_on_step" and not snapshot.allow_ocr_on_step:
            return snapshot.governor_pressure_reasons or ("hardware_governor_degraded",)
        return ()

    async def _save_local_task_observation_state(
        self,
        session: LocalTaskSession,
        *,
        tier: str,
        status: str,
        summary: str,
        output_ref: str = "",
        text_preview: str = "",
        backend: str = "",
        warnings: tuple[str, ...] = (),
    ) -> LocalTaskSession:
        updated = replace(
            session,
            last_observation_tier=tier,
            last_observation_status=status,
            last_observation_summary=summary,
            last_observation_output_ref=output_ref,
            last_observation_text_preview=text_preview[: self.OBSERVATION_TEXT_PREVIEW_CHARS],
            last_observation_backend=backend,
            last_observation_warnings=tuple(dict.fromkeys(str(item) for item in warnings if str(item).strip())),
            last_observation_at=utc_now(),
            updated_at=utc_now(),
        )
        await self.storage.save_local_task_session(updated)
        await self._publish_dashboard_local_task_session(updated)
        return updated

    async def _run_observation_step_if_enabled(
        self,
        session: LocalTaskSession,
        *,
        request: CapabilityRequest,
        result: CapabilityExecutionResult,
        profile: UserSettingsProfile,
    ) -> LocalTaskSession:
        session = self._session_with_observation_governor(
            session,
            profile=profile,
            updated_at=utc_now(),
        )
        plan = self._observation_step_plan_for_profile(session.session_id, profile)
        if plan is None or not self._should_run_observation_step(request, result):
            await self.storage.save_local_task_session(session)
            await self._publish_dashboard_local_task_session(session)
            return session

        gate_reasons = self._observation_step_gate_reasons(plan.tier)
        if gate_reasons:
            updated = await self._save_local_task_observation_state(
                session,
                tier=plan.tier,
                status="degraded",
                summary=(
                    f"{plan.tier} skipped after '{request.action_name()}' because runtime headroom is low."
                ),
                warnings=(*plan.warnings, *gate_reasons),
            )
            await self._emit_event(
                "observation.step_degraded",
                {
                    "session_id": updated.session_id,
                    "tier": plan.tier,
                    "status": updated.last_observation_status,
                    "reason": gate_reasons[0],
                    "reasons": list(gate_reasons),
                    "trigger_action": request.action_name(),
                },
            )
            return updated

        capture_path = plan.capture_directory / f"step_{stable_hash(request.request_id)[:12]}.png"
        try:
            await asyncio.to_thread(
                self.capability_executor.capture_observation_frame,
                capture_path,
                region=plan.region_of_interest,
                max_width=plan.max_width,
                max_height=plan.max_height,
                image_format="Png",
            )
            if plan.tier == "ocr_on_step":
                bounded_text, backend_name, ocr_warnings, _text_length, _truncated = await asyncio.to_thread(
                    self.capability_executor.extract_bounded_ocr_text,
                    capture_path,
                    region="full_image",
                    languages=(),
                )
                updated = await self._save_local_task_observation_state(
                    session,
                    tier=plan.tier,
                    status="succeeded",
                    summary=f"OCR-on-step captured '{capture_path.name}' after '{request.action_name()}'.",
                    output_ref=str(capture_path),
                    text_preview=bounded_text,
                    backend=backend_name,
                    warnings=(*plan.warnings, *ocr_warnings),
                )
                await self._emit_event(
                    "observation.step_completed",
                    {
                        "session_id": updated.session_id,
                        "tier": plan.tier,
                        "status": updated.last_observation_status,
                        "trigger_action": request.action_name(),
                        "output_ref": updated.last_observation_output_ref,
                        "backend": updated.last_observation_backend,
                        "warnings": list(updated.last_observation_warnings),
                    },
                )
                return updated

            bounded_text, backend_name, ocr_warnings, _text_length, _truncated = await asyncio.to_thread(
                self.capability_executor.extract_bounded_ocr_text,
                capture_path,
                region="full_image",
                languages=(),
            )
            inspection = await self.model_manager.inspect_image(
                capture_path,
                request_text="Summarize the visible UI state for the bounded local task step.",
                extracted_text=bounded_text,
                role=ModelRole.VISION,
            )
            await self._publish_dashboard_model_registry_view()
            if inspection.status == "inspected":
                updated = await self._save_local_task_observation_state(
                    session,
                    tier=plan.tier,
                    status="succeeded",
                    summary=inspection.summary or f"Vision-on-step inspected '{capture_path.name}'.",
                    output_ref=str(capture_path),
                    text_preview=inspection.extracted_text or bounded_text,
                    backend=inspection.inspection_backend,
                    warnings=(*plan.warnings, *ocr_warnings, *inspection.warnings),
                )
                await self._emit_event(
                    "observation.step_completed",
                    {
                        "session_id": updated.session_id,
                        "tier": plan.tier,
                        "status": updated.last_observation_status,
                        "trigger_action": request.action_name(),
                        "output_ref": updated.last_observation_output_ref,
                        "backend": updated.last_observation_backend,
                        "warnings": list(updated.last_observation_warnings),
                    },
                )
                return updated
            route_reason = inspection.degraded_reason or "vision_route_unavailable"
            route_warning = f"vision_route_{route_reason}"
            route_summary = (
                "Vision-on-step fell back to CPU OCR because the vision route is unavailable "
                f"({route_reason})."
            )
            updated = await self._save_local_task_observation_state(
                session,
                tier=plan.tier,
                status="degraded",
                summary=route_summary,
                output_ref=str(capture_path),
                text_preview=bounded_text,
                backend=backend_name,
                warnings=(*plan.warnings, route_warning, *ocr_warnings, *inspection.warnings),
            )
            await self._emit_event(
                "observation.step_degraded",
                {
                    "session_id": updated.session_id,
                    "tier": plan.tier,
                    "status": updated.last_observation_status,
                    "trigger_action": request.action_name(),
                    "output_ref": updated.last_observation_output_ref,
                    "backend": updated.last_observation_backend,
                    "warnings": list(updated.last_observation_warnings),
                },
            )
            return updated
        except Exception as exc:
            updated = await self._save_local_task_observation_state(
                session,
                tier=plan.tier,
                status="degraded",
                summary=f"{plan.tier} failed after '{request.action_name()}'.",
                warnings=(*plan.warnings, "observation_step_failed"),
            )
            await self._emit_event(
                "observation.step_degraded",
                {
                    "session_id": updated.session_id,
                    "tier": plan.tier,
                    "status": updated.last_observation_status,
                    "reason": "observation_step_failed",
                    "detail": str(exc),
                    "trigger_action": request.action_name(),
                },
            )
            return updated

    async def _stop_continuous_capture_task(
        self,
        *,
        reason: str,
        session_id: str = "",
    ) -> None:
        target_session_id = session_id or self._continuous_capture_session_id
        active_task = self._continuous_capture_task
        if active_task is not None:
            self._continuous_capture_task = None
            self._continuous_capture_session_id = ""
            active_task.cancel()
            try:
                await active_task
            except asyncio.CancelledError:
                pass
        if not target_session_id:
            return
        session = await self.storage.load_local_task_session(target_session_id)
        if session is None or not session.continuous_capture_active:
            return
        updated = replace(
            session,
            continuous_capture_active=False,
            updated_at=utc_now(),
        )
        await self.storage.save_local_task_session(updated)
        await self._publish_dashboard_local_task_session(updated)
        await self._emit_event(
            "observation.continuous_capture_stopped",
            {
                "session_id": target_session_id,
                "reason": reason,
                "captured_frames": updated.continuous_capture_frame_count,
                "retained_frames": updated.continuous_capture_retained_frame_count,
            },
        )

    async def _sync_continuous_capture_for_session(
        self,
        session: LocalTaskSession,
        *,
        profile: UserSettingsProfile,
        reason: str,
    ) -> LocalTaskSession:
        snapshot = self.model_manager.health_snapshot()
        session = self._session_with_observation_governor(
            session,
            profile=profile,
            snapshot=snapshot,
            updated_at=utc_now(),
        )
        plan = self._continuous_capture_plan_for_profile(session.session_id, profile)
        if (
            self._requested_observation_tier(profile) == "continuous_capture"
            and session.effective_observation_tier != "continuous_capture"
        ):
            await self._stop_continuous_capture_task(
                reason=f"hardware_governor:{session.observation_degraded_reason or 'degraded'}",
                session_id=session.session_id,
            )
            await self.storage.save_local_task_session(session)
            await self._publish_dashboard_local_task_session(session)
            await self._emit_event(
                "observation.continuous_capture_degraded",
                {
                    "session_id": session.session_id,
                    "requested_tier": session.requested_observation_tier,
                    "effective_tier": session.effective_observation_tier,
                    "reason": session.observation_degraded_reason or "hardware_governor_degraded",
                    "degraded_features": list(session.observation_degraded_features),
                    "trigger": reason,
                },
            )
            return session
        if plan is None:
            await self._stop_continuous_capture_task(reason=reason, session_id=session.session_id)
            await self.storage.save_local_task_session(session)
            await self._publish_dashboard_local_task_session(session)
            return await self.storage.load_local_task_session(session.session_id) or session
        if self._continuous_capture_task is not None and self._continuous_capture_session_id != session.session_id:
            await self._stop_continuous_capture_task(reason="continuous_capture_replaced")
        if self._continuous_capture_task is not None and self._continuous_capture_session_id == session.session_id:
            if (
                session.continuous_capture_active
                and session.continuous_capture_region == plan.region_of_interest
                and session.continuous_capture_fps == plan.fps
                and session.continuous_capture_max_width == plan.max_width
                and session.continuous_capture_max_height == plan.max_height
                and session.continuous_capture_warnings == plan.warnings
            ):
                return session
            await self._stop_continuous_capture_task(
                reason="continuous_capture_reconfigured",
                session_id=session.session_id,
            )
        plan.capture_directory.mkdir(parents=True, exist_ok=True)
        updated = replace(
            session,
            continuous_capture_active=True,
            continuous_capture_directory=str(plan.capture_directory),
            continuous_capture_region=plan.region_of_interest,
            continuous_capture_fps=plan.fps,
            continuous_capture_max_width=plan.max_width,
            continuous_capture_max_height=plan.max_height,
            continuous_capture_warnings=plan.warnings,
            updated_at=utc_now(),
        )
        await self.storage.save_local_task_session(updated)
        await self._publish_dashboard_local_task_session(updated)
        await self._emit_event(
            "observation.continuous_capture_started",
            {
                "session_id": updated.session_id,
                "capture_directory": updated.continuous_capture_directory,
                "fps": updated.continuous_capture_fps,
                "max_width": updated.continuous_capture_max_width,
                "max_height": updated.continuous_capture_max_height,
                "frame_history": plan.frame_history,
                "diff_threshold": plan.diff_threshold,
                "region_of_interest": updated.continuous_capture_region,
                "reason": reason,
                "warnings": list(updated.continuous_capture_warnings),
            },
        )
        self._continuous_capture_session_id = updated.session_id
        self._continuous_capture_task = asyncio.create_task(
            self._continuous_capture_loop(plan),
            name=f"continuous-capture:{updated.session_id}",
        )
        return updated

    async def _continuous_capture_loop(self, plan: _ContinuousCapturePlan) -> None:
        retained_paths = sorted(plan.capture_directory.glob("frame_*.jpg"))
        while len(retained_paths) > plan.frame_history:
            obsolete = retained_paths.pop(0)
            obsolete.unlink(missing_ok=True)
        previous_bytes = retained_paths[-1].read_bytes() if retained_paths else None
        session = await self.storage.load_local_task_session(plan.session_id)
        frame_index = 0 if session is None else session.continuous_capture_frame_count
        try:
            while not self._shutdown_requested:
                session = await self.storage.load_local_task_session(plan.session_id)
                if session is None or session.status != LocalTaskSessionState.RUNNING or session.kill_switch_engaged:
                    break
                profile = await self.storage.load_user_settings_profile(session.profile_name)
                if profile is None:
                    profile = self.dashboard.app_state_snapshot().user_settings
                session = self._session_with_observation_governor(
                    session,
                    profile=profile,
                    updated_at=utc_now(),
                )
                if (
                    session.requested_observation_tier == "continuous_capture"
                    and session.effective_observation_tier != "continuous_capture"
                ):
                    degraded = replace(
                        session,
                        continuous_capture_active=False,
                        last_control_reason="continuous_capture_degraded",
                        updated_at=utc_now(),
                    )
                    await self.storage.save_local_task_session(degraded)
                    await self._publish_dashboard_local_task_session(degraded)
                    await self._emit_event(
                        "observation.continuous_capture_degraded",
                        {
                            "session_id": degraded.session_id,
                            "requested_tier": degraded.requested_observation_tier,
                            "effective_tier": degraded.effective_observation_tier,
                            "reason": degraded.observation_degraded_reason or "hardware_governor_degraded",
                            "degraded_features": list(degraded.observation_degraded_features),
                            "captured_frames": degraded.continuous_capture_frame_count,
                        },
                    )
                    break
                frame_index += 1
                frame_path = plan.capture_directory / f"frame_{frame_index:04d}.jpg"
                await asyncio.to_thread(
                    self.capability_executor.capture_continuous_frame,
                    frame_path,
                    region=plan.region_of_interest,
                    max_width=plan.max_width,
                    max_height=plan.max_height,
                )
                frame_bytes = await asyncio.to_thread(frame_path.read_bytes)
                diff_ratio = 1.0 if previous_bytes is None else self._byte_diff_ratio(previous_bytes, frame_bytes)
                previous_bytes = frame_bytes
                retained = not retained_paths or diff_ratio >= plan.diff_threshold
                if retained:
                    retained_paths.append(frame_path)
                    while len(retained_paths) > plan.frame_history:
                        obsolete = retained_paths.pop(0)
                        obsolete.unlink(missing_ok=True)
                else:
                    frame_path.unlink(missing_ok=True)
                updated = replace(
                    session,
                    continuous_capture_active=True,
                    continuous_capture_directory=str(plan.capture_directory),
                    continuous_capture_frame_count=frame_index,
                    continuous_capture_retained_frame_count=len(retained_paths),
                    continuous_capture_last_frame_path=(
                        str(retained_paths[-1]) if retained_paths else session.continuous_capture_last_frame_path
                    ),
                    continuous_capture_region=plan.region_of_interest,
                    continuous_capture_fps=plan.fps,
                    continuous_capture_max_width=plan.max_width,
                    continuous_capture_max_height=plan.max_height,
                    continuous_capture_last_diff_ratio=diff_ratio,
                    continuous_capture_warnings=plan.warnings,
                    continuous_capture_last_capture_at=utc_now(),
                    updated_at=utc_now(),
                )
                await self.storage.save_local_task_session(updated)
                await self._publish_dashboard_local_task_session(updated)
                if retained:
                    await self._emit_event(
                        "observation.continuous_capture_frame",
                        {
                            "session_id": plan.session_id,
                            "frame_path": str(frame_path),
                            "captured_frames": frame_index,
                            "retained_frames": len(retained_paths),
                            "diff_ratio": diff_ratio,
                            "diff_threshold": plan.diff_threshold,
                        },
                    )
                await asyncio.sleep(plan.interval_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            session = await self.storage.load_local_task_session(plan.session_id)
            if session is not None and session.status == LocalTaskSessionState.RUNNING:
                paused = replace(
                    session,
                    status=LocalTaskSessionState.PAUSED,
                    continuous_capture_active=False,
                    last_control_reason="continuous_capture_failed",
                    last_error=str(exc),
                    updated_at=utc_now(),
                )
                self._local_task_emergency_stop.set()
                await self.storage.save_local_task_session(paused)
                await self._emit_event(
                    "local_task_session.paused",
                    {
                        "session_id": paused.session_id,
                        "status": paused.status.value,
                        "reason": paused.last_control_reason,
                    },
                )
                await self._emit_event(
                    "observation.continuous_capture_failed",
                    {
                        "session_id": paused.session_id,
                        "error": str(exc),
                    },
                )
                await self._publish_dashboard_local_task_session(paused)
        finally:
            if self._continuous_capture_session_id == plan.session_id:
                self._continuous_capture_session_id = ""
                self._continuous_capture_task = None

    async def _recover_local_task_session(self) -> None:
        session = await self.storage.load_active_local_task_session()
        if session is None:
            await self._stop_continuous_capture_task(reason="session_not_active")
            await self._publish_dashboard_local_task_session(None)
            return
        if session.status == LocalTaskSessionState.RUNNING:
            session = replace(
                session,
                status=LocalTaskSessionState.PAUSED,
                continuous_capture_active=False,
                pause_requested=False,
                last_control_reason="recovered_after_restart",
                updated_at=utc_now(),
            )
            await self.storage.save_local_task_session(session)
            await self._emit_event(
                "local_task_session.recovered",
                {
                    "session_id": session.session_id,
                    "status": session.status.value,
                    "reason": session.last_control_reason,
                },
            )
        elif session.status in self._terminal_local_task_session_states():
            await self.storage.save_active_local_task_session_id("")
        await self._stop_continuous_capture_task(reason="session_recovered", session_id=session.session_id)
        await self._publish_dashboard_local_task_session(session)

    async def start_local_task_session(
        self,
        label: str,
        *,
        active_profile: UserSettingsProfile | None = None,
    ) -> LocalTaskSession:
        """Start one explicit local task execution session."""
        normalized_label = label.strip() or "Local task session"
        profile = active_profile or (
            self.dashboard.app_state_snapshot().user_settings if self._started else self._default_user_settings_profile()
        )
        existing = await self.storage.load_active_local_task_session()
        if existing is not None and existing.status not in self._terminal_local_task_session_states():
            raise RuntimeError(
                f"Local task session '{existing.session_id}' is already {existing.status.value}; stop it before starting a new one."
            )
        now = utc_now()
        session = LocalTaskSession(
            session_id=self._local_task_session_id(label=normalized_label),
            label=normalized_label,
            profile_name=profile.profile_name,
            status=LocalTaskSessionState.RUNNING,
            control_mode="local_task",
            requested_observation_tier=self._requested_observation_tier(profile),
            effective_observation_tier=self._requested_observation_tier(profile),
            last_control_reason="started",
            created_at=now,
            updated_at=now,
        )
        session = self._session_with_observation_governor(session, profile=profile)
        self._local_task_emergency_stop.clear()
        await self.storage.save_local_task_session(session)
        await self.storage.save_active_local_task_session_id(session.session_id)
        await self._emit_event(
            "local_task_session.started",
            {
                "session_id": session.session_id,
                "label": session.label,
                "profile_name": session.profile_name,
                "status": session.status.value,
            },
        )
        session = await self._sync_continuous_capture_for_session(
            session,
            profile=profile,
            reason="session_started",
        )
        await self._publish_dashboard_local_task_session(session)
        return session

    async def pause_local_task_session(self, session_id: str, *, reason: str) -> bool:
        """Pause one explicit local task session."""
        session = await self.storage.load_local_task_session(session_id)
        if session is None or session.status != LocalTaskSessionState.RUNNING:
            return False
        paused = replace(
            session,
            status=LocalTaskSessionState.PAUSED,
            continuous_capture_active=False,
            pause_requested=False,
            last_control_reason=reason,
            updated_at=utc_now(),
        )
        self._local_task_emergency_stop.set()
        await self.storage.save_local_task_session(paused)
        await self._stop_continuous_capture_task(reason=reason, session_id=session_id)
        await self._emit_event(
            "local_task_session.paused",
            {
                "session_id": paused.session_id,
                "status": paused.status.value,
                "reason": reason,
            },
        )
        await self._publish_dashboard_local_task_session(paused)
        return True

    async def resume_local_task_session(self, session_id: str, *, reason: str) -> bool:
        """Resume one paused local task session."""
        session = await self.storage.load_local_task_session(session_id)
        if session is None or session.status != LocalTaskSessionState.PAUSED or session.kill_switch_engaged:
            return False
        resumed = replace(
            session,
            status=LocalTaskSessionState.RUNNING,
            continuous_capture_active=False,
            stop_requested=False,
            last_control_reason=reason,
            updated_at=utc_now(),
            ended_at=None,
        )
        self._local_task_emergency_stop.clear()
        await self.storage.save_local_task_session(resumed)
        await self.storage.save_active_local_task_session_id(resumed.session_id)
        profile = await self.storage.load_user_settings_profile(resumed.profile_name)
        if profile is None:
            profile = self.dashboard.app_state_snapshot().user_settings
        resumed = self._session_with_observation_governor(resumed, profile=profile)
        resumed = await self._sync_continuous_capture_for_session(
            resumed,
            profile=profile,
            reason="session_resumed",
        )
        await self._emit_event(
            "local_task_session.resumed",
            {
                "session_id": resumed.session_id,
                "status": resumed.status.value,
                "reason": reason,
            },
        )
        await self._publish_dashboard_local_task_session(resumed)
        return True

    async def stop_local_task_session(self, session_id: str, *, reason: str) -> bool:
        """Stop one local task session and clear it as the active executor boundary."""
        session = await self.storage.load_local_task_session(session_id)
        if session is None or session.status in self._terminal_local_task_session_states():
            return False
        stopped = replace(
            session,
            status=LocalTaskSessionState.STOPPED,
            continuous_capture_active=False,
            stop_requested=True,
            pause_requested=False,
            last_control_reason=reason,
            updated_at=utc_now(),
            ended_at=utc_now(),
        )
        self._local_task_emergency_stop.set()
        await self.storage.save_local_task_session(stopped)
        await self.storage.save_active_local_task_session_id("")
        await self._stop_continuous_capture_task(reason=reason, session_id=session_id)
        await self._emit_event(
            "local_task_session.stopped",
            {
                "session_id": stopped.session_id,
                "status": stopped.status.value,
                "reason": reason,
            },
        )
        await self._publish_dashboard_local_task_session(stopped)
        return True

    async def kill_local_task_session(self, session_id: str, *, reason: str) -> bool:
        """Trigger the kill-switch for one local task session."""
        session = await self.storage.load_local_task_session(session_id)
        if session is None or session.status == LocalTaskSessionState.KILLED:
            return False
        killed = replace(
            session,
            status=LocalTaskSessionState.KILLED,
            continuous_capture_active=False,
            stop_requested=True,
            pause_requested=False,
            kill_switch_engaged=True,
            last_control_reason=reason,
            updated_at=utc_now(),
            ended_at=utc_now(),
        )
        self._local_task_emergency_stop.set()
        await self.storage.save_local_task_session(killed)
        await self.storage.save_active_local_task_session_id("")
        await self._stop_continuous_capture_task(reason=reason, session_id=session_id)
        await self._emit_event(
            "local_task_session.killed",
            {
                "session_id": killed.session_id,
                "status": killed.status.value,
                "reason": reason,
            },
        )
        await self._publish_dashboard_local_task_session(killed)
        return True

    async def _pause_active_local_task_session(self, *, reason: str) -> None:
        session = await self.storage.load_active_local_task_session()
        if session is None or session.status != LocalTaskSessionState.RUNNING:
            return
        await self.pause_local_task_session(session.session_id, reason=reason)

    def _save_dashboard_settings_request(self, profile: UserSettingsProfile) -> None:
        if self._loop is None:
            raise RuntimeError("Orchestrator event loop is not available for dashboard settings persistence.")
        self._submit_dashboard_coroutine(
            self._save_dashboard_settings_and_refresh(profile),
        )

    def _handle_dashboard_action_request(self, action: str, payload: dict[str, Any]) -> None:
        if self._loop is None:
            raise RuntimeError("Orchestrator event loop is not available for dashboard actions.")
        self._submit_dashboard_coroutine(
            self._run_dashboard_action(action=action, payload=payload),
        )

    async def _save_dashboard_settings_and_refresh(self, profile: UserSettingsProfile) -> None:
        await self.storage.save_user_settings_profile(profile)
        await self._apply_runtime_settings_profile(profile)
        await self._publish_dashboard_settings_profiles(active_profile_name=profile.profile_name)
        await self._publish_dashboard_readiness_report(active_profile=profile)
        await self._publish_dashboard_capability_registry_view(active_profile=profile)
        await self._publish_dashboard_notice(
            f"Saved settings profile '{profile.profile_name}'.",
            severity="info",
        )

    async def _run_dashboard_action(self, *, action: str, payload: dict[str, Any]) -> None:
        try:
            if action == "settings.refresh_profiles":
                await self._publish_dashboard_settings_profiles(
                    active_profile_name=str(payload.get("active_profile_name", "default") or "default")
                )
            elif action == "settings.load_profile":
                profile_name = str(payload.get("profile_name", "")).strip() or "default"
                profile = await self.storage.load_user_settings_profile(profile_name)
                if profile is None:
                    await self._publish_dashboard_notice(
                        f"Unknown settings profile '{profile_name}'.",
                        severity="warning",
                    )
                else:
                    await self._apply_runtime_settings_profile(profile)
                    await self._publish_dashboard_settings_profiles(active_profile_name=profile.profile_name)
                    await self._publish_dashboard_readiness_report(active_profile=profile)
                    await self._publish_dashboard_capability_registry_view(active_profile=profile)
                    await self._publish_dashboard_notice(
                        f"Loaded settings profile '{profile.profile_name}'.",
                        severity="info",
                    )
            elif action == "settings.import_profile":
                import_path = Path(str(payload.get("path", "")).strip())
                profile = await self.storage.import_user_settings_profile(import_path)
                await self._apply_runtime_settings_profile(profile)
                await self._publish_dashboard_settings_profiles(active_profile_name=profile.profile_name)
                await self._publish_dashboard_readiness_report(active_profile=profile)
                await self._publish_dashboard_capability_registry_view(active_profile=profile)
                await self._publish_dashboard_notice(
                    f"Imported settings profile '{profile.profile_name}' from '{import_path}'.",
                    severity="info",
                )
            elif action == "settings.export_profile":
                export_path = Path(str(payload.get("path", "")).strip())
                profile_name = str(payload.get("profile_name", "")).strip() or "default"
                await self.storage.export_user_settings_profile(profile_name, export_path)
                await self._publish_dashboard_notice(
                    f"Exported settings profile '{profile_name}' to '{export_path}'.",
                    severity="info",
                )
            elif action.startswith("model."):
                await self._run_dashboard_model_action(action=action, payload=payload)
            elif action == "history.refresh":
                await self._publish_dashboard_task_history()
            elif action == "history.inspect_task":
                await self._publish_dashboard_task_detail(str(payload.get("task_id", "")).strip())
            elif action == "history.export_task_debug":
                task_id = str(payload.get("task_id", "")).strip()
                export_path = Path(str(payload.get("path", "")).strip())
                actual_path = await self.storage.export_trace_debug_view(task_id, export_path)
                await self._publish_dashboard_task_detail(task_id, trace_debug_export_path=actual_path)
                await self._publish_dashboard_notice(
                    f"Exported trace debug for '{task_id}' to '{actual_path}'.",
                    severity="info",
                )
            elif action == "knowledge.refresh":
                await self._publish_dashboard_knowledge_library()
            elif action == "knowledge.ingest_text":
                source_ref = str(payload.get("source_ref", "")).strip()
                title = str(payload.get("title", "")).strip() or source_ref
                content = str(payload.get("content", "")).strip()
                await self.storage.ingest_document(
                    source_ref=source_ref,
                    title=title,
                    content=content,
                    metadata={
                        "corpus_origin": "user_local",
                        "corpus_tier": "user",
                        "archived": False,
                    },
                    embed_document=self.model_manager.embed_document,
                    embedding_model_name=self.config.preflight.backends.embedding_model,
                )
                await self._publish_dashboard_knowledge_library()
                await self._publish_dashboard_notice(
                    f"Ingested knowledge source '{source_ref}'.",
                    severity="info",
                )
            elif action == "knowledge.archive_source":
                source_ref = str(payload.get("source_ref", "")).strip()
                document = await self.storage.set_document_archived(source_ref, archived=True)
                await self._publish_dashboard_knowledge_library()
                await self._publish_dashboard_notice(
                    f"Archived knowledge source '{document.source_ref if document else source_ref}'.",
                    severity="info" if document is not None else "warning",
                )
            elif action == "knowledge.unarchive_source":
                source_ref = str(payload.get("source_ref", "")).strip()
                document = await self.storage.set_document_archived(source_ref, archived=False)
                await self._publish_dashboard_knowledge_library()
                await self._publish_dashboard_notice(
                    f"Restored knowledge source '{document.source_ref if document else source_ref}'.",
                    severity="info" if document is not None else "warning",
                )
            elif action == "knowledge.rebuild_source":
                source_ref = str(payload.get("source_ref", "")).strip()
                document = await self.storage.rebuild_document(
                    source_ref,
                    embed_document=self.model_manager.embed_document,
                    embedding_model_name=self.config.preflight.backends.embedding_model,
                )
                await self._publish_dashboard_knowledge_library()
                await self._publish_dashboard_notice(
                    f"Rebuilt knowledge source '{document.source_ref}'.",
                    severity="info",
                )
            elif action == "knowledge.remove_source":
                source_ref = str(payload.get("source_ref", "")).strip()
                removed = await self.storage.remove_document(source_ref)
                await self._publish_dashboard_knowledge_library()
                await self._publish_dashboard_notice(
                    f"Removed knowledge source '{source_ref}'." if removed else f"Unknown source '{source_ref}'.",
                    severity="info" if removed else "warning",
                )
            elif action == "audio.transcribe_file":
                audio_path = Path(str(payload.get("path", "")).strip())
                if not str(audio_path).strip():
                    await self._publish_dashboard_notice(
                        "Provide a local .wav path before transcribing audio.",
                        severity="warning",
                    )
                else:
                    transcription = await self.model_manager.transcribe_audio(audio_path)
                    await self._publish_dashboard_audio_input(transcription)
                    await self._publish_dashboard_notice(
                        (
                            f"Transcribed audio from '{audio_path}'."
                            if transcription.status == "transcribed"
                            else f"Audio transcription returned '{transcription.status}' for '{audio_path}'."
                        ),
                        severity="info" if transcription.status == "transcribed" else "warning",
                    )
            elif action == "audio.use_transcript_as_question":
                audio_input = self.dashboard.app_state_snapshot().audio_input
                question_text = audio_input.normalized_question or audio_input.transcript_text
                if not question_text:
                    await self._publish_dashboard_notice(
                        "Transcribe a local audio file before importing a voice question.",
                        severity="warning",
                    )
                else:
                    imported = replace(audio_input, imported_into_question=True)
                    await self._publish_dashboard_audio_transcript_import(imported, question_text)
                    await self._publish_dashboard_notice(
                        "Loaded the current audio transcript into the question box.",
                        severity="info",
                    )
            elif action == "audio.synthesize_text":
                output_path = Path(str(payload.get("path", "")).strip())
                source_text = str(payload.get("text", "")).strip()
                if not source_text or not str(output_path).strip():
                    await self._publish_dashboard_notice(
                        "Provide text and a local .wav output path before synthesizing speech.",
                        severity="warning",
                    )
                else:
                    audio_output = await self.model_manager.synthesize_text(source_text, output_path=output_path)
                    await self._publish_dashboard_audio_output(audio_output)
                    await self._publish_dashboard_notice(
                        (
                            f"Synthesized voice output to '{audio_output.target_path}'."
                            if audio_output.status == "synthesized"
                            else f"Voice synthesis returned '{audio_output.status}' for '{output_path}'."
                        ),
                        severity="info" if audio_output.status == "synthesized" else "warning",
                    )
            elif action == "audio.speak_answer":
                output_path = Path(str(payload.get("path", "")).strip())
                answer_text = self.dashboard.app_state_snapshot().active_task.answer_text.strip()
                if not answer_text:
                    await self._publish_dashboard_notice(
                        "Run a task with a completed answer before synthesizing voice output for it.",
                        severity="warning",
                    )
                elif not str(output_path).strip():
                    await self._publish_dashboard_notice(
                        "Provide a local .wav output path before speaking the current answer.",
                        severity="warning",
                    )
                else:
                    audio_output = await self.model_manager.synthesize_text(answer_text, output_path=output_path)
                    await self._publish_dashboard_audio_output(audio_output)
                    await self._publish_dashboard_notice(
                        (
                            f"Synthesized the current answer to '{audio_output.target_path}'."
                            if audio_output.status == "synthesized"
                            else f"Answer synthesis returned '{audio_output.status}' for '{output_path}'."
                        ),
                        severity="info" if audio_output.status == "synthesized" else "warning",
                    )
            elif action == "audio.clear":
                self.dashboard.publish_event({"stage": "dashboard.audio_input_cleared"})
                await self._publish_dashboard_notice(
                    "Cleared the current audio transcript.",
                    severity="info",
                )
            elif action == "audio.clear_output":
                self.dashboard.publish_event({"stage": "dashboard.audio_output_cleared"})
                await self._publish_dashboard_notice(
                    "Cleared the current voice output state.",
                    severity="info",
                )
            elif action == "translation.translate_text":
                source_text = str(payload.get("text", "")).strip()
                source_language = str(payload.get("source_language", "")).strip() or self.config.translation.default_source_language
                target_language = str(payload.get("target_language", "")).strip() or self.config.translation.default_target_language
                if not source_text:
                    await self._publish_dashboard_notice(
                        "Provide text before requesting translation.",
                        severity="warning",
                    )
                else:
                    translation_output = await self.model_manager.translate_text(
                        source_text,
                        source_language=source_language,
                        target_language=target_language,
                        source_scope="free_text",
                    )
                    await self._publish_dashboard_translation_output(translation_output)
                    await self._publish_dashboard_notice(
                        (
                            f"Translated text to {translation_output.target_language or target_language}."
                            if translation_output.status == "translated"
                            else f"Translation returned '{translation_output.status}'."
                        ),
                        severity="info" if translation_output.status == "translated" else "warning",
                    )
            elif action == "translation.translate_answer":
                answer_text = self.dashboard.app_state_snapshot().active_task.answer_text.strip()
                source_language = str(payload.get("source_language", "")).strip() or self.config.translation.default_source_language
                target_language = str(payload.get("target_language", "")).strip() or self.config.translation.default_target_language
                if not answer_text:
                    await self._publish_dashboard_notice(
                        "Run a task with a completed answer before translating the answer text.",
                        severity="warning",
                    )
                else:
                    translation_output = await self.model_manager.translate_text(
                        answer_text,
                        source_language=source_language,
                        target_language=target_language,
                        source_scope="answer",
                    )
                    await self._publish_dashboard_translation_output(translation_output)
                    await self._publish_dashboard_notice(
                        (
                            f"Translated the current answer to {translation_output.target_language or target_language}."
                            if translation_output.status == "translated"
                            else f"Answer translation returned '{translation_output.status}'."
                        ),
                        severity="info" if translation_output.status == "translated" else "warning",
                    )
            elif action == "translation.use_as_question":
                translation_output = self.dashboard.app_state_snapshot().translation_output
                if not translation_output.translated_text:
                    await self._publish_dashboard_notice(
                        "Translate text before importing it into the question box.",
                        severity="warning",
                    )
                else:
                    imported = replace(translation_output, imported_into_question=True)
                    await self._publish_dashboard_translation_import(imported, imported.translated_text)
                    await self._publish_dashboard_notice(
                        "Loaded the current translation into the question box.",
                        severity="info",
                    )
            elif action == "translation.clear":
                self.dashboard.publish_event({"stage": "dashboard.translation_output_cleared"})
                await self._publish_dashboard_notice(
                    "Cleared the current translation result.",
                    severity="info",
                )
            elif action == "code.analyze_file":
                source_path = Path(str(payload.get("path", "")).strip())
                request_text = str(payload.get("request_text", "")).strip() or self.config.code_specialist.default_request
                if not str(source_path).strip():
                    await self._publish_dashboard_notice(
                        "Provide a local source path before analyzing code.",
                        severity="warning",
                    )
                else:
                    code_output = await self.model_manager.analyze_code_file(
                        source_path,
                        request_text=request_text,
                    )
                    await self._publish_dashboard_code_output(code_output)
                    await self._publish_dashboard_notice(
                        (
                            f"Analyzed code file '{source_path}'."
                            if code_output.status == "analyzed"
                            else f"Code analysis returned '{code_output.status}' for '{source_path}'."
                        ),
                        severity="info" if code_output.status == "analyzed" else "warning",
                    )
            elif action == "code.analyze_snippet":
                source_text = str(payload.get("text", "")).strip()
                request_text = str(payload.get("request_text", "")).strip() or self.config.code_specialist.default_request
                if not source_text:
                    await self._publish_dashboard_notice(
                        "Paste or provide code text before running the code specialist.",
                        severity="warning",
                    )
                else:
                    code_output = await self.model_manager.analyze_code(
                        text=source_text,
                        request_text=request_text,
                        source_scope="snippet",
                    )
                    await self._publish_dashboard_code_output(code_output)
                    await self._publish_dashboard_notice(
                        (
                            "Analyzed the current code snippet."
                            if code_output.status == "analyzed"
                            else f"Code analysis returned '{code_output.status}'."
                        ),
                        severity="info" if code_output.status == "analyzed" else "warning",
                    )
            elif action == "code.clear":
                self.dashboard.publish_event({"stage": "dashboard.code_output_cleared"})
                await self._publish_dashboard_notice(
                    "Cleared the current code-specialist result.",
                    severity="info",
                )
            elif action == "coding.run_task":
                app_state = self.dashboard.app_state_snapshot()
                request_payload = {
                    "task_type": str(payload.get("task_type", "code_review")).strip() or "code_review",
                    "prompt": str(payload.get("prompt", "")).strip(),
                    "language": str(
                        payload.get(
                            "language",
                            app_state.user_settings.coding.get("default_language", self.config.coding_mode.default_language),
                        )
                    ).strip()
                    or self.config.coding_mode.default_language,
                    "framework": str(payload.get("framework", "")).strip(),
                    "source_scope": str(payload.get("source_scope", "snippet")).strip() or "snippet",
                    "source_path": str(payload.get("path", "")).strip(),
                    "source_text": str(payload.get("text", "")),
                    "tests_text": str(payload.get("tests_text", "")),
                    "idle_practice": False,
                    "metadata": {
                        "source": "dashboard",
                    },
                }
                if not request_payload["prompt"] and not request_payload["source_path"] and not request_payload["source_text"]:
                    await self._publish_dashboard_notice(
                        "Provide a coding prompt, source path, or code text before running Coding Mode.",
                        severity="warning",
                    )
                else:
                    coding_output = await self.coding_mode.run_task(
                        CodingTaskRequest.from_dict(request_payload),
                        user_settings=app_state.user_settings,
                        event_callback=self._emit_event,
                    )
                    await self._publish_dashboard_coding_output(coding_output)
                    await self._publish_dashboard_coding_patterns()
                    await self._publish_dashboard_recent_coding_activity()
                    await self._publish_dashboard_notice(
                        (
                            "Completed the coding task."
                            if coding_output.status == "completed"
                            else f"Coding task returned '{coding_output.status}'."
                        ),
                        severity="info" if coding_output.status == "completed" else "warning",
                    )
            elif action == "coding.practice_once":
                app_state = self.dashboard.app_state_snapshot()
                practice_output = await self.coding_mode.run_idle_practice_cycle(
                    user_settings=app_state.user_settings,
                    event_callback=self._emit_event,
                )
                await self._publish_dashboard_coding_practice(practice_output)
                await self._publish_dashboard_coding_output(practice_output.task_result)
                await self._publish_dashboard_coding_patterns()
                await self._publish_dashboard_recent_coding_activity()
                await self._publish_dashboard_notice(
                    f"Completed Coding Dojo practice session '{practice_output.session_id}'.",
                    severity="info",
                )
            elif action == "coding.clear":
                self.dashboard.publish_event({"stage": "dashboard.coding_output_cleared"})
                self.dashboard.publish_event({"stage": "dashboard.coding_practice_cleared"})
                await self._publish_dashboard_notice(
                    "Cleared the current Coding Mode results.",
                    severity="info",
                )
            elif action == "readiness.refresh":
                await self._publish_dashboard_readiness_report()
            elif action == "readiness.export_report":
                export_path = Path(str(payload.get("path", "")).strip())
                if not str(export_path).strip():
                    await self._publish_dashboard_notice(
                        "Provide an export directory before writing the packaged preflight report.",
                        severity="warning",
                    )
                else:
                    artifacts = await self.export_packaged_preflight_report(
                        export_path,
                        active_profile=self.dashboard.app_state_snapshot().user_settings,
                    )
                    await self._publish_dashboard_notice(
                        f"Exported the packaged preflight report to '{artifacts['report_path']}'.",
                        severity="info",
                    )
            elif action == "local_task_session.start":
                label = str(payload.get("label", "")).strip() or "Local task session"
                session = await self.start_local_task_session(label)
                await self._publish_dashboard_notice(
                    f"Started local task session '{session.session_id}'.",
                    severity="info",
                )
            elif action == "local_task_session.pause":
                session_id = str(payload.get("session_id", "")).strip()
                if not session_id:
                    await self._publish_dashboard_notice(
                        "Provide a local task session id before requesting pause.",
                        severity="warning",
                    )
                elif await self.pause_local_task_session(session_id, reason="dashboard_pause_requested"):
                    await self._publish_dashboard_notice(
                        f"Paused local task session '{session_id}'.",
                        severity="info",
                    )
                else:
                    await self._publish_dashboard_notice(
                        f"Unable to pause local task session '{session_id}'.",
                        severity="warning",
                    )
            elif action == "local_task_session.resume":
                session_id = str(payload.get("session_id", "")).strip()
                if not session_id:
                    await self._publish_dashboard_notice(
                        "Provide a local task session id before requesting resume.",
                        severity="warning",
                    )
                elif await self.resume_local_task_session(session_id, reason="dashboard_resume_requested"):
                    await self._publish_dashboard_notice(
                        f"Resumed local task session '{session_id}'.",
                        severity="info",
                    )
                else:
                    await self._publish_dashboard_notice(
                        f"Unable to resume local task session '{session_id}'.",
                        severity="warning",
                    )
            elif action == "local_task_session.stop":
                session_id = str(payload.get("session_id", "")).strip()
                if not session_id:
                    await self._publish_dashboard_notice(
                        "Provide a local task session id before requesting stop.",
                        severity="warning",
                    )
                elif await self.stop_local_task_session(session_id, reason="dashboard_stop_requested"):
                    await self._publish_dashboard_notice(
                        f"Stopped local task session '{session_id}'.",
                        severity="info",
                    )
                else:
                    await self._publish_dashboard_notice(
                        f"Unable to stop local task session '{session_id}'.",
                        severity="warning",
                    )
            elif action == "local_task_session.kill":
                session_id = str(payload.get("session_id", "")).strip()
                if not session_id:
                    await self._publish_dashboard_notice(
                        "Provide a local task session id before engaging the kill-switch.",
                        severity="warning",
                    )
                elif await self.kill_local_task_session(session_id, reason="dashboard_kill_switch_requested"):
                    await self._publish_dashboard_notice(
                        f"Kill-switch engaged for local task session '{session_id}'.",
                        severity="warning",
                    )
                else:
                    await self._publish_dashboard_notice(
                        f"Unable to engage the kill-switch for local task session '{session_id}'.",
                        severity="warning",
                    )
            elif action == "long_horizon.pause":
                session_id = str(payload.get("session_id", "")).strip()
                if not session_id:
                    await self._publish_dashboard_notice(
                        "Provide a long-horizon session id before requesting pause.",
                        severity="warning",
                    )
                elif await self._request_long_horizon_pause(session_id, reason="dashboard_pause_requested"):
                    await self._publish_dashboard_notice(
                        f"Pause requested for long-horizon session '{session_id}'.",
                        severity="info",
                    )
                else:
                    await self._publish_dashboard_notice(
                        f"Unable to pause long-horizon session '{session_id}'.",
                        severity="warning",
                    )
            elif action == "long_horizon.resume":
                session_id = str(payload.get("session_id", "")).strip()
                if not session_id:
                    await self._publish_dashboard_notice(
                        "Provide a long-horizon session id before requesting resume.",
                        severity="warning",
                    )
                else:
                    await self._publish_dashboard_notice(
                        f"Resuming long-horizon session '{session_id}'.",
                        severity="info",
                    )
                    await self._resume_long_horizon_session(session_id)
            elif action == "long_horizon.cancel":
                session_id = str(payload.get("session_id", "")).strip()
                if not session_id:
                    await self._publish_dashboard_notice(
                        "Provide a long-horizon session id before requesting cancel.",
                        severity="warning",
                    )
                elif await self._request_long_horizon_cancel(session_id, reason="dashboard_cancel_requested"):
                    await self._publish_dashboard_notice(
                        f"Cancelled long-horizon session '{session_id}'.",
                        severity="info",
                    )
                else:
                    await self._publish_dashboard_notice(
                        f"Unable to cancel long-horizon session '{session_id}'.",
                        severity="warning",
                    )
            elif action == "examples.refresh":
                await self._publish_dashboard_examples(
                    selected_sample_id=str(payload.get("sample_id", "")).strip() or None
                )
            elif action == "examples.load_demo_pack":
                status = await self.phase11_content.load_demo_pack(
                    storage=self.storage,
                    embed_document=self.model_manager.embed_document,
                    embedding_model_name=self.config.preflight.backends.embedding_model,
                )
                await self._publish_dashboard_examples()
                await self._publish_dashboard_knowledge_library()
                await self._publish_dashboard_notice(
                    f"Loaded Phase 11 demo pack '{status.pack_version}' into local storage.",
                    severity="info",
                )
            elif action == "examples.select_sample":
                sample_id = str(payload.get("sample_id", "")).strip()
                if not sample_id:
                    await self._publish_dashboard_notice(
                        "Select a sample id before loading a Phase 11 example.",
                        severity="warning",
                    )
                else:
                    await self._publish_dashboard_selected_sample(sample_id)
            elif action == "examples.run_sample_task":
                sample_id = str(payload.get("sample_id", "")).strip()
                sample_task = self.phase11_content.get_sample_task(sample_id)
                if sample_task is None:
                    await self._publish_dashboard_notice(
                        f"Unknown Phase 11 sample '{sample_id}'.",
                        severity="warning",
                    )
                else:
                    status = await self.phase11_content.build_demo_pack_status(storage=self.storage)
                    if sample_task.requires_demo_pack and not status.loaded:
                        await self._publish_dashboard_notice(
                            "Load the Phase 11 demo pack before running demo-pack sample tasks.",
                            severity="warning",
                        )
                    else:
                        await self._publish_dashboard_selected_sample(sample_task.sample_id)
                        await self._publish_dashboard_notice(
                            f"Running sample task '{sample_task.title}'.",
                            severity="info",
                        )
                        await self.run_task(
                            sample_task.question,
                            sample_task.recommended_thinking_minutes,
                        )
            elif action == "examples.export_verified_trace_example":
                export_path_raw = str(payload.get("path", "")).strip()
                if not export_path_raw:
                    await self._publish_dashboard_notice(
                        "Provide an export path before exporting the packaged verified trace example.",
                        severity="warning",
                    )
                else:
                    export_path = Path(export_path_raw)
                    actual_path = await self.phase11_content.export_packaged_verified_trace_example(export_path)
                    await self._publish_dashboard_notice(
                        f"Exported the packaged Phase 11 verified trace example to '{actual_path}'.",
                        severity="info",
                    )
            else:
                await self._publish_dashboard_notice(
                    f"Unknown dashboard action '{action}'.",
                    severity="warning",
                )
        except Exception as exc:
            await self._publish_dashboard_notice(
                f"{action} failed: {exc}",
                severity="error",
            )
            self.logger.warning("Dashboard action %s failed: %s", action, exc)

    async def _publish_dashboard_notice(self, message: str, *, severity: str) -> None:
        self.dashboard.publish_event(
            {
                "stage": "dashboard.notice",
                "message": message,
                "severity": severity,
            }
        )

    async def _publish_dashboard_settings_profiles(self, *, active_profile_name: str) -> None:
        profiles = await self.storage.list_user_settings_profiles()
        self.dashboard.publish_event(
            {
                "stage": "dashboard.settings_profiles_loaded",
                "profiles": [profile.to_dict() for profile in profiles],
                "active_profile_name": active_profile_name,
            }
        )

    async def _publish_dashboard_task_history(self) -> None:
        results = tuple(
            sorted(
                await self.storage.list_task_results(limit=24),
                key=lambda result: result.completed_at,
                reverse=True,
            )
        )
        history = tuple(self._build_dashboard_task_history_entry(result) for result in results)
        self.dashboard.publish_event(
            {
                "stage": "dashboard.task_history_loaded",
                "history": [entry.to_dict() for entry in history],
            }
        )
        if results:
            await self._publish_dashboard_task_detail(results[0].task_id)

    async def _publish_dashboard_task_detail(
        self,
        task_id: str,
        *,
        trace_debug_export_path: Path | None = None,
    ) -> None:
        if not task_id:
            return
        result = await self.storage.get_task_result(task_id)
        if result is None:
            await self._publish_dashboard_notice(
                f"Unknown task '{task_id}'.",
                severity="warning",
            )
            return
        detail = await self._build_dashboard_task_inspector(
            result,
            trace_debug_export_path=trace_debug_export_path,
        )
        self.dashboard.publish_event(
            {
                "stage": "dashboard.task_detail_loaded",
                "task": detail.to_dict(),
            }
        )

    async def _publish_dashboard_knowledge_library(self) -> None:
        documents = await self.storage.list_source_documents()
        sources: list[DashboardKnowledgeSource] = []
        for document in documents:
            sources.append(
                DashboardKnowledgeSource(
                    document_id=document.document_id,
                    source_ref=document.source_ref,
                    title=document.title,
                    chunk_count=await self.storage.count_document_chunks(document.document_id),
                    embedding_model=await self.storage.get_document_embedding_model(document.document_id),
                    archived=bool(document.metadata.get("archived", False)),
                    corpus_origin=str(document.metadata.get("corpus_origin", "")),
                    corpus_tier=str(document.metadata.get("corpus_tier", "")),
                    updated_at=document.updated_at,
                    metadata=dict(document.metadata),
                )
            )
        self.dashboard.publish_event(
            {
                "stage": "dashboard.knowledge_library_loaded",
                "sources": [source.to_dict() for source in sources],
            }
        )

    async def _publish_dashboard_readiness_report(
        self,
        *,
        active_profile: UserSettingsProfile | None = None,
    ) -> None:
        report = self._build_dashboard_readiness_report(active_profile=active_profile)
        self.dashboard.publish_event(
            {
                "stage": "dashboard.readiness_loaded",
                "report": report.to_dict(),
            }
        )

    async def _publish_dashboard_capability_registry_view(
        self,
        *,
        active_profile: UserSettingsProfile | None = None,
        recent_decisions: tuple[CapabilityPolicyDecision, ...] | None = None,
        recent_audits: tuple[CapabilityAuditRecord, ...] | None = None,
    ) -> None:
        profile = active_profile or (
            self.dashboard.app_state_snapshot().user_settings if self._started else self._default_user_settings_profile()
        )
        existing_view = await self.storage.load_capability_registry_view()
        resolved_recent_decisions = tuple(recent_decisions or existing_view.recent_decisions[-6:])
        resolved_recent_audits = tuple(recent_audits or (await self.storage.list_capability_audits())[-8:])
        view = self.capability_policy.build_registry_view(
            profile=profile,
            snapshot=self.model_manager.health_snapshot(),
            recent_decisions=resolved_recent_decisions,
            recent_audits=resolved_recent_audits,
        )
        await self.storage.save_capability_registry_view(view)
        self.dashboard.publish_event(
            {
                "stage": "dashboard.capability_registry_loaded",
                "capability_registry_view": view.to_dict(),
            }
        )

    async def _publish_dashboard_model_registry_view(self) -> None:
        recent_optimizer_suggestions = (await self.storage.list_optimizer_suggestion_records())[-4:]
        self.model_manager.apply_governor_advisory_inputs(recent_optimizer_suggestions)
        view = self.model_manager.registry_view(
            advisory_available=True,
            optimizer_subscriptions=("reasoner", "critic", "compressor", "dashboard"),
            recent_optimizer_suggestions=recent_optimizer_suggestions,
        )
        await self.storage.save_model_registry_view(view)
        self.dashboard.publish_event(
            {
                "stage": "dashboard.model_registry_loaded",
                "model_registry_view": view.to_dict(),
            }
        )

    async def _publish_dashboard_model_role_action(self, report: ModelRoleActionReport) -> None:
        self.dashboard.publish_event(
            {
                "stage": "dashboard.model_role_action_reported",
                "model_role_action": report.to_dict(),
            }
        )

    async def _publish_dashboard_examples(self, *, selected_sample_id: str | None = None) -> None:
        sample_tasks = self.phase11_content.load_sample_tasks()
        pack_status = await self.phase11_content.build_demo_pack_status(storage=self.storage)
        current_selected_id = self.dashboard.app_state_snapshot().selected_sample_task.sample_id
        resolved_selected_id = selected_sample_id or current_selected_id
        selected_sample = self.phase11_content.get_sample_task(resolved_selected_id)
        if selected_sample is None and sample_tasks:
            selected_sample = sample_tasks[0]
        self.dashboard.publish_event(
            {
                "stage": "dashboard.examples_loaded",
                "demo_pack_status": pack_status.to_dict(),
                "sample_tasks": [sample_task.to_dict() for sample_task in sample_tasks],
                "selected_sample_task": selected_sample.to_dict() if selected_sample is not None else {},
            }
        )

    async def _publish_dashboard_selected_sample(self, sample_id: str) -> None:
        sample_task = self.phase11_content.get_sample_task(sample_id)
        if sample_task is None:
            await self._publish_dashboard_notice(
                f"Unknown Phase 11 sample '{sample_id}'.",
                severity="warning",
            )
            return
        self.dashboard.publish_event(
            {
                "stage": "dashboard.sample_task_selected",
                "sample_task": sample_task.to_dict(),
            }
        )
        await self._publish_dashboard_notice(
            f"Loaded sample task '{sample_task.title}'.",
            severity="info",
        )

    async def _publish_dashboard_audio_input(self, audio_input: AudioTranscriptionResult) -> None:
        self.dashboard.publish_event(
            {
                "stage": "dashboard.audio_input_loaded",
                "audio_input": audio_input.to_dict(),
            }
        )

    async def _publish_dashboard_audio_output(self, audio_output: AudioSynthesisResult) -> None:
        self.dashboard.publish_event(
            {
                "stage": "dashboard.audio_output_loaded",
                "audio_output": audio_output.to_dict(),
            }
        )

    async def _publish_dashboard_translation_output(self, translation_output: TextTranslationResult) -> None:
        self.dashboard.publish_event(
            {
                "stage": "dashboard.translation_output_loaded",
                "translation_output": translation_output.to_dict(),
            }
        )

    async def _publish_dashboard_translation_import(
        self,
        translation_output: TextTranslationResult,
        question_text: str,
    ) -> None:
        self.dashboard.publish_event(
            {
                "stage": "dashboard.translation_imported",
                "translation_output": translation_output.to_dict(),
                "question_text": question_text,
            }
        )

    async def _publish_dashboard_code_output(self, code_output: CodeSpecialistResult) -> None:
        self.dashboard.publish_event(
            {
                "stage": "dashboard.code_output_loaded",
                "code_output": code_output.to_dict(),
            }
        )

    async def _publish_dashboard_coding_output(self, coding_output: CodingTaskResult) -> None:
        self.dashboard.publish_event(
            {
                "stage": "dashboard.coding_output_loaded",
                "coding_output": coding_output.to_dict(),
            }
        )

    async def _publish_dashboard_coding_practice(self, practice_output: PracticeSessionResult) -> None:
        self.dashboard.publish_event(
            {
                "stage": "dashboard.coding_practice_loaded",
                "coding_practice": practice_output.to_dict(),
            }
        )

    async def _publish_dashboard_coding_patterns(self) -> None:
        list_patterns = getattr(self.storage, "list_coding_patterns", None)
        if list_patterns is None:
            return
        patterns = await list_patterns(limit=8)
        self.dashboard.publish_event(
            {
                "stage": "dashboard.coding_patterns_loaded",
                "coding_patterns": [pattern.to_dict() for pattern in patterns],
            }
        )

    async def _publish_dashboard_recent_coding_activity(self) -> None:
        list_results = getattr(self.storage, "list_coding_task_results", None)
        list_practice = getattr(self.storage, "list_coding_practice_sessions", None)
        if list_results is None or list_practice is None:
            return
        recent_results = await list_results(limit=1)
        recent_practice = await list_practice(limit=1)
        if recent_results:
            await self._publish_dashboard_coding_output(recent_results[0])
        if recent_practice:
            await self._publish_dashboard_coding_practice(recent_practice[0])

    async def _publish_dashboard_audio_transcript_import(
        self,
        audio_input: AudioTranscriptionResult,
        question_text: str,
    ) -> None:
        self.dashboard.publish_event(
            {
                "stage": "dashboard.audio_transcript_imported",
                "audio_input": audio_input.to_dict(),
                "question_text": question_text,
            }
        )

    async def _apply_runtime_settings_profile(self, profile: UserSettingsProfile) -> None:
        """Apply one settings profile across runtime routing and the dashboard shell."""
        self.model_manager.apply_user_settings_profile(profile)
        self.dashboard.apply_user_settings(profile)
        active_session = None
        if hasattr(self.storage, "load_active_local_task_session"):
            active_session = await self.storage.load_active_local_task_session()
        if active_session is not None and active_session.profile_name == profile.profile_name:
            active_session = self._session_with_observation_governor(
                active_session,
                profile=profile,
                updated_at=utc_now(),
            )
            await self.storage.save_local_task_session(active_session)
            await self._sync_continuous_capture_for_session(
                active_session,
                profile=profile,
                reason="settings_updated",
            )
        await self._publish_dashboard_capability_registry_view(active_profile=profile)
        await self._publish_dashboard_model_registry_view()

    async def evaluate_capability_request(
        self,
        request: CapabilityRequest | dict[str, Any],
        *,
        active_profile: UserSettingsProfile | None = None,
    ) -> CapabilityPolicyDecision:
        """Evaluate one typed capability request against the current policy state."""
        capability_request = (
            request if isinstance(request, CapabilityRequest) else CapabilityRequest.from_dict(request)
        )
        profile = active_profile or (
            self.dashboard.app_state_snapshot().user_settings if self._started else self._default_user_settings_profile()
        )
        snapshot = self.model_manager.health_snapshot()
        return self.capability_policy.evaluate(capability_request, profile=profile, snapshot=snapshot)

    @staticmethod
    def _local_task_session_gate_decision(
        request: CapabilityRequest,
        *,
        reason_code: str,
        detail: str,
    ) -> CapabilityPolicyDecision:
        return CapabilityPolicyDecision(
            request_id=request.request_id,
            capability_type=request.capability_type,
            action_name=request.action_name(),
            outcome=CapabilityPolicyOutcome.DENIED,
            availability=CapabilityAvailabilityStatus.AVAILABLE,
            requires_approval=False,
            reason_codes=(reason_code,),
            detail=detail,
            warnings=(),
            decided_at=utc_now(),
        )

    @staticmethod
    def _local_task_control_mode(request: CapabilityRequest) -> str:
        if request.capability_type == CapabilityType.SHELL_COMMAND:
            return "shell_task"
        if request.capability_type == CapabilityType.BROWSER_ACTION:
            return "browser_task"
        if request.capability_type in {
            CapabilityType.APP_WINDOW_FOCUS,
            CapabilityType.SCREENSHOT,
            CapabilityType.OCR_REQUEST,
            CapabilityType.DESKTOP_INPUT,
        }:
            return "desktop_control"
        return "local_task"

    @staticmethod
    def _bind_request_to_local_task_session(
        session: LocalTaskSession,
        request: CapabilityRequest,
    ) -> LocalTaskSession:
        return replace(
            session,
            control_mode=Orchestrator._local_task_control_mode(request),
            current_target=request.target_summary(),
            last_request_id=request.request_id,
            updated_at=utc_now(),
        )

    @staticmethod
    def _queue_pending_approval(
        session: LocalTaskSession,
        request: CapabilityRequest,
    ) -> LocalTaskSession:
        pending_approvals = [
            item
            for item in session.pending_approvals
            if item.request_id != request.request_id
        ]
        pending_approvals.append(
            LocalTaskPendingApproval(
                request_id=request.request_id,
                capability_type=request.capability_type,
                action_name=request.action_name(),
                summary=request.summary or f"Approval required for {request.action_name()}.",
                target=request.target_summary(),
                requested_at=request.requested_at,
            )
        )
        return replace(
            session,
            pending_approvals=tuple(pending_approvals),
            updated_at=utc_now(),
        )

    @staticmethod
    def _clear_pending_approval(
        session: LocalTaskSession,
        request_id: str,
    ) -> LocalTaskSession:
        return replace(
            session,
            pending_approvals=tuple(
                item for item in session.pending_approvals if item.request_id != request_id
            ),
            updated_at=utc_now(),
        )

    @staticmethod
    def _finalize_local_task_session_after_request(
        session: LocalTaskSession,
        request: CapabilityRequest,
        result: CapabilityExecutionResult,
    ) -> LocalTaskSession:
        return replace(
            Orchestrator._clear_pending_approval(session, request.request_id),
            control_mode=Orchestrator._local_task_control_mode(request),
            current_target=request.target_summary(),
            last_action_summary=result.summary,
            last_request_id=request.request_id,
            last_error=result.detail if result.status == CapabilityExecutionStatus.FAILED else "",
            updated_at=utc_now(),
        )

    @staticmethod
    def _local_task_request_fingerprint(request: CapabilityRequest) -> str:
        payload: dict[str, Any] = {
            "capability_type": request.capability_type.value,
            "action_name": request.action_name(),
            "target": request.target_summary(),
            "destructive": request.destructive,
            "cross_app": request.cross_app,
            "metadata": dict(request.metadata),
        }
        if request.file_operation is not None:
            payload["file_operation"] = request.file_operation.to_dict()
        if request.shell_command is not None:
            payload["shell_command"] = request.shell_command.to_dict()
        if request.browser_action is not None:
            payload["browser_action"] = request.browser_action.to_dict()
        if request.app_focus is not None:
            payload["app_focus"] = request.app_focus.to_dict()
        if request.clipboard_action is not None:
            payload["clipboard_action"] = request.clipboard_action.to_dict()
        if request.screenshot is not None:
            payload["screenshot"] = request.screenshot.to_dict()
        if request.ocr_request is not None:
            payload["ocr_request"] = request.ocr_request.to_dict()
        if request.desktop_input is not None:
            payload["desktop_input"] = request.desktop_input.to_dict()
        return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))

    @classmethod
    def _track_local_task_request_state(
        cls,
        session: LocalTaskSession,
        request: CapabilityRequest,
    ) -> tuple[LocalTaskSession, bool]:
        fingerprint = cls._local_task_request_fingerprint(request)
        repeated_count = (
            session.repeated_request_count + 1
            if fingerprint == session.last_request_fingerprint
            else 1
        )
        next_session = replace(
            session,
            last_request_fingerprint=fingerprint,
            repeated_request_count=repeated_count,
            updated_at=utc_now(),
        )
        return next_session, repeated_count > cls.LOCAL_TASK_MAX_REPEATED_REQUESTS

    def _local_task_abort_reason(self) -> str | None:
        return "emergency_stop_requested" if self._local_task_emergency_stop.is_set() else None

    async def _pause_local_task_session_for_recovery(
        self,
        session: LocalTaskSession,
        *,
        reason: str,
        error: str,
    ) -> LocalTaskSession:
        paused = replace(
            session,
            status=LocalTaskSessionState.PAUSED,
            pause_requested=False,
            last_control_reason=reason,
            last_error=error,
            updated_at=utc_now(),
        )
        self._local_task_emergency_stop.set()
        await self.storage.save_local_task_session(paused)
        await self._emit_event(
            "local_task_session.paused",
            {
                "session_id": paused.session_id,
                "status": paused.status.value,
                "reason": reason,
                "detail": error,
            },
        )
        await self._publish_dashboard_local_task_session(paused)
        return paused

    async def _apply_local_task_safety_recovery(
        self,
        session: LocalTaskSession,
        request: CapabilityRequest,
        result: CapabilityExecutionResult,
    ) -> LocalTaskSession:
        warning_set = {str(item) for item in result.warnings}
        if "loop_guard_triggered" in warning_set:
            return await self._pause_local_task_session_for_recovery(
                session,
                reason="loop_guard_triggered",
                error=result.detail,
            )
        if warning_set & {"emergency_stop_requested"}:
            return await self._pause_local_task_session_for_recovery(
                session,
                reason="emergency_stop_requested",
                error=result.detail,
            )
        if warning_set & {"foreground_target_mismatch", "target_window_not_found"}:
            return await self._pause_local_task_session_for_recovery(
                session,
                reason="target_validation_failed",
                error=result.detail,
            )
        if result.status == CapabilityExecutionStatus.FAILED and "shell_timeout" in warning_set:
            return await self._pause_local_task_session_for_recovery(
                session,
                reason="timeout_recovery",
                error=result.detail,
            )
        return session

    async def run_capability_request(
        self,
        request: CapabilityRequest | dict[str, Any],
        *,
        approval_granted: bool | None = None,
        active_profile: UserSettingsProfile | None = None,
    ) -> CapabilityExecutionResult:
        """Run one typed capability request through session, policy, audit, and execution layers."""
        capability_request = (
            request if isinstance(request, CapabilityRequest) else CapabilityRequest.from_dict(request)
        )
        profile = active_profile or (
            self.dashboard.app_state_snapshot().user_settings if self._started else self._default_user_settings_profile()
        )
        session = await self.storage.load_active_local_task_session()
        if session is None:
            decision = self._local_task_session_gate_decision(
                capability_request,
                reason_code="session_not_active",
                detail="Start an explicit local task session before running capability requests.",
            )
        elif session.kill_switch_engaged or session.status == LocalTaskSessionState.KILLED:
            await self.storage.save_active_local_task_session_id("")
            decision = self._local_task_session_gate_decision(
                capability_request,
                reason_code="kill_switch_engaged",
                detail="The kill-switch is engaged for this local task session. Start a new session before retrying.",
            )
        elif session.status == LocalTaskSessionState.PAUSED:
            decision = self._local_task_session_gate_decision(
                capability_request,
                reason_code="session_paused",
                detail="Resume the active local task session before running capability requests.",
            )
        elif session.status in self._terminal_local_task_session_states():
            await self.storage.save_active_local_task_session_id("")
            decision = self._local_task_session_gate_decision(
                capability_request,
                reason_code="session_not_running",
                detail=(
                    f"Local task session '{session.session_id}' is {session.status.value}. "
                    "Start a new session before running capability requests."
                ),
            )
        else:
            session = self._bind_request_to_local_task_session(session, capability_request)
            session, loop_detected = self._track_local_task_request_state(session, capability_request)
            if loop_detected:
                loop_detail = (
                    "loop_guard_triggered: repeated identical capability requests exceeded the local safety threshold; "
                    "the session was paused for recovery."
                )
                session = await self._pause_local_task_session_for_recovery(
                    session,
                    reason="loop_guard_triggered",
                    error=loop_detail,
                )
                decision = self._local_task_session_gate_decision(
                    capability_request,
                    reason_code="loop_guard_triggered",
                    detail=loop_detail,
                )
            else:
                self._local_task_emergency_stop.clear()
                await self.storage.save_local_task_session(session)
                await self._publish_dashboard_local_task_session(session)
                decision = await self.evaluate_capability_request(capability_request, active_profile=profile)
        audit_records: list[CapabilityAuditRecord] = [
            self._capability_audit_record(
                request=capability_request,
                event_type=CapabilityAuditEventType.REQUESTED,
                summary=capability_request.summary or f"Requested {capability_request.capability_type.value}.",
                detail=capability_request.target_summary() or "",
                session_id=session.session_id if session is not None else "",
            ),
            self._capability_audit_record(
                request=capability_request,
                event_type=CapabilityAuditEventType.POLICY_DECISION,
                summary=f"Policy decided {decision.outcome.value} for {decision.action_name}.",
                detail=decision.detail,
                policy_outcome=decision.outcome.value,
                reason_codes=decision.reason_codes,
                session_id=session.session_id if session is not None else "",
            ),
        ]
        effective_decision = decision
        if decision.outcome == CapabilityPolicyOutcome.REQUIRES_APPROVAL and approval_granted is None and session is not None:
            session = self._queue_pending_approval(session, capability_request)
            await self.storage.save_local_task_session(session)
            await self._publish_dashboard_local_task_session(session)
            audit_records.append(
                self._capability_audit_record(
                    request=capability_request,
                    event_type=CapabilityAuditEventType.WARNING,
                    summary=f"Approval is pending for {decision.action_name}.",
                    detail=decision.detail,
                    policy_outcome=decision.outcome.value,
                    reason_codes=decision.reason_codes,
                    session_id=session.session_id,
                )
            )
            result = CapabilityExecutionResult(
                request_id=capability_request.request_id,
                capability_type=capability_request.capability_type,
                action_name=capability_request.action_name(),
                status=CapabilityExecutionStatus.BLOCKED,
                summary=f"Approval pending for {capability_request.action_name()}.",
                detail=decision.detail,
                executor_kind="session_gate",
                output_ref=capability_request.target_summary(),
                warnings=(*decision.warnings, "approval_pending"),
                metadata={
                    "reason_codes": list(decision.reason_codes),
                    "session_id": session.session_id,
                },
                completed_at=utc_now(),
            )
            audit_records.append(
                self._capability_audit_record(
                    request=capability_request,
                    event_type=CapabilityAuditEventType.EXECUTOR_RESULT,
                    summary=result.summary,
                    detail=result.detail,
                    policy_outcome=effective_decision.outcome.value,
                    reason_codes=effective_decision.reason_codes,
                    session_id=session.session_id,
                )
            )
            await self.storage.record_capability_audits(audit_records)
            existing_view = await self.storage.load_capability_registry_view()
            await self._publish_dashboard_capability_registry_view(
                active_profile=profile,
                recent_decisions=(*existing_view.recent_decisions[-5:], decision),
                recent_audits=(*existing_view.recent_audits[-4:], *tuple(audit_records)[-4:]),
            )
            return result
        if decision.outcome == CapabilityPolicyOutcome.REQUIRES_APPROVAL:
            if session is not None:
                session = self._clear_pending_approval(session, capability_request.request_id)
            approval_record_type = (
                CapabilityAuditEventType.APPROVAL_GRANTED
                if approval_granted
                else CapabilityAuditEventType.APPROVAL_DENIED
            )
            audit_records.append(
                self._capability_audit_record(
                    request=capability_request,
                    event_type=approval_record_type,
                    summary=(
                        f"Approval granted for {decision.action_name}."
                        if approval_granted
                        else f"Approval denied for {decision.action_name}."
                    ),
                    detail=decision.detail,
                    policy_outcome=decision.outcome.value,
                    reason_codes=decision.reason_codes,
                    session_id=session.session_id if session is not None else "",
                )
            )
            if approval_granted:
                effective_decision = replace(
                    decision,
                    outcome=CapabilityPolicyOutcome.ALLOWED,
                    availability=CapabilityAvailabilityStatus.AVAILABLE,
                    requires_approval=False,
                    detail=f"{decision.detail} Approval granted for bounded execution.",
                )
        if decision.outcome in {CapabilityPolicyOutcome.DENIED, CapabilityPolicyOutcome.DEGRADED}:
            audit_records.append(
                self._capability_audit_record(
                    request=capability_request,
                    event_type=CapabilityAuditEventType.WARNING,
                    summary=f"Capability request surfaced {decision.outcome.value}.",
                    detail=decision.detail,
                    policy_outcome=decision.outcome.value,
                    reason_codes=decision.reason_codes,
                    session_id=session.session_id if session is not None else "",
                )
            )
        result = await self.capability_executor.execute(
            capability_request,
            decision=effective_decision,
            profile=profile,
            should_abort=self._local_task_abort_reason,
        )
        if session is not None and session.status == LocalTaskSessionState.RUNNING:
            session = self._finalize_local_task_session_after_request(session, capability_request, result)
            session = await self._apply_local_task_safety_recovery(session, capability_request, result)
            if session.status == LocalTaskSessionState.RUNNING:
                session = await self._run_observation_step_if_enabled(
                    session,
                    request=capability_request,
                    result=result,
                    profile=profile,
                )
        audit_records.append(
            self._capability_audit_record(
                request=capability_request,
                event_type=CapabilityAuditEventType.EXECUTOR_RESULT,
                summary=result.summary,
                detail=result.detail,
                policy_outcome=effective_decision.outcome.value,
                reason_codes=effective_decision.reason_codes,
                session_id=session.session_id if session is not None else "",
            )
        )
        await self.storage.record_capability_audits(audit_records)
        existing_view = await self.storage.load_capability_registry_view()
        await self._publish_dashboard_capability_registry_view(
            active_profile=profile,
            recent_decisions=(*existing_view.recent_decisions[-5:], decision),
            recent_audits=(*existing_view.recent_audits[-4:], *tuple(audit_records)[-4:]),
        )
        return result

    @staticmethod
    def _capability_audit_record(
        *,
        request: CapabilityRequest,
        event_type: CapabilityAuditEventType,
        summary: str,
        detail: str = "",
        policy_outcome: str = "",
        reason_codes: tuple[str, ...] = (),
        session_id: str = "",
    ) -> CapabilityAuditRecord:
        audit_id = f"{request.request_id}:{event_type.value}"
        return CapabilityAuditRecord(
            audit_id=audit_id,
            request_id=request.request_id,
            capability_type=request.capability_type,
            action_name=request.action_name(),
            event_type=event_type,
            summary=summary,
            detail=detail,
            policy_outcome=policy_outcome,
            reason_codes=reason_codes,
            metadata={
                "target": request.target_summary(),
                "summary": request.summary,
                "session_id": session_id,
            },
            created_at=utc_now(),
        )

    async def _run_dashboard_model_action(self, *, action: str, payload: dict[str, Any]) -> None:
        role = self._resolve_dashboard_model_role(str(payload.get("role", "")).strip())
        if role is None:
            await self._publish_dashboard_notice(
                "Select a valid local-AI role before using control-plane actions.",
                severity="warning",
            )
            return

        if action == "model.install_guidance":
            guidance = self.model_manager.install_guidance_for_role(role)
            report = ModelRoleActionReport(
                role=role,
                action="install_guidance",
                ok=bool(guidance),
                summary=f"Install guidance for {role.value}.",
                detail=guidance[0] if guidance else f"No install guidance is available for {role.value}.",
                guidance=guidance,
                warnings=self.model_manager.recent_fallback_reasons_for_role(role),
            )
        elif action == "model.inspect_fallback":
            fallback_reasons = self.model_manager.recent_fallback_reasons_for_role(role)
            route_decisions = self.model_manager.recent_route_decisions_for_role(role, limit=1)
            report = ModelRoleActionReport(
                role=role,
                action="inspect_fallback",
                ok=bool(fallback_reasons or route_decisions),
                summary=(
                    f"Fallback inspection for {role.value}: {fallback_reasons[0]}"
                    if fallback_reasons
                    else f"No fallback reason is currently recorded for {role.value}."
                ),
                detail=(
                    "Recent fallback reasons: " + ", ".join(fallback_reasons)
                    if fallback_reasons
                    else "The selected role has not recorded a recent fallback or routing block."
                ),
                route_decision=route_decisions[-1] if route_decisions else None,
                guidance=self.model_manager.install_guidance_for_role(role),
                warnings=fallback_reasons,
            )
        elif action == "model.enable_role":
            report = await self._toggle_dashboard_model_role(role, enabled=True)
        elif action == "model.disable_role":
            report = await self._toggle_dashboard_model_role(role, enabled=False)
        elif action == "model.warm_role":
            decision = await self.model_manager.warm_role(role)
            report = ModelRoleActionReport(
                role=role,
                action="warm_role",
                ok=decision.allowed,
                summary=(
                    f"Warmed {role.value} via {decision.selected_backend or '(none)'} / "
                    f"{decision.selected_model_identifier or '(none)'}."
                    if decision.allowed
                    else f"Unable to warm {role.value}: {decision.fallback_reason or 'routing blocked'}."
                ),
                detail=(
                    "Role is ready for bounded use in the integrated local-AI control plane."
                    if decision.allowed
                    else "Warm was blocked before the role could become active."
                ),
                route_decision=decision,
                guidance=self.model_manager.install_guidance_for_role(role),
                warnings=self.model_manager.recent_fallback_reasons_for_role(role),
            )
        elif action == "model.unload_role":
            unloaded = self.model_manager.unload_optional_role(role)
            report = ModelRoleActionReport(
                role=role,
                action="unload_role",
                ok=unloaded,
                summary=(
                    f"Unloaded resident state for {role.value}."
                    if unloaded
                    else f"{role.value} could not be unloaded from the control plane."
                ),
                detail=(
                    "Optional heavy state was dropped or the role has no resident heavy state to keep loaded."
                    if unloaded
                    else "Mandatory baseline roles stay resident and cannot be unloaded here."
                ),
                guidance=self.model_manager.install_guidance_for_role(role),
                warnings=self.model_manager.recent_fallback_reasons_for_role(role),
            )
        elif action == "model.test_ping":
            decision, detail = await self.model_manager.test_role_ping(role)
            report = ModelRoleActionReport(
                role=role,
                action="test_ping",
                ok=decision.allowed,
                summary=(
                    f"Test ping succeeded for {role.value}."
                    if decision.allowed
                    else f"Test ping blocked for {role.value}: {decision.fallback_reason or 'routing blocked'}."
                ),
                detail=detail,
                route_decision=decision,
                guidance=self.model_manager.install_guidance_for_role(role),
                warnings=self.model_manager.recent_fallback_reasons_for_role(role),
            )
        else:
            await self._publish_dashboard_notice(
                f"Unknown local-AI control-plane action '{action}'.",
                severity="warning",
            )
            return

        await self._publish_dashboard_model_registry_view()
        await self._publish_dashboard_model_role_action(report)
        await self._publish_dashboard_notice(
            report.summary,
            severity="info" if report.ok else "warning",
        )

    async def _toggle_dashboard_model_role(
        self,
        role: ModelRole,
        *,
        enabled: bool,
    ) -> ModelRoleActionReport:
        current = self.dashboard.app_state_snapshot().user_settings
        if role in {ModelRole.GENERATION, ModelRole.EMBEDDING} and not enabled:
            return ModelRoleActionReport(
                role=role,
                action="disable_role",
                ok=False,
                summary=f"{role.value} is part of the baseline runtime and stays enabled.",
                detail="Disable optional specialist roles here, not the generation or embedding base pair.",
                guidance=self.model_manager.install_guidance_for_role(role),
            )

        enabled_roles = [
            str(item)
            for item in current.models.get("enabled_roles", ())
            if str(item).strip()
        ]
        if enabled and role.value not in enabled_roles:
            enabled_roles.append(role.value)
        if not enabled:
            enabled_roles = [item for item in enabled_roles if item != role.value]
        next_profile = replace(
            current,
            models={
                **current.models,
                "enabled_roles": tuple(enabled_roles),
            },
            updated_at=utc_now(),
        )
        next_profile.validate()
        await self.storage.save_user_settings_profile(next_profile)
        await self._apply_runtime_settings_profile(next_profile)
        await self._publish_dashboard_settings_profiles(active_profile_name=next_profile.profile_name)
        await self._publish_dashboard_readiness_report(active_profile=next_profile)

        registered = self.model_manager.list_registered_models(role=role)
        enabled_registrations = tuple(
            registration
            for registration in registered
            if registration.enabled and registration.backend != "unconfigured"
        )
        return ModelRoleActionReport(
            role=role,
            action="enable_role" if enabled else "disable_role",
            ok=True,
            summary=f"{'Enabled' if enabled else 'Disabled'} {role.value}.",
            detail=(
                "Active registrations: "
                + (
                    ", ".join(
                        f"{registration.backend} / {registration.model_identifier}"
                        for registration in enabled_registrations
                    )
                    if enabled_registrations
                    else "(none)"
                )
            ),
            guidance=self.model_manager.install_guidance_for_role(role),
            warnings=self.model_manager.recent_fallback_reasons_for_role(role),
        )

    @staticmethod
    def _resolve_dashboard_model_role(raw_value: str) -> ModelRole | None:
        normalized = str(raw_value).strip()
        if not normalized:
            return None
        try:
            return ModelRole(normalized)
        except ValueError:
            return None

    def _build_dashboard_task_history_entry(self, result: TaskResult) -> DashboardTaskHistoryEntry:
        metadata = dict(result.reasoning.context_frames[0].metadata) if result.reasoning.context_frames else {}
        return DashboardTaskHistoryEntry(
            task_id=result.task_id,
            question=result.plan.question,
            answer_preview=(result.answer_text[:180] + "...") if len(result.answer_text) > 180 else result.answer_text,
            critique_result=result.critique.result.value,
            degraded_reason=result.critique.degraded_reason,
            warning_count=len(result.warnings),
            candidate_trace_count=len(result.reasoning.candidate_traces),
            citation_count=len(self.translation.summarize_answer_metadata(evidence=result.evidence, reasoning=result.reasoning)["citation_refs"]),
            selected_strategy=str(metadata.get("sa", "")),
            selected_verifier=str(metadata.get("sv", result.critique.verifier_type)),
            used_web_fallback=result.evidence.used_web_fallback,
            completed_at=result.completed_at,
        )

    async def _build_dashboard_task_inspector(
        self,
        result: TaskResult,
        *,
        trace_debug_export_path: Path | None = None,
    ) -> DashboardTaskInspector:
        metadata = self.translation.summarize_answer_metadata(
            evidence=result.evidence,
            reasoning=result.reasoning,
        )
        context_metadata = dict(result.reasoning.context_frames[0].metadata) if result.reasoning.context_frames else {}
        specialist_roles_used, specialist_role_explanations = self._task_specialist_usage(
            evidence=result.evidence,
            reasoning=result.reasoning,
        )
        advisor_summaries = await self._task_advisor_summaries(result.task_id)
        lifecycle_entries: list[str] = []
        for record in (await self.storage.list_optimizer_proposal_records())[-3:]:
            lifecycle_entries.append(
                f"proposal:{record.proposal_id}:{record.lifecycle_stage.value}:{record.mean_simulation_score:.2f}"
            )
        for record in (await self.storage.list_optimizer_activation_records())[-3:]:
            lifecycle_entries.append(
                f"activation:{record.proposal_id}:{record.decision.value}:{record.reason or 'no_reason'}"
            )
        for record in (await self.storage.list_optimizer_rollback_records())[-3:]:
            lifecycle_entries.append(
                f"rollback:{record.proposal_id}:{record.proposal_macro_name}:{'applied' if record.applied else 'prepared'}"
            )
        return DashboardTaskInspector(
            task_id=result.task_id,
            question=result.plan.question,
            answer_text=result.answer_text,
            critique_result=result.critique.result.value,
            degraded_reason=result.critique.degraded_reason,
            warning_count=len(result.warnings),
            warnings=result.warnings,
            citation_refs=tuple(str(item) for item in metadata["citation_refs"]),
            repair_actions=result.critique.repair_actions,
            failure_categories=result.critique.failure_categories,
            supporting_evidence_ids=tuple(str(item) for item in metadata["supporting_evidence_ids"]),
            candidate_trace_count=len(result.reasoning.candidate_traces),
            selected_strategy=str(context_metadata.get("sa", "")),
            selected_verifier=str(context_metadata.get("sv", result.critique.verifier_type)),
            used_web_fallback=result.evidence.used_web_fallback,
            trace_debug_export_path=str(trace_debug_export_path or ""),
            optimizer_lifecycle=tuple(lifecycle_entries),
            specialist_roles_used=specialist_roles_used,
            specialist_role_explanations=specialist_role_explanations,
            advisor_summaries=advisor_summaries,
            completed_at=result.completed_at,
        )

    def build_packaged_launch_report(
        self,
        *,
        active_profile: UserSettingsProfile | None = None,
        first_run: bool = False,
    ) -> PackagedLaunchReport:
        """Return the packaged-app launch decision before or after startup.

        Input:
        - `active_profile`: optional settings profile to evaluate instead of the
          live dashboard state.
        - `first_run`: whether the packaged app is planning its first startup
          without a previously saved default settings profile.

        Output:
        - A typed `PackagedLaunchReport` describing stub or real mode readiness.

        Failure behavior:
        - The method does not raise for missing optional runtime dependencies;
          those become typed readiness blockers in the returned report.
        """
        state = self._collect_readiness_state(active_profile=active_profile)
        readiness = self._build_dashboard_readiness_report_from_state(state)
        requested_mode = "stub" if state.requested_stub_mode else "real"
        prelaunch_real_ready = state.primary_generation_ready and state.primary_embedding_ready
        used_stub_fallback = requested_mode == "real" and not prelaunch_real_ready
        effective_mode = "stub" if requested_mode == "stub" or used_stub_fallback else "real"
        launch_ready = True if effective_mode == "stub" else prelaunch_real_ready
        blocking_reason = ""
        blocking_detail = ""
        if requested_mode == "stub":
            summary = (
                "Packaged launch will start in stub mode for first-run local validation."
                if first_run
                else "Packaged launch will stay in stub mode until you explicitly re-enable real-mode prerequisites."
            )
        elif used_stub_fallback:
            blocking_reason, blocking_detail = self._packaged_launch_blocker(state)
            summary = (
                "Real-mode packaged launch is not ready; the app should fall back to stub mode and surface setup guidance."
            )
        else:
            summary = "Packaged launch is ready for real mode."
        return PackagedLaunchReport(
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            launch_ready=launch_ready,
            used_stub_fallback=used_stub_fallback,
            summary=summary,
            blocking_reason=blocking_reason,
            blocking_detail=blocking_detail,
            guidance=readiness.guidance,
            readiness_report=readiness,
        )

    def build_packaged_startup_plan(
        self,
        *,
        startup_profile: UserSettingsProfile | None = None,
    ) -> _PackagedStartupPlan:
        """Plan packaged startup while preserving requested vs effective runtime mode.

        Input:
        - `startup_profile`: persisted default profile if one already exists.

        Output:
        - An internal packaged-startup plan carrying the requested profile,
          effective startup profile, launch report, and runtime config.

        Failure behavior:
        - The method does not raise for missing optional runtime dependencies;
          those remain encoded in the contained packaged launch report.
        """
        first_run = startup_profile is None
        if first_run:
            default_profile = self._default_user_settings_profile()
            requested_profile = replace(
                default_profile,
                runtime={
                    **default_profile.runtime,
                    "stub_mode": True,
                },
            )
            effective_profile = requested_profile
            persist_effective_profile = True
        else:
            requested_profile = startup_profile
            effective_profile = requested_profile
            persist_effective_profile = False
        launch_report = self.build_packaged_launch_report(
            active_profile=requested_profile,
            first_run=first_run,
        )
        if launch_report.used_stub_fallback:
            effective_profile = replace(
                requested_profile,
                runtime={
                    **requested_profile.runtime,
                    "stub_mode": True,
                },
            )
        runtime_config = self._config_for_user_settings_profile(effective_profile)
        if first_run:
            startup_notice = (
                "Packaged startup is beginning in stub mode for first-run validation. "
                "Review readiness guidance before enabling real mode."
            )
            startup_notice_severity = "info"
        elif launch_report.used_stub_fallback:
            startup_notice = f"{launch_report.summary} {launch_report.blocking_detail}".strip()
            startup_notice_severity = "warning"
        elif launch_report.effective_mode == "real":
            startup_notice = "Packaged startup is ready for real mode."
            startup_notice_severity = "info"
        else:
            startup_notice = (
                "Packaged startup is using stub mode because the saved profile still requests a lightweight local path."
            )
            startup_notice_severity = "info"
        return _PackagedStartupPlan(
            requested_profile=requested_profile,
            effective_profile=effective_profile,
            launch_report=launch_report,
            runtime_config=runtime_config,
            first_run=first_run,
            persist_effective_profile=persist_effective_profile,
            startup_notice=startup_notice,
            startup_notice_severity=startup_notice_severity,
        )

    def _packaged_default_model_bundle(self) -> dict[str, str]:
        """Return the pinned default local model bundle used by readiness and onboarding surfaces."""
        backends = self.config.preflight.backends
        return {
            "generation": f"{backends.generation_backend}:{backends.generation_model}",
            "generation_fallback": f"{backends.generation_fallback_backend}:{backends.generation_fallback_model}",
            "embedding": f"{backends.embedding_backend}:{backends.embedding_model}",
            "embedding_fallback": f"{backends.embedding_fallback_backend}:{backends.embedding_fallback_model}",
        }

    def _packaged_data_paths(self) -> dict[str, str]:
        """Return the local paths that packaged onboarding and diagnostics should point to."""
        return {
            "sqlite_path": str(self.config.storage.sqlite_path),
            "logs_dir": str(self.config.storage.logs_dir),
            "models_dir": str(self.config.backend_runtime.models_dir),
        }

    def _packaged_onboarding_lines(
        self,
        *,
        launch_report: PackagedLaunchReport,
    ) -> tuple[str, ...]:
        """Return first-run onboarding guidance for packaged startup and reopened preflight reports."""
        hardware = self.config.preflight.hardware
        bundle = self._packaged_default_model_bundle()
        data_paths = self._packaged_data_paths()
        guide_path = str(_LOCAL_MODEL_SETUP_GUIDE)
        return (
            (
                "Stub mode is the default packaged first-run path so the app can validate storage, UI, "
                "history, knowledge management, and local task controls without heavy model dependencies."
            ),
            (
                "Real mode stays opt-in and only becomes the effective packaged mode when both the pinned "
                "generation and embedding backends are locally ready."
            ),
            (
                f"Default local model bundle: generation={bundle['generation']}, "
                f"generation_fallback={bundle['generation_fallback']}, embedding={bundle['embedding']}, "
                f"embedding_fallback={bundle['embedding_fallback']}."
            ),
            f"Hardware target: {hardware.max_vram_gb:.0f}GB VRAM / {hardware.max_ram_gb:.0f}GB RAM.",
            (
                "Privacy boundary: the base runtime stays local-first; optional cloud helpers remain auxiliary-only "
                "and can offload approved content only after the matching capability is explicitly enabled."
            ),
            (
                f"User data stays on this machine by default in '{data_paths['sqlite_path']}' plus "
                f"'{data_paths['logs_dir']}', while local model files live under '{data_paths['models_dir']}'."
            ),
            (
                f"Current packaged launch decision: requested={launch_report.requested_mode}, "
                f"effective={launch_report.effective_mode}, fallback={'yes' if launch_report.used_stub_fallback else 'no'}."
            ),
            f"Exact local backend and model setup steps are documented in '{guide_path}'.",
        )

    def _packaged_setup_steps(self) -> tuple[str, ...]:
        """Return actionable setup steps aligned to the pinned default local model bundle."""
        bundle = self._packaged_default_model_bundle()
        models_dir = self._packaged_data_paths()["models_dir"]
        return (
            "Create a Python 3.11+ environment and install the repo with only the extras you need.",
            "Install the primary embedding dependency with `python -m pip install -e .[embeddings]`.",
            "Install the persistent vector dependency with `python -m pip install -e .[vector]` if you want the default Chroma store.",
            "Install the fallback generation dependency with `python -m pip install -e .[llama-cpp]` if you want local GGUF fallback.",
            (
                f"Install Ollama and pull the primary generation model `{bundle['generation']}` plus the "
                f"embedding fallback model `{bundle['embedding_fallback']}` if you plan to use the Ollama embedding path."
            ),
            (
                f"Place the GGUF fallback model `{bundle['generation_fallback'].split(':', 1)[1]}` under "
                f"`{models_dir}` or update the configured path."
            ),
            "Refresh the Readiness tab or export a packaged preflight report after setup changes so the effective launch mode is recalculated from the same shared checks.",
        )

    def _packaged_preflight_payload(
        self,
        *,
        profile: UserSettingsProfile,
        launch_report: PackagedLaunchReport,
    ) -> dict[str, Any]:
        """Build a reopenable packaged preflight report payload."""
        cloud_mode = str(profile.cloud.get("mode", CloudOffloadMode.AUXILIARY_ONLY.value))
        return {
            "generated_at": utc_now().isoformat(),
            "launch_report": launch_report.to_dict(),
            "default_model_bundle": self._packaged_default_model_bundle(),
            "hardware_budget": {
                "max_vram_gb": self.config.preflight.hardware.max_vram_gb,
                "max_ram_gb": self.config.preflight.hardware.max_ram_gb,
            },
            "data_paths": self._packaged_data_paths(),
            "heavy_slot_policy": {
                "default_active_pair": ("generation", "embedding"),
                "max_active_heavy_backends": 2,
                "sidecars_do_not_consume_heavy_slots": True,
            },
            "privacy": {
                "cloud_mode": cloud_mode,
                "allow_cloud_content": bool(profile.privacy.get("allow_cloud_content", False)),
                "summary": (
                    "Local-first by default; optional cloud helpers remain auxiliary-only and only handle approved "
                    "content after explicit enablement."
                ),
            },
            "onboarding_guidance": self._packaged_onboarding_lines(launch_report=launch_report),
            "setup_steps": self._packaged_setup_steps(),
            "setup_guide_path": str(_LOCAL_MODEL_SETUP_GUIDE),
        }

    async def export_packaged_preflight_report(
        self,
        export_dir: Path,
        *,
        active_profile: UserSettingsProfile | None = None,
        launch_report: PackagedLaunchReport | None = None,
    ) -> dict[str, str]:
        """Export a reopenable packaged preflight report plus onboarding/setup artifacts."""
        report_dir = Path(export_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        profile = active_profile
        if profile is None:
            if self._started:
                profile = self.dashboard.app_state_snapshot().user_settings
            else:
                profile = self._default_user_settings_profile()
        launch = launch_report or self.build_packaged_launch_report(active_profile=profile)
        payload = self._packaged_preflight_payload(profile=profile, launch_report=launch)

        report_path = report_dir / "packaged_preflight_report.json"
        onboarding_path = report_dir / "packaged_onboarding.txt"
        setup_path = report_dir / _LOCAL_MODEL_SETUP_GUIDE.name

        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        onboarding_path.write_text(
            "\n".join(
                (
                    "Quester.AI Packaged Onboarding",
                    "",
                    *(f"- {line}" for line in payload["onboarding_guidance"]),
                    "",
                    "Setup Steps:",
                    *(f"{index}. {line}" for index, line in enumerate(payload["setup_steps"], start=1)),
                )
            )
            + "\n",
            encoding="utf-8",
        )
        if _LOCAL_MODEL_SETUP_GUIDE.exists():
            shutil.copy2(_LOCAL_MODEL_SETUP_GUIDE, setup_path)
        else:
            setup_path.write_text("\n".join(payload["setup_steps"]) + "\n", encoding="utf-8")
        return {
            "report_path": str(report_path),
            "onboarding_path": str(onboarding_path),
            "setup_path": str(setup_path),
        }

    def _write_packaged_startup_diagnostics(
        self,
        error: BaseException,
        *,
        export_dir: Path | None = None,
        launch_report: PackagedLaunchReport | None = None,
    ) -> Path:
        """Persist startup diagnostics so packaged failures are exportable and actionable."""
        diagnostics_dir = Path(export_dir) if export_dir is not None else self.config.storage.logs_dir
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path = diagnostics_dir / "packaged_startup_diagnostics.json"
        diagnostics_payload = {
            "generated_at": utc_now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "launch_report": None if launch_report is None else launch_report.to_dict(),
            "safe_recovery": {
                "recommended_mode": "stub",
                "summary": "Retry packaged startup in stub mode, inspect readiness, then re-enable real mode only after the reported blockers are resolved.",
            },
            "data_paths": self._packaged_data_paths(),
            "setup_guide_path": str(_LOCAL_MODEL_SETUP_GUIDE),
        }
        diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2), encoding="utf-8")
        return diagnostics_path

    def _build_runtime_recovery_launch_report(
        self,
        *,
        launch_report: PackagedLaunchReport,
        error: BaseException,
        diagnostics_path: Path,
    ) -> PackagedLaunchReport:
        """Translate an unexpected real-mode startup failure into a readable stub-recovery launch report."""
        guidance = tuple(
            dict.fromkeys(
                (
                    *launch_report.guidance,
                    f"Startup diagnostics were written to '{diagnostics_path}'.",
                    "The packaged app recovered into stub mode after a real-mode startup failure.",
                )
            )
        )
        return replace(
            launch_report,
            effective_mode="stub",
            launch_ready=True,
            used_stub_fallback=True,
            summary="Packaged startup hit a real-mode exception and recovered in stub mode.",
            blocking_reason="startup_exception",
            blocking_detail=f"{type(error).__name__}: {error}",
            guidance=guidance,
        )

    async def dispatch_auxiliary_cloud_job(
        self,
        *,
        capability: CloudOffloadCapability,
        payload: dict[str, Any],
        active_profile: UserSettingsProfile | None = None,
        local_fallback: Callable[[], Awaitable[Any] | Any] | None = None,
        job_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CloudOffloadRecord:
        """Dispatch one auxiliary cloud helper job without making cloud mandatory.

        Input:
        - `capability`: the helper category to evaluate.
        - `payload`: bounded provider-agnostic request payload.
        - `active_profile`: optional settings profile override.
        - `local_fallback`: optional local callable to run if the helper is blocked or fails.
        - `job_id`: optional stable ID override for persistence and audit lookup.
        - `metadata`: optional audit metadata merged into the persisted record.

        Output:
        - A typed `CloudOffloadRecord` describing the final cloud or local-fallback outcome.

        Failure behavior:
        - The method does not raise for cloud-policy, provider-availability, or provider-dispatch failures.
          Those conditions are returned and persisted as typed outcomes instead.
        """
        profile = active_profile or (
            self.dashboard.app_state_snapshot().user_settings if self._started else self._default_user_settings_profile()
        )
        title, payload_class, privacy_class, requires_content_approval = _CLOUD_OFFLOAD_CAPABILITY_DETAILS[capability]
        provider_name = self._cloud_provider_name(profile)
        provider_family = self._cloud_provider_family(profile)
        fallback_behavior = self._cloud_fallback_behavior(profile)
        resolved_job_id = str(job_id or f"{capability.value}:{stable_hash(json.dumps(payload, sort_keys=True, default=str))[:12]}")
        payload_bytes = len(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        dispatch_metadata = dict(metadata or {})
        dispatch_metadata.setdefault("title", title)

        async def _blocked_record(reason: str, detail: str) -> CloudOffloadRecord:
            fallback_used = False
            outcome = CloudOffloadOutcome.BLOCKED
            summary = "Auxiliary cloud helper was blocked before dispatch."
            if local_fallback is not None:
                maybe_result = local_fallback()
                if inspect.isawaitable(maybe_result):
                    await maybe_result
                fallback_used = True
                outcome = CloudOffloadOutcome.LOCAL_FALLBACK
                summary = "Auxiliary cloud helper was skipped and local fallback ran."
            return CloudOffloadRecord(
                dispatch_id=f"cloud_{stable_hash(f'{resolved_job_id}:{reason}:{utc_now().isoformat()}')[:16]}",
                job_id=resolved_job_id,
                capability=capability,
                provider_name=provider_name,
                provider_family=provider_family,
                payload_class=payload_class,
                privacy_class=privacy_class,
                outcome=outcome,
                summary=summary,
                detail=detail,
                fallback_behavior=fallback_behavior,
                bytes_sent=payload_bytes,
                local_fallback_used=fallback_used,
                fallback_reason=reason,
                metadata=dispatch_metadata,
            )

        cloud_mode = CloudOffloadMode(str(profile.cloud.get("mode", CloudOffloadMode.AUXILIARY_ONLY.value)).strip())
        if cloud_mode == CloudOffloadMode.DISABLED:
            record = await _blocked_record(
                "cloud_mode_disabled",
                f"{title} stays local-only because global cloud mode is disabled.",
            )
        elif self._cloud_capability_modes(profile)[capability] == CloudOffloadMode.DISABLED:
            record = await _blocked_record(
                "cloud_capability_disabled",
                f"{title} stays local-only because this helper category is disabled.",
            )
        elif requires_content_approval and not bool(profile.privacy.get("allow_cloud_content", False)):
            record = await _blocked_record(
                "cloud_content_not_approved",
                f"{title} requires approved-content offload, but Allow cloud content is off.",
            )
        else:
            contract = self._cloud_job_contract_for_capability(profile=profile, capability=capability)
            if contract is None:
                record = await _blocked_record(
                    "cloud_contract_unavailable",
                    f"{title} has no valid cloud contract, so the helper remains local-only.",
                )
            else:
                record = await self.cloud_offload.dispatch(
                    provider_name=provider_name,
                    contract=replace(contract, job_id=resolved_job_id),
                    payload=payload,
                    local_fallback=local_fallback,
                    metadata=dispatch_metadata,
                )
        await self.storage.record_cloud_offload_record(record)
        await self._emit_event(
            "cloud.offload_recorded",
            {
                "dispatch_id": record.dispatch_id,
                "job_id": record.job_id,
                "capability": record.capability.value,
                "provider_name": record.provider_name,
                "provider_family": record.provider_family,
                "outcome": record.outcome.value,
                "fallback_reason": record.fallback_reason,
                "local_fallback_used": record.local_fallback_used,
                "bytes_sent": record.bytes_sent,
                "latency_ms": record.latency_ms,
            },
        )
        return record

    async def export_packaged_support_bundle(
        self,
        export_dir: Path,
        *,
        active_profile: UserSettingsProfile | None = None,
        launch_report: PackagedLaunchReport | None = None,
        diagnostics_path: Path | None = None,
    ) -> PackagedSupportBundle:
        """Export a small support bundle for packaged-launch smoke and troubleshooting.

        Input:
        - `export_dir`: destination directory for the bundle files.
        - `active_profile`: optional settings profile override.
        - `launch_report`: optional precomputed packaged-launch report.
        - `diagnostics_path`: optional packaged startup diagnostics file to include.

        Output:
        - A typed `PackagedSupportBundle` containing the generated file paths.

        Failure behavior:
        - Raises filesystem exceptions if the export directory cannot be created
          or written.
        """
        bundle_dir = Path(export_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        profile = active_profile
        if profile is None:
            if self._started:
                profile = self.dashboard.app_state_snapshot().user_settings
            else:
                profile = self._default_user_settings_profile()
        launch = launch_report or self.build_packaged_launch_report(active_profile=profile)
        app_state = self.dashboard.app_state_snapshot()
        preflight_artifacts = await self.export_packaged_preflight_report(
            bundle_dir,
            active_profile=profile,
            launch_report=launch,
        )

        launch_report_path = bundle_dir / "launch_report.json"
        readiness_report_path = bundle_dir / "readiness_report.json"
        user_settings_path = bundle_dir / "user_settings_profile.json"
        app_state_path = bundle_dir / "app_state.json"
        support_readme_path = bundle_dir / "support_bundle.txt"
        manifest_path = bundle_dir / "support_bundle_manifest.json"
        diagnostics_copy_path = bundle_dir / "packaged_startup_diagnostics.json"

        launch_report_path.write_text(json.dumps(launch.to_dict(), indent=2), encoding="utf-8")
        readiness_report_path.write_text(
            json.dumps(launch.readiness_report.to_dict(), indent=2),
            encoding="utf-8",
        )
        user_settings_path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        app_state_path.write_text(json.dumps(app_state.to_dict(), indent=2), encoding="utf-8")
        support_readme_path.write_text(
            "\n".join(
                (
                    "Quester.AI Packaged Support Bundle",
                    "",
                    f"Requested mode: {launch.requested_mode}",
                    f"Effective mode: {launch.effective_mode}",
                    f"Launch ready: {'yes' if launch.launch_ready else 'no'}",
                    f"Used stub fallback: {'yes' if launch.used_stub_fallback else 'no'}",
                    f"Summary: {launch.summary}",
                    (
                        f"Blocking reason: {launch.blocking_reason} | {launch.blocking_detail}"
                        if launch.blocking_reason or launch.blocking_detail
                        else "Blocking reason: none"
                    ),
                    "",
                    "Onboarding:",
                    *(f"- {item}" for item in self._packaged_onboarding_lines(launch_report=launch)),
                    "",
                    "Setup Steps:",
                    *(f"{index}. {item}" for index, item in enumerate(self._packaged_setup_steps(), start=1)),
                    "",
                    "Guidance:",
                    *(f"- {item}" for item in launch.guidance),
                )
            )
            + "\n",
            encoding="utf-8",
        )

        diagnostics_output = ""
        if diagnostics_path is not None and diagnostics_path.exists():
            if diagnostics_path.resolve() != diagnostics_copy_path.resolve():
                shutil.copy2(diagnostics_path, diagnostics_copy_path)
            diagnostics_output = str(diagnostics_copy_path)
        elif diagnostics_copy_path.exists():
            diagnostics_output = str(diagnostics_copy_path)

        if (
            CloudOffloadMode(str(profile.cloud.get("mode", CloudOffloadMode.AUXILIARY_ONLY.value)).strip())
            != CloudOffloadMode.DISABLED
            and self._cloud_capability_modes(profile)[CloudOffloadCapability.EXPORT] == CloudOffloadMode.AUXILIARY_ONLY
        ):
            await self.dispatch_auxiliary_cloud_job(
                capability=CloudOffloadCapability.EXPORT,
                payload={
                    "bundle_dir": str(bundle_dir),
                    "manifest_path": str(manifest_path),
                    "launch_ready": launch.launch_ready,
                    "used_stub_fallback": launch.used_stub_fallback,
                    "copied_log_names": (
                        self.config.storage.events_log_name,
                        self.config.storage.cloud_offload_log_name,
                        self.config.storage.trace_log_name,
                        self.config.storage.web_log_name,
                        self.config.storage.status_log_name,
                    ),
                },
                active_profile=profile,
                job_id=f"support_bundle:{bundle_dir.name}",
                metadata={
                    "bundle_dir": str(bundle_dir),
                    "manifest_path": str(manifest_path),
                },
            )

        copied_artifact_paths: list[str] = []
        for artifact_name in (
            self.config.storage.events_log_name,
            self.config.storage.cloud_offload_log_name,
            self.config.storage.trace_log_name,
            self.config.storage.web_log_name,
            self.config.storage.status_log_name,
        ):
            source_path = self.config.storage.logs_dir / artifact_name
            destination_path = bundle_dir / artifact_name
            if source_path.exists():
                shutil.copy2(source_path, destination_path)
            else:
                destination_path.write_text("", encoding="utf-8")
            copied_artifact_paths.append(str(destination_path))

        bundle = PackagedSupportBundle(
            bundle_dir=str(bundle_dir),
            manifest_path=str(manifest_path),
            launch_report_path=str(launch_report_path),
            readiness_report_path=str(readiness_report_path),
            preflight_report_path=str(preflight_artifacts["report_path"]),
            onboarding_guide_path=str(preflight_artifacts["onboarding_path"]),
            setup_guide_path=str(preflight_artifacts["setup_path"]),
            user_settings_path=str(user_settings_path),
            app_state_path=str(app_state_path),
            support_readme_path=str(support_readme_path),
            diagnostics_path=diagnostics_output,
            copied_artifact_paths=tuple(copied_artifact_paths),
        )
        manifest_path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
        return bundle

    def _build_dashboard_readiness_report(
        self,
        *,
        active_profile: UserSettingsProfile | None = None,
    ) -> DashboardReadinessReport:
        return self._build_dashboard_readiness_report_from_state(
            self._collect_readiness_state(active_profile=active_profile)
        )

    def _build_dashboard_readiness_report_from_state(
        self,
        state: _ReadinessState,
    ) -> DashboardReadinessReport:
        real_mode_ready = (
            not state.requested_stub_mode
            and state.snapshot.started
            and state.snapshot.generation_backend not in {"stub_generation"}
            and state.snapshot.embedding_backend not in {"stub_embedding"}
            and state.primary_generation_ready
            and state.primary_embedding_ready
        )
        checks = (
            DashboardReadinessCheck(
                check_id="stub_mode",
                title="Stub Mode",
                status="ready" if state.requested_stub_mode else "disabled",
                detail=(
                    "Stub mode is enabled and the local lightweight pipeline can run without heavy model dependencies."
                    if state.requested_stub_mode
                    else "Stub mode is disabled; real backends are expected."
                ),
                recovery_actions=(
                    ("Enable stub mode for first-run local validation.",)
                    if not state.requested_stub_mode
                    else ()
                ),
            ),
            DashboardReadinessCheck(
                check_id="runtime_health",
                title="Runtime Health",
                status="ready" if state.snapshot.started else "blocked",
                detail=(
                    f"gen={state.snapshot.generation_backend}, embed={state.snapshot.embedding_backend}, "
                    f"fallback={'yes' if state.snapshot.fallback_active else 'no'}"
                ),
                recovery_actions=(
                    ("Restart the local app or inspect backend logs if startup failed.",)
                    if not state.snapshot.started or state.snapshot.last_error
                    else ()
                ),
            ),
            DashboardReadinessCheck(
                check_id="ollama_service",
                title="Ollama Service",
                status=state.ollama_status,
                detail=state.ollama_detail,
                recovery_actions=(
                    ("Start the local Ollama service before enabling real-mode Ollama backends.",)
                    if (
                        state.any_uses_ollama
                        and not state.requested_stub_mode
                        and not state.ollama_service_ready
                    )
                    else ()
                ),
            ),
            DashboardReadinessCheck(
                check_id="vector_backend",
                title="Vector Backend Dependency",
                status="ready" if state.chromadb_available else "degraded",
                detail=(
                    "chromadb import is available."
                    if state.chromadb_available
                    else "chromadb is missing; the runtime can still fall back to the simple in-memory index."
                ),
                recovery_actions=(
                    ("Install chromadb to restore the primary persistent vector backend.",)
                    if not state.chromadb_available
                    else ()
                ),
            ),
            DashboardReadinessCheck(
                check_id="embedding_dependency",
                title="Embedding Dependency",
                status="ready" if state.sentence_transformers_available else "degraded",
                detail=(
                    "sentence-transformers is importable."
                    if state.sentence_transformers_available
                    else "sentence-transformers is missing; real embedding mode will fall back or stay unavailable."
                ),
                recovery_actions=(
                    ("Install sentence-transformers for the default local embedding path.",)
                    if not state.sentence_transformers_available
                    else ()
                ),
            ),
            DashboardReadinessCheck(
                check_id="llama_cpp_model_file",
                title="llama.cpp Model File",
                status=state.llama_cpp_model_status,
                detail=state.llama_cpp_model_detail,
                recovery_actions=(
                    ("Place the configured GGUF model file in the models directory or update the configured path.",)
                    if state.llama_cpp_model_required and not state.llama_cpp_model_ready
                    else ()
                ),
            ),
            DashboardReadinessCheck(
                check_id="llama_cpp_dependency",
                title="llama.cpp Fallback Dependency",
                status="ready" if state.llama_cpp_available else "degraded",
                detail=(
                    "llama_cpp is importable."
                    if state.llama_cpp_available
                    else "llama_cpp is missing; the configured generation fallback backend cannot load yet."
                ),
                recovery_actions=(
                    ("Install llama-cpp-python if you want the local fallback generation backend.",)
                    if not state.llama_cpp_available
                    else ()
                ),
            ),
            self._build_specialist_role_readiness_check(state.profile),
        )
        capability_registry = self.capability_policy.build_registry_view(profile=state.profile, snapshot=state.snapshot)
        capabilities = (
            self._build_desktop_control_summary_capability(capability_registry, state.profile),
            *(
                self._build_dashboard_capability_from_registration(item)
                for item in capability_registry.registrations
            ),
            self._build_observation_tier_summary_capability(state.profile, snapshot=state.snapshot),
            self._build_cloud_offload_summary_capability(state.profile),
            *(
                self._build_cloud_offload_capability(state.profile, capability)
                for capability in CloudOffloadCapability
            ),
            DashboardCapabilityAvailability(
                capability_name="real_mode",
                status="ready" if real_mode_ready else "degraded",
                reason=(
                    "stub_mode_enabled"
                    if state.requested_stub_mode
                    else (
                        "fallback_active"
                        if state.snapshot.fallback_active
                        else (
                            "missing_ollama_service"
                            if state.any_uses_ollama and not state.ollama_service_ready
                            else "missing_real_dependencies"
                        )
                    )
                ),
                detail="Real mode requires local dependencies plus non-stub generation and embedding backends to be available.",
                recovery_actions=(
                    ("Disable stub mode only after real dependencies and local services are installed.",)
                    if state.requested_stub_mode
                    else ("Install missing real-mode dependencies or local services.",)
                ),
            ),
        )
        guidance = (
            "The local app shell is complete enough for stub-mode runs, profile management, history inspection, and knowledge management.",
            "Use readiness warnings to decide whether to stay in lightweight stub mode or finish real-backend setup.",
            (
                "Pinned default model bundle: "
                f"{self.config.preflight.backends.generation_backend}:{self.config.preflight.backends.generation_model} "
                "for generation, "
                f"{self.config.preflight.backends.generation_fallback_backend}:{self.config.preflight.backends.generation_fallback_model} "
                "for generation fallback, "
                f"{self.config.preflight.backends.embedding_backend}:{self.config.preflight.backends.embedding_model} "
                "for embeddings, and "
                f"{self.config.preflight.backends.embedding_fallback_backend}:{self.config.preflight.backends.embedding_fallback_model} "
                "for embedding fallback."
            ),
            (
                f"Hardware target remains {self.config.preflight.hardware.max_vram_gb:.0f}GB VRAM / "
                f"{self.config.preflight.hardware.max_ram_gb:.0f}GB RAM, and user data stays local under "
                f"'{self.config.storage.sqlite_path}' plus '{self.config.storage.logs_dir}'."
            ),
            (
                "Stub mode is the first-run packaged path; real mode stays opt-in, while cloud helpers remain "
                "auxiliary-only and can offload approved content only after explicit enablement."
            ),
            f"Step-by-step local backend and model setup lives in '{_LOCAL_MODEL_SETUP_GUIDE}'.",
            "Optional specialist roles can be enabled individually in Settings without replacing the base generation and embedding runtime.",
            "The live control tier now covers bounded file, shell, browser, app/window, screenshot, OCR, approval-gated desktop input, and auxiliary cloud helpers that always preserve local fallback.",
        )
        return DashboardReadinessReport(
            stub_mode_ready=True,
            real_mode_ready=real_mode_ready,
            checks=checks,
            capabilities=capabilities,
            guidance=guidance,
        )

    @staticmethod
    def _dashboard_capability_status(registration: CapabilityRegistration) -> str:
        if registration.enabled and registration.status == CapabilityAvailabilityStatus.AVAILABLE:
            return "ready"
        if registration.status == CapabilityAvailabilityStatus.REQUIRES_APPROVAL:
            return "requires_approval"
        if registration.status == CapabilityAvailabilityStatus.DEGRADED:
            return "degraded"
        if registration.status == CapabilityAvailabilityStatus.DENIED_BY_POLICY:
            return "blocked_by_policy"
        if registration.enabled:
            return "degraded"
        return "visible_not_enabled"

    def _build_dashboard_capability_from_registration(
        self,
        registration: CapabilityRegistration,
    ) -> DashboardCapabilityAvailability:
        return DashboardCapabilityAvailability(
            capability_name=registration.capability_type.value,
            status=self._dashboard_capability_status(registration),
            reason=registration.reason,
            detail=registration.detail,
            recovery_actions=(
                ("Enable the capability in Settings before requesting it.",)
                if registration.reason == "capability_not_enabled"
                else (
                    ("Adjust allowlists or disable risky request flags.",)
                    if "allowlist" in registration.reason or "blocked" in registration.reason
                    else (
                        ("Wait for resource pressure to clear before retrying.",)
                        if registration.reason == "resource_pressure"
                        else ()
                    )
                )
            ),
        )

    def _build_desktop_control_summary_capability(
        self,
        registry_view: CapabilityRegistryView,
        profile: UserSettingsProfile,
    ) -> DashboardCapabilityAvailability:
        desktop_enabled = bool(profile.desktop.get("enabled"))
        any_enabled = any(item.enabled for item in registry_view.registrations)
        live_ready = tuple(
            item.capability_type.value
            for item in registry_view.registrations
            if item.executor_kind == "live" and item.enabled
        )
        ready_status = "visible_not_enabled"
        ready_reason = "desktop_mode_disabled"
        if desktop_enabled and not any_enabled:
            ready_reason = "desktop_capabilities_not_enabled"
        elif desktop_enabled and live_ready:
            ready_status = "ready"
            ready_reason = "live_control_tier_available"
        return DashboardCapabilityAvailability(
            capability_name="desktop_control",
            status=ready_status,
            reason=ready_reason,
            detail=(
                "The typed capability, policy, and session boundaries are in place. Bounded live execution now exists for "
                f"{', '.join(live_ready) if live_ready else 'the currently enabled local task families'}, while "
                "heavier observation tiers remain separate and desktop input stays approval-gated."
            ),
            recovery_actions=(
                ("Enable desktop mode and specific capabilities only when you are validating a bounded local task.",)
                if not desktop_enabled or not any_enabled
                else (
                    "Keep tasks inside allowlists and use heavier observation tiers only when screenshot-on-demand or OCR are insufficient.",
                )
            ),
        )

    def _build_observation_tier_summary_capability(
        self,
        profile: UserSettingsProfile,
        snapshot: ModelHealthSnapshot | None = None,
    ) -> DashboardCapabilityAvailability:
        snapshot = snapshot or self.model_manager.health_snapshot()
        observation_tier = self._requested_observation_tier(profile)
        effective_tier = self._effective_observation_tier(observation_tier, snapshot)
        enabled_roles = {
            str(item).strip()
            for item in profile.models.get("enabled_roles", ())
            if str(item).strip()
        }
        vision_role_ready = any(
            registration.enabled
            and registration.backend != "unconfigured"
            and registration.model_identifier != "placeholder"
            and not bool(registration.metadata.get("placeholder"))
            and not registration.missing_dependencies
            for registration in self.model_manager.list_registered_models(role=ModelRole.VISION)
        ) and ModelRole.VISION.value in enabled_roles
        if effective_tier != observation_tier:
            pressure_detail = (
                ", ".join(snapshot.governor_pressure_reasons)
                if snapshot.governor_pressure_reasons
                else "hardware governor pressure"
            )
            return DashboardCapabilityAvailability(
                capability_name="observation_tiers",
                status="degraded",
                reason=f"hardware_governor_{observation_tier}_degraded",
                detail=(
                    f"Requested {observation_tier} is temporarily degraded to {effective_tier} because "
                    f"{pressure_detail}."
                ),
                recovery_actions=("Wait for pressure to clear before retrying heavier observation tiers.",),
            )
        if observation_tier == "screenshot_on_demand":
            return DashboardCapabilityAvailability(
                capability_name="observation_tiers",
                status="ready",
                reason="screenshot_on_demand_live",
                detail=(
                    "Screenshot-on-demand is live as the lightest observation tier. OCR-on-step and continuous capture "
                    "are separate opt-in live tiers, while vision-on-step currently degrades through CPU OCR until a "
                    "routed vision backend lands."
                ),
                recovery_actions=("Keep heavier observation tiers disabled unless the current task actually needs them.",),
            )
        if observation_tier == "ocr_on_step":
            return DashboardCapabilityAvailability(
                capability_name="observation_tiers",
                status="ready",
                reason="ocr_on_step_live",
                detail=(
                    "Screenshot-on-demand and per-step CPU-first OCR are active. Vision-on-step and continuous capture "
                    "remain separate, gated observation tiers."
                ),
                recovery_actions=("Use OCR-on-step for bounded text extraction from screenshots or selected regions.",),
            )
        if observation_tier == "vision_on_step":
            if vision_role_ready:
                return DashboardCapabilityAvailability(
                    capability_name="observation_tiers",
                    status="ready",
                    reason="vision_on_step_routed",
                    detail=(
                        "Vision-on-step is active with an optional routed vision role. CPU OCR still runs first as a "
                        "bounded seed, and the heavier visual role swaps in only on demand."
                    ),
                    recovery_actions=(
                        "Keep the vision role enabled only for steps that genuinely need UI interpretation beyond CPU OCR.",
                    ),
                )
            return DashboardCapabilityAvailability(
                capability_name="observation_tiers",
                status="degraded",
                reason="vision_on_step_cpu_fallback",
                detail=(
                    "Vision-on-step is now an explicit opt-in per-step mode, but until the routed vision executor lands "
                    "it captures bounded screenshots and falls back to CPU OCR with visible route and degrade reasons."
                ),
                recovery_actions=(
                    "Enable a concrete vision role when you need routed visual inspection; use ocr_on_step for lighter text-first tasks.",
                ),
            )
        if observation_tier == "continuous_capture":
            return DashboardCapabilityAvailability(
                capability_name="observation_tiers",
                status="ready",
                reason="continuous_capture_live",
                detail=(
                    "Continuous capture is active behind low-FPS, downscaled, diff-based, and region-aware caps. "
                    "OCR-on-step and vision-on-step remain separate observation tiers."
                ),
                recovery_actions=("Keep continuous capture task-scoped and rely on the strict capture caps for bounded sampling.",),
            )
        return DashboardCapabilityAvailability(
            capability_name="observation_tiers",
            status="blocked_by_policy",
            reason="phase_22_partial_live_execution",
            detail=(
                "The active observation tier still depends on later Phase 22 work. Screenshot-on-demand, CPU-first OCR, "
                "and bounded continuous capture are available today, but vision-on-step is not."
            ),
            recovery_actions=("Keep observation tier at screenshot_on_demand, ocr_on_step, or continuous_capture until later observation phases land.",),
        )

    def _cloud_capability_modes(self, profile: UserSettingsProfile) -> dict[CloudOffloadCapability, CloudOffloadMode]:
        raw_modes = dict(profile.cloud.get("capability_modes", {}))
        resolved: dict[CloudOffloadCapability, CloudOffloadMode] = {}
        for capability in CloudOffloadCapability:
            raw_mode = str(raw_modes.get(capability.value, CloudOffloadMode.DISABLED.value)).strip()
            try:
                resolved[capability] = CloudOffloadMode(raw_mode)
            except ValueError:
                resolved[capability] = CloudOffloadMode.DISABLED
        return resolved

    def _cloud_provider_name(self, profile: UserSettingsProfile) -> str:
        return str(profile.cloud.get("provider", "stub_cloud")).strip() or "stub_cloud"

    def _cloud_provider_family(self, profile: UserSettingsProfile) -> str:
        configured_family = str(profile.cloud.get("provider_family", "provider_agnostic")).strip() or "provider_agnostic"
        return self.cloud_offload.provider_family(self._cloud_provider_name(profile), fallback=configured_family)

    def _cloud_fallback_behavior(self, profile: UserSettingsProfile) -> CloudFallbackBehavior:
        return CloudFallbackBehavior(
            str(profile.cloud.get("fallback_behavior", CloudFallbackBehavior.RETRY_THEN_LOCAL.value)).strip()
        )

    def _cloud_job_contract_for_capability(
        self,
        *,
        profile: UserSettingsProfile,
        capability: CloudOffloadCapability,
    ) -> CloudJobContract | None:
        cloud_mode = CloudOffloadMode(str(profile.cloud.get("mode", CloudOffloadMode.AUXILIARY_ONLY.value)).strip())
        if cloud_mode == CloudOffloadMode.DISABLED:
            return None
        capability_mode = self._cloud_capability_modes(profile)[capability]
        if capability_mode == CloudOffloadMode.DISABLED:
            return None
        _title, payload_class, privacy_class, requires_content_approval = _CLOUD_OFFLOAD_CAPABILITY_DETAILS[capability]
        allow_cloud_content = bool(profile.privacy.get("allow_cloud_content", False))
        if requires_content_approval and not allow_cloud_content:
            return None
        return CloudJobContract(
            job_id=f"readiness:{capability.value}",
            capability=capability,
            payload_class=payload_class,
            privacy_class=privacy_class,
            max_payload_bytes=int(profile.cloud.get("max_payload_bytes", 1024 * 256) or 1024 * 256),
            max_retries=int(profile.cloud.get("max_retries", 1) or 0),
            fallback_behavior=self._cloud_fallback_behavior(profile),
            dispatch_mode=capability_mode,
            provider_family=self._cloud_provider_family(profile),
            content_approved=privacy_class == CloudJobPrivacyClass.APPROVED_CONTENT and allow_cloud_content,
        )

    def _build_cloud_offload_record(
        self,
        *,
        profile: UserSettingsProfile,
        contract: CloudJobContract,
        outcome: CloudOffloadOutcome,
        summary: str,
        detail: str,
        fallback_reason: str = "",
        local_fallback_used: bool = False,
        response_ref: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CloudOffloadRecord:
        return CloudOffloadRecord(
            dispatch_id=f"cloud_{stable_hash(f'{contract.job_id}:{outcome.value}:{utc_now().isoformat()}')[:16]}",
            job_id=contract.job_id,
            capability=contract.capability,
            provider_name=self._cloud_provider_name(profile),
            provider_family=self._cloud_provider_family(profile),
            payload_class=contract.payload_class,
            privacy_class=contract.privacy_class,
            outcome=outcome,
            summary=summary,
            detail=detail,
            fallback_behavior=contract.fallback_behavior,
            local_fallback_used=local_fallback_used,
            fallback_reason=fallback_reason,
            response_ref=response_ref,
            metadata=dict(metadata or {}),
        )

    def _build_cloud_offload_summary_capability(
        self,
        profile: UserSettingsProfile,
    ) -> DashboardCapabilityAvailability:
        cloud_mode = CloudOffloadMode(str(profile.cloud.get("mode", CloudOffloadMode.AUXILIARY_ONLY.value)).strip())
        capability_modes = self._cloud_capability_modes(profile)
        enabled_capabilities = [
            capability
            for capability, mode in capability_modes.items()
            if mode == CloudOffloadMode.AUXILIARY_ONLY
        ]
        provider_name = self._cloud_provider_name(profile)
        provider_family = self._cloud_provider_family(profile)
        fallback_behavior = str(
            profile.cloud.get("fallback_behavior", CloudFallbackBehavior.RETRY_THEN_LOCAL.value)
        ).strip()
        if cloud_mode == CloudOffloadMode.DISABLED:
            return DashboardCapabilityAvailability(
                capability_name="cloud_offload",
                status="visible_not_enabled",
                reason="cloud_mode_disabled",
                detail=(
                    "Cloud offload is globally disabled. Per-capability cloud selections remain stored, but the "
                    "runtime stays fully local-first until you explicitly re-enable auxiliary mode."
                ),
                recovery_actions=(
                    "Switch cloud mode to auxiliary_only only for specific helper categories you actually want to test.",
                ),
            )
        if not enabled_capabilities:
            return DashboardCapabilityAvailability(
                capability_name="cloud_offload",
                status="visible_not_enabled",
                reason="cloud_capabilities_not_enabled",
                detail=(
                    "Cloud offload remains auxiliary-only and provider-agnostic, but no individual capability is "
                    "enabled yet. The baseline local pipeline remains the only active path."
                ),
                recovery_actions=("Enable only the cloud helper categories you explicitly want to evaluate.",),
            )
        enabled_labels = [
            _CLOUD_OFFLOAD_CAPABILITY_DETAILS[capability][0]
            for capability in enabled_capabilities
        ]
        eligible_capabilities = [
            capability
            for capability in enabled_capabilities
            if self._cloud_job_contract_for_capability(profile=profile, capability=capability) is not None
        ]
        if not eligible_capabilities:
            return DashboardCapabilityAvailability(
                capability_name="cloud_offload",
                status="blocked_by_policy",
                reason="cloud_content_not_approved",
                detail=(
                    "Cloud helper categories are selected, but each enabled category still requires approved content "
                    "and Allow cloud content is off. Local fallback remains the only active path."
                ),
                recovery_actions=(
                    "Leave content approval off unless you explicitly want approved task data to become eligible for auxiliary offload.",
                ),
            )
        if not any(
            self.cloud_offload.provider_available(provider_name, capability=capability)
            for capability in eligible_capabilities
        ):
            return DashboardCapabilityAvailability(
                capability_name="cloud_offload",
                status="degraded",
                reason="cloud_provider_unavailable",
                detail=(
                    "Auxiliary cloud helpers are enabled for "
                    f"{', '.join(enabled_labels)} with provider={provider_name}, provider_family={provider_family}, "
                    f"and fallback={fallback_behavior}, but the configured provider is unavailable so local execution "
                    "remains authoritative."
                ),
                recovery_actions=(
                    "Switch to an available provider or disable helper categories you are not actively evaluating.",
                ),
            )
        return DashboardCapabilityAvailability(
            capability_name="cloud_offload",
            status="ready",
            reason="cloud_auxiliary_available",
            detail=(
                "Auxiliary cloud helpers are available for "
                f"{', '.join(enabled_labels)} with provider={provider_name}, provider_family={provider_family}, "
                f"and fallback={fallback_behavior}. Local execution remains authoritative if the helper fails."
            ),
            recovery_actions=(
                "Keep cloud helpers auxiliary-only and enable only the categories you explicitly want to test.",
            ),
        )

    def _build_cloud_offload_capability(
        self,
        profile: UserSettingsProfile,
        capability: CloudOffloadCapability,
    ) -> DashboardCapabilityAvailability:
        title, payload_class, privacy_class, requires_content_approval = _CLOUD_OFFLOAD_CAPABILITY_DETAILS[capability]
        capability_name = f"cloud_{capability.value}"
        cloud_mode = CloudOffloadMode(str(profile.cloud.get("mode", CloudOffloadMode.AUXILIARY_ONLY.value)).strip())
        if cloud_mode == CloudOffloadMode.DISABLED:
            return DashboardCapabilityAvailability(
                capability_name=capability_name,
                status="visible_not_enabled",
                reason="cloud_mode_disabled",
                detail=f"{title} is forced local-only while global cloud mode is disabled.",
                recovery_actions=("Keep the global cloud mode disabled unless you need an auxiliary helper category.",),
            )
        capability_mode = self._cloud_capability_modes(profile)[capability]
        if capability_mode == CloudOffloadMode.DISABLED:
            return DashboardCapabilityAvailability(
                capability_name=capability_name,
                status="visible_not_enabled",
                reason="cloud_capability_disabled",
                detail=f"{title} is disabled independently, so enabling another cloud helper will not enable this one.",
                recovery_actions=(f"Enable {title} only if you want this specific helper category to become eligible later.",),
            )
        if requires_content_approval and not bool(profile.privacy.get("allow_cloud_content", False)):
            return DashboardCapabilityAvailability(
                capability_name=capability_name,
                status="blocked_by_policy",
                reason="cloud_content_not_approved",
                detail=(
                    f"{title} is configured as auxiliary_only, but its payload={payload_class.value} and "
                    f"privacy={privacy_class.value} require explicit content approval before any future dispatch."
                ),
                recovery_actions=("Turn on Allow cloud content only if you explicitly accept approved-content offload.",),
            )
        contract = self._cloud_job_contract_for_capability(profile=profile, capability=capability)
        if contract is None:
            return DashboardCapabilityAvailability(
                capability_name=capability_name,
                status="blocked_by_policy",
                reason="cloud_contract_unavailable",
                detail=f"{title} has no valid cloud contract yet, so it stays local-only.",
                recovery_actions=("Review cloud settings and keep the capability disabled until you need it.",),
            )
        provider_name = self._cloud_provider_name(profile)
        provider_family = self._cloud_provider_family(profile)
        if not self.cloud_offload.provider_available(provider_name, capability=capability):
            return DashboardCapabilityAvailability(
                capability_name=capability_name,
                status="degraded",
                reason="cloud_provider_unavailable",
                detail=(
                    f"{title} is configured with payload={contract.payload_class.value}, "
                    f"privacy={contract.privacy_class.value}, provider={provider_name}, "
                    f"provider_family={provider_family}, and fallback={contract.fallback_behavior.value}, "
                    "but the configured provider is unavailable so local fallback remains authoritative."
                ),
                recovery_actions=("Switch to an available provider or disable this helper category.",),
            )
        return DashboardCapabilityAvailability(
            capability_name=capability_name,
            status="ready",
            reason="cloud_auxiliary_ready",
            detail=(
                f"{title} is staged with payload={contract.payload_class.value}, privacy={contract.privacy_class.value}, "
                f"bytes<={contract.max_payload_bytes}, retries={contract.max_retries}, "
                f"fallback={contract.fallback_behavior.value}, provider={provider_name}, "
                f"provider_family={provider_family}. Local fallback remains authoritative if the helper fails."
            ),
            recovery_actions=("Keep this helper auxiliary-only and disable it when you do not need it.",),
        )

    def _collect_readiness_state(
        self,
        *,
        active_profile: UserSettingsProfile | None = None,
    ) -> _ReadinessState:
        profile = active_profile or (
            self.dashboard.app_state_snapshot().user_settings if self._started else self._default_user_settings_profile()
        )
        runtime_config = self._config_for_user_settings_profile(profile)
        snapshot = self.model_manager.health_snapshot()
        chromadb_available = self._dependency_available("chromadb")
        sentence_transformers_available = self._dependency_available("sentence_transformers")
        llama_cpp_available = self._dependency_available("llama_cpp")
        backends = runtime_config.preflight.backends
        primary_generation_backend = backends.generation_backend
        primary_embedding_backend = backends.embedding_backend
        primary_uses_ollama = primary_generation_backend == "ollama" or primary_embedding_backend == "ollama_embeddings"
        any_uses_ollama = (
            primary_uses_ollama
            or backends.generation_fallback_backend == "ollama"
            or backends.embedding_fallback_backend == "ollama_embeddings"
        )
        if runtime_config.preflight.flags.stub_mode:
            ollama_service_ready = False
            ollama_status = "disabled" if any_uses_ollama else "not_required"
            ollama_detail = (
                "Stub mode is enabled; Ollama service probing is skipped until real mode is requested."
                if any_uses_ollama
                else "No configured backend requires Ollama service access."
            )
        elif any_uses_ollama:
            ollama_service_ready, ollama_detail = self._probe_ollama_service(config=runtime_config)
            if ollama_service_ready:
                ollama_status = "ready"
            else:
                ollama_status = "blocked" if primary_uses_ollama else "degraded"
        else:
            ollama_service_ready = True
            ollama_status = "not_required"
            ollama_detail = "No configured backend requires Ollama service access."

        llama_cpp_model_ready, llama_cpp_model_detail, llama_cpp_model_required, llama_cpp_model_blocking = (
            self._check_llama_cpp_model_files(config=runtime_config)
        )
        if not llama_cpp_model_required:
            llama_cpp_model_status = "not_required"
        elif llama_cpp_model_ready:
            llama_cpp_model_status = "ready"
        else:
            llama_cpp_model_status = "blocked" if llama_cpp_model_blocking else "degraded"

        primary_generation_ready = (
            (primary_generation_backend != "ollama" or ollama_service_ready)
            and (primary_generation_backend != "llama_cpp" or (llama_cpp_available and llama_cpp_model_ready))
        )
        primary_embedding_ready = (
            (primary_embedding_backend != "sentence_transformers" or sentence_transformers_available)
            and (primary_embedding_backend != "ollama_embeddings" or ollama_service_ready)
        )
        return _ReadinessState(
            profile=profile,
            requested_stub_mode=runtime_config.preflight.flags.stub_mode,
            snapshot=snapshot,
            sentence_transformers_available=sentence_transformers_available,
            chromadb_available=chromadb_available,
            llama_cpp_available=llama_cpp_available,
            primary_generation_backend=primary_generation_backend,
            primary_embedding_backend=primary_embedding_backend,
            primary_uses_ollama=primary_uses_ollama,
            any_uses_ollama=any_uses_ollama,
            ollama_service_ready=ollama_service_ready,
            ollama_status=ollama_status,
            ollama_detail=ollama_detail,
            llama_cpp_model_ready=llama_cpp_model_ready,
            llama_cpp_model_detail=llama_cpp_model_detail,
            llama_cpp_model_required=llama_cpp_model_required,
            llama_cpp_model_blocking=llama_cpp_model_blocking,
            llama_cpp_model_status=llama_cpp_model_status,
            primary_generation_ready=primary_generation_ready,
            primary_embedding_ready=primary_embedding_ready,
        )

    def _build_specialist_role_readiness_check(
        self,
        profile: UserSettingsProfile,
    ) -> DashboardReadinessCheck:
        enabled_roles = {
            str(item)
            for item in profile.models.get("enabled_roles", ())
            if str(item).strip()
        }
        specialist_roles = (
            ModelRole.RERANKER,
            ModelRole.SPEECH_TO_TEXT,
            ModelRole.TEXT_TO_SPEECH,
            ModelRole.VAD,
            ModelRole.TRANSLATION,
            ModelRole.CODE_SPECIALIST,
        )
        enabled_specialists = [role for role in specialist_roles if role.value in enabled_roles]
        summaries: list[str] = []
        blocked_roles: list[str] = []
        for role in specialist_roles:
            registrations = self.model_manager.list_registered_models(role=role)
            selected = self._select_readiness_registration(role, registrations)
            role_enabled = role.value in enabled_roles
            if selected is None:
                backend_summary = "(none)"
                role_ready = False
                reason = "no registration"
            else:
                backend_summary = f"{selected.backend} / {selected.model_identifier}"
                role_ready = (
                    role_enabled
                    and selected.enabled
                    and selected.backend != "unconfigured"
                    and not selected.missing_dependencies
                )
                if not role_enabled:
                    reason = "disabled"
                elif selected.backend == "unconfigured":
                    reason = "not installed"
                elif selected.missing_dependencies:
                    reason = f"missing {', '.join(selected.missing_dependencies)}"
                else:
                    reason = "ready"
            summaries.append(
                f"{role.value}: {backend_summary} | {'enabled' if role_enabled else 'disabled'} | {reason}"
            )
            if role_enabled and not role_ready:
                blocked_roles.append(role.value)

        if not enabled_specialists:
            status = "disabled"
        elif blocked_roles:
            status = "degraded"
        else:
            status = "ready"

        recovery_actions: tuple[str, ...]
        if blocked_roles:
            recovery_actions = (
                "Install or configure the enabled specialist backends shown in the local AI control plane.",
                "Disable unused specialist roles in Settings to preserve the default lightweight runtime.",
            )
        else:
            recovery_actions = (
                "Enable only the specialist roles you want to keep local and lightweight.",
            )
        return DashboardReadinessCheck(
            check_id="specialist_roles",
            title="Specialist Roles",
            status=status,
            detail=" ; ".join(summaries),
            recovery_actions=recovery_actions,
        )

    @staticmethod
    def _select_readiness_registration(
        role: ModelRole,
        registrations,
    ):
        role_registrations = tuple(registrations)
        if not role_registrations:
            return None
        for registration in role_registrations:
            if registration.enabled and registration.backend != "unconfigured":
                return registration
        for registration in role_registrations:
            if registration.backend != "unconfigured":
                return registration
        for registration in role_registrations:
            if registration.role == role:
                return registration
        return role_registrations[0]

    def _packaged_launch_blocker(self, state: _ReadinessState) -> tuple[str, str]:
        if state.primary_generation_backend == "ollama" and not state.ollama_service_ready:
            return ("missing_ollama_service", state.ollama_detail)
        if state.primary_generation_backend == "llama_cpp" and not state.llama_cpp_available:
            return (
                "missing_llama_cpp_dependency",
                "llama_cpp is missing; the primary local generation backend cannot load yet.",
            )
        if state.primary_generation_backend == "llama_cpp" and not state.llama_cpp_model_ready:
            return ("missing_llama_cpp_model_file", state.llama_cpp_model_detail)
        if (
            state.primary_embedding_backend == "sentence_transformers"
            and not state.sentence_transformers_available
        ):
            return (
                "missing_embedding_dependency",
                "sentence-transformers is missing; the default embedding backend is unavailable.",
            )
        if state.primary_embedding_backend == "ollama_embeddings" and not state.ollama_service_ready:
            return ("missing_ollama_service", state.ollama_detail)
        return ("", "")

    def _packaged_runtime_config(
        self,
        launch_report: PackagedLaunchReport,
    ) -> AppConfig:
        desired_stub_mode = launch_report.effective_mode != "real"
        if self.config.preflight.flags.stub_mode == desired_stub_mode:
            return self.config
        next_flags = replace(self.config.preflight.flags, stub_mode=desired_stub_mode)
        return replace(self.config, preflight=replace(self.config.preflight, flags=next_flags))

    def _dependency_available(self, module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None

    def _probe_ollama_service(self, *, config: AppConfig | None = None) -> tuple[bool, str]:
        runtime_config = config or self.config
        base_url = runtime_config.backend_runtime.ollama_base_url.rstrip("/")
        request_url = f"{base_url}/api/tags"
        timeout_s = min(2.0, max(0.25, float(runtime_config.backend_runtime.request_timeout_s)))
        request = urllib_request.Request(request_url, headers={"User-Agent": "QuesterAI/0.1"})
        try:
            with urllib_request.urlopen(request, timeout=timeout_s) as response:
                status_code = getattr(response, "status", None) or response.getcode()
            if int(status_code) >= 400:
                return False, f"Ollama probe at {request_url} returned HTTP {status_code}."
            return True, f"Ollama responded successfully at {request_url}."
        except (urllib_error.URLError, TimeoutError, ValueError) as exc:
            return False, f"Ollama service probe failed at {request_url}: {exc}"

    def _check_llama_cpp_model_files(
        self,
        *,
        config: AppConfig | None = None,
    ) -> tuple[bool, str, bool, bool]:
        runtime_config = config or self.config
        backends = runtime_config.preflight.backends
        configured_models: list[tuple[str, Path]] = []
        if backends.generation_backend == "llama_cpp":
            configured_models.append(
                ("primary_generation", self._resolve_llama_cpp_model_path(backends.generation_model, config=runtime_config))
            )
        if backends.generation_fallback_backend == "llama_cpp":
            configured_models.append(
                (
                    "fallback_generation",
                    self._resolve_llama_cpp_model_path(backends.generation_fallback_model, config=runtime_config),
                )
            )
        if not configured_models:
            return True, "No configured backend requires a llama.cpp model file.", False, False
        missing = [(role, path) for role, path in configured_models if not path.exists()]
        if not missing:
            detail = "; ".join(
                f"{role} model file present at {path}"
                for role, path in configured_models
            )
            return True, detail, True, False
        detail = "; ".join(f"{role} model file missing at {path}" for role, path in missing)
        blocking = any(role == "primary_generation" for role, _ in missing)
        return False, detail, True, blocking

    def _resolve_llama_cpp_model_path(self, model_name: str, *, config: AppConfig | None = None) -> Path:
        runtime_config = config or self.config
        model_path = Path(model_name)
        if model_path.is_absolute():
            return model_path
        return runtime_config.backend_runtime.models_dir / model_path

    def _submit_dashboard_coroutine(self, coroutine: Any) -> None:
        if self._loop is None:
            raise RuntimeError("Orchestrator event loop is not available for dashboard submissions.")
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        except Exception:
            coroutine.close()
            raise
        self._dashboard_futures.add(future)
        future.add_done_callback(self._log_dashboard_future_exception)

    async def _cancel_pending_dashboard_futures(self) -> None:
        pending = tuple(self._dashboard_futures)
        self._dashboard_futures.clear()
        for future in pending:
            future.cancel()
        if pending:
            await asyncio.sleep(0)

    def _log_dashboard_future_exception(self, future: ConcurrentFuture[Any]) -> None:
        self._dashboard_futures.discard(future)
        try:
            future.result()
        except ConcurrentCancelledError:
            return
        except Exception as exc:  # pragma: no cover - defensive callback logging
            self.logger.warning("Dashboard-triggered background action failed: %s", exc)

    async def _record_status(
        self,
        component: str,
        state: AgentState,
        *,
        task_id: str | None = None,
        message: str = "",
        severity: SeverityLevel = SeverityLevel.LOW,
    ) -> None:
        status = AgentStatus(
            component=component,
            state=state,
            task_id=task_id,
            severity=severity,
            message=message,
        )
        await self.storage.record_agent_status(status)

    async def _run_component(
        self,
        component: str,
        *,
        task_id: str | None,
        start_stage: str,
        done_stage: str,
        start_payload: dict[str, Any],
        run,
    ):
        await self._record_status(
            component,
            AgentState.RUNNING,
            task_id=task_id,
            message=f"{component} started",
        )
        await self._emit_event(start_stage, start_payload)
        retry_limit = self.config.preflight.flags.max_component_retries
        retry_backoff_s = self.config.preflight.flags.retry_backoff_s
        attempt = 0
        while True:
            before_snapshot = self.model_manager.health_snapshot()
            try:
                result = await run()
                after_snapshot = self.model_manager.health_snapshot()
                await self._emit_model_state_change_events(
                    component=component,
                    task_id=task_id,
                    before_snapshot=before_snapshot,
                    after_snapshot=after_snapshot,
                )
                return result
            except asyncio.CancelledError:
                await self._record_status(
                    component,
                    AgentState.ERROR,
                    task_id=task_id,
                    message=f"{component} cancelled",
                    severity=SeverityLevel.MEDIUM,
                )
                await self._emit_event(
                    done_stage.replace("_done", "_cancelled"),
                    {
                        "task_id": task_id,
                        "component": component,
                    },
                )
                raise
            except ResourcePressureError as exc:
                snapshot = self.model_manager.health_snapshot()
                await self._emit_runtime_condition_event(
                    "runtime.resource_pressure_detected",
                    category="resource_pressure",
                    component="model_manager",
                    reason=str(exc),
                    severity=SeverityLevel.HIGH,
                    task_id=task_id,
                    metadata={
                        "trigger_component": component,
                        "generation_backend": snapshot.generation_backend,
                        "embedding_backend": snapshot.embedding_backend,
                        "available_ram_gb": snapshot.available_ram_gb,
                        "total_ram_gb": snapshot.total_ram_gb,
                        "generation_backend_vram_gb": snapshot.generation_backend_vram_gb,
                        "embedding_backend_vram_gb": snapshot.embedding_backend_vram_gb,
                    },
                )
                await self._emit_health_snapshot(snapshot=snapshot, task_id=task_id)
                await self._record_status(
                    component,
                    AgentState.ERROR,
                    task_id=task_id,
                    message=str(exc),
                    severity=SeverityLevel.HIGH,
                )
                await self._emit_event(
                    done_stage.replace("_done", "_failed"),
                    {
                        "task_id": task_id,
                        "component": component,
                        "error": str(exc),
                    },
                )
                raise
            except (ModelTimeoutError, BackendUnavailableError, WebLookupTimeoutError) as exc:
                if attempt < retry_limit:
                    attempt += 1
                    await self._record_status(
                        component,
                        AgentState.RUNNING,
                        task_id=task_id,
                        message=f"{component} retrying after transient failure ({attempt}/{retry_limit})",
                        severity=SeverityLevel.MEDIUM,
                    )
                    await self._emit_event(
                        done_stage.replace("_done", "_retrying"),
                        {
                            "task_id": task_id,
                            "component": component,
                            "attempt": attempt,
                            "error": str(exc),
                        },
                    )
                    if retry_backoff_s > 0:
                        await asyncio.sleep(retry_backoff_s * attempt)
                    continue
                await self._record_status(
                    component,
                    AgentState.ERROR,
                    task_id=task_id,
                    message=str(exc),
                    severity=SeverityLevel.HIGH,
                )
                await self._emit_event(
                    done_stage.replace("_done", "_failed"),
                    {
                        "task_id": task_id,
                        "component": component,
                        "error": str(exc),
                    },
                )
                raise
            except Exception as exc:
                await self._record_status(
                    component,
                    AgentState.ERROR,
                    task_id=task_id,
                    message=str(exc),
                    severity=SeverityLevel.HIGH,
                )
                await self._emit_event(
                    done_stage.replace("_done", "_failed"),
                    {
                        "task_id": task_id,
                        "component": component,
                        "error": str(exc),
                    },
                )
                raise

    def _build_answer_text(
        self,
        *,
        evidence,
        reasoning,
        critique,
    ) -> str:
        return self.translation.render_answer(
            evidence=evidence,
            reasoning=reasoning,
            critique=critique,
        )

    async def _emit_model_state_change_events(
        self,
        *,
        component: str,
        task_id: str | None,
        before_snapshot: ModelHealthSnapshot,
        after_snapshot: ModelHealthSnapshot,
    ) -> None:
        fallback_changed = (
            after_snapshot.fallback_active
            and (
                not before_snapshot.fallback_active
                or before_snapshot.fallback_reason != after_snapshot.fallback_reason
                or before_snapshot.generation_backend != after_snapshot.generation_backend
                or before_snapshot.embedding_backend != after_snapshot.embedding_backend
            )
        )
        if fallback_changed:
            await self._emit_runtime_condition_event(
                "runtime.fallback_activated",
                category="fallback",
                component="model_manager",
                reason=after_snapshot.fallback_reason or "fallback_active",
                severity=SeverityLevel.MEDIUM,
                task_id=task_id,
                metadata={
                    "trigger_component": component,
                    "generation_backend": after_snapshot.generation_backend,
                    "embedding_backend": after_snapshot.embedding_backend,
                },
            )
            await self._emit_health_snapshot(snapshot=after_snapshot, task_id=task_id)

    def _model_health_payload(
        self,
        *,
        snapshot: ModelHealthSnapshot,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "started": snapshot.started,
            "generation_backend": snapshot.generation_backend,
            "embedding_backend": snapshot.embedding_backend,
            "active_generation_jobs": snapshot.active_generation_jobs,
            "active_embedding_jobs": snapshot.active_embedding_jobs,
            "active_heavy_roles": list(snapshot.active_heavy_roles),
            "heavy_slot_limit": snapshot.heavy_slot_limit,
            "last_used_at": snapshot.last_used_at,
            "fallback_active": snapshot.fallback_active,
            "fallback_reason": snapshot.fallback_reason or "",
            "available_ram_gb": snapshot.available_ram_gb,
            "total_ram_gb": snapshot.total_ram_gb,
            "generation_backend_vram_gb": snapshot.generation_backend_vram_gb,
            "embedding_backend_vram_gb": snapshot.embedding_backend_vram_gb,
            "governor_active": snapshot.governor_active,
            "governor_pressure_reasons": list(snapshot.governor_pressure_reasons),
            "governor_degraded_features": list(snapshot.governor_degraded_features),
            "queue_pressure": snapshot.queue_pressure,
            "backend_health_degraded": snapshot.backend_health_degraded,
            "allow_continuous_capture": snapshot.allow_continuous_capture,
            "allow_ocr_on_step": snapshot.allow_ocr_on_step,
            "allow_vision_on_step": snapshot.allow_vision_on_step,
            "allow_optional_heavy_residency": snapshot.allow_optional_heavy_residency,
            "allow_background_work": snapshot.allow_background_work,
            "governor_summary": snapshot.governor_summary,
            "telemetry_enabled": snapshot.telemetry_enabled,
            "last_error": snapshot.last_error or "",
        }
        if task_id is not None:
            payload["task_id"] = task_id
        return payload

    def _default_user_settings_profile(self) -> UserSettingsProfile:
        return UserSettingsProfile(
            profile_name="default",
            runtime={
                "stub_mode": self.config.preflight.flags.stub_mode,
                "allow_web_fallback": self.config.preflight.flags.allow_web_fallback,
                "enable_self_optimizer": self.config.preflight.flags.enable_self_optimizer,
                "generation_backend": self.config.preflight.backends.generation_backend,
                "embedding_backend": self.config.preflight.backends.embedding_backend,
                "vector_store_backend": self.config.preflight.backends.vector_store_backend,
            },
            retrieval={
                "allow_web_fallback": self.config.preflight.flags.allow_web_fallback,
                "provider": self.config.web.provider,
                "reranking": self.config.retrieval.enable_reranking,
            },
            reasoning={
                "thinking_minutes": 30,
                "mode": "auto",
            },
            long_horizon={
                "enabled": False,
                "wall_clock_minutes": 120,
                "cycle_budget_minutes": 120,
                "checkpoint_interval_minutes": 120,
                "duty_cycle_ratio": 0.75,
                "cooldown_seconds": 0.05,
                "max_resume_count": 5,
            },
            optimizer={
                "activation_policy": "proposal_only",
                "replay_limit": self.config.self_optimizer.replay_history_limit,
            },
            models={
                "preferred_by_role": {
                    "generation": (
                        f"{self.config.preflight.backends.generation_backend}:"
                        f"{self.config.preflight.backends.generation_model}"
                    ),
                    "embedding": (
                        f"{self.config.preflight.backends.embedding_backend}:"
                        f"{self.config.preflight.backends.embedding_model}"
                    ),
                },
                "enabled_roles": ("generation", "embedding"),
            },
            desktop={
                "enabled": False,
                "approval_policy": "approve_risky_only",
            },
            observation={
                "tier": "screenshot_on_demand",
                "continuous_capture": False,
                "ocr_on_step": False,
                "vision_on_step": False,
            },
            cloud={
                "enabled": False,
                "mode": CloudOffloadMode.AUXILIARY_ONLY.value,
                "provider": "stub_cloud",
                "provider_family": "provider_agnostic",
                "max_payload_bytes": 1024 * 256,
                "max_retries": 1,
                "fallback_behavior": CloudFallbackBehavior.RETRY_THEN_LOCAL.value,
                "capability_modes": {
                    capability.value: CloudOffloadMode.DISABLED.value
                    for capability in CloudOffloadCapability
                },
            },
            privacy={
                "log_runtime_events": True,
                "allow_cloud_content": False,
                "log_level": self.config.logging.level,
            },
            ui={
                "show_debug_pane": True,
                "app_shell": "tkinter",
            },
        )

    def _reasoning_event_payload(
        self,
        *,
        task_id: str,
        reasoning: CompressedTrace,
    ) -> dict[str, Any]:
        metadata = dict(reasoning.context_frames[0].metadata) if reasoning.context_frames else {}
        return {
            "task_id": task_id,
            "candidate_count": int(metadata.get("cc", len(reasoning.candidate_traces) or 1)),
            "candidate_trace_count": len(reasoning.candidate_traces),
            "selected_candidate_id": str(metadata.get("cid", "")),
            "selected_strategy": str(metadata.get("sa", "")),
            "selected_verifier": str(metadata.get("sv", "")),
            "candidate_score": float(metadata.get("ss", 0.0) or 0.0),
            "degraded_reason": str(metadata.get("dr", "")),
            "reasoning_mode": str(metadata.get("rm", "")),
        }

    def _research_event_payload(
        self,
        *,
        task_id: str,
        evidence,
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "local_result_count": len(evidence.local_results),
            "web_result_count": len(evidence.web_results),
            "used_web_fallback": evidence.used_web_fallback,
            "local_source_refs": [item.source_ref for item in evidence.local_results[:6]],
            "web_source_refs": [item.source_ref for item in evidence.web_results[:6]],
        }

    def _critique_event_payload(
        self,
        *,
        task_id: str,
        critique: CritiqueReport,
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "critique_result": critique.result.value,
            "verifier_type": critique.verifier_type,
            "candidate_score": critique.candidate_score,
            "repair_actions": list(critique.repair_actions),
            "degraded_reason": critique.degraded_reason,
            "failure_categories": list(critique.failure_categories),
            "provenance_coverage": critique.provenance_coverage,
            "drift_score": critique.drift_score,
            "proof_hash_match": critique.proof_hash_match,
        }

    def _completion_event_payload(
        self,
        *,
        task_id: str,
        evidence,
        reasoning: CompressedTrace,
        critique: CritiqueReport,
        answer_text: str,
        warning_count: int,
    ) -> dict[str, Any]:
        answer_metadata = self.translation.summarize_answer_metadata(
            evidence=evidence,
            reasoning=reasoning,
        )
        specialist_roles_used, specialist_role_explanations = self._task_specialist_usage(
            evidence=evidence,
            reasoning=reasoning,
        )
        return {
            **self._reasoning_event_payload(task_id=task_id, reasoning=reasoning),
            **self._critique_event_payload(task_id=task_id, critique=critique),
            "answer_text": answer_text,
            "supporting_evidence_ids": answer_metadata["supporting_evidence_ids"],
            "citation_refs": answer_metadata["citation_refs"],
            "specialist_roles_used": list(specialist_roles_used),
            "specialist_role_explanations": list(specialist_role_explanations),
            "warning_count": warning_count,
        }

    def _task_specialist_usage(
        self,
        *,
        evidence,
        reasoning: CompressedTrace,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        roles_used: list[str] = []
        explanations: list[str] = []
        local_results = tuple(evidence.local_results)
        reranker_used = any(bool(item.metadata.get("specialist_reranked")) for item in local_results)
        reranker_backend = ""
        reranker_reason = ""
        for item in local_results:
            backend = str(item.metadata.get("specialist_reranker_backend", "")).strip()
            reason = str(item.metadata.get("specialist_reranker_reason", "")).strip()
            if backend and not reranker_backend:
                reranker_backend = backend
            if reason and not reranker_reason:
                reranker_reason = reason
        if reranker_used:
            roles_used.append(ModelRole.RERANKER.value)
            explanations.append(
                f"reranker used via {reranker_backend or '(unknown)'} to reorder local retrieval candidates"
            )
        elif local_results:
            explanations.append(f"reranker not used: {reranker_reason or 'not_needed_or_disabled'}")
        else:
            explanations.append("reranker not used: no_local_candidates")

        selected_strategy = ""
        if reasoning.context_frames:
            selected_strategy = str(reasoning.context_frames[0].metadata.get("sa", "")).strip()
        if selected_strategy:
            explanations.append(f"selected reasoning strategy: {selected_strategy}")
        return tuple(dict.fromkeys(roles_used)), tuple(dict.fromkeys(explanations))

    async def _task_advisor_summaries(
        self,
        task_id: str,
        *,
        limit: int = 3,
    ) -> tuple[str, ...]:
        if not task_id:
            return ()
        records = await self.storage.list_optimizer_suggestion_records()
        summaries: list[str] = []
        for record in records:
            if task_id not in record.source_task_ids:
                continue
            summaries.append(f"{record.kind.value}: {record.summary}")
        return tuple(dict.fromkeys(summaries))[: max(1, limit)]

    async def _run_repair_loop(
        self,
        *,
        task_id: str,
        plan,
        evidence,
        budget,
        reasoner_handoff: ResearchReasonerHandoff,
        reasoning: CompressedTrace,
        critique: CritiqueReport,
    ) -> tuple[CompressedTrace, CritiqueReport, tuple[str, ...]]:
        max_repair_attempts = max(0, min(2, max(1, budget.critic_passes)))
        repair_history: list[str] = []
        current_reasoner_handoff = reasoner_handoff
        current_reasoning = reasoning
        current_critique = critique
        for attempt in range(1, max_repair_attempts + 1):
            if current_critique.is_valid:
                break
            repair_action = self._select_next_repair_action(
                repair_actions=current_critique.repair_actions,
                repair_history=tuple(repair_history),
            )
            if repair_action is None:
                break
            await self._emit_event(
                "pipeline.repair_started",
                {
                    "task_id": task_id,
                    "repair_action": repair_action,
                    "repair_attempt": attempt,
                    "critique_result": current_critique.result.value,
                },
            )
            previous_reasoning = current_reasoning
            current_reasoner_handoff, current_reasoning, applied = await self._apply_repair_action(
                task_id=task_id,
                plan=plan,
                evidence=evidence,
                budget=budget,
                reasoner_handoff=current_reasoner_handoff,
                reasoning=current_reasoning,
                critique=current_critique,
                repair_action=repair_action,
                repair_attempt=attempt,
            )
            if not applied:
                await self._emit_event(
                    "pipeline.repair_skipped",
                    {
                        "task_id": task_id,
                        "repair_action": repair_action,
                        "repair_attempt": attempt,
                    },
                )
                break
            repair_history.append(repair_action)
            await self._emit_event(
                "pipeline.repair_applied",
                {
                    "task_id": task_id,
                    "repair_action": repair_action,
                    "repair_attempt": attempt,
                    "trace_changed": current_reasoning != previous_reasoning,
                },
            )
            if current_reasoning != previous_reasoning:
                await self.storage.record_reasoning_trace(current_reasoning)
                await self.storage.record_reasoning_log(
                    ReasoningLog(
                        task_id=task_id,
                        compressed_chain=current_reasoning.tokens,
                        macros_used=current_reasoning.macros_used,
                    )
                )
            critic_handoff = ReasonerCriticHandoff.from_inputs(
                plan=plan,
                evidence=evidence,
                trace=current_reasoning,
                budget=budget,
                repair_attempt_count=attempt,
                repair_history=tuple(repair_history),
                final_text_policy=self.critic.service.final_text_policy,
                implementation_mode=self.critic.service.implementation_mode,
            )
            current_critique = await self._run_component(
                "critic",
                task_id=task_id,
                start_stage="pipeline.critic_repair_started",
                done_stage="pipeline.critic_repair_done",
                start_payload={
                    "task_id": task_id,
                    "repair_action": repair_action,
                    "repair_attempt": attempt,
                    "proof_hash": critic_handoff.proof_hash,
                },
                run=lambda critic_handoff=critic_handoff: self.critic.review_from_handoff(critic_handoff),
            )
            await self._emit_event(
                "pipeline.critic_repair_done",
                {
                    **self._critique_event_payload(task_id=task_id, critique=current_critique),
                    "repair_action": repair_action,
                    "repair_attempt": attempt,
                },
            )
            if current_critique.is_valid:
                break
        await self._emit_event(
            "pipeline.repair_completed",
            {
                "task_id": task_id,
                "repair_actions_applied": list(repair_history),
                "final_critique_result": current_critique.result.value,
            },
        )
        return current_reasoning, current_critique, tuple(repair_history)

    async def _apply_repair_action(
        self,
        *,
        task_id: str,
        plan,
        evidence,
        budget,
        reasoner_handoff: ResearchReasonerHandoff,
        reasoning: CompressedTrace,
        critique: CritiqueReport,
        repair_action: str,
        repair_attempt: int,
    ) -> tuple[ResearchReasonerHandoff, CompressedTrace, bool]:
        if repair_action == "replace_answer_with_tool_result":
            repaired_answer = self._repair_answer_from_tools(plan.question, evidence=evidence)
            if not repaired_answer:
                return reasoner_handoff, reasoning, False
            return (
                reasoner_handoff,
                self._replace_trace_answer(
                    reasoning,
                    answer_text=repaired_answer,
                    selected_strategy=self._tool_repair_strategy(plan.question, evidence=evidence),
                    selected_verifier=self._tool_repair_verifier(plan.question, evidence=evidence),
                    candidate_score=1.0,
                    verified=True,
                    degraded_reason="",
                    supporting_evidence_ids=(),
                    repair_action=repair_action,
                    repair_attempt=repair_attempt,
                ),
                True,
            )
        if repair_action == "abstain_due_to_low_grounding":
            return (
                reasoner_handoff,
                self._replace_trace_answer(
                    reasoning,
                    answer_text="Insufficient evidence to produce a verified answer.",
                    selected_strategy="abstain",
                    selected_verifier=critique.verifier_type or "tool.evidence_grounding",
                    candidate_score=0.0,
                    verified=False,
                    degraded_reason=critique.degraded_reason or "low_evidence_support",
                    supporting_evidence_ids=(),
                    repair_action=repair_action,
                    repair_attempt=repair_attempt,
                ),
                True,
            )
        if repair_action == "rebuild_trace_projection":
            return reasoner_handoff, CompressedTrace.from_dict(reasoning.to_dict()), True
        if repair_action == "reload_runtime_subset":
            return reasoner_handoff, reasoning, True
        if repair_action == "rerun_reasoner":
            repaired_handoff = ResearchReasonerHandoff.from_inputs(
                plan=plan,
                evidence=evidence,
                budget=budget,
                reasoning_mode="deep" if reasoner_handoff.reasoning_mode != "deep" else reasoner_handoff.reasoning_mode,
                final_text_policy=reasoner_handoff.final_text_policy,
                implementation_mode=reasoner_handoff.implementation_mode,
            )
            repaired_reasoning = await self._run_component(
                "reasoner",
                task_id=task_id,
                start_stage="pipeline.reasoner_repair_started",
                done_stage="pipeline.reasoner_repair_done",
                start_payload={
                    "task_id": task_id,
                    "repair_action": repair_action,
                    "repair_attempt": repair_attempt,
                    "reasoning_mode": repaired_handoff.reasoning_mode,
                },
                run=lambda repaired_handoff=repaired_handoff: self.reasoner.reason_from_handoff(repaired_handoff),
            )
            await self._emit_event(
                "pipeline.reasoner_repair_done",
                {
                    "task_id": task_id,
                    "repair_action": repair_action,
                    "repair_attempt": repair_attempt,
                    "reasoning_mode": repaired_handoff.reasoning_mode,
                },
            )
            await self._record_status(
                "reasoner",
                AgentState.IDLE,
                task_id=task_id,
                message=f"reasoning repair completed via {repair_action}",
            )
            return repaired_handoff, repaired_reasoning, True
        return reasoner_handoff, reasoning, False

    def _build_warnings(self, *, evidence, critique) -> list[str]:
        warnings: list[str] = []
        if evidence.used_web_fallback and not evidence.web_results:
            warnings.append("web_fallback_returned_no_results")
        if not critique.is_valid:
            warnings.append("critique_reported_issues")
        if critique.result == CritiqueResult.DEGRADED:
            warnings.append("critique_degraded")
        snapshot = self.model_manager.health_snapshot()
        if snapshot.fallback_active:
            warnings.append(f"model_fallback_active:{snapshot.fallback_reason or 'unknown'}")
        if self.dashboard.dropped_events > 0:
            warnings.append(f"dashboard_dropped_events:{self.dashboard.dropped_events}")
        return warnings

    def _build_repair_warnings(self, repair_history: tuple[str, ...]) -> list[str]:
        if not repair_history:
            return []
        return [f"repair_applied:{','.join(repair_history)}"]

    def _build_performance_metric(
        self,
        *,
        task_id: str,
        started_at: float,
        budget,
    ) -> PerformanceMetric:
        snapshot = self.model_manager.health_snapshot()
        vram_usage = snapshot.generation_backend_vram_gb or snapshot.embedding_backend_vram_gb or 0.0
        return PerformanceMetric(
            task_id=task_id,
            time=round(time.perf_counter() - started_at, 4),
            vram_usage=float(vram_usage),
            iterations=budget.reasoner_passes + budget.critic_passes,
        )

    def _select_next_repair_action(
        self,
        *,
        repair_actions: tuple[str, ...],
        repair_history: tuple[str, ...],
    ) -> str | None:
        for action in repair_actions:
            if action and action not in repair_history:
                return action
        return None

    def _repair_answer_from_tools(self, question: str, *, evidence) -> str:
        arithmetic_answer = evaluate_arithmetic_question(question)
        if arithmetic_answer is not None:
            return arithmetic_answer
        python_answer = evaluate_python_expression_question(question)
        if python_answer is not None:
            return python_answer
        python_code_answer = evaluate_python_code_question(question)
        if python_code_answer is not None:
            return python_code_answer
        python_unit_test_answer = evaluate_python_unit_test_question(question)
        if python_unit_test_answer is not None:
            return python_unit_test_answer
        evidence_count_answer = expected_evidence_count(
            question,
            len(evidence.local_results) + len(evidence.web_results),
        )
        return evidence_count_answer or ""

    def _tool_repair_strategy(self, question: str, *, evidence) -> str:
        if evaluate_arithmetic_question(question) is not None:
            return "tool_arithmetic"
        if evaluate_python_expression_question(question) is not None:
            return "tool_python_expression"
        if evaluate_python_code_question(question) is not None:
            return "tool_python_code_execution"
        if evaluate_python_unit_test_question(question) is not None:
            return "tool_python_unit_test"
        if expected_evidence_count(question, len(evidence.local_results) + len(evidence.web_results)) is not None:
            return "tool_evidence_count"
        return "tool_repair"

    def _tool_repair_verifier(self, question: str, *, evidence) -> str:
        if evaluate_arithmetic_question(question) is not None:
            return "tool.python_ast_arithmetic"
        if evaluate_python_expression_question(question) is not None:
            return "tool.python_expression"
        if evaluate_python_code_question(question) is not None:
            return "tool.python_code_execution"
        if evaluate_python_unit_test_question(question) is not None:
            return "tool.python_unit_test"
        if expected_evidence_count(question, len(evidence.local_results) + len(evidence.web_results)) is not None:
            return "tool.evidence_count"
        return "tool.repair"

    def _replace_trace_answer(
        self,
        trace: CompressedTrace,
        *,
        answer_text: str,
        selected_strategy: str,
        selected_verifier: str,
        candidate_score: float,
        verified: bool,
        degraded_reason: str,
        supporting_evidence_ids: tuple[str, ...],
        repair_action: str,
        repair_attempt: int,
    ) -> CompressedTrace:
        payload = trace.to_dict()
        selected_candidate_id = self._selected_candidate_id(trace)
        raw_steps = [dict(step) for step in payload.get("operation_stream", [])]
        for step in raw_steps:
            metadata = dict(step.get("metadata", {}))
            if step.get("opcode") == "infer":
                metadata["selected_strategy"] = selected_strategy
                metadata["selected_verifier"] = selected_verifier
                metadata["candidate_score"] = round(candidate_score, 3)
                metadata["verified"] = verified
                metadata["candidate_id"] = selected_candidate_id or "cand_repair"
            if step.get("opcode") == "bind":
                metadata["selected_strategy"] = selected_strategy
                metadata["selected_verifier"] = selected_verifier
                metadata["candidate_score"] = round(candidate_score, 3)
                metadata["verified"] = verified
                metadata["candidate_id"] = selected_candidate_id or "cand_repair"
            if step.get("opcode") == "check":
                metadata["tool_check"] = selected_verifier or selected_strategy
                metadata["candidate_id"] = selected_candidate_id or "cand_repair"
            if step.get("opcode") == "emit":
                metadata["answer_text"] = answer_text
                metadata["selected_strategy"] = selected_strategy
                metadata["selected_verifier"] = selected_verifier
                metadata["candidate_score"] = round(candidate_score, 3)
                metadata["verified"] = verified
                metadata["degraded_reason"] = degraded_reason
                metadata["supporting_evidence_ids"] = list(supporting_evidence_ids)
                metadata["candidate_id"] = selected_candidate_id or "cand_repair"
            step["metadata"] = metadata
        payload["operation_stream"] = raw_steps
        context_frames = [dict(frame) for frame in payload.get("context_frames", [])]
        if context_frames:
            frame_metadata = dict(context_frames[0].get("metadata", {}))
            frame_metadata["cid"] = selected_candidate_id or "cand_repair"
            frame_metadata["ta"] = answer_text
            frame_metadata["sa"] = selected_strategy
            frame_metadata["sv"] = selected_verifier
            frame_metadata["ss"] = round(candidate_score, 3)
            frame_metadata["vv"] = verified
            frame_metadata["dr"] = degraded_reason
            frame_metadata["si"] = list(supporting_evidence_ids)
            candidate_summary = list(frame_metadata.get("candidate_summary", []))
            if candidate_summary:
                candidate_summary[0] = {
                    "candidate_id": candidate_summary[0].get("candidate_id", "cand_repair"),
                    "strategy": selected_strategy,
                    "verifier_type": selected_verifier,
                    "verified": verified,
                    "total_score": round(candidate_score, 3),
                    "proof_hash_stability": 1.0,
                }
                frame_metadata["candidate_summary"] = candidate_summary
            context_frames[0]["metadata"] = frame_metadata
            payload["context_frames"] = context_frames
        payload["decode_hints"] = [
            DecodeHint(
                hint_id="d0",
                template="verified_answer",
                entity_ids=("a",),
                metadata={
                    "answer_text": answer_text,
                    "selected_strategy": selected_strategy,
                    "selected_verifier": selected_verifier,
                    "candidate_score": round(candidate_score, 3),
                    "verified": verified,
                    "degraded_reason": degraded_reason,
                    "candidate_count": self._extract_trace_candidate_count(trace),
                    "supporting_evidence_ids": list(supporting_evidence_ids),
                    "candidate_id": selected_candidate_id or "cand_repair",
                },
            ).to_dict()
        ]
        payload["candidate_traces"] = [
            self._repair_candidate_trace(
                candidate_trace,
                answer_text=answer_text,
                selected_strategy=selected_strategy,
                selected_verifier=selected_verifier,
                candidate_score=candidate_score,
                verified=verified,
                degraded_reason=degraded_reason,
                supporting_evidence_ids=supporting_evidence_ids,
            ).to_dict()
            if candidate_trace.candidate_id == (selected_candidate_id or candidate_trace.candidate_id)
            else candidate_trace.to_dict()
            for candidate_trace in trace.candidate_traces
        ]
        reasoner_notes = str(payload.get("reasoner_notes", "")).strip()
        repair_note = f"repair_action={repair_action}\nrepair_attempt={repair_attempt}"
        payload["reasoner_notes"] = f"{reasoner_notes}\n{repair_note}".strip()
        operation_stream = tuple(OperationStep.from_dict(step) for step in raw_steps)
        payload["proof_hash"] = self._compute_trace_proof_hash(
            task_id=trace.task_id,
            tokens=trace.tokens,
            operation_stream=operation_stream,
            evidence_handles=trace.evidence_handles,
        )
        return CompressedTrace.from_dict(payload)

    def _repair_candidate_trace(
        self,
        candidate_trace: CandidateTrace,
        *,
        answer_text: str,
        selected_strategy: str,
        selected_verifier: str,
        candidate_score: float,
        verified: bool,
        degraded_reason: str,
        supporting_evidence_ids: tuple[str, ...],
    ) -> CandidateTrace:
        raw_steps = [step.to_dict() for step in candidate_trace.operation_stream]
        for step in raw_steps:
            metadata = dict(step.get("metadata", {}))
            if step.get("opcode") == "bind":
                metadata["selected_strategy"] = selected_strategy
                metadata["selected_verifier"] = selected_verifier
                metadata["candidate_score"] = round(candidate_score, 3)
                metadata["verified"] = verified
            if step.get("opcode") == "check":
                metadata["tool_check"] = selected_verifier or selected_strategy
            if step.get("opcode") == "emit":
                metadata["answer_text"] = answer_text
                metadata["selected_strategy"] = selected_strategy
                metadata["selected_verifier"] = selected_verifier
                metadata["candidate_score"] = round(candidate_score, 3)
                metadata["verified"] = verified
                metadata["degraded_reason"] = degraded_reason
                metadata["supporting_evidence_ids"] = list(supporting_evidence_ids)
            step["metadata"] = metadata
        operation_stream = tuple(OperationStep.from_dict(step) for step in raw_steps)
        proof_hash = self._compute_trace_proof_hash(
            task_id=candidate_trace.candidate_id,
            tokens=candidate_trace.tokens,
            operation_stream=operation_stream,
            evidence_handles=supporting_evidence_ids or candidate_trace.supporting_evidence_ids,
        )
        return CandidateTrace(
            candidate_id=candidate_trace.candidate_id,
            answer_text=answer_text,
            strategy=selected_strategy,
            verifier_type=selected_verifier,
            verified=verified,
            total_score=round(candidate_score, 3),
            agreement_score=candidate_trace.agreement_score,
            evidence_support_score=candidate_trace.evidence_support_score,
            proof_hash_stability=1.0,
            degraded_reason=degraded_reason,
            supporting_evidence_ids=supporting_evidence_ids or candidate_trace.supporting_evidence_ids,
            tokens=candidate_trace.tokens,
            expanded_preview=candidate_trace.expanded_preview,
            operation_stream=operation_stream,
            decode_hints=(
                DecodeHint(
                    hint_id=f"{candidate_trace.candidate_id}_hint",
                    template="verified_answer",
                    entity_ids=("a",),
                    metadata={
                        "candidate_id": candidate_trace.candidate_id,
                        "answer_text": answer_text,
                        "selected_strategy": selected_strategy,
                        "selected_verifier": selected_verifier,
                        "candidate_score": round(candidate_score, 3),
                        "verified": verified,
                        "degraded_reason": degraded_reason,
                        "supporting_evidence_ids": list(
                            supporting_evidence_ids or candidate_trace.supporting_evidence_ids
                        ),
                    },
                ),
            ),
            proof_hash=proof_hash,
            created_at=candidate_trace.created_at,
        )

    def _compute_trace_proof_hash(
        self,
        *,
        task_id: str,
        tokens: tuple[str, ...],
        operation_stream: tuple[OperationStep, ...],
        evidence_handles: tuple[str, ...],
    ) -> str:
        proof_payload = {
            "task_id": task_id,
            "tokens": list(tokens),
            "operation_stream": [step.to_dict() for step in operation_stream],
            "evidence_handles": list(evidence_handles),
        }
        return stable_hash(json.dumps(proof_payload, sort_keys=True, separators=(",", ":")))

    def _extract_trace_candidate_count(self, trace: CompressedTrace) -> int:
        for step in reversed(trace.operation_stream):
            if step.opcode != "emit":
                continue
            raw_count = step.metadata.get("candidate_count", 1)
            try:
                return max(1, int(raw_count))
            except (TypeError, ValueError):
                return 1
        if trace.context_frames:
            raw_count = trace.context_frames[0].metadata.get("cc", 1)
            try:
                return max(1, int(raw_count))
            except (TypeError, ValueError):
                return 1
        return 1

    def _critique_status_state(self, critique: CritiqueReport) -> AgentState:
        if critique.is_valid or critique.result == CritiqueResult.DEGRADED:
            return AgentState.IDLE
        return AgentState.ERROR

    def _critique_status_message(
        self,
        critique: CritiqueReport,
        *,
        repair_history: tuple[str, ...],
    ) -> str:
        if critique.is_valid and repair_history:
            return f"critique repaired via {', '.join(repair_history)}"
        if critique.is_valid:
            return "critique completed"
        if critique.result == CritiqueResult.DEGRADED:
            if repair_history:
                return f"degraded answer after repair via {', '.join(repair_history)}"
            return "degraded answer emitted"
        return "critique reported issues"

    def _critique_status_severity(self, critique: CritiqueReport) -> SeverityLevel:
        if critique.is_valid:
            return SeverityLevel.LOW
        if critique.result == CritiqueResult.DEGRADED:
            return SeverityLevel.MEDIUM
        return SeverityLevel.HIGH

    def _research_status_message(self, evidence) -> str:
        if evidence.used_web_fallback and not evidence.web_results:
            return "web fallback degraded to local-only evidence"
        if evidence.used_web_fallback:
            return "used bounded web fallback"
        return "research completed with local evidence"

    def _research_status_severity(self, evidence) -> SeverityLevel:
        if evidence.used_web_fallback and not evidence.web_results:
            return SeverityLevel.HIGH
        if evidence.used_web_fallback:
            return SeverityLevel.MEDIUM
        return SeverityLevel.LOW

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("Orchestrator.start() must be called before run_task() or run_pipeline().")

    def _select_reasoning_mode(self, budget) -> str:
        if budget.reasoner_passes > 1 or budget.critic_passes > 1 or budget.max_web_queries > 1:
            return "deep"
        return "fast"

    def _extract_trace_answer(self, trace) -> str:
        for step in reversed(trace.operation_stream):
            answer_text = str(step.metadata.get("answer_text", "")).strip()
            if step.opcode == "emit" and answer_text:
                return answer_text
        for hint in trace.decode_hints:
            answer_text = str(hint.metadata.get("answer_text", "")).strip()
            if answer_text:
                return answer_text
        return ""

    async def _load_recent_reasoning_logs(self, *, limit: int) -> list[ReasoningLog]:
        payloads = await self.storage.list_reasoning_history()
        logs: list[ReasoningLog] = []
        for payload in reversed(payloads):
            if "compressed_chain" not in payload:
                continue
            try:
                logs.append(ReasoningLog.from_dict(payload))
            except (KeyError, TypeError, ValueError):
                continue
            if len(logs) >= limit:
                break
        logs.reverse()
        return logs

    async def __aenter__(self) -> "Orchestrator":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()


AppOrchestrator = Orchestrator


async def run_once(
    question: str,
    config: AppConfig = APP_CONFIG,
    thinking_minutes: int = 1,
) -> TaskResult:
    """Convenience helper for one-shot runs."""
    async with Orchestrator(config=config) as app:
        return await app.run_task(question, thinking_minutes)


async def _run_dashboard_session(
    app: Orchestrator,
    *,
    support_bundle_dir: Path | None = None,
    launch_report: PackagedLaunchReport | None = None,
    diagnostics_path: Path | None = None,
    startup_notice: str = "",
    startup_notice_severity: str = "info",
) -> TaskResult | None:
    """Start an app session, optionally publish startup context, and always stop cleanly."""
    await app.start()
    try:
        if startup_notice:
            await app._publish_dashboard_notice(startup_notice, severity=startup_notice_severity)
        if not app.config.dashboard.enable_ui:
            result = await app.run_task("What should I build first?", thinking_minutes=1)
            if support_bundle_dir is not None:
                await app.export_packaged_support_bundle(
                    support_bundle_dir,
                    launch_report=launch_report,
                    diagnostics_path=diagnostics_path,
                )
            return result
        if support_bundle_dir is not None:
            await app.export_packaged_support_bundle(
                support_bundle_dir,
                launch_report=launch_report,
                diagnostics_path=diagnostics_path,
            )
        while app.dashboard.ui_running:
            await asyncio.sleep(0.1)
        return None
    finally:
        await app.stop()


async def run_dashboard_app(
    config: AppConfig = APP_CONFIG,
    *,
    support_bundle_dir: Path | None = None,
) -> TaskResult | None:
    """Launch the developer/source app shell without packaged startup indirection."""
    app = Orchestrator(config=config)
    return await _run_dashboard_session(app, support_bundle_dir=support_bundle_dir)


async def run_packaged_dashboard_app(
    config: AppConfig = APP_CONFIG,
    *,
    support_bundle_dir: Path | None = None,
) -> TaskResult | None:
    """Launch the packaged app shell using the shared readiness/preflight contract."""
    startup_profile = await _load_startup_settings_profile(config)
    planning_app = Orchestrator(config=config)
    packaged_plan = planning_app.build_packaged_startup_plan(startup_profile=startup_profile)
    packaged_app = Orchestrator(
        config=packaged_plan.runtime_config,
        startup_profile_override=packaged_plan.effective_profile,
        persist_startup_profile_override=packaged_plan.persist_effective_profile,
    )
    try:
        return await _run_dashboard_session(
            packaged_app,
            support_bundle_dir=support_bundle_dir,
            launch_report=packaged_plan.launch_report,
            startup_notice=packaged_plan.startup_notice,
            startup_notice_severity=packaged_plan.startup_notice_severity,
        )
    except Exception as exc:
        diagnostics_path = planning_app._write_packaged_startup_diagnostics(
            exc,
            export_dir=support_bundle_dir,
            launch_report=packaged_plan.launch_report,
        )
        if packaged_plan.launch_report.effective_mode != "real":
            raise

        recovery_profile = replace(
            packaged_plan.effective_profile,
            runtime={
                **packaged_plan.effective_profile.runtime,
                "stub_mode": True,
            },
        )
        recovery_launch_report = planning_app._build_runtime_recovery_launch_report(
            launch_report=packaged_plan.launch_report,
            error=exc,
            diagnostics_path=diagnostics_path,
        )
        recovery_app = Orchestrator(
            config=planning_app._config_for_user_settings_profile(recovery_profile),
            startup_profile_override=recovery_profile,
            persist_startup_profile_override=True,
        )
        recovery_notice = (
            f"{recovery_launch_report.summary} Review '{diagnostics_path}' before re-enabling real mode."
        )
        return await _run_dashboard_session(
            recovery_app,
            support_bundle_dir=support_bundle_dir,
            launch_report=recovery_launch_report,
            diagnostics_path=diagnostics_path,
            startup_notice=recovery_notice,
            startup_notice_severity="warning",
        )


async def _load_startup_settings_profile(config: AppConfig) -> UserSettingsProfile | None:
    """Load the persisted default settings profile before packaged startup planning."""
    startup_config = replace(
        config,
        preflight=replace(
            config.preflight,
            backends=replace(
                config.preflight.backends,
                vector_store_backend="simple_inmemory",
                vector_store_fallback_backend="simple_inmemory",
            ),
        ),
    )
    storage = StorageManager(config=startup_config)
    await storage.start()
    try:
        return await storage.load_user_settings_profile("default")
    finally:
        await storage.stop()


def main() -> None:
    """CLI entrypoint for the developer/source app shell or one-shot headless smoke runs."""
    result = asyncio.run(run_dashboard_app())
    if result is not None:
        print(
            "Pipeline completed for task",
            result.task_id,
            "with validity",
            result.critique.is_valid,
        )


def packaged_main() -> None:
    """Packaged-app entrypoint that honors the packaged startup plan and stub fallback contract."""
    result = asyncio.run(run_packaged_dashboard_app())
    if result is not None:
        print(
            "Pipeline completed for task",
            result.task_id,
            "with validity",
            result.critique.is_valid,
        )


if __name__ == "__main__":
    main()
