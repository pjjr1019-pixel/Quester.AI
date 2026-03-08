"""Async orchestrator that wires agents and services into one pipeline."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import time
from concurrent.futures import CancelledError as ConcurrentCancelledError
from concurrent.futures import Future as ConcurrentFuture
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from acceptance_thresholds import PHASE12_ACCEPTANCE_THRESHOLDS
from compressor import CompressorAgent
from config import APP_CONFIG, AppConfig, BudgetPolicy
from critic import CriticAgent
from data_structures import (
    AgentState,
    AgentStatus,
    AudioSynthesisResult,
    AudioTranscriptionResult,
    CandidateTrace,
    CodeSpecialistResult,
    CompressedTrace,
    CritiqueReport,
    CritiqueResult,
    DashboardCapabilityAvailability,
    DashboardKnowledgeSource,
    ModelRegistryView,
    ModelRoleActionReport,
    DashboardReadinessCheck,
    DashboardReadinessReport,
    DashboardTaskHistoryEntry,
    DashboardTaskInspector,
    DecodeHint,
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


@dataclass(slots=True, frozen=True)
class _ReadinessState:
    """Internal snapshot reused by dashboard and packaged launch checks."""

    profile: UserSettingsProfile
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


class Orchestrator:
    """Single owner of startup, pipeline execution, and clean shutdown."""

    COMPRESSION_HISTORY_SCAN_LIMIT = PHASE12_ACCEPTANCE_THRESHOLDS.compression.max_recent_reasoning_logs

    def __init__(
        self,
        config: AppConfig = APP_CONFIG,
        *,
        storage: StorageManager | None = None,
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
        self.translation = translation or TranslationService()
        self.phase11_content = phase11_content or Phase11ContentLoader(config=config)
        self._started = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dashboard_futures: set[ConcurrentFuture[Any]] = set()
        self._active_long_horizon_tasks: dict[str, asyncio.Task[TaskResult]] = {}
        self._shutdown_requested = False
        self.storage.add_runtime_event_listener(self._forward_runtime_event_to_dashboard)
        self.storage.add_agent_status_listener(self._forward_agent_status_to_dashboard)

    async def start(self) -> None:
        """Start all services and agents in dependency-safe order."""
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        await self.storage.start()
        await self.model_manager.start()
        await self.planner.start()
        await self.researcher.start()
        await self.reasoner.start()
        await self.critic.start()
        await self.compressor.start()
        await self.dashboard.start()
        await self.self_optimizer.start()
        settings_profile = await self.storage.load_user_settings_profile("default")
        if settings_profile is None:
            settings_profile = self._default_user_settings_profile()
            await self.storage.save_user_settings_profile(settings_profile)
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
        await self._publish_dashboard_readiness_report()
        await self._publish_dashboard_examples()

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
            elif action == "readiness.refresh":
                await self._publish_dashboard_readiness_report()
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

    async def _publish_dashboard_model_registry_view(self) -> None:
        recent_optimizer_suggestions = (await self.storage.list_optimizer_suggestion_records())[-4:]
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
        await self._publish_dashboard_model_registry_view()

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
    ) -> PackagedLaunchReport:
        """Return the packaged-app launch decision before or after startup.

        Input:
        - `active_profile`: optional settings profile to evaluate instead of the
          live dashboard state.

        Output:
        - A typed `PackagedLaunchReport` describing stub or real mode readiness.

        Failure behavior:
        - The method does not raise for missing optional runtime dependencies;
          those become typed readiness blockers in the returned report.
        """
        state = self._collect_readiness_state(active_profile=active_profile)
        readiness = self._build_dashboard_readiness_report_from_state(state)
        requested_mode = "stub" if self.config.preflight.flags.stub_mode else "real"
        prelaunch_real_ready = state.primary_generation_ready and state.primary_embedding_ready
        used_stub_fallback = requested_mode == "real" and not prelaunch_real_ready
        effective_mode = "stub" if requested_mode == "stub" or used_stub_fallback else "real"
        launch_ready = True if effective_mode == "stub" else prelaunch_real_ready
        blocking_reason = ""
        blocking_detail = ""
        if requested_mode == "stub":
            summary = "Packaged launch will start in stub mode for first-run local validation."
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

    async def export_packaged_support_bundle(
        self,
        export_dir: Path,
        *,
        active_profile: UserSettingsProfile | None = None,
        launch_report: PackagedLaunchReport | None = None,
    ) -> PackagedSupportBundle:
        """Export a small support bundle for packaged-launch smoke and troubleshooting.

        Input:
        - `export_dir`: destination directory for the bundle files.
        - `active_profile`: optional settings profile override.
        - `launch_report`: optional precomputed packaged-launch report.

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

        launch_report_path = bundle_dir / "launch_report.json"
        readiness_report_path = bundle_dir / "readiness_report.json"
        user_settings_path = bundle_dir / "user_settings_profile.json"
        app_state_path = bundle_dir / "app_state.json"
        support_readme_path = bundle_dir / "support_bundle.txt"
        manifest_path = bundle_dir / "support_bundle_manifest.json"

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
                    "Guidance:",
                    *(f"- {item}" for item in launch.guidance),
                )
            )
            + "\n",
            encoding="utf-8",
        )

        copied_artifact_paths: list[str] = []
        for artifact_name in (
            self.config.storage.events_log_name,
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
            user_settings_path=str(user_settings_path),
            app_state_path=str(app_state_path),
            support_readme_path=str(support_readme_path),
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
            not self.config.preflight.flags.stub_mode
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
                status="ready" if self.config.preflight.flags.stub_mode else "disabled",
                detail=(
                    "Stub mode is enabled and the local lightweight pipeline can run without heavy model dependencies."
                    if self.config.preflight.flags.stub_mode
                    else "Stub mode is disabled; real backends are expected."
                ),
                recovery_actions=(
                    ("Enable stub mode for first-run local validation.",)
                    if not self.config.preflight.flags.stub_mode
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
                        and not self.config.preflight.flags.stub_mode
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
        capabilities = (
            DashboardCapabilityAvailability(
                capability_name="desktop_control",
                status="blocked_by_policy" if state.profile.desktop.get("enabled") else "visible_not_enabled",
                reason="phase_20_21_not_implemented",
                detail=(
                    "Desktop-control settings are configuration placeholders only; no local task-session executor "
                    "or approval-gated control runtime is active yet."
                ),
                recovery_actions=("Keep the toggle off until the typed desktop capability phases land.",),
            ),
            DashboardCapabilityAvailability(
                capability_name="observation_tiers",
                status=(
                    "blocked_by_policy"
                    if str(state.profile.observation.get("tier", "screenshot_on_demand")) != "screenshot_on_demand"
                    else "visible_not_enabled"
                ),
                reason="phase_22_not_implemented",
                detail=(
                    "Continuous capture, OCR-on-step, and vision-on-step remain visible for forward-compatible "
                    "settings only; the runtime still stays on screenshot-on-demand."
                ),
                recovery_actions=("Keep observation tier at screenshot_on_demand until the observation phases land.",),
            ),
            DashboardCapabilityAvailability(
                capability_name="cloud_offload",
                status="blocked_by_policy" if state.profile.cloud.get("mode") != "disabled" else "visible_not_enabled",
                reason="phase_23_not_implemented",
                detail=(
                    "Auxiliary cloud-offload settings are visible for future compatibility, but no provider adapter "
                    "or fallback routing path is active yet."
                ),
                recovery_actions=("Leave cloud mode disabled until auxiliary cloud adapters exist.",),
            ),
            DashboardCapabilityAvailability(
                capability_name="real_mode",
                status="ready" if real_mode_ready else "degraded",
                reason=(
                    "stub_mode_enabled"
                    if self.config.preflight.flags.stub_mode
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
                    if self.config.preflight.flags.stub_mode
                    else ("Install missing real-mode dependencies or local services.",)
                ),
            ),
        )
        guidance = (
            "The local app shell is complete enough for stub-mode runs, profile management, history inspection, and knowledge management.",
            "Use readiness warnings to decide whether to stay in lightweight stub mode or finish real-backend setup.",
            "Optional specialist roles can be enabled individually in Settings without replacing the base generation and embedding runtime.",
            "Desktop control, advanced observation tiers, and cloud offload remain visible but capability-gated future phases.",
        )
        return DashboardReadinessReport(
            stub_mode_ready=True,
            real_mode_ready=real_mode_ready,
            checks=checks,
            capabilities=capabilities,
            guidance=guidance,
        )

    def _collect_readiness_state(
        self,
        *,
        active_profile: UserSettingsProfile | None = None,
    ) -> _ReadinessState:
        profile = active_profile or (
            self.dashboard.app_state_snapshot().user_settings if self._started else self._default_user_settings_profile()
        )
        snapshot = self.model_manager.health_snapshot()
        chromadb_available = self._dependency_available("chromadb")
        sentence_transformers_available = self._dependency_available("sentence_transformers")
        llama_cpp_available = self._dependency_available("llama_cpp")
        backends = self.config.preflight.backends
        primary_generation_backend = backends.generation_backend
        primary_embedding_backend = backends.embedding_backend
        primary_uses_ollama = primary_generation_backend == "ollama" or primary_embedding_backend == "ollama_embeddings"
        any_uses_ollama = (
            primary_uses_ollama
            or backends.generation_fallback_backend == "ollama"
            or backends.embedding_fallback_backend == "ollama_embeddings"
        )
        if self.config.preflight.flags.stub_mode:
            ollama_service_ready = False
            ollama_status = "disabled" if any_uses_ollama else "not_required"
            ollama_detail = (
                "Stub mode is enabled; Ollama service probing is skipped until real mode is requested."
                if any_uses_ollama
                else "No configured backend requires Ollama service access."
            )
        elif any_uses_ollama:
            ollama_service_ready, ollama_detail = self._probe_ollama_service()
            if ollama_service_ready:
                ollama_status = "ready"
            else:
                ollama_status = "blocked" if primary_uses_ollama else "degraded"
        else:
            ollama_service_ready = True
            ollama_status = "not_required"
            ollama_detail = "No configured backend requires Ollama service access."

        llama_cpp_model_ready, llama_cpp_model_detail, llama_cpp_model_required, llama_cpp_model_blocking = (
            self._check_llama_cpp_model_files()
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
        if launch_report.effective_mode == "real" or self.config.preflight.flags.stub_mode:
            return self.config
        stub_flags = replace(self.config.preflight.flags, stub_mode=True)
        return replace(self.config, preflight=replace(self.config.preflight, flags=stub_flags))

    def _dependency_available(self, module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None

    def _probe_ollama_service(self) -> tuple[bool, str]:
        base_url = self.config.backend_runtime.ollama_base_url.rstrip("/")
        request_url = f"{base_url}/api/tags"
        timeout_s = min(2.0, max(0.25, float(self.config.backend_runtime.request_timeout_s)))
        request = urllib_request.Request(request_url, headers={"User-Agent": "QuesterAI/0.1"})
        try:
            with urllib_request.urlopen(request, timeout=timeout_s) as response:
                status_code = getattr(response, "status", None) or response.getcode()
            if int(status_code) >= 400:
                return False, f"Ollama probe at {request_url} returned HTTP {status_code}."
            return True, f"Ollama responded successfully at {request_url}."
        except (urllib_error.URLError, TimeoutError, ValueError) as exc:
            return False, f"Ollama service probe failed at {request_url}: {exc}"

    def _check_llama_cpp_model_files(self) -> tuple[bool, str, bool, bool]:
        backends = self.config.preflight.backends
        configured_models: list[tuple[str, Path]] = []
        if backends.generation_backend == "llama_cpp":
            configured_models.append(
                ("primary_generation", self._resolve_llama_cpp_model_path(backends.generation_model))
            )
        if backends.generation_fallback_backend == "llama_cpp":
            configured_models.append(
                ("fallback_generation", self._resolve_llama_cpp_model_path(backends.generation_fallback_model))
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

    def _resolve_llama_cpp_model_path(self, model_name: str) -> Path:
        model_path = Path(model_name)
        if model_path.is_absolute():
            return model_path
        return self.config.backend_runtime.models_dir / model_path

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
            "last_used_at": snapshot.last_used_at,
            "fallback_active": snapshot.fallback_active,
            "fallback_reason": snapshot.fallback_reason or "",
            "available_ram_gb": snapshot.available_ram_gb,
            "total_ram_gb": snapshot.total_ram_gb,
            "generation_backend_vram_gb": snapshot.generation_backend_vram_gb,
            "embedding_backend_vram_gb": snapshot.embedding_backend_vram_gb,
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
                "mode": "auxiliary_only",
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


async def run_dashboard_app(
    config: AppConfig = APP_CONFIG,
    *,
    support_bundle_dir: Path | None = None,
) -> TaskResult | None:
    """Launch the local app shell and keep it alive while the dashboard window is open."""
    planning_app = Orchestrator(config=config)
    launch_report = planning_app.build_packaged_launch_report()
    runtime_config = planning_app._packaged_runtime_config(launch_report)
    async with Orchestrator(config=runtime_config) as app:
        if launch_report.used_stub_fallback:
            await app._publish_dashboard_notice(
                f"{launch_report.summary} {launch_report.blocking_detail}".strip(),
                severity="warning",
            )
        if not runtime_config.dashboard.enable_ui:
            result = await app.run_task("What should I build first?", thinking_minutes=1)
            if support_bundle_dir is not None:
                await app.export_packaged_support_bundle(support_bundle_dir, launch_report=launch_report)
            return result
        if support_bundle_dir is not None:
            await app.export_packaged_support_bundle(support_bundle_dir, launch_report=launch_report)
        while app.dashboard.ui_running:
            await asyncio.sleep(0.1)
        return None


def main() -> None:
    """CLI entrypoint for the local app shell or one-shot headless smoke runs."""
    result = asyncio.run(run_dashboard_app())
    if result is not None:
        print(
            "Pipeline completed for task",
            result.task_id,
            "with validity",
            result.critique.is_valid,
        )


if __name__ == "__main__":
    main()
