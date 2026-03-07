"""Async orchestrator that wires agents and services into one pipeline."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from compressor import CompressorAgent
from config import APP_CONFIG, AppConfig, BudgetPolicy
from critic import CriticAgent
from data_structures import (
    AgentState,
    AgentStatus,
    CandidateTrace,
    CompressedTrace,
    CritiqueReport,
    CritiqueResult,
    DecodeHint,
    OperationStep,
    PerformanceMetric,
    ReasonerCriticHandoff,
    ResearchReasonerHandoff,
    ReasoningLog,
    RuntimeEvent,
    SeverityLevel,
    TaskResult,
)
from dashboard import DashboardService
from model_manager import ModelManager
from planner import PlannerAgent
from reasoner import ReasonerAgent
from researcher import ResearcherAgent
from retrieval import stable_hash
from runtime_errors import BackendUnavailableError, ModelTimeoutError, WebLookupTimeoutError
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


class Orchestrator:
    """Single owner of startup, pipeline execution, and clean shutdown."""

    def __init__(self, config: AppConfig = APP_CONFIG):
        self.config = config
        self.config.validate()
        self.logger = configure_logging(config.logging)

        self.storage = StorageManager(config=config, logger=self.logger.getChild("storage"))
        self.model_manager = ModelManager(config=config, logger=self.logger.getChild("model"))
        self.dashboard = DashboardService(config=config)

        self.planner = PlannerAgent(model_manager=self.model_manager, config=config)
        self.researcher = ResearcherAgent(
            model_manager=self.model_manager,
            storage=self.storage,
            config=config,
        )
        self.reasoner = ReasonerAgent(
            model_manager=self.model_manager,
            storage=self.storage,
            config=config,
        )
        self.critic = CriticAgent(
            model_manager=self.model_manager,
            storage=self.storage,
            config=config,
        )
        self.compressor = CompressorAgent(model_manager=self.model_manager, config=config)
        self.self_optimizer = SelfOptimizer(storage=self.storage, config=config)
        self.translation = TranslationService()
        self._started = False

    async def start(self) -> None:
        """Start all services and agents in dependency-safe order."""
        if self._started:
            return
        await self.storage.start()
        await self.model_manager.start()
        await self.planner.start()
        await self.researcher.start()
        await self.reasoner.start()
        await self.critic.start()
        await self.compressor.start()
        await self.dashboard.start()
        await self.self_optimizer.start()
        self._started = True
        await self._emit_event(
            "orchestrator.started",
            {"stub_mode": self.config.preflight.flags.stub_mode},
        )
        await self._record_status(
            "orchestrator",
            AgentState.IDLE,
            message="orchestrator started",
        )

    async def stop(self) -> None:
        """Stop all services and agents in reverse dependency order."""
        if not self._started:
            return
        await self._record_status(
            "orchestrator",
            AgentState.STOPPING,
            message="orchestrator stopping",
        )
        await self._emit_event("orchestrator.stopping", {})
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

    async def run_task(self, question: str, thinking_minutes: int) -> TaskResult:
        """Execute the foreground workflow from Planner to Compressor."""
        self._require_started()
        await self.storage.enter_foreground_task()
        started_at = time.perf_counter()
        budget = BudgetPolicy.from_minutes(
            thinking_minutes,
            calibration=self.config.budget_calibration,
        )
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
                await self._emit_event("pipeline.researcher_done", {"task_id": task_id})
                await self._record_status(
                    "researcher",
                    AgentState.IDLE,
                    task_id=task_id,
                    message=self._research_status_message(evidence),
                    severity=self._research_status_severity(evidence),
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
                recent_reasoning_logs = await self._load_recent_reasoning_logs(limit=6)

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
                await self.storage.record_task_result(result)
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
                raise
        finally:
            await self.storage.exit_foreground_task()

    async def run_pipeline(self, question: str) -> TaskResult:
        """Compatibility wrapper that uses the smallest valid budget."""
        return await self.run_task(question, thinking_minutes=1)

    async def _emit_event(self, stage: str, payload: dict[str, Any]) -> None:
        event = RuntimeEvent(
            stage=stage,
            payload=dict(payload),
        )
        await self.storage.record_runtime_event(event)
        self.dashboard.publish_event(
            {
                "stage": stage,
                **event.payload,
                "timestamp": event.timestamp.isoformat(),
            }
        )

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
        self.dashboard.publish_event(
            {
                "stage": "status.updated",
                **status.to_dict(),
            }
        )

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
            try:
                return await run()
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
        return {
            **self._reasoning_event_payload(task_id=task_id, reasoning=reasoning),
            **self._critique_event_payload(task_id=task_id, critique=critique),
            "answer_text": answer_text,
            "supporting_evidence_ids": answer_metadata["supporting_evidence_ids"],
            "citation_refs": answer_metadata["citation_refs"],
            "warning_count": warning_count,
        }

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
        selected_candidate_id = ""
        if trace.context_frames:
            selected_candidate_id = str(trace.context_frames[0].metadata.get("cid", "")).strip()
        if not selected_candidate_id and trace.candidate_traces:
            selected_candidate_id = trace.candidate_traces[0].candidate_id
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


def main() -> None:
    """CLI entrypoint for manual Phase 1 smoke runs."""
    result = asyncio.run(run_once("What should I build first?"))
    print(
        "Pipeline completed for task",
        result.task_id,
        "with validity",
        result.critique.is_valid,
    )


if __name__ == "__main__":
    main()
