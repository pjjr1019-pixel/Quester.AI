"""Async orchestrator that wires agents and services into one pipeline."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from compressor import CompressorAgent
from config import APP_CONFIG, AppConfig, BudgetPolicy
from critic import CriticAgent
from data_structures import (
    AgentState,
    AgentStatus,
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
from self_optimizer import SelfOptimizer
from storage import StorageManager
from utils import configure_logging


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
            await self._emit_event("pipeline.reasoner_done", {"task_id": task_id})
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
            await self._emit_event("pipeline.critic_done", {"task_id": task_id})
            await self._record_status(
                "critic",
                AgentState.IDLE if critique.is_valid else AgentState.ERROR,
                task_id=task_id,
                message="critique completed" if critique.is_valid else "critique reported issues",
                severity=SeverityLevel.LOW if critique.is_valid else SeverityLevel.HIGH,
            )

            compression = await self._run_component(
                "compressor",
                task_id=task_id,
                start_stage="pipeline.compressor_started",
                done_stage="pipeline.compressor_done",
                start_payload={"task_id": task_id},
                run=lambda: self.compressor.propose(reasoning, logs=[reasoning_log]),
            )
            await self._emit_event("pipeline.compressor_done", {"task_id": task_id})
            await self._record_status(
                "compressor",
                AgentState.IDLE,
                task_id=task_id,
                message="compression proposal generation completed",
            )

            warnings = self._build_warnings(evidence=evidence, critique=critique)
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

            result = TaskResult(
                task_id=task_id,
                plan=plan,
                evidence=evidence,
                reasoning=reasoning,
                critique=critique,
                compression=tuple(compression),
                answer_text=self._build_answer_text(evidence=evidence, critique=critique),
                warnings=tuple(warnings),
                metrics=(metric,),
            )
            await self.storage.record_task_result(result)
            await self._emit_event(
                "pipeline.completed",
                {
                    "task_id": task_id,
                    "warning_count": len(result.warnings),
                },
            )
            await self._record_status(
                "orchestrator",
                AgentState.IDLE,
                task_id=task_id,
                message="pipeline completed",
            )
            return result
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
        try:
            result = await run()
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
        return result

    def _build_answer_text(
        self,
        *,
        evidence,
        critique,
    ) -> str:
        return (
            f"Collected {len(evidence.local_results)} local evidence item(s) and "
            f"{len(evidence.web_results)} web evidence item(s). "
            f"Critique valid: {critique.is_valid}."
        )

    def _build_warnings(self, *, evidence, critique) -> list[str]:
        warnings: list[str] = []
        if evidence.used_web_fallback and not evidence.web_results:
            warnings.append("web_fallback_returned_no_results")
        if not critique.is_valid:
            warnings.append("critique_reported_issues")
        snapshot = self.model_manager.health_snapshot()
        if snapshot.fallback_active:
            warnings.append(f"model_fallback_active:{snapshot.fallback_reason or 'unknown'}")
        return warnings

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
            raise RuntimeError("AppOrchestrator.start() must be called before run_pipeline().")

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
