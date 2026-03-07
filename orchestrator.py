"""Async orchestrator that wires agents and services into one pipeline."""

from __future__ import annotations

import asyncio
from typing import Any

from compressor import CompressorAgent
from config import APP_CONFIG, AppConfig, BudgetPolicy
from critic import CriticAgent
from data_structures import TaskResult
from dashboard import DashboardService
from model_manager import ModelManager
from planner import PlannerAgent
from reasoner import ReasonerAgent
from researcher import ResearcherAgent
from self_optimizer import SelfOptimizer
from storage import StorageManager
from utils import configure_logging, utc_now_iso


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
        self.reasoner = ReasonerAgent(model_manager=self.model_manager, config=config)
        self.critic = CriticAgent(model_manager=self.model_manager, config=config)
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

    async def stop(self) -> None:
        """Stop all services and agents in reverse dependency order."""
        if not self._started:
            return
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
        budget = BudgetPolicy.from_minutes(thinking_minutes)
        await self._emit_event(
            "pipeline.received",
            {
                "question": question,
                "thinking_minutes": thinking_minutes,
                "budget": budget.to_dict(),
            },
        )

        plan = await self.planner.plan(question, budget)
        await self._emit_event("pipeline.planner_done", {"task_id": plan.task_id})

        evidence = await self.researcher.research(plan, budget)
        await self._emit_event("pipeline.researcher_done", {"task_id": plan.task_id})

        reasoning = await self.reasoner.reason(plan, evidence, budget)
        await self._emit_event("pipeline.reasoner_done", {"task_id": plan.task_id})

        critique = await self.critic.review(plan, evidence, reasoning, budget)
        await self._emit_event("pipeline.critic_done", {"task_id": plan.task_id})

        compression = await self.compressor.propose(reasoning, logs=[])
        await self._emit_event("pipeline.compressor_done", {"task_id": plan.task_id})

        result = TaskResult(
            task_id=plan.task_id,
            plan=plan,
            evidence=evidence,
            reasoning=reasoning,
            critique=critique,
            compression=tuple(compression),
        )
        await self._emit_event("pipeline.completed", {"task_id": plan.task_id})
        return result

    async def run_pipeline(self, question: str) -> TaskResult:
        """Compatibility wrapper that uses the smallest valid budget."""
        return await self.run_task(question, thinking_minutes=1)

    async def _emit_event(self, stage: str, payload: dict[str, Any]) -> None:
        event_payload = dict(payload)
        event_payload["timestamp"] = utc_now_iso()
        await self.storage.log_event(stage, event_payload)
        self.dashboard.publish_event({"stage": stage, **event_payload})

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
