"""Planner agent wrapper around the typed planning service."""

from __future__ import annotations

import logging

from config import APP_CONFIG, AppConfig
from data_structures import Plan, ResourceBudget
from model_manager import ModelManager
from planner_service import PlannerService


class PlannerAgent:
    """Builds a task plan from the user question."""

    def __init__(
        self,
        model_manager: ModelManager,
        config: AppConfig = APP_CONFIG,
        service: PlannerService | None = None,
    ):
        self.model_manager = model_manager
        self.config = config
        self.logger = logging.getLogger("quester.planner")
        self.service = service or PlannerService(model_manager=model_manager, config=config)
        self._started = False

    async def start(self) -> None:
        """Mark the planner ready to accept planning requests."""
        if self._started:
            return
        self._started = True

    async def stop(self) -> None:
        """Reject future planning requests until the planner is restarted."""
        self._started = False

    async def plan(self, question: str, budget: ResourceBudget) -> Plan:
        """Return a typed plan for downstream pipeline stages."""
        if not self._started:
            raise RuntimeError("PlannerAgent must be started before use.")
        return await self.service.plan(question, budget)
