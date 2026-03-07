"""Planner agent scaffold."""

from __future__ import annotations

import logging
import uuid

from config import APP_CONFIG, AppConfig
from data_structures import Plan, PlanStep, ResourceBudget
from model_manager import ModelManager
from prompts import PLANNER_PROMPT
from utils import utc_now_iso


class PlannerAgent:
    """Builds a task plan from the user question."""

    def __init__(self, model_manager: ModelManager, config: AppConfig = APP_CONFIG):
        self.model_manager = model_manager
        self.config = config
        self.logger = logging.getLogger("quester.planner")
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def plan(self, question: str, budget: ResourceBudget) -> Plan:
        """Return a typed plan for downstream pipeline stages."""
        if not self._started:
            raise RuntimeError("PlannerAgent must be started before use.")
        prompt = f"{PLANNER_PROMPT}\nQuestion: {question}"
        planner_notes = await self.model_manager.generate(prompt)
        steps = (
            PlanStep(step_id="step_1", description="Interpret user intent"),
            PlanStep(step_id="step_2", description="Gather relevant evidence", depends_on=("step_1",)),
            PlanStep(step_id="step_3", description="Reason over evidence", depends_on=("step_2",)),
            PlanStep(step_id="step_4", description="Validate reasoning", depends_on=("step_3",)),
            PlanStep(
                step_id="step_5",
                description="Suggest compression improvements",
                depends_on=("step_4",),
            ),
        )
        return Plan(
            task_id=str(uuid.uuid4()),
            question=question,
            steps=steps,
            required_evidence=("local vectors", "supporting context"),
            success_criteria=("answer is coherent", "critic marks output valid"),
            budget=budget,
            planner_notes=f"{planner_notes}\ncreated_at={utc_now_iso()}",
        )
