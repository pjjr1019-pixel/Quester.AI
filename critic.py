"""Critic agent scaffold."""

from __future__ import annotations

import logging

from config import APP_CONFIG, AppConfig
from data_structures import (
    CompressedTrace,
    CritiqueReport,
    CritiqueResult,
    EvidenceBundle,
    Plan,
    ResourceBudget,
)
from model_manager import ModelManager
from prompts import CRITIC_PROMPT


class CriticAgent:
    """Validates reasoning output against basic consistency checks."""

    def __init__(self, model_manager: ModelManager, config: AppConfig = APP_CONFIG):
        self.model_manager = model_manager
        self.config = config
        self.logger = logging.getLogger("quester.critic")
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def review(
        self,
        plan: Plan,
        evidence: EvidenceBundle,
        trace: CompressedTrace,
        budget: ResourceBudget,
    ) -> CritiqueReport:
        """Return a typed baseline critique report."""
        if not self._started:
            raise RuntimeError("CriticAgent must be started before use.")
        prompt = (
            f"{CRITIC_PROMPT}\n"
            f"Task: {plan.question}\n"
            f"TraceLength: {len(trace.tokens)}\n"
            f"CriticPasses: {budget.critic_passes}"
        )
        notes = await self.model_manager.generate(prompt)
        has_evidence = bool(evidence.local_results) or bool(evidence.web_results)
        issues: list[str] = []
        if not has_evidence:
            issues.append("No evidence found in local or web sources.")
        if budget.critic_passes < 1:
            issues.append("Critic budget is invalid.")
        is_valid = len(issues) == 0
        return CritiqueReport(
            task_id=plan.task_id,
            is_valid=is_valid,
            issues=tuple(issues),
            fixed_trace=trace if is_valid else None,
            evidence_coverage=1.0 if has_evidence else 0.0,
            critic_notes=notes,
            result=CritiqueResult.VALID if is_valid else CritiqueResult.INVALID,
        )
