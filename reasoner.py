"""Reasoner agent scaffold."""

from __future__ import annotations

import logging

from config import APP_CONFIG, AppConfig
from data_structures import CompressedTrace, EvidenceBundle, Plan, ResourceBudget
from model_manager import ModelManager
from prompts import REASONER_PROMPT


class ReasonerAgent:
    """Produces a compressed-style reasoning trace placeholder."""

    def __init__(self, model_manager: ModelManager, config: AppConfig = APP_CONFIG):
        self.model_manager = model_manager
        self.config = config
        self.logger = logging.getLogger("quester.reasoner")
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def reason(
        self,
        plan: Plan,
        evidence: EvidenceBundle,
        budget: ResourceBudget,
    ) -> CompressedTrace:
        """Build a typed baseline reasoning trace for critic validation."""
        if not self._started:
            raise RuntimeError("ReasonerAgent must be started before use.")
        prompt = (
            f"{REASONER_PROMPT}\n"
            f"Task: {plan.question}\n"
            f"LocalEvidenceCount: {len(evidence.local_results)}\n"
            f"ReasonerPasses: {budget.reasoner_passes}\n"
            f"MacroDepth: {budget.macro_depth}"
        )
        notes = await self.model_manager.generate(prompt)
        base_chain = [
            "@read_question",
            "@extract_constraints",
            "@match_local_evidence",
            "@compose_answer",
        ]
        for _ in range(max(0, budget.reasoner_passes - 1)):
            base_chain.append("@refine_answer")
        chain = tuple(base_chain)
        return CompressedTrace(
            task_id=plan.task_id,
            tokens=chain,
            expanded_preview=(
                "Read question and constraints",
                "Match local evidence",
                "Compose answer candidate",
                f"Reasoning passes: {budget.reasoner_passes}",
            ),
            macros_used=(),
            confidence=0.8,
            reasoner_notes=notes,
        )
