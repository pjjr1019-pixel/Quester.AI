"""Shared plan-building logic used by PlannerAgent."""

from __future__ import annotations

import uuid

from agent_schema import parse_planner_output, planner_output_schema
from config import APP_CONFIG, AppConfig
from data_structures import Plan, PlanStep, ResourceBudget
from model_manager import ModelManager
from prompts import PLANNER_PROMPT
from structured_generation import StructuredGenerationService


class PlannerService:
    """Build typed plans while keeping PlannerAgent thin."""

    output_contract = "planner_plan_v1"
    implementation_mode = "deterministic_stub"

    def __init__(self, model_manager: ModelManager, config: AppConfig = APP_CONFIG):
        self.model_manager = model_manager
        self.config = config
        self.structured_generation = StructuredGenerationService(model_manager=model_manager, config=config)

    async def plan(self, question: str, budget: ResourceBudget) -> Plan:
        prompt = (
            f"{PLANNER_PROMPT}\n"
            f"Question: {question}\n"
            f"Budget: {budget.to_dict()}\n"
            f"OutputContract: {self.output_contract}"
        )
        decode_result = await self.structured_generation.decode_json_output(
            prompt=prompt,
            schema=planner_output_schema(),
            parser=parse_planner_output,
            fallback_factory=lambda error_message: self._build_deterministic_plan(
                question=question,
                budget=budget,
                planner_notes=self._fallback_notes(error_message),
            ),
            max_tokens=self.config.model_tuning.default_max_tokens,
        )
        if decode_result.used_fallback:
            return decode_result.value
        return self._normalize_plan(
            decode_result.value,
            question=question,
            budget=budget,
            planner_notes=self._structured_notes(decode_result),
        )

    def _normalize_plan(
        self,
        plan: Plan,
        *,
        question: str,
        budget: ResourceBudget,
        planner_notes: str,
    ) -> Plan:
        return Plan(
            task_id=plan.task_id or str(uuid.uuid4()),
            question=question,
            steps=plan.steps,
            required_evidence=plan.required_evidence,
            success_criteria=plan.success_criteria,
            budget=budget,
            planner_notes=planner_notes,
            created_at=plan.created_at,
        )

    def _build_deterministic_plan(
        self,
        *,
        question: str,
        budget: ResourceBudget,
        planner_notes: str,
    ) -> Plan:
        steps: list[PlanStep] = [
            PlanStep(step_id="step_1", description="Interpret user intent"),
            PlanStep(
                step_id="step_2",
                description=f"Gather up to {budget.retrieval_top_k} local evidence items",
                depends_on=("step_1",),
            ),
        ]
        previous_step = "step_2"
        if budget.max_web_queries > 1:
            steps.append(
                PlanStep(
                    step_id="step_3",
                    description=(
                        f"Use up to {budget.max_web_queries} bounded web queries only if local "
                        "evidence is insufficient"
                    ),
                    depends_on=(previous_step,),
                )
            )
            previous_step = "step_3"
        for pass_index in range(1, budget.reasoner_passes + 1):
            step_id = f"reason_{pass_index}"
            steps.append(
                PlanStep(
                    step_id=step_id,
                    description=f"Reason over evidence pass {pass_index} of {budget.reasoner_passes}",
                    depends_on=(previous_step,),
                )
            )
            previous_step = step_id
        for pass_index in range(1, budget.critic_passes + 1):
            step_id = f"critic_{pass_index}"
            steps.append(
                PlanStep(
                    step_id=step_id,
                    description=f"Validate reasoning pass {pass_index} of {budget.critic_passes}",
                    depends_on=(previous_step,),
                )
            )
            previous_step = step_id
        steps.append(
            PlanStep(
                step_id="step_finalize",
                description="Suggest compression improvements",
                depends_on=(previous_step,),
            )
        )
        return Plan(
            task_id=str(uuid.uuid4()),
            question=question,
            steps=tuple(steps),
            required_evidence=(
                f"up to {budget.retrieval_top_k} local evidence items",
                "supporting context",
            )
            + (
                (
                    f"up to {budget.max_web_queries} bounded web lookups if freshness or missing evidence requires it",
                )
                if budget.max_web_queries > 1
                else ()
            ),
            success_criteria=(
                "answer is coherent",
                f"reasoning completes in {budget.reasoner_passes} pass(es)",
                f"critic completes in {budget.critic_passes} pass(es)",
            ),
            budget=budget,
            planner_notes=planner_notes,
        )

    def _structured_notes(self, decode_result) -> str:
        note_parts = [
            "planner_output_mode=structured_json",
            f"output_contract={self.output_contract}",
            f"implementation_mode={self.implementation_mode}",
            f"used_repair={'yes' if decode_result.used_repair else 'no'}",
            f"used_fallback={'yes' if decode_result.used_fallback else 'no'}",
            f"raw_output={decode_result.raw_text}",
        ]
        if decode_result.repaired_text is not None:
            note_parts.append(f"repaired_output={decode_result.repaired_text}")
        return "\n".join(note_parts)

    def _fallback_notes(self, error_message: str | None) -> str:
        note_parts = [
            "planner_output_mode=deterministic_fallback",
            f"output_contract={self.output_contract}",
            f"implementation_mode={self.implementation_mode}",
            f"parse_error={error_message or 'unknown'}",
        ]
        return "\n".join(note_parts)
