"""Phase 7 boundary and thin-agent tests."""

from __future__ import annotations

import json
import shutil
import unittest
from dataclasses import replace
from pathlib import Path

from agent_schema import (
    compressor_output_schema,
    critic_output_schema,
    planner_output_schema,
    reasoner_critic_handoff_schema,
    reasoner_output_schema,
    research_reasoner_handoff_schema,
)
from config import APP_CONFIG
from data_structures import ResourceBudget
from orchestrator import Orchestrator
from planner_service import PlannerService


class _QueuedGenerationManager:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        _ = max_tokens
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("No fake responses remaining.")
        return self._responses.pop(0)


class Phase7BoundaryTests(unittest.IsolatedAsyncioTestCase):
    """Verify typed handoffs and boundary schemas are wired into the pipeline."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase7_boundaries.sqlite3")
        self.test_logs = Path("test_phase7_boundaries_logs")
        backends = replace(
            APP_CONFIG.preflight.backends,
            vector_store_backend="simple_inmemory",
            vector_store_fallback_backend="simple_inmemory",
        )
        preflight = replace(
            APP_CONFIG.preflight,
            backends=backends,
            flags=replace(
                APP_CONFIG.preflight.flags,
                stub_mode=True,
                enable_self_optimizer=False,
            ),
        )
        storage_cfg = replace(APP_CONFIG.storage, sqlite_path=self.test_db, logs_dir=self.test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        self.test_config = replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)
        self.orchestrator = Orchestrator(config=self.test_config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_pipeline_uses_typed_reasoner_and_critic_handoffs(self) -> None:
        result = await self.orchestrator.run_task(
            "What typed boundaries exist between phases 7 agents?",
            thinking_minutes=30,
        )

        reasoner_handoff = self.orchestrator.reasoner.last_handoff
        critic_handoff = self.orchestrator.critic.last_handoff

        self.assertIsNotNone(reasoner_handoff)
        self.assertIsNotNone(critic_handoff)
        assert reasoner_handoff is not None
        assert critic_handoff is not None
        self.assertEqual(reasoner_handoff.plan.task_id, result.task_id)
        self.assertEqual(reasoner_handoff.evidence.task_id, result.task_id)
        self.assertEqual(reasoner_handoff.evidence_handles, result.reasoning.evidence_handles)
        self.assertEqual(reasoner_handoff.final_text_policy, "post_verification")
        self.assertEqual(critic_handoff.plan.task_id, result.task_id)
        self.assertEqual(critic_handoff.trace.task_id, result.task_id)
        self.assertEqual(critic_handoff.proof_hash, result.reasoning.proof_hash)
        self.assertEqual(critic_handoff.evidence_handles, result.reasoning.evidence_handles)
        self.assertEqual(critic_handoff.final_text_policy, "post_verification")

    def test_boundary_schema_helpers_lock_required_shapes(self) -> None:
        self.assertEqual(planner_output_schema()["title"], "planner_plan_v1")
        self.assertEqual(research_reasoner_handoff_schema()["title"], "research_reasoner_handoff_v1")
        self.assertEqual(reasoner_output_schema()["title"], "compressed_trace_v1")
        self.assertEqual(reasoner_critic_handoff_schema()["title"], "reasoner_critic_handoff_v1")
        self.assertEqual(critic_output_schema()["title"], "critique_report_v1")
        self.assertEqual(compressor_output_schema()["title"], "macro_proposal_list_v1")


class PlannerStructuredOutputTests(unittest.IsolatedAsyncioTestCase):
    """Verify the planner uses the structured JSON path with bounded repair."""

    def _payload(self, *, task_id: str = "plan-1", question: str = "ignored") -> str:
        return json.dumps(
            {
                "task_id": task_id,
                "question": question,
                "steps": [
                    {
                        "step_id": "step_a",
                        "description": "Inspect the question",
                        "depends_on": [],
                        "status": "pending",
                    },
                    {
                        "step_id": "step_b",
                        "description": "Return a plan",
                        "depends_on": ["step_a"],
                        "status": "pending",
                    },
                ],
                "required_evidence": ["local docs"],
                "success_criteria": ["produce typed plan"],
                "planner_notes": "model-notes",
                "created_at": "2026-03-07T12:00:00+00:00",
            }
        )

    async def test_planner_uses_structured_json_when_available(self) -> None:
        manager = _QueuedGenerationManager([self._payload(task_id="plan-structured")])
        service = PlannerService(model_manager=manager, config=APP_CONFIG)
        budget = ResourceBudget(
            retrieval_top_k=6,
            max_web_queries=2,
            reasoner_passes=2,
            critic_passes=2,
            macro_depth=3,
        )

        plan = await service.plan("Actual planner question", budget)

        self.assertEqual(plan.task_id, "plan-structured")
        self.assertEqual(plan.question, "Actual planner question")
        self.assertEqual(plan.budget, budget)
        self.assertEqual(tuple(step.step_id for step in plan.steps), ("step_a", "step_b"))
        self.assertIn("planner_output_mode=structured_json", plan.planner_notes)
        self.assertIn("used_repair=no", plan.planner_notes)
        self.assertEqual(len(manager.prompts), 1)

    async def test_planner_repairs_invalid_json_once_before_accepting(self) -> None:
        manager = _QueuedGenerationManager(
            [
                "not valid json at all",
                self._payload(task_id="plan-repaired"),
            ]
        )
        service = PlannerService(model_manager=manager, config=APP_CONFIG)
        budget = ResourceBudget(
            retrieval_top_k=6,
            max_web_queries=2,
            reasoner_passes=2,
            critic_passes=2,
            macro_depth=3,
        )

        plan = await service.plan("Planner repair question", budget)

        self.assertEqual(plan.task_id, "plan-repaired")
        self.assertIn("planner_output_mode=structured_json", plan.planner_notes)
        self.assertIn("used_repair=yes", plan.planner_notes)
        self.assertIn("used_fallback=no", plan.planner_notes)
        self.assertEqual(len(manager.prompts), 2)

    async def test_planner_falls_back_after_bounded_repair_failure(self) -> None:
        manager = _QueuedGenerationManager(
            [
                "still not json",
                "repair also failed",
            ]
        )
        service = PlannerService(model_manager=manager, config=APP_CONFIG)
        budget = ResourceBudget(
            retrieval_top_k=4,
            max_web_queries=1,
            reasoner_passes=1,
            critic_passes=1,
            macro_depth=2,
        )

        plan = await service.plan("Fallback planner question", budget)

        self.assertEqual(plan.question, "Fallback planner question")
        self.assertEqual(plan.budget, budget)
        self.assertEqual(plan.steps[0].step_id, "step_1")
        self.assertIn("planner_output_mode=deterministic_fallback", plan.planner_notes)
        self.assertIn("parse_error=", plan.planner_notes)
        self.assertEqual(len(manager.prompts), 2)
