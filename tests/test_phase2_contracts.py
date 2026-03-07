"""Phase 2 contract and typed-pipeline tests."""

from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from inspect import signature
from pathlib import Path

from config import APP_CONFIG, BudgetPolicy
from data_structures import (
    AgentState,
    AgentStatus,
    CompressedTrace,
    CritiqueReport,
    CritiqueResult,
    EvidenceBundle,
    EvidenceItem,
    KnowledgeVector,
    LifecycleStatus,
    Macro,
    MacroProposal,
    PerformanceMetric,
    Plan,
    PlanStep,
    ReasoningLog,
    ResourceBudget,
    RuntimeEvent,
    SourceType,
    TaskResult,
    TaskState,
)
from macro_engine import MacroEngine
from orchestrator import Orchestrator
from planner import PlannerAgent
from researcher import ResearcherAgent
from reasoner import ReasonerAgent
from critic import CriticAgent
from compressor import CompressorAgent
from self_optimizer import SelfOptimizer


def _build_contract_examples() -> dict[str, object]:
    budget = ResourceBudget(
        retrieval_top_k=5,
        max_web_queries=2,
        reasoner_passes=2,
        critic_passes=2,
        macro_depth=3,
    )
    plan_step = PlanStep(
        step_id="step_1",
        description="Interpret intent",
        depends_on=(),
        status=TaskState.PENDING,
    )
    plan = Plan(
        task_id="task-1",
        question="What is phase 2?",
        steps=(plan_step,),
        required_evidence=("local documents",),
        success_criteria=("typed output",),
        budget=budget,
        planner_notes="notes",
    )
    local_item = EvidenceItem(
        id="ev-1",
        content="Local evidence sample",
        source_type=SourceType.LOCAL,
        source_ref="local://sample",
        score=0.8,
        metadata={"topic": "phase2"},
        vector_preview=(0.1, 0.2),
    )
    evidence = EvidenceBundle(
        task_id=plan.task_id,
        local_results=(local_item,),
        web_results=(),
        used_web_fallback=False,
    )
    trace = CompressedTrace(
        task_id=plan.task_id,
        tokens=("@read_question", "@compose_answer"),
        expanded_preview=("Read question", "Compose answer"),
        macros_used=("@compose_answer",),
        confidence=0.75,
        reasoner_notes="reasoner-note",
    )
    critique = CritiqueReport(
        task_id=plan.task_id,
        is_valid=True,
        issues=(),
        fixed_trace=trace,
        evidence_coverage=1.0,
        critic_notes="critic-note",
        result=CritiqueResult.VALID,
    )
    proposal = MacroProposal(
        proposal_id="proposal-1",
        macro=Macro(macro_name="compose_answer", expansion=("@compose_answer",), version=1),
        reason="Repeated token.",
        examples=("@compose_answer",),
        simulation_score=0.5,
        approved=False,
    )
    task_result = TaskResult(
        task_id=plan.task_id,
        plan=plan,
        evidence=evidence,
        reasoning=trace,
        critique=critique,
        compression=(proposal,),
    )
    return {
        "ResourceBudget": budget,
        "PlanStep": plan_step,
        "Plan": plan,
        "Macro": proposal.macro,
        "KnowledgeVector": KnowledgeVector(
            id="vec-1",
            vector=(0.1, 0.2, 0.3),
            source=SourceType.LOCAL,
        ),
        "ReasoningLog": ReasoningLog(
            task_id=plan.task_id,
            compressed_chain=trace.tokens,
            macros_used=trace.macros_used,
        ),
        "PerformanceMetric": PerformanceMetric(
            task_id=plan.task_id,
            time=0.12,
            vram_usage=1.4,
            iterations=2,
        ),
        "EvidenceItem": local_item,
        "EvidenceBundle": evidence,
        "CompressedTrace": trace,
        "CritiqueReport": critique,
        "MacroProposal": proposal,
        "TaskResult": task_result,
        "RuntimeEvent": RuntimeEvent(stage="pipeline.completed", payload={"task_id": plan.task_id}),
        "LifecycleStatus": LifecycleStatus(component="planner", state=AgentState.RUNNING),
        "AgentStatus": AgentStatus(component="reasoner", state=AgentState.RUNNING, task_id=plan.task_id),
    }


class DataStructureRoundTripTests(unittest.TestCase):
    """Validate typed contract serialization and reconstruction."""

    def test_contract_round_trip(self) -> None:
        for name, instance in _build_contract_examples().items():
            payload = instance.to_dict()
            rebuilt = type(instance).from_dict(payload)
            self.assertEqual(instance, rebuilt, msg=f"Round-trip failed for {name}")


class DataStructureValidationTests(unittest.TestCase):
    """Validate constructors fail on malformed data."""

    def test_invalid_budget_raises(self) -> None:
        with self.assertRaises(ValueError):
            ResourceBudget(retrieval_top_k=0)

    def test_invalid_plan_raises(self) -> None:
        with self.assertRaises(ValueError):
            Plan(
                task_id="task-2",
                question="Invalid plan",
                steps=(),
                required_evidence=(),
                success_criteria=(),
            )

    def test_invalid_evidence_score_raises(self) -> None:
        with self.assertRaises(ValueError):
            EvidenceItem(
                id="ev-2",
                content="bad score",
                source_type=SourceType.LOCAL,
                source_ref="local://bad",
                score=1.5,
            )


class MacroEngineContractTests(unittest.TestCase):
    """Validate macro engine typed interfaces."""

    def test_macro_engine_returns_typed_trace(self) -> None:
        engine = MacroEngine()
        engine.register_macro("@macro_step", ("expand 1", "expand 2"))
        trace = engine.compress(["@macro_step", "@compose_answer"], task_id="task-macro")
        self.assertIsInstance(trace, CompressedTrace)
        self.assertTrue(engine.verify_round_trip(trace))


class PublicInterfaceChecklistTests(unittest.TestCase):
    """Validate the top-level public API checklist is actually true."""

    def test_budget_policy_returns_resource_budget(self) -> None:
        budget = BudgetPolicy.from_minutes(30)
        self.assertIsInstance(budget, ResourceBudget)
        self.assertEqual(budget.reasoner_passes, 2)

    def test_public_method_signatures_exist(self) -> None:
        self.assertIn("budget", signature(PlannerAgent.plan).parameters)
        self.assertIn("budget", signature(ResearcherAgent.research).parameters)
        self.assertIn("budget", signature(ReasonerAgent.reason).parameters)
        self.assertIn("budget", signature(CriticAgent.review).parameters)
        self.assertIn("logs", signature(CompressorAgent.propose).parameters)
        self.assertIn("thinking_minutes", signature(Orchestrator.run_task).parameters)
        self.assertIn("MacroProposal", str(signature(SelfOptimizer.run_cycle).return_annotation))


class Phase2TypedPipelineTests(unittest.IsolatedAsyncioTestCase):
    """Validate startup/shutdown and typed pipeline output in stub mode."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase2.sqlite3")
        self.test_logs = Path("test_phase2_logs")
        preflight = replace(
            APP_CONFIG.preflight,
            flags=replace(
                APP_CONFIG.preflight.flags,
                stub_mode=True,
                enable_self_optimizer=False,
            ),
        )
        storage = replace(APP_CONFIG.storage, sqlite_path=self.test_db, logs_dir=self.test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        self.test_config = replace(
            APP_CONFIG,
            preflight=preflight,
            storage=storage,
            dashboard=dashboard,
        )
        self.orchestrator = Orchestrator(config=self.test_config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_pipeline_returns_typed_task_result(self) -> None:
        result = await self.orchestrator.run_task("Phase 2 typed question", thinking_minutes=30)
        self.assertIsInstance(result, TaskResult)
        self.assertIsInstance(result.plan, Plan)
        self.assertIsInstance(result.evidence, EvidenceBundle)
        self.assertIsInstance(result.reasoning, CompressedTrace)
        self.assertIsInstance(result.critique, CritiqueReport)
        self.assertIsInstance(result.compression, tuple)
        self.assertTrue(result.critique.is_valid)
        self.assertEqual(result.plan.budget.reasoner_passes, 2)


if __name__ == "__main__":
    unittest.main()
