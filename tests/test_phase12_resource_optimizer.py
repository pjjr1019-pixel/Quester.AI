"""Phase 12 resource and optimizer tests that were still implicit or indirect."""

from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import APP_CONFIG
from critique_service import CritiqueService
from data_structures import (
    CompressionRuntimeSubset,
    DecoderEntry,
    EvidenceBundle,
    EvidenceItem,
    Macro,
    MacroProposal,
    OpcodeEntry,
    OptimizerActivationDecision,
    OptimizerLifecycleStage,
    Plan,
    PlanStep,
    ResearchReasonerHandoff,
    ResourceBudget,
    SourceType,
    SymbolTableSnapshot,
)
from orchestrator import Orchestrator
from reasoning_service import ReasoningService
from self_optimizer import SelfOptimizer


def _build_test_config(
    *,
    sqlite_name: str,
    logs_name: str,
    dashboard_queue_maxsize: int = 8,
):
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
            allow_web_fallback=True,
        ),
    )
    storage_cfg = replace(APP_CONFIG.storage, sqlite_path=Path(sqlite_name), logs_dir=Path(logs_name))
    dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
    concurrency = replace(APP_CONFIG.concurrency, dashboard_queue_maxsize=dashboard_queue_maxsize)
    return replace(
        APP_CONFIG,
        preflight=preflight,
        storage=storage_cfg,
        dashboard=dashboard,
        concurrency=concurrency,
    )


class Phase12ResourceAndOptimizerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase12_resource_optimizer.sqlite3")
        self.test_logs = Path("test_phase12_resource_optimizer_logs")
        self.orchestrator: Orchestrator | None = None

    async def asyncTearDown(self) -> None:
        if self.orchestrator is not None:
            await self.orchestrator.stop()
            self.orchestrator = None
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_run_task_surfaces_dashboard_backpressure_as_runtime_condition_and_warning(self) -> None:
        config = _build_test_config(
            sqlite_name=str(self.test_db),
            logs_name=str(self.test_logs),
            dashboard_queue_maxsize=1,
        )
        orchestrator = Orchestrator(config=config)
        self.orchestrator = orchestrator
        await orchestrator.start()

        result = await orchestrator.run_task("What is 2 + 2?", thinking_minutes=30)
        events = await orchestrator.storage.list_runtime_events(stage="runtime.backpressure_detected")

        self.assertTrue(events)
        self.assertEqual(events[-1].payload["reason"], "dashboard_queue_overflow")
        self.assertGreaterEqual(int(events[-1].payload["metadata"]["dropped_events"]), 1)
        self.assertTrue(any(str(item).startswith("dashboard_dropped_events:") for item in result.warnings))

    async def test_optimizer_cycle_records_simulation_before_activation_and_never_mutates_active_macros(self) -> None:
        config = _build_test_config(
            sqlite_name=str(self.test_db),
            logs_name=str(self.test_logs),
        )
        orchestrator = Orchestrator(config=config)
        self.orchestrator = orchestrator
        await orchestrator.start()

        await orchestrator.run_task("What is 2 + 2?", thinking_minutes=30)
        active_macros_before = await orchestrator.storage.list_macros(active_only=True)

        proposals = await orchestrator.self_optimizer.run_cycle()

        proposal_records = await orchestrator.storage.list_optimizer_proposal_records()
        activation_records = await orchestrator.storage.list_optimizer_activation_records()
        active_macros_after = await orchestrator.storage.list_macros(active_only=True)

        self.assertTrue(proposals)
        self.assertEqual(active_macros_before, active_macros_after)
        self.assertTrue(activation_records)
        for activation in activation_records:
            lifecycle = [
                record.lifecycle_stage.value
                for record in proposal_records
                if record.proposal_id == activation.proposal_id
            ]
            self.assertIn("proposed", lifecycle)
            self.assertIn("simulated", lifecycle)
            self.assertTrue(any(stage in {"validated", "rejected"} for stage in lifecycle))
            self.assertFalse(activation.activation_applied)


class Phase12OptimizerDecisionTests(unittest.TestCase):
    def _proposal(self, *, validation_passed: bool) -> MacroProposal:
        return MacroProposal(
            proposal_id="proposal:test",
            macro=Macro(
                macro_name="proposal_test",
                expansion=("@compose_answer",),
                version=1,
            ),
            reason="Phase 16 optimizer gate test proposal.",
            examples=("@compose_answer",),
            simulation_score=0.8,
            approved=False,
            validation_passed=validation_passed,
        )

    def test_activation_outcome_requires_replay_then_validation_then_policy_block(self) -> None:
        optimizer = SelfOptimizer(storage=SimpleNamespace())

        deferred = optimizer._evaluate_activation_outcome(
            proposal=self._proposal(validation_passed=True),
            summary={"sample_count": 0},
        )
        validation_rejected = optimizer._evaluate_activation_outcome(
            proposal=self._proposal(validation_passed=False),
            summary={"sample_count": 2, "validation_ready": False},
        )
        simulation_rejected = optimizer._evaluate_activation_outcome(
            proposal=self._proposal(validation_passed=True),
            summary={"sample_count": 2, "validation_ready": False},
        )
        contradiction_rejected = optimizer._evaluate_activation_outcome(
            proposal=self._proposal(validation_passed=True),
            summary={
                "sample_count": 2,
                "validation_ready": True,
                "activation_eligible": False,
                "rejection_reasons": ("contradiction_risk_too_high",),
            },
        )
        blocked = optimizer._evaluate_activation_outcome(
            proposal=self._proposal(validation_passed=True),
            summary={
                "sample_count": 2,
                "validation_ready": True,
                "activation_eligible": True,
            },
        )

        self.assertEqual(
            deferred,
            (
                OptimizerActivationDecision.DEFERRED,
                OptimizerLifecycleStage.SIMULATED,
                "replay_samples_unavailable",
            ),
        )
        self.assertEqual(
            validation_rejected,
            (
                OptimizerActivationDecision.REJECTED,
                OptimizerLifecycleStage.REJECTED,
                "proposal_validation_failed",
            ),
        )
        self.assertEqual(
            simulation_rejected,
            (
                OptimizerActivationDecision.REJECTED,
                OptimizerLifecycleStage.REJECTED,
                "simulation_gate_failed",
            ),
        )
        self.assertEqual(
            contradiction_rejected,
            (
                OptimizerActivationDecision.REJECTED,
                OptimizerLifecycleStage.REJECTED,
                "contradiction_risk_too_high",
            ),
        )
        self.assertEqual(
            blocked,
            (
                OptimizerActivationDecision.BLOCKED,
                OptimizerLifecycleStage.ACTIVATION_BLOCKED,
                "proposal_only_policy",
            ),
        )


class Phase12DeepModeBoundTests(unittest.TestCase):
    def _model_manager_stub(self):
        return type(
            "ModelManagerStub",
            (),
            {"health_snapshot": lambda self: SimpleNamespace(generation_backend="stub_generation")},
        )()

    def _build_research_handoff(
        self,
        *,
        question: str,
        budget: ResourceBudget,
    ) -> ResearchReasonerHandoff:
        plan = Plan(
            task_id="task-deep-cap",
            question=question,
            steps=(PlanStep(step_id="step_1", description="Inspect the evidence"),),
            required_evidence=("supporting evidence",),
            success_criteria=("return a verified answer",),
            budget=budget,
        )
        evidence = EvidenceBundle(
            task_id=plan.task_id,
            local_results=(
                EvidenceItem(
                    id="ev-1",
                    content="First supporting fact for the bounded deep-mode test.",
                    source_type=SourceType.LOCAL,
                    source_ref="local://ev-1",
                    score=0.95,
                ),
                EvidenceItem(
                    id="ev-2",
                    content="Second supporting fact for the bounded deep-mode test.",
                    source_type=SourceType.LOCAL,
                    source_ref="local://ev-2",
                    score=0.91,
                ),
            ),
            web_results=(),
            used_web_fallback=False,
        )
        return ResearchReasonerHandoff.from_inputs(
            plan=plan,
            evidence=evidence,
            budget=budget,
            reasoning_mode="deep",
        )

    def test_deep_mode_candidate_count_remains_capped_under_large_budget(self) -> None:
        service = ReasoningService(
            model_manager=type(
                "ModelManagerStub",
                (),
                {"health_snapshot": lambda self: SimpleNamespace(generation_backend="stub_generation")},
            )()
        )
        budget = ResourceBudget(
            retrieval_top_k=10,
            max_web_queries=5,
            reasoner_passes=12,
            critic_passes=10,
            macro_depth=4,
        )
        handoff = type("Handoff", (), {})()
        handoff.plan = type("PlanPayload", (), {"task_id": "task-deep-cap", "question": "What is 2 + 2?"})()
        handoff.evidence = type(
            "EvidencePayload",
            (),
            {
                "local_results": (
                    type("Evidence", (), {"id": "ev-1", "content": "2 + 2 = 4", "source_type": type("Source", (), {"value": "local"})()})(),
                    type("Evidence", (), {"id": "ev-2", "content": "Arithmetic confirms 4", "source_type": type("Source", (), {"value": "local"})()})(),
                ),
                "web_results": (),
            },
        )()
        handoff.budget = budget
        handoff.reasoning_mode = "deep"
        handoff.evidence_handles = ("ev-1", "ev-2")

        trace = service._build_deterministic_trace(
            handoff,
            CompressionRuntimeSubset(task_id="task-deep-cap"),
        )

        self.assertLessEqual(len(trace.candidate_traces), 5)
        self.assertEqual(trace.context_frames[0].metadata["cc"], len(trace.candidate_traces))

    def test_deep_mode_tool_candidate_generation_remains_capped_under_large_budget(self) -> None:
        service = ReasoningService(model_manager=self._model_manager_stub())
        budget = ResourceBudget(
            retrieval_top_k=10,
            max_web_queries=5,
            reasoner_passes=12,
            critic_passes=10,
            macro_depth=4,
        )
        handoff = self._build_research_handoff(
            question="Bound the candidate set even when every tool path can contribute.",
            budget=budget,
        )

        with (
            patch("reasoning_service.evaluate_arithmetic_question", return_value="arith-answer"),
            patch("reasoning_service.evaluate_python_expression_question", return_value="expr-answer"),
            patch("reasoning_service.evaluate_python_code_question", return_value="code-answer"),
            patch("reasoning_service.evaluate_python_unit_test_question", return_value="unit-answer"),
            patch("reasoning_service.expected_evidence_count", return_value="count-answer"),
        ):
            candidates = service._build_answer_candidates(handoff)
            trace = service._build_deterministic_trace(
                handoff,
                CompressionRuntimeSubset(task_id=handoff.plan.task_id),
            )

        self.assertEqual(len(candidates), 5)
        self.assertEqual(len(trace.candidate_traces), 5)
        self.assertEqual(trace.context_frames[0].metadata["cc"], 5)
        self.assertTrue(
            all(str(candidate["verifier_type"]).startswith("tool.") for candidate in candidates),
        )

    def test_critic_verification_depth_remains_capped_above_supported_threshold(self) -> None:
        reasoner_service = ReasoningService(model_manager=self._model_manager_stub())
        critic_service = CritiqueService(model_manager=self._model_manager_stub())
        base_budget = ResourceBudget(
            retrieval_top_k=10,
            max_web_queries=5,
            reasoner_passes=4,
            critic_passes=3,
            macro_depth=4,
        )
        research_handoff = self._build_research_handoff(
            question="What is 2 + 2?",
            budget=base_budget,
        )
        trace = reasoner_service._build_deterministic_trace(
            research_handoff,
            CompressionRuntimeSubset(task_id=research_handoff.plan.task_id),
        )
        critic_handoff_capped = reasoner_service.build_critic_handoff(
            plan=research_handoff.plan,
            evidence=research_handoff.evidence,
            trace=trace,
            budget=base_budget,
        )
        larger_budget = ResourceBudget(
            retrieval_top_k=base_budget.retrieval_top_k,
            max_web_queries=base_budget.max_web_queries,
            reasoner_passes=base_budget.reasoner_passes,
            critic_passes=10,
            macro_depth=base_budget.macro_depth,
        )
        critic_handoff_oversized = reasoner_service.build_critic_handoff(
            plan=research_handoff.plan,
            evidence=research_handoff.evidence,
            trace=trace,
            budget=larger_budget,
        )
        symbol_refs = set(trace.symbol_table_refs)
        for step in trace.operation_stream:
            symbol_refs.update(argument for argument in step.args if argument.startswith("sym_"))
            if step.output_ref.startswith("sym_"):
                symbol_refs.add(step.output_ref)
        runtime_subset = CompressionRuntimeSubset(
            task_id=research_handoff.plan.task_id,
            macros=(),
            opcodes=tuple(
                OpcodeEntry(opcode_name=name, description=f"{name} opcode")
                for name in critic_handoff_oversized.required_opcode_names
            ),
            decoders=tuple(
                DecoderEntry(decoder_name=name, template=f"{name}: {{value}}")
                for name in critic_handoff_oversized.required_decoder_names
            ),
            symbol_table=SymbolTableSnapshot(
                task_id=research_handoff.plan.task_id,
                symbols={ref: ref for ref in sorted(symbol_refs)},
                metadata={"scope": "task"},
            ),
        )

        _, checks_capped, _ = critic_service._run_deterministic_checks(
            critic_handoff_capped,
            runtime_subset,
        )
        _, checks_oversized, _ = critic_service._run_deterministic_checks(
            critic_handoff_oversized,
            runtime_subset,
        )

        self.assertEqual(tuple(checks_capped), tuple(checks_oversized))
        self.assertEqual(len(checks_oversized), len(set(checks_oversized)))
        self.assertIn("check.plan_budget_alignment", checks_oversized)
        self.assertIn("tool.python_ast_arithmetic", checks_oversized)


if __name__ == "__main__":
    unittest.main()
