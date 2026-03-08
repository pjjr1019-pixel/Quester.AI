"""Phase 7 boundary and thin-agent tests."""

from __future__ import annotations

import asyncio
import json
import shutil
from types import SimpleNamespace
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
from compression_service import CompressionService
from config import APP_CONFIG
from data_structures import (
    CompressedTrace,
    CompressionRuntimeSubset,
    CritiqueReport,
    CritiqueResult,
    DecoderEntry,
    EvidenceBundle,
    EvidenceItem,
    OpcodeEntry,
    OptimizerActivationDecision,
    OptimizerLifecycleStage,
    PerformanceMetric,
    Plan,
    PlanStep,
    ReasonerCriticHandoff,
    ReasoningLog,
    ResearchReasonerHandoff,
    ResourceBudget,
    SourceType,
    SymbolTableSnapshot,
    UserSettingsProfile,
)
from critique_service import CritiqueService
from dashboard import DashboardService
from model_manager import ModelHealthSnapshot
from orchestrator import Orchestrator
from planner_service import PlannerService
from reasoning_service import ReasoningService
from runtime_errors import ModelTimeoutError, ResourcePressureError
from self_optimizer import SelfOptimizer
from translation_service import TranslationService


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

    def health_snapshot(self):
        return SimpleNamespace(generation_backend="queued_generation")


class _FakeCritiqueStorage:
    def __init__(self, subset: CompressionRuntimeSubset) -> None:
        self.subset = subset

    async def load_active_compression_runtime(
        self,
        task_id: str,
        *,
        macro_names: tuple[str, ...],
        opcode_names: tuple[str, ...],
        decoder_names: tuple[str, ...],
    ) -> CompressionRuntimeSubset:
        _ = (task_id, macro_names, opcode_names, decoder_names)
        return self.subset


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

    async def test_pipeline_completed_event_surfaces_reasoning_and_critique_metadata(self) -> None:
        await self.orchestrator.run_task("What is 2 + 2?", thinking_minutes=30)

        completed_events = await self.orchestrator.storage.list_runtime_events(stage="pipeline.completed")

        self.assertEqual(len(completed_events), 1)
        payload = completed_events[0].payload
        self.assertIn("candidate_trace_count", payload)
        self.assertIn("citation_refs", payload)
        self.assertIn("failure_categories", payload)
        self.assertEqual(payload["critique_result"], "valid")

    def test_boundary_schema_helpers_lock_required_shapes(self) -> None:
        self.assertEqual(planner_output_schema()["title"], "planner_plan_v1")
        self.assertEqual(research_reasoner_handoff_schema()["title"], "research_reasoner_handoff_v1")
        self.assertEqual(reasoner_output_schema()["title"], "compressed_trace_v1")
        self.assertEqual(reasoner_critic_handoff_schema()["title"], "reasoner_critic_handoff_v1")
        self.assertEqual(critic_output_schema()["title"], "critique_report_v1")
        self.assertIn("candidate_traces", reasoner_output_schema()["properties"])
        self.assertIn("verifier_type", critic_output_schema()["properties"])
        self.assertIn("candidate_score", critic_output_schema()["properties"])
        self.assertIn("repair_actions", critic_output_schema()["properties"])
        self.assertIn("failure_categories", critic_output_schema()["properties"])
        self.assertIn("provenance_coverage", critic_output_schema()["properties"])
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

    async def test_planner_repairs_schema_invalid_object_once_before_accepting(self) -> None:
        manager = _QueuedGenerationManager(
            [
                json.dumps({"task_id": "bad-plan", "question": "ignored"}),
                self._payload(task_id="plan-schema-repaired"),
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

        plan = await service.plan("Schema repair planner question", budget)

        self.assertEqual(plan.task_id, "plan-schema-repaired")
        self.assertIn("planner_output_mode=structured_json", plan.planner_notes)
        self.assertIn("used_repair=yes", plan.planner_notes)
        self.assertEqual(len(manager.prompts), 2)


class ReasonerStructuredOutputTests(unittest.IsolatedAsyncioTestCase):
    """Verify the reasoner uses the structured JSON path with bounded fallback."""

    def _build_handoff(self) -> ResearchReasonerHandoff:
        budget = ResourceBudget(
            retrieval_top_k=6,
            max_web_queries=2,
            reasoner_passes=2,
            critic_passes=2,
            macro_depth=3,
        )
        plan = Plan(
            task_id="reasoner-task",
            question="What evidence supports the answer?",
            steps=(PlanStep(step_id="step_1", description="Inspect evidence"),),
            required_evidence=("local docs",),
            success_criteria=("return typed trace",),
            budget=budget,
        )
        evidence = EvidenceBundle(
            task_id=plan.task_id,
            local_results=(
                EvidenceItem(
                    id="ev-1",
                    content="First evidence item",
                    source_type=SourceType.LOCAL,
                    source_ref="local://ev-1",
                    score=0.8,
                ),
            ),
            web_results=(),
            used_web_fallback=False,
        )
        return ResearchReasonerHandoff.from_inputs(plan=plan, evidence=evidence, budget=budget)

    def _payload(self, *, task_id: str = "wrong-task") -> str:
        return json.dumps(
            {
                "task_id": task_id,
                "tokens": [
                    "@read_question",
                    "@extract_constraints",
                    "@match_local_evidence",
                    "@review_evidence_1",
                    "@reason_pass_1",
                    "@synthesize_pass_1",
                    "@refine_answer_1",
                    "@reason_pass_2",
                    "@synthesize_pass_2",
                    "@compose_answer",
                ],
                "expanded_preview": [
                    "Read question and constraints",
                    "Review 1 evidence item(s)",
                    "Reasoning pass 1 of 2",
                    "Reasoning pass 2 of 2",
                ],
                "confidence": 0.91,
                "ir_version": "1",
                "operation_stream": [
                    {
                        "op_id": "o0",
                        "opcode": "lookup",
                        "args": ["sym_question"],
                        "output_ref": "sym_evidence_set",
                        "context_frame_id": "cf0",
                        "evidence_handles": ["ev-1"],
                    },
                    {
                        "op_id": "o1",
                        "opcode": "bind",
                        "args": ["sym_question", "sym_evidence_1", "sym_answer"],
                        "output_ref": "sym_answer",
                        "context_frame_id": "cf0",
                        "evidence_handles": ["ev-1"],
                    },
                    {
                        "op_id": "o2",
                        "opcode": "emit",
                        "args": ["sym_answer"],
                        "context_frame_id": "cf0",
                        "evidence_handles": ["ev-1"],
                    },
                ],
                "decode_hints": [
                    {
                        "hint_id": "d0",
                        "template": "verified_answer",
                        "entity_ids": ["a"],
                    }
                ],
                "proof_hash": "stale-proof-hash",
            }
        )

    async def test_reasoner_uses_structured_json_when_available(self) -> None:
        manager = _QueuedGenerationManager([self._payload()])
        service = ReasoningService(model_manager=manager, storage=None, config=APP_CONFIG)
        handoff = self._build_handoff()

        trace = await service.reason(handoff)

        self.assertEqual(trace.task_id, handoff.plan.task_id)
        self.assertEqual(trace.tokens[-1], "@compose_answer")
        self.assertEqual(trace.confidence, 0.91)
        self.assertEqual(trace.evidence_handles, handoff.evidence_handles)
        self.assertEqual(trace.canonical_graph_builder, "reasoner_stub_v1")
        self.assertIn("reasoner_output_mode=structured_json", trace.reasoner_notes)
        self.assertIn("used_repair=no", trace.reasoner_notes)
        self.assertNotEqual(trace.proof_hash, "stale-proof-hash")
        self.assertEqual(len(manager.prompts), 1)

    async def test_reasoner_repairs_invalid_json_once_before_accepting(self) -> None:
        manager = _QueuedGenerationManager(["not valid json", self._payload(task_id="repaired-task")])
        service = ReasoningService(model_manager=manager, storage=None, config=APP_CONFIG)
        handoff = self._build_handoff()

        trace = await service.reason(handoff)

        self.assertEqual(trace.task_id, handoff.plan.task_id)
        self.assertIn("reasoner_output_mode=structured_json", trace.reasoner_notes)
        self.assertIn("used_repair=yes", trace.reasoner_notes)
        self.assertIn("used_fallback=no", trace.reasoner_notes)
        self.assertEqual(len(manager.prompts), 2)

    async def test_reasoner_falls_back_after_bounded_repair_failure(self) -> None:
        manager = _QueuedGenerationManager(["still not json", "repair also failed"])
        service = ReasoningService(model_manager=manager, storage=None, config=APP_CONFIG)
        handoff = self._build_handoff()

        trace = await service.reason(handoff)

        self.assertEqual(trace.task_id, handoff.plan.task_id)
        self.assertIn("reasoner_output_mode=deterministic_fallback", trace.reasoner_notes)
        self.assertIn("parse_error=", trace.reasoner_notes)
        self.assertEqual(len(manager.prompts), 2)

    async def test_reasoner_deep_mode_builds_candidate_selection_trace(self) -> None:
        manager = _QueuedGenerationManager(["still not json", "repair also failed"])
        service = ReasoningService(model_manager=manager, storage=None, config=APP_CONFIG)
        base_handoff = self._build_handoff()
        deep_budget = ResourceBudget(
            retrieval_top_k=8,
            max_web_queries=3,
            reasoner_passes=3,
            critic_passes=2,
            macro_depth=4,
        )
        deep_plan = replace(
            base_handoff.plan,
            question="What is 2 + 2?",
            budget=deep_budget,
        )
        handoff = replace(
            base_handoff,
            plan=deep_plan,
            budget=deep_budget,
            reasoning_mode="deep",
        )

        trace = await service.reason(handoff)

        self.assertIn("@compare_candidates", trace.tokens)
        self.assertIn("@select_verified_candidate", trace.tokens)
        self.assertEqual(trace.context_frames[0].metadata["rm"], "deep")
        self.assertGreaterEqual(trace.context_frames[0].metadata["cc"], 2)
        self.assertEqual(trace.operation_stream[-1].metadata["answer_text"], "4")
        self.assertEqual(trace.operation_stream[-1].metadata["selected_strategy"], "tool_arithmetic")

    async def test_reasoner_deep_mode_materializes_candidate_traces(self) -> None:
        manager = _QueuedGenerationManager(["still not json", "repair also failed"])
        service = ReasoningService(model_manager=manager, storage=None, config=APP_CONFIG)
        base_handoff = self._build_handoff()
        deep_budget = ResourceBudget(
            retrieval_top_k=8,
            max_web_queries=3,
            reasoner_passes=3,
            critic_passes=2,
            macro_depth=4,
        )
        deep_plan = replace(base_handoff.plan, question="What is 2 + 2?", budget=deep_budget)
        handoff = replace(base_handoff, plan=deep_plan, budget=deep_budget, reasoning_mode="deep")

        trace = await service.reason(handoff)

        self.assertGreaterEqual(len(trace.candidate_traces), 3)
        self.assertTrue(all(candidate.operation_stream for candidate in trace.candidate_traces))
        self.assertEqual(trace.candidate_traces[0].candidate_id, trace.context_frames[0].metadata["cid"])

    async def test_reasoner_budget_scales_deep_candidate_count(self) -> None:
        manager = _QueuedGenerationManager(["still not json", "repair also failed", "still not json", "repair also failed"])
        service = ReasoningService(model_manager=manager, storage=None, config=APP_CONFIG)
        base_handoff = self._build_handoff()
        medium_budget = ResourceBudget(
            retrieval_top_k=6,
            max_web_queries=2,
            reasoner_passes=2,
            critic_passes=2,
            macro_depth=3,
        )
        large_budget = ResourceBudget(
            retrieval_top_k=10,
            max_web_queries=5,
            reasoner_passes=4,
            critic_passes=3,
            macro_depth=4,
        )
        expanded_evidence = EvidenceBundle(
            task_id=base_handoff.plan.task_id,
            local_results=(
                *base_handoff.evidence.local_results,
                EvidenceItem(
                    id="ev-2",
                    content="Second evidence item with different wording",
                    source_type=SourceType.LOCAL,
                    source_ref="local://ev-2",
                    score=0.7,
                ),
            ),
            web_results=(),
            used_web_fallback=False,
        )
        medium_plan = replace(base_handoff.plan, question="What is 2 + 2?", budget=medium_budget)
        large_plan = replace(base_handoff.plan, question="What is 2 + 2?", budget=large_budget)

        medium_trace = await service.reason(
            replace(
                base_handoff,
                plan=medium_plan,
                evidence=expanded_evidence,
                budget=medium_budget,
                reasoning_mode="deep",
            )
        )
        large_trace = await service.reason(
            replace(
                base_handoff,
                plan=large_plan,
                evidence=expanded_evidence,
                budget=large_budget,
                reasoning_mode="deep",
            )
        )

        self.assertLess(len(medium_trace.candidate_traces), len(large_trace.candidate_traces))


class CriticStructuredOutputTests(unittest.IsolatedAsyncioTestCase):
    """Verify the critic uses the structured JSON path after deterministic checks pass."""

    async def _build_handoff(self) -> ReasonerCriticHandoff:
        handoff = ReasonerStructuredOutputTests()._build_handoff()
        manager = _QueuedGenerationManager(["not json", "still not json"])
        service = ReasoningService(model_manager=manager, storage=None, config=APP_CONFIG)
        trace = await service.reason(handoff)
        return ReasonerCriticHandoff.from_inputs(
            plan=handoff.plan,
            evidence=handoff.evidence,
            trace=trace,
            budget=handoff.budget,
        )

    def _build_storage(self, handoff: ReasonerCriticHandoff) -> _FakeCritiqueStorage:
        symbol_refs = set(handoff.trace.symbol_table_refs)
        symbol_refs.update(
            argument
            for step in handoff.trace.operation_stream
            for argument in step.args
            if argument.startswith("sym_")
        )
        symbol_refs.update(
            step.output_ref
            for step in handoff.trace.operation_stream
            if step.output_ref.startswith("sym_")
        )
        subset = CompressionRuntimeSubset(
            task_id=handoff.plan.task_id,
            macros=(),
            opcodes=tuple(
                OpcodeEntry(opcode_name=name, description=f"{name} opcode")
                for name in handoff.required_opcode_names
            ),
            decoders=tuple(
                DecoderEntry(decoder_name=name, template=f"{name}: {{value}}")
                for name in handoff.required_decoder_names
            ),
            symbol_table=SymbolTableSnapshot(
                task_id=handoff.plan.task_id,
                symbols={ref: ref for ref in symbol_refs},
                metadata={"scope": "task"},
            ),
            proof_hashes=(),
        )
        return _FakeCritiqueStorage(subset)

    def _payload(self, *, task_id: str = "wrong-task", result: str = "valid") -> str:
        return json.dumps(
            {
                "task_id": task_id,
                "is_valid": result == "valid",
                "issues": [],
                "evidence_coverage": 0.75,
                "critic_notes": "model-critic",
                "result": result,
            }
        )

    async def test_critic_uses_structured_json_when_checks_pass(self) -> None:
        handoff = await self._build_handoff()
        manager = _QueuedGenerationManager([self._payload()])
        service = CritiqueService(
            model_manager=manager,
            storage=self._build_storage(handoff),
            config=APP_CONFIG,
        )

        report = await service.review(handoff)

        self.assertEqual(report.task_id, handoff.plan.task_id)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.fixed_trace, handoff.trace)
        self.assertEqual(report.evidence_coverage, 0.75)
        self.assertEqual(report.result, CritiqueResult.VALID)
        self.assertIn("critic_output_mode=structured_json", report.critic_notes)
        self.assertIn("used_repair=no", report.critic_notes)
        self.assertEqual(len(manager.prompts), 1)

    async def test_critic_repairs_invalid_json_once_before_accepting(self) -> None:
        handoff = await self._build_handoff()
        manager = _QueuedGenerationManager(["not valid json", self._payload(task_id="repaired-task")])
        service = CritiqueService(
            model_manager=manager,
            storage=self._build_storage(handoff),
            config=APP_CONFIG,
        )

        report = await service.review(handoff)

        self.assertEqual(report.task_id, handoff.plan.task_id)
        self.assertIn("critic_output_mode=structured_json", report.critic_notes)
        self.assertIn("used_repair=yes", report.critic_notes)
        self.assertIn("used_fallback=no", report.critic_notes)
        self.assertEqual(len(manager.prompts), 2)

    async def test_critic_falls_back_after_bounded_repair_failure(self) -> None:
        handoff = await self._build_handoff()
        manager = _QueuedGenerationManager(["still not json", "repair also failed"])
        service = CritiqueService(
            model_manager=manager,
            storage=self._build_storage(handoff),
            config=APP_CONFIG,
        )

        report = await service.review(handoff)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.fixed_trace, handoff.trace)
        self.assertIn("critic_output_mode=deterministic_fallback", report.critic_notes)
        self.assertIn("parse_error=", report.critic_notes)
        self.assertEqual(len(manager.prompts), 2)

    async def test_critic_tool_verification_rejects_wrong_arithmetic_answer(self) -> None:
        base_handoff = ReasonerStructuredOutputTests()._build_handoff()
        deep_budget = ResourceBudget(
            retrieval_top_k=8,
            max_web_queries=3,
            reasoner_passes=3,
            critic_passes=2,
            macro_depth=4,
        )
        deep_plan = replace(
            base_handoff.plan,
            question="What is 2 + 2?",
            budget=deep_budget,
        )
        reasoner_handoff = replace(
            base_handoff,
            plan=deep_plan,
            budget=deep_budget,
            reasoning_mode="deep",
        )
        reasoner_manager = _QueuedGenerationManager(["still not json", "repair also failed"])
        reasoner_service = ReasoningService(model_manager=reasoner_manager, storage=None, config=APP_CONFIG)
        trace = await reasoner_service.reason(reasoner_handoff)
        emit_step = replace(trace.operation_stream[-1], metadata={**trace.operation_stream[-1].metadata, "answer_text": "5"})
        decode_hint = replace(trace.decode_hints[0], metadata={**trace.decode_hints[0].metadata, "answer_text": "5"})
        wrong_trace = replace(
            trace,
            operation_stream=tuple((*trace.operation_stream[:-1], emit_step)),
            decode_hints=(decode_hint,),
        )
        handoff = ReasonerCriticHandoff.from_inputs(
            plan=deep_plan,
            evidence=reasoner_handoff.evidence,
            trace=wrong_trace,
            budget=deep_budget,
        )
        manager = _QueuedGenerationManager([self._payload()])
        service = CritiqueService(
            model_manager=manager,
            storage=self._build_storage(handoff),
            config=APP_CONFIG,
        )

        report = await service.review(handoff)

        self.assertFalse(report.is_valid)
        self.assertIn("Arithmetic tool verification failed", report.issues[0])
        self.assertEqual(len(manager.prompts), 0)

    async def test_critic_records_python_expression_verifier_metadata(self) -> None:
        base_handoff = ReasonerStructuredOutputTests()._build_handoff()
        deep_budget = ResourceBudget(
            retrieval_top_k=8,
            max_web_queries=3,
            reasoner_passes=3,
            critic_passes=2,
            macro_depth=4,
        )
        deep_plan = replace(
            base_handoff.plan,
            question="What does `sum([1, 2, 3])` return?",
            budget=deep_budget,
        )
        reasoner_handoff = replace(
            base_handoff,
            plan=deep_plan,
            budget=deep_budget,
            reasoning_mode="deep",
        )
        reasoner_manager = _QueuedGenerationManager(["still not json", "repair also failed"])
        reasoner_service = ReasoningService(model_manager=reasoner_manager, storage=None, config=APP_CONFIG)
        trace = await reasoner_service.reason(reasoner_handoff)
        handoff = ReasonerCriticHandoff.from_inputs(
            plan=deep_plan,
            evidence=reasoner_handoff.evidence,
            trace=trace,
            budget=deep_budget,
        )
        manager = _QueuedGenerationManager([self._payload()])
        service = CritiqueService(
            model_manager=manager,
            storage=self._build_storage(handoff),
            config=APP_CONFIG,
        )

        report = await service.review(handoff)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.verifier_type, "tool.python_expression")
        self.assertEqual(report.candidate_score, 1.0)

    async def test_critic_records_python_code_execution_verifier_metadata(self) -> None:
        base_handoff = ReasonerStructuredOutputTests()._build_handoff()
        deep_budget = ResourceBudget(
            retrieval_top_k=8,
            max_web_queries=3,
            reasoner_passes=3,
            critic_passes=2,
            macro_depth=4,
        )
        question = (
            "Given this Python code:\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```\n"
            "What does `add(2, 3)` return?"
        )
        deep_plan = replace(base_handoff.plan, question=question, budget=deep_budget)
        reasoner_handoff = replace(
            base_handoff,
            plan=deep_plan,
            budget=deep_budget,
            reasoning_mode="deep",
        )
        reasoner_manager = _QueuedGenerationManager(["still not json", "repair also failed"])
        reasoner_service = ReasoningService(model_manager=reasoner_manager, storage=None, config=APP_CONFIG)
        trace = await reasoner_service.reason(reasoner_handoff)
        handoff = ReasonerCriticHandoff.from_inputs(
            plan=deep_plan,
            evidence=reasoner_handoff.evidence,
            trace=trace,
            budget=deep_budget,
        )
        manager = _QueuedGenerationManager([self._payload()])
        service = CritiqueService(model_manager=manager, storage=self._build_storage(handoff), config=APP_CONFIG)

        report = await service.review(handoff)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.verifier_type, "tool.python_code_execution")
        self.assertEqual(report.candidate_score, 1.0)

    async def test_critic_records_python_unit_test_verifier_metadata(self) -> None:
        base_handoff = ReasonerStructuredOutputTests()._build_handoff()
        deep_budget = ResourceBudget(
            retrieval_top_k=8,
            max_web_queries=3,
            reasoner_passes=3,
            critic_passes=2,
            macro_depth=4,
        )
        question = (
            "Will this unit test pass?\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n\n"
            "def test_add():\n"
            "    assert add(2, 2) == 4\n"
            "```"
        )
        deep_plan = replace(base_handoff.plan, question=question, budget=deep_budget)
        reasoner_handoff = replace(
            base_handoff,
            plan=deep_plan,
            budget=deep_budget,
            reasoning_mode="deep",
        )
        reasoner_manager = _QueuedGenerationManager(["still not json", "repair also failed"])
        reasoner_service = ReasoningService(model_manager=reasoner_manager, storage=None, config=APP_CONFIG)
        trace = await reasoner_service.reason(reasoner_handoff)
        handoff = ReasonerCriticHandoff.from_inputs(
            plan=deep_plan,
            evidence=reasoner_handoff.evidence,
            trace=trace,
            budget=deep_budget,
        )
        manager = _QueuedGenerationManager([self._payload()])
        service = CritiqueService(model_manager=manager, storage=self._build_storage(handoff), config=APP_CONFIG)

        report = await service.review(handoff)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.verifier_type, "tool.python_unit_test")
        self.assertEqual(report.candidate_score, 1.0)

    async def test_critic_marks_weakly_grounded_answer_as_degraded(self) -> None:
        handoff = await self._build_handoff()
        emit_step = replace(
            handoff.trace.operation_stream[-1],
            metadata={**handoff.trace.operation_stream[-1].metadata, "answer_text": "unsupported hallucination"},
        )
        decode_hint = replace(
            handoff.trace.decode_hints[0],
            metadata={**handoff.trace.decode_hints[0].metadata, "answer_text": "unsupported hallucination"},
        )
        weak_trace = replace(
            handoff.trace,
            operation_stream=tuple((*handoff.trace.operation_stream[:-1], emit_step)),
            decode_hints=(decode_hint,),
        )
        degraded_handoff = ReasonerCriticHandoff.from_inputs(
            plan=handoff.plan,
            evidence=handoff.evidence,
            trace=weak_trace,
            budget=handoff.budget,
        )
        manager = _QueuedGenerationManager([self._payload()])
        service = CritiqueService(
            model_manager=manager,
            storage=self._build_storage(degraded_handoff),
            config=APP_CONFIG,
        )

        report = await service.review(degraded_handoff)

        self.assertFalse(report.is_valid)
        self.assertEqual(report.result, CritiqueResult.DEGRADED)
        self.assertEqual(report.degraded_reason, "low_evidence_support")
        self.assertIn("abstain_due_to_low_grounding", report.repair_actions)
        self.assertEqual(len(manager.prompts), 0)


class CompressorStructuredOutputTests(unittest.IsolatedAsyncioTestCase):
    """Verify the compressor uses the structured JSON path with array payloads."""

    async def _build_trace(self) -> CompressedTrace:
        handoff = ReasonerStructuredOutputTests()._build_handoff()
        manager = _QueuedGenerationManager(["still not json", "repair also failed"])
        service = ReasoningService(model_manager=manager, storage=None, config=APP_CONFIG)
        return await service.reason(handoff)

    def _payload(self) -> str:
        return json.dumps(
            [
                {
                    "proposal_id": "proposal-structured",
                    "macro": {
                        "macro_name": "motif_structured",
                        "expansion": ["Read question and constraints", "Review 1 evidence item(s)"],
                        "version": 1,
                        "opcode_pattern": ["lookup", "bind"],
                        "invariants": [
                            "deterministic_round_trip",
                            "provenance_preserving",
                            "uncertainty_preserving",
                        ],
                        "semantic_kind": "motif_macro",
                    },
                    "reason": "Structured compressor proposal",
                    "examples": ["Read question and constraints | Review 1 evidence item(s)"],
                    "simulation_score": 0.8,
                    "approved": False,
                }
            ]
        )

    async def test_compressor_uses_structured_json_when_available(self) -> None:
        trace = await self._build_trace()
        manager = _QueuedGenerationManager([self._payload()])
        service = CompressionService(model_manager=manager, config=APP_CONFIG)

        proposals = await service.propose(trace, logs=[])

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].proposal_id, "proposal-structured")
        self.assertEqual(proposals[0].macro.macro_name, "motif_structured")
        self.assertEqual(len(manager.prompts), 1)

    async def test_compressor_fallback_surfaces_graph_or_subproof_macros(self) -> None:
        base_handoff = ReasonerStructuredOutputTests()._build_handoff()
        deep_budget = ResourceBudget(
            retrieval_top_k=8,
            max_web_queries=3,
            reasoner_passes=3,
            critic_passes=2,
            macro_depth=4,
        )
        deep_plan = replace(base_handoff.plan, question="What is 2 + 2?", budget=deep_budget)
        reasoner_handoff = replace(
            base_handoff,
            plan=deep_plan,
            budget=deep_budget,
            reasoning_mode="deep",
        )
        reasoner_manager = _QueuedGenerationManager(["still not json", "repair also failed"])
        reasoner_service = ReasoningService(model_manager=reasoner_manager, storage=None, config=APP_CONFIG)
        trace = await reasoner_service.reason(reasoner_handoff)
        manager = _QueuedGenerationManager(["not json", "still not json"])
        service = CompressionService(model_manager=manager, config=APP_CONFIG)

        proposals = await service.propose(trace, logs=[])

        self.assertTrue(
            any(
                proposal.macro.semantic_kind
                in {"candidate_subproof_macro", "graph_path_macro", "symbol_bundle_macro"}
                for proposal in proposals
            )
        )


class CritiqueReportContractTests(unittest.TestCase):
    """Verify additive critique-report fields remain round-trippable."""

    def test_critique_report_round_trips_additive_fields(self) -> None:
        report = CritiqueReport(
            task_id="critique-roundtrip",
            is_valid=False,
            issues=("weak grounding",),
            fixed_trace=None,
            evidence_coverage=0.4,
            critic_notes="notes",
            result=CritiqueResult.DEGRADED,
            verifier_type="tool.evidence_grounding",
            proof_hash_match=False,
            candidate_score=0.4,
            repair_actions=("abstain_due_to_low_grounding",),
            degraded_reason="low_evidence_support",
            failure_categories=("evidence_coverage",),
            provenance_coverage=0.4,
            macro_violations=(),
            drift_score=0.21,
        )

        self.assertEqual(CritiqueReport.from_dict(report.to_dict()), report)


class TranslationServiceTests(unittest.IsolatedAsyncioTestCase):
    """Verify final answer translation renders verified and degraded states."""

    async def test_translation_service_renders_verified_answer(self) -> None:
        base_handoff = ReasonerStructuredOutputTests()._build_handoff()
        deep_budget = ResourceBudget(
            retrieval_top_k=8,
            max_web_queries=3,
            reasoner_passes=3,
            critic_passes=2,
            macro_depth=4,
        )
        deep_plan = replace(base_handoff.plan, question="What is 2 + 2?", budget=deep_budget)
        handoff = replace(base_handoff, plan=deep_plan, budget=deep_budget, reasoning_mode="deep")
        manager = _QueuedGenerationManager(["still not json", "repair also failed"])
        service = ReasoningService(model_manager=manager, storage=None, config=APP_CONFIG)
        trace = await service.reason(handoff)
        translator = TranslationService()
        critique = CritiqueReport(
            task_id=deep_plan.task_id,
            is_valid=True,
            issues=(),
            fixed_trace=trace,
            evidence_coverage=1.0,
            result=CritiqueResult.VALID,
            verifier_type="tool.python_ast_arithmetic",
            candidate_score=1.0,
            repair_actions=("preserve_trace",),
        )

        answer_text = translator.render_answer(
            evidence=handoff.evidence,
            reasoning=trace,
            critique=critique,
        )

        self.assertIn("Verified answer: 4.", answer_text)
        self.assertIn("Verification: tool.python_ast_arithmetic.", answer_text)

    async def test_translation_service_renders_degraded_answer(self) -> None:
        handoff = ReasonerStructuredOutputTests()._build_handoff()
        manager = _QueuedGenerationManager(["still not json", "repair also failed"])
        service = ReasoningService(model_manager=manager, storage=None, config=APP_CONFIG)
        trace = await service.reason(handoff)
        emit_step = replace(
            trace.operation_stream[-1],
            metadata={**trace.operation_stream[-1].metadata, "answer_text": "unsupported hallucination"},
        )
        decode_hint = replace(
            trace.decode_hints[0],
            metadata={**trace.decode_hints[0].metadata, "answer_text": "unsupported hallucination"},
        )
        weak_trace = replace(
            trace,
            operation_stream=tuple((*trace.operation_stream[:-1], emit_step)),
            decode_hints=(decode_hint,),
        )
        translator = TranslationService()
        critique = CritiqueReport(
            task_id=handoff.plan.task_id,
            is_valid=False,
            issues=("weak grounding",),
            fixed_trace=None,
            evidence_coverage=0.2,
            result=CritiqueResult.DEGRADED,
            verifier_type="tool.evidence_grounding",
            candidate_score=0.1,
            repair_actions=("abstain_due_to_low_grounding",),
            degraded_reason="low_evidence_support",
        )

        answer_text = translator.render_answer(
            evidence=handoff.evidence,
            reasoning=weak_trace,
            critique=critique,
        )

        self.assertIn("Degraded answer: unsupported hallucination.", answer_text)
        self.assertIn("Reason: low_evidence_support.", answer_text)

    async def test_translation_service_renders_source_refs_for_grounded_answer(self) -> None:
        handoff = ReasonerStructuredOutputTests()._build_handoff()
        manager = _QueuedGenerationManager(["still not json", "repair also failed"])
        service = ReasoningService(model_manager=manager, storage=None, config=APP_CONFIG)
        trace = await service.reason(handoff)
        translator = TranslationService()
        critique = CritiqueReport(
            task_id=handoff.plan.task_id,
            is_valid=True,
            issues=(),
            fixed_trace=trace,
            evidence_coverage=1.0,
            result=CritiqueResult.VALID,
            verifier_type="tool.evidence_grounding",
            candidate_score=0.9,
            repair_actions=("preserve_trace",),
            provenance_coverage=1.0,
        )

        answer_text = translator.render_answer(
            evidence=handoff.evidence,
            reasoning=trace,
            critique=critique,
        )

        self.assertIn("Sources: local://ev-1.", answer_text)


class DashboardBackpressureTests(unittest.TestCase):
    """Verify dropped dashboard events remain visible to consumers."""

    def test_dashboard_marks_overflow_events(self) -> None:
        config = replace(
            APP_CONFIG,
            dashboard=replace(APP_CONFIG.dashboard, enable_ui=False),
            concurrency=replace(APP_CONFIG.concurrency, dashboard_queue_maxsize=1),
        )
        dashboard = DashboardService(config=config)

        dashboard.publish_event({"stage": "pipeline.started"})
        dashboard.publish_event({"stage": "pipeline.completed"})
        event = dashboard._events.get_nowait()

        self.assertEqual(event["stage"], "pipeline.completed")
        self.assertEqual(event["dropped_events"], 1)
        self.assertEqual(event["queue_overflow"], "evicted_oldest")


class DashboardAppStateTests(unittest.TestCase):
    """Verify the dashboard projects a typed app-state from runtime events."""

    def test_dashboard_app_state_tracks_health_task_and_conditions(self) -> None:
        config = replace(
            APP_CONFIG,
            dashboard=replace(APP_CONFIG.dashboard, enable_ui=False),
        )
        dashboard = DashboardService(config=config)
        dashboard.apply_user_settings(
            UserSettingsProfile(
                profile_name="quiet",
                reasoning={"thinking_minutes": 60, "mode": "deep"},
                ui={"show_debug_pane": False, "app_shell": "tkinter"},
            )
        )

        dashboard.publish_event(
            {
                "stage": "runtime.health_snapshot",
                "generation_backend": "stub_generation",
                "embedding_backend": "stub_embedding",
                "active_generation_jobs": 0,
                "active_embedding_jobs": 0,
                "fallback_active": False,
                "telemetry_enabled": False,
            }
        )
        dashboard.publish_event(
            {
                "stage": "pipeline.received",
                "question": "What is 2 + 2?",
                "thinking_minutes": 30,
            }
        )
        dashboard.publish_event(
            {
                "stage": "status.updated",
                "component": "planner",
                "state": "running",
                "task_id": "task-1",
                "severity": "low",
                "message": "planner started",
            }
        )
        dashboard.publish_event(
            {
                "stage": "runtime.degraded",
                "category": "degraded",
                "component": "researcher",
                "reason": "web_fallback_returned_no_results",
                "severity": "medium",
                "task_id": "task-1",
                "metadata": {"web_result_count": 0},
            }
        )
        dashboard.publish_event(
            {
                "stage": "pipeline.completed",
                "task_id": "task-1",
                "answer_text": "Verified answer: 4.",
                "citation_refs": ["local://ev-1"],
                "warning_count": 1,
                "candidate_trace_count": 1,
                "critique_result": "valid",
            }
        )

        state = dashboard.app_state_snapshot()

        self.assertEqual(state.user_settings.profile_name, "quiet")
        self.assertEqual(state.runtime_health.generation_backend, "stub_generation")
        self.assertEqual(state.active_task.question, "What is 2 + 2?")
        self.assertEqual(state.active_task.answer_text, "Verified answer: 4.")
        self.assertEqual(state.active_task.citation_refs, ("local://ev-1",))
        self.assertEqual(state.statuses["planner"].message, "planner started")
        self.assertEqual(state.recent_conditions[0].reason, "web_fallback_returned_no_results")

    def test_dashboard_controller_callbacks_receive_requests(self) -> None:
        config = replace(
            APP_CONFIG,
            dashboard=replace(APP_CONFIG.dashboard, enable_ui=False),
        )
        dashboard = DashboardService(config=config)
        submitted: list[tuple[str, int]] = []
        saved_profiles: list[UserSettingsProfile] = []
        dashboard.attach_controller(
            submit_task=lambda question, minutes: submitted.append((question, minutes)),
            save_settings=lambda profile: saved_profiles.append(profile),
        )

        save_result = dashboard.request_settings_save(
            UserSettingsProfile(
                profile_name="local_ui",
                reasoning={"thinking_minutes": 45, "mode": "deep"},
                ui={"show_debug_pane": False, "app_shell": "tkinter"},
            )
        )
        run_result = dashboard.request_task_submission("What is 2 + 2?", 45)

        self.assertTrue(save_result)
        self.assertTrue(run_result)
        self.assertEqual(submitted, [("What is 2 + 2?", 45)])
        self.assertEqual(saved_profiles[0].profile_name, "local_ui")
        self.assertEqual(dashboard.app_state_snapshot().user_settings.profile_name, "local_ui")

    def test_dashboard_tracks_history_knowledge_and_readiness_in_headless_mode(self) -> None:
        config = replace(
            APP_CONFIG,
            dashboard=replace(APP_CONFIG.dashboard, enable_ui=False),
        )
        dashboard = DashboardService(config=config)

        dashboard.publish_event(
            {
                "stage": "dashboard.settings_profiles_loaded",
                "profiles": [
                    UserSettingsProfile(profile_name="default").to_dict(),
                    UserSettingsProfile(profile_name="deep", reasoning={"thinking_minutes": 90, "mode": "deep"}).to_dict(),
                ],
            }
        )
        dashboard.publish_event(
            {
                "stage": "dashboard.task_history_loaded",
                "history": [
                    {
                        "task_id": "task-history-1",
                        "question": "What is 2 + 2?",
                        "answer_preview": "Verified answer: 4.",
                        "critique_result": "valid",
                        "warning_count": 0,
                        "candidate_trace_count": 1,
                    }
                ],
            }
        )
        dashboard.publish_event(
            {
                "stage": "dashboard.task_detail_loaded",
                "task": {
                    "task_id": "task-history-1",
                    "question": "What is 2 + 2?",
                    "answer_text": "Verified answer: 4.",
                    "citation_refs": ["local://sample"],
                    "optimizer_lifecycle": ["proposal:p1:proposed:0.75"],
                },
            }
        )
        dashboard.publish_event(
            {
                "stage": "dashboard.knowledge_library_loaded",
                "sources": [
                    {
                        "document_id": "doc-1",
                        "source_ref": "local://knowledge/doc1",
                        "title": "Knowledge Doc",
                        "chunk_count": 2,
                        "embedding_model": "stub-embed",
                        "archived": False,
                    }
                ],
            }
        )
        dashboard.publish_event(
            {
                "stage": "dashboard.readiness_loaded",
                "report": {
                    "stub_mode_ready": True,
                    "real_mode_ready": False,
                    "checks": [
                        {
                            "check_id": "stub_mode",
                            "title": "Stub Mode",
                            "status": "ready",
                            "detail": "Stub mode is ready.",
                        }
                    ],
                    "capabilities": [
                        {
                            "capability_name": "desktop_control",
                            "status": "blocked_by_policy",
                            "reason": "phase_20_21_not_implemented",
                        }
                    ],
                    "guidance": ["Keep heavy capabilities disabled by default."],
                },
            }
        )

        state = dashboard.app_state_snapshot()

        self.assertEqual(tuple(profile.profile_name for profile in state.settings_profiles), ("default", "deep"))
        self.assertEqual(state.task_history[0].task_id, "task-history-1")
        self.assertEqual(state.selected_task.citation_refs, ("local://sample",))
        self.assertEqual(state.knowledge_sources[0].source_ref, "local://knowledge/doc1")
        self.assertTrue(state.readiness_report.stub_mode_ready)
        self.assertEqual(state.readiness_report.capabilities[0].capability_name, "desktop_control")


class OrchestratorRepairLoopTests(unittest.IsolatedAsyncioTestCase):
    """Verify the orchestrator applies bounded repair actions before finalizing results."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase7_repair.sqlite3")
        self.test_logs = Path("test_phase7_repair_logs")
        preflight = replace(
            APP_CONFIG.preflight,
            flags=replace(
                APP_CONFIG.preflight.flags,
                stub_mode=True,
                enable_self_optimizer=False,
            ),
        )
        storage_cfg = replace(APP_CONFIG.storage, sqlite_path=self.test_db, logs_dir=self.test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        self.config = replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    def _mutate_trace_answer(
        self,
        trace: CompressedTrace,
        *,
        answer_text: str,
        selected_strategy: str,
        selected_verifier: str,
        verified: bool,
        degraded_reason: str,
    ) -> CompressedTrace:
        payload = trace.to_dict()
        raw_steps = [dict(step) for step in payload.get("operation_stream", [])]
        for step in raw_steps:
            metadata = dict(step.get("metadata", {}))
            if step.get("opcode") == "bind":
                metadata["selected_strategy"] = selected_strategy
                metadata["selected_verifier"] = selected_verifier
                metadata["verified"] = verified
            if step.get("opcode") == "emit":
                metadata["answer_text"] = answer_text
                metadata["selected_strategy"] = selected_strategy
                metadata["selected_verifier"] = selected_verifier
                metadata["verified"] = verified
                metadata["degraded_reason"] = degraded_reason
            step["metadata"] = metadata
        payload["operation_stream"] = raw_steps
        payload["decode_hints"] = [
            {
                "hint_id": "d0",
                "template": "verified_answer",
                "entity_ids": ["a"],
                "metadata": {
                    "answer_text": answer_text,
                    "selected_strategy": selected_strategy,
                    "selected_verifier": selected_verifier,
                    "verified": verified,
                    "degraded_reason": degraded_reason,
                    "candidate_count": 1,
                },
            }
        ]
        context_frames = [dict(frame) for frame in payload.get("context_frames", [])]
        if context_frames:
            context_metadata = dict(context_frames[0].get("metadata", {}))
            context_metadata["ta"] = answer_text
            context_metadata["sa"] = selected_strategy
            context_metadata["sv"] = selected_verifier
            context_metadata["vv"] = verified
            context_metadata["dr"] = degraded_reason
            context_frames[0]["metadata"] = context_metadata
            payload["context_frames"] = context_frames
        return CompressedTrace.from_dict(payload)

    async def test_repair_loop_replaces_wrong_tool_answer(self) -> None:
        original_reason = self.orchestrator.reasoner.reason_from_handoff
        attempts = 0

        async def wrong_once(handoff):
            nonlocal attempts
            attempts += 1
            trace = await original_reason(handoff)
            if attempts == 1:
                return self._mutate_trace_answer(
                    trace,
                    answer_text="5",
                    selected_strategy="tool_arithmetic",
                    selected_verifier="tool.python_ast_arithmetic",
                    verified=False,
                    degraded_reason="",
                )
            return trace

        self.orchestrator.reasoner.reason_from_handoff = wrong_once

        result = await self.orchestrator.run_task("What is 2 + 2?", thinking_minutes=30)

        self.assertEqual(attempts, 1)
        self.assertTrue(result.critique.is_valid)
        self.assertIn("Verified answer: 4.", result.answer_text)
        self.assertIn("repair_applied:replace_answer_with_tool_result", result.warnings)

    async def test_repair_loop_abstains_when_grounding_is_weak(self) -> None:
        original_reason = self.orchestrator.reasoner.reason_from_handoff

        async def hallucinate_once(handoff):
            trace = await original_reason(handoff)
            return self._mutate_trace_answer(
                trace,
                answer_text="unsupported hallucination",
                selected_strategy="top_evidence",
                selected_verifier="tool.evidence_grounding",
                verified=False,
                degraded_reason="low_evidence_support",
            )

        self.orchestrator.reasoner.reason_from_handoff = hallucinate_once

        result = await self.orchestrator.run_task(
            "What evidence supports the answer?",
            thinking_minutes=30,
        )

        self.assertEqual(result.critique.result, CritiqueResult.DEGRADED)
        self.assertIn("Degraded answer: Insufficient evidence to produce a verified answer.", result.answer_text)
        self.assertIn("repair_applied:abstain_due_to_low_grounding", result.warnings)
        self.assertIn("critique_degraded", result.warnings)

        degraded_events = await self.orchestrator.storage.list_runtime_events(stage="runtime.degraded")

        self.assertTrue(degraded_events)
        self.assertTrue(
            any(
                event.payload["component"] == "critic"
                and event.payload["reason"] == "low_evidence_support"
                for event in degraded_events
            )
        )


class OrchestratorRetryTests(unittest.IsolatedAsyncioTestCase):
    """Verify transient component failures are retried once by the orchestrator."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase7_retry.sqlite3")
        self.test_logs = Path("test_phase7_retry_logs")
        preflight = replace(
            APP_CONFIG.preflight,
            flags=replace(
                APP_CONFIG.preflight.flags,
                stub_mode=True,
                enable_self_optimizer=False,
                max_component_retries=1,
                retry_backoff_s=0.0,
            ),
        )
        storage_cfg = replace(APP_CONFIG.storage, sqlite_path=self.test_db, logs_dir=self.test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        self.config = replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_run_component_retries_transient_failure_once(self) -> None:
        attempts = 0

        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ModelTimeoutError("first call timed out")
            return "ok"

        result = await self.orchestrator._run_component(
            "test_component",
            task_id="task-1",
            start_stage="pipeline.test_component_started",
            done_stage="pipeline.test_component_done",
            start_payload={"task_id": "task-1"},
            run=flaky,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 2)

    async def test_run_task_cancellation_surfaces_event_and_clears_foreground_count(self) -> None:
        started = asyncio.Event()

        async def slow_plan(question: str, budget) -> Plan:
            _ = (question, budget)
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        self.orchestrator.planner.plan = slow_plan

        task = asyncio.create_task(
            self.orchestrator.run_task("Cancel this task", thinking_minutes=30),
            name="phase9-cancel-test",
        )
        await started.wait()
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        cancelled_events = await self.orchestrator.storage.list_runtime_events(stage="pipeline.cancelled")
        foreground_count = await self.orchestrator.storage.get_foreground_task_count()

        self.assertEqual(len(cancelled_events), 1)
        self.assertEqual(cancelled_events[0].payload["question"], "Cancel this task")
        self.assertEqual(foreground_count, 0)

    async def test_run_component_surfaces_fallback_event_when_model_state_changes(self) -> None:
        snapshots = iter(
            (
                ModelHealthSnapshot(
                    started=True,
                    generation_backend="primary_gen",
                    embedding_backend="primary_embed",
                    active_generation_jobs=0,
                    active_embedding_jobs=0,
                    last_used_at=None,
                    fallback_active=False,
                    fallback_reason=None,
                    available_ram_gb=None,
                    total_ram_gb=None,
                    generation_backend_vram_gb=None,
                    embedding_backend_vram_gb=None,
                    telemetry_enabled=False,
                    last_error=None,
                ),
                ModelHealthSnapshot(
                    started=True,
                    generation_backend="fallback_gen",
                    embedding_backend="primary_embed",
                    active_generation_jobs=0,
                    active_embedding_jobs=0,
                    last_used_at=None,
                    fallback_active=True,
                    fallback_reason="generation fallback activated: low memory",
                    available_ram_gb=0.9,
                    total_ram_gb=8.0,
                    generation_backend_vram_gb=5.8,
                    embedding_backend_vram_gb=0.1,
                    telemetry_enabled=False,
                    last_error="low memory",
                ),
            )
        )
        self.orchestrator.model_manager.health_snapshot = lambda: next(snapshots)

        async def succeed() -> str:
            return "ok"

        result = await self.orchestrator._run_component(
            "reasoner",
            task_id="task-fallback",
            start_stage="pipeline.reasoner_started",
            done_stage="pipeline.reasoner_done",
            start_payload={"task_id": "task-fallback"},
            run=succeed,
        )

        fallback_events = await self.orchestrator.storage.list_runtime_events(stage="runtime.fallback_activated")

        self.assertEqual(result, "ok")
        self.assertEqual(len(fallback_events), 1)
        self.assertEqual(fallback_events[0].payload["component"], "model_manager")
        self.assertEqual(
            fallback_events[0].payload["reason"],
            "generation fallback activated: low memory",
        )

    async def test_run_component_surfaces_resource_pressure_event(self) -> None:
        async def pressure() -> str:
            raise ResourcePressureError("available RAM 0.90GB is below headroom 1.00GB")

        with self.assertRaises(ResourcePressureError):
            await self.orchestrator._run_component(
                "reasoner",
                task_id="task-pressure",
                start_stage="pipeline.reasoner_started",
                done_stage="pipeline.reasoner_done",
                start_payload={"task_id": "task-pressure"},
                run=pressure,
            )

        pressure_events = await self.orchestrator.storage.list_runtime_events(
            stage="runtime.resource_pressure_detected"
        )

        self.assertEqual(len(pressure_events), 1)
        self.assertEqual(pressure_events[0].payload["component"], "model_manager")
        self.assertIn("available RAM 0.90GB", pressure_events[0].payload["reason"])

    async def test_dashboard_receives_service_level_web_lookup_events_via_storage(self) -> None:
        await self.orchestrator.run_task(
            "What is the latest runtime status today?",
            thinking_minutes=30,
        )

        state = self.orchestrator.dashboard.app_state_snapshot()

        self.assertEqual(state.active_task.web_query, "What is the latest runtime status today?")
        self.assertTrue(state.active_task.used_web_fallback)
        self.assertTrue(state.active_task.web_source_refs)

    async def test_dashboard_actions_populate_phase10_shell_surfaces(self) -> None:
        await self.orchestrator.storage.save_user_settings_profile(
            UserSettingsProfile(
                profile_name="deep_shell",
                reasoning={"thinking_minutes": 90, "mode": "deep"},
            )
        )
        await self.orchestrator.storage.ingest_document(
            source_ref="local://phase10/doc",
            title="Phase 10 Doc",
            content="Phase 10 knowledge content used by the local app shell test.",
            metadata={"corpus_origin": "user_local", "corpus_tier": "user", "archived": False},
            embed_document=self.orchestrator.model_manager.embed_document,
            embedding_model_name="stub-embed",
        )
        await self.orchestrator.run_task("What is 2 + 2?", thinking_minutes=30)

        await self.orchestrator._run_dashboard_action(
            action="settings.refresh_profiles",
            payload={"active_profile_name": "default"},
        )
        await self.orchestrator._run_dashboard_action(action="knowledge.refresh", payload={})
        await self.orchestrator._run_dashboard_action(action="history.refresh", payload={})
        await self.orchestrator._run_dashboard_action(action="readiness.refresh", payload={})

        state = self.orchestrator.dashboard.app_state_snapshot()

        self.assertIn("deep_shell", tuple(profile.profile_name for profile in state.settings_profiles))
        self.assertTrue(state.task_history)
        self.assertEqual(state.selected_task.task_id, state.task_history[0].task_id)
        self.assertTrue(state.knowledge_sources)
        self.assertEqual(state.knowledge_sources[0].source_ref, "local://phase10/doc")
        self.assertTrue(state.readiness_report.checks)
        self.assertTrue(state.readiness_report.capabilities)


class SelfOptimizerCycleTests(unittest.IsolatedAsyncioTestCase):
    """Verify the self-optimizer uses persisted traces instead of a fixed stub proposal."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase7_optimizer.sqlite3")
        self.test_logs = Path("test_phase7_optimizer_logs")
        preflight = replace(
            APP_CONFIG.preflight,
            flags=replace(
                APP_CONFIG.preflight.flags,
                stub_mode=True,
                enable_self_optimizer=False,
            ),
        )
        storage_cfg = replace(APP_CONFIG.storage, sqlite_path=self.test_db, logs_dir=self.test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        self.config = replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_run_cycle_builds_trace_driven_proposals(self) -> None:
        await self.orchestrator.run_task("What is 2 + 2?", thinking_minutes=30)
        optimizer = SelfOptimizer(storage=self.orchestrator.storage, config=self.config)

        proposals = await optimizer.run_cycle()
        replay_samples = await self.orchestrator.storage.list_optimizer_replay_samples()
        replay_evaluations = await self.orchestrator.storage.list_optimizer_replay_evaluations()
        proposal_records = await self.orchestrator.storage.list_optimizer_proposal_records()
        activation_records = await self.orchestrator.storage.list_optimizer_activation_records()
        rollback_records = await self.orchestrator.storage.list_optimizer_rollback_records()

        self.assertTrue(proposals)
        self.assertTrue(replay_samples)
        self.assertTrue(replay_evaluations)
        self.assertTrue(proposal_records)
        self.assertTrue(activation_records)
        self.assertTrue(all(not proposal.approved for proposal in proposals))
        self.assertTrue(all(proposal.reason for proposal in proposals))
        self.assertTrue(all(0.0 <= proposal.simulation_score <= 1.0 for proposal in proposals))
        self.assertEqual(optimizer.last_evaluations, replay_evaluations)
        self.assertTrue(
            all(
                evaluation.proposal_id in {proposal.proposal_id for proposal in proposals}
                for evaluation in replay_evaluations
            )
        )
        self.assertTrue(all(0.0 <= evaluation.aggregate_score <= 1.0 for evaluation in replay_evaluations))
        self.assertTrue(
            {record.lifecycle_stage for record in proposal_records}.issuperset(
                {OptimizerLifecycleStage.PROPOSED, OptimizerLifecycleStage.SIMULATED}
            )
        )
        self.assertTrue(
            all(
                record.decision
                in {
                    OptimizerActivationDecision.BLOCKED,
                    OptimizerActivationDecision.REJECTED,
                    OptimizerActivationDecision.DEFERRED,
                }
                for record in activation_records
            )
        )
        blocked_records = tuple(
            record for record in activation_records if record.decision == OptimizerActivationDecision.BLOCKED
        )
        self.assertEqual(
            {record.proposal_id for record in activation_records},
            {proposal.proposal_id for proposal in proposals},
        )
        if blocked_records:
            self.assertEqual(
                {record.rollback_record_id for record in blocked_records},
                {record.rollback_record_id for record in rollback_records},
            )

    async def test_optimizer_defers_cycles_during_foreground_work_and_exports_only_verified_deep_traces(self) -> None:
        await self.orchestrator.run_task("What is 2 + 2?", thinking_minutes=30)
        await self.orchestrator.run_task("What is 2 + 2?", thinking_minutes=1)
        optimizer = SelfOptimizer(storage=self.orchestrator.storage, config=self.config)

        await self.orchestrator.storage.enter_foreground_task()
        try:
            proposals = await optimizer.run_cycle()
        finally:
            await self.orchestrator.storage.exit_foreground_task()
        export_path = self.test_logs / "optimizer_verified_deep_exports.jsonl"
        exports = await optimizer.export_verified_deep_traces(export_path=export_path)
        deferred_events = await self.orchestrator.storage.list_runtime_events(stage="self_optimizer.cycle_deferred")

        self.assertEqual(proposals, [])
        self.assertTrue(deferred_events)
        self.assertTrue(exports)
        self.assertTrue(all(export.reasoning.context_frames[0].metadata.get("rm") == "deep" for export in exports))
        self.assertTrue(export_path.exists())
