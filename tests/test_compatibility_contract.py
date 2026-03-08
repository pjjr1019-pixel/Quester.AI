"""Compatibility contract tests for future-phase migrations."""

from __future__ import annotations

import json
import shutil
import time
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from inspect import signature
from pathlib import Path

from compressor import CompressorAgent
from config import APP_CONFIG, BudgetPolicy
from critic import CriticAgent
from dashboard import DashboardService
from data_structures import (
    AgentState,
    AgentStatus,
    CompressedTrace,
    CritiqueReport,
    CritiqueResult,
    EvidenceBundle,
    EvidenceItem,
    Macro,
    MacroProposal,
    PerformanceMetric,
    Plan,
    PlanStep,
    ReasoningLog,
    ResourceBudget,
    RuntimeEvent,
    SeverityLevel,
    SourceType,
    TaskResult,
    TaskState,
)
from macro_engine import MacroEngine
from model_manager import ModelHealthSnapshot, ModelManager
from orchestrator import Orchestrator
from planner import PlannerAgent
from reasoner import ReasonerAgent
from researcher import ResearcherAgent
from self_optimizer import SelfOptimizer
from storage import StorageManager

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "compatibility"
_FIXTURE_TIMESTAMP = datetime(2026, 3, 7, 12, 0, tzinfo=UTC)


def _unlink_with_retries(path: Path, *, attempts: int = 10, delay_s: float = 0.05) -> None:
    """Best-effort Windows-friendly file cleanup for test SQLite files."""
    for attempt in range(attempts):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_s)


def _load_compatibility_fixture(name: str) -> dict[str, object]:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="ascii"))


def _assert_payload_subset(
    testcase: unittest.TestCase,
    actual: object,
    expected: object,
    *,
    path: str = "root",
) -> None:
    if isinstance(expected, dict):
        testcase.assertIsInstance(actual, dict, msg=f"{path} should be a dict.")
        actual_dict = actual
        for key, value in expected.items():
            testcase.assertIn(key, actual_dict, msg=f"Missing key at {path}.{key}")
            _assert_payload_subset(testcase, actual_dict[key], value, path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        testcase.assertIsInstance(actual, list, msg=f"{path} should be a list.")
        actual_list = actual
        testcase.assertEqual(
            len(actual_list),
            len(expected),
            msg=f"List length mismatch at {path}",
        )
        for index, value in enumerate(expected):
            _assert_payload_subset(testcase, actual_list[index], value, path=f"{path}[{index}]")
        return
    testcase.assertEqual(actual, expected, msg=f"Value mismatch at {path}")


def _build_frozen_compressed_trace() -> CompressedTrace:
    return CompressedTrace(
        task_id="fixture-task",
        tokens=("@lookup", "@emit"),
        expanded_preview=("Lookup evidence", "Emit answer"),
        macros_used=("@emit",),
        confidence=0.72,
        reasoner_notes="fixture trace",
        created_at=_FIXTURE_TIMESTAMP,
    )


def _build_frozen_task_result() -> TaskResult:
    plan = Plan(
        task_id="fixture-task",
        question="What remains stable?",
        steps=(PlanStep(step_id="step_1", description="Inspect evidence"),),
        required_evidence=("local docs",),
        success_criteria=("return typed answer",),
        budget=ResourceBudget(
            retrieval_top_k=5,
            max_web_queries=1,
            reasoner_passes=2,
            critic_passes=1,
            macro_depth=3,
        ),
        planner_notes="fixture planner notes",
        created_at=_FIXTURE_TIMESTAMP,
    )
    evidence = EvidenceBundle(
        task_id=plan.task_id,
        local_results=(
            EvidenceItem(
                id="ev-local-1",
                content="Fixture local evidence",
                source_type=SourceType.LOCAL,
                source_ref="local://fixture",
                score=0.81,
                metadata={"topic": "compatibility"},
                vector_preview=(0.1, 0.2),
            ),
        ),
        web_results=(),
        used_web_fallback=False,
        created_at=_FIXTURE_TIMESTAMP,
    )
    trace = _build_frozen_compressed_trace()
    critique = CritiqueReport(
        task_id=plan.task_id,
        is_valid=True,
        issues=(),
        fixed_trace=trace,
        evidence_coverage=1.0,
        critic_notes="fixture critique",
        result=CritiqueResult.VALID,
        created_at=_FIXTURE_TIMESTAMP,
    )
    proposal = MacroProposal(
        proposal_id="proposal-1",
        macro=Macro(macro_name="emit_answer", expansion=("@emit",), version=1),
        reason="Repeated emit token",
        examples=("@emit",),
        simulation_score=0.5,
        approved=False,
        created_at=_FIXTURE_TIMESTAMP,
    )
    metric = PerformanceMetric(
        task_id=plan.task_id,
        time=0.25,
        vram_usage=0.0,
        iterations=2,
    )
    return TaskResult(
        task_id=plan.task_id,
        plan=plan,
        evidence=evidence,
        reasoning=trace,
        critique=critique,
        compression=(proposal,),
        answer_text="Stable answer text",
        warnings=("warning.compatibility",),
        metrics=(metric,),
        completed_at=_FIXTURE_TIMESTAMP,
    )


def _build_examples() -> dict[str, object]:
    budget = ResourceBudget(
        retrieval_top_k=5,
        max_web_queries=2,
        reasoner_passes=2,
        critic_passes=2,
        macro_depth=3,
    )
    step = PlanStep(step_id="step_1", description="Interpret intent")
    plan = Plan(
        task_id="compat-task",
        question="What should stay compatible?",
        steps=(step,),
        required_evidence=("local documents",),
        success_criteria=("valid typed payload",),
        budget=budget,
        planner_notes="compat-notes",
    )
    evidence_item = EvidenceItem(
        id="ev-1",
        content="Local evidence sample",
        source_type=SourceType.LOCAL,
        source_ref="local://sample",
        score=0.8,
        metadata={"topic": "compat"},
        vector_preview=(0.1, 0.2),
    )
    evidence = EvidenceBundle(
        task_id=plan.task_id,
        local_results=(evidence_item,),
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
        "PlanStep": step,
        "Plan": plan,
        "Macro": proposal.macro,
        "ReasoningLog": ReasoningLog(
            task_id=plan.task_id,
            compressed_chain=trace.tokens,
            macros_used=trace.macros_used,
        ),
        "EvidenceItem": evidence_item,
        "EvidenceBundle": evidence,
        "CompressedTrace": trace,
        "CritiqueReport": critique,
        "MacroProposal": proposal,
        "TaskResult": task_result,
        "RuntimeEvent": RuntimeEvent(stage="pipeline.completed", payload={"task_id": plan.task_id}),
        "AgentStatus": AgentStatus(
            component="reasoner",
            state=AgentState.RUNNING,
            task_id=plan.task_id,
            severity=SeverityLevel.LOW,
            message="compat message",
        ),
    }


class CompatibilityPublicApiTests(unittest.TestCase):
    """Verify current public signatures remain stable during migrations."""

    def test_agent_and_orchestrator_signatures_match_contract(self) -> None:
        self.assertEqual(tuple(signature(Orchestrator.start).parameters), ("self",))
        self.assertEqual(tuple(signature(Orchestrator.stop).parameters), ("self",))
        self.assertEqual(tuple(signature(PlannerAgent.plan).parameters), ("self", "question", "budget"))
        self.assertEqual(
            tuple(signature(ResearcherAgent.research).parameters),
            ("self", "plan", "budget"),
        )
        self.assertEqual(
            tuple(signature(ReasonerAgent.reason).parameters),
            ("self", "plan", "evidence", "budget"),
        )
        self.assertEqual(
            tuple(signature(CriticAgent.review).parameters),
            ("self", "plan", "evidence", "trace", "budget"),
        )
        self.assertEqual(
            tuple(signature(CompressorAgent.propose).parameters),
            ("self", "trace", "logs"),
        )
        self.assertEqual(
            tuple(signature(Orchestrator.run_task).parameters),
            ("self", "question", "thinking_minutes"),
        )
        self.assertEqual(
            tuple(signature(Orchestrator.run_pipeline).parameters),
            ("self", "question"),
        )

    def test_model_manager_signatures_match_contract(self) -> None:
        self.assertEqual(tuple(signature(ModelManager.start).parameters), ("self",))
        self.assertEqual(tuple(signature(ModelManager.stop).parameters), ("self",))
        self.assertEqual(tuple(signature(ModelManager.generate).parameters), ("self", "prompt", "max_tokens"))
        self.assertEqual(tuple(signature(ModelManager.embed).parameters), ("self", "text"))
        self.assertEqual(tuple(signature(ModelManager.embed_query).parameters), ("self", "text"))
        self.assertEqual(tuple(signature(ModelManager.embed_document).parameters), ("self", "text"))
        self.assertEqual(tuple(signature(ModelManager.health_snapshot).parameters), ("self",))

    def test_macro_and_optimizer_signatures_match_contract(self) -> None:
        self.assertEqual(tuple(signature(MacroEngine.compress).parameters), ("self", "steps", "task_id"))
        self.assertEqual(tuple(signature(MacroEngine.expand).parameters), ("self", "tokens"))
        self.assertEqual(tuple(signature(MacroEngine.verify_round_trip).parameters), ("self", "trace"))
        self.assertEqual(tuple(signature(SelfOptimizer.run_cycle).parameters), ("self",))

    def test_dashboard_service_signatures_match_contract(self) -> None:
        self.assertEqual(tuple(signature(DashboardService.start).parameters), ("self",))
        self.assertEqual(tuple(signature(DashboardService.stop).parameters), ("self",))
        self.assertEqual(tuple(signature(DashboardService.app_state_snapshot).parameters), ("self",))
        self.assertEqual(
            tuple(signature(DashboardService.attach_controller).parameters),
            ("self", "submit_task", "save_settings", "perform_action"),
        )
        self.assertEqual(tuple(signature(DashboardService.apply_user_settings).parameters), ("self", "profile"))
        self.assertEqual(
            tuple(signature(DashboardService.request_task_submission).parameters),
            ("self", "question", "thinking_minutes"),
        )
        self.assertEqual(tuple(signature(DashboardService.request_settings_save).parameters), ("self", "profile"))
        self.assertEqual(tuple(signature(DashboardService.request_action).parameters), ("self", "action", "payload"))
        self.assertEqual(tuple(signature(DashboardService.publish_event).parameters), ("self", "event"))


class CompatibilityEnumTests(unittest.TestCase):
    """Verify current enum values remain valid."""

    def test_enum_values_keep_existing_meanings(self) -> None:
        self.assertEqual(TaskState.PENDING.value, "pending")
        self.assertEqual(TaskState.RUNNING.value, "running")
        self.assertEqual(TaskState.COMPLETED.value, "completed")
        self.assertEqual(TaskState.FAILED.value, "failed")

        self.assertEqual(SourceType.LOCAL.value, "local")
        self.assertEqual(SourceType.WEB.value, "web")

        self.assertEqual(AgentState.IDLE.value, "idle")
        self.assertEqual(AgentState.RUNNING.value, "running")
        self.assertEqual(AgentState.ERROR.value, "error")

        self.assertEqual(CritiqueResult.VALID.value, "valid")
        self.assertEqual(CritiqueResult.INVALID.value, "invalid")
        self.assertEqual(CritiqueResult.DEGRADED.value, "degraded")


class CompatibilityFieldContractTests(unittest.TestCase):
    """Verify stable field projections remain present in serialized payloads."""

    def test_required_field_projections_exist(self) -> None:
        examples = _build_examples()
        expected_fields = {
            "ResourceBudget": {"retrieval_top_k", "max_web_queries", "reasoner_passes", "critic_passes", "macro_depth"},
            "PlanStep": {"step_id", "description", "depends_on", "status", "notes"},
            "Plan": {"task_id", "question", "steps", "required_evidence", "success_criteria", "budget", "planner_notes"},
            "Macro": {"macro_name", "expansion", "version"},
            "ReasoningLog": {"compressed_chain", "macros_used"},
            "EvidenceItem": {"content", "source_type", "source_ref", "score", "metadata", "vector_preview"},
            "EvidenceBundle": {"local_results", "web_results", "used_web_fallback"},
            "CompressedTrace": {"tokens", "expanded_preview", "macros_used", "confidence", "reasoner_notes"},
            "CritiqueReport": {"is_valid", "issues", "fixed_trace", "evidence_coverage", "critic_notes", "result"},
            "MacroProposal": {"proposal_id", "macro", "reason", "examples", "simulation_score", "approved"},
            "TaskResult": {"plan", "evidence", "reasoning", "critique", "compression", "answer_text", "warnings", "metrics"},
            "RuntimeEvent": {"stage", "payload", "timestamp"},
            "AgentStatus": {"component", "state", "task_id", "severity", "message"},
        }
        for name, keys in expected_fields.items():
            payload = examples[name].to_dict()
            self.assertTrue(keys.issubset(payload.keys()), msg=f"Missing compatibility fields for {name}")

    def test_model_health_snapshot_fields_remain_additive(self) -> None:
        expected_fields = {
            "started",
            "generation_backend",
            "embedding_backend",
            "active_generation_jobs",
            "active_embedding_jobs",
            "last_used_at",
            "fallback_active",
            "fallback_reason",
            "available_ram_gb",
            "total_ram_gb",
            "generation_backend_vram_gb",
            "embedding_backend_vram_gb",
            "telemetry_enabled",
            "last_error",
        }
        self.assertTrue(expected_fields.issubset(ModelHealthSnapshot.__dataclass_fields__.keys()))


class CompatibilitySerializationTests(unittest.TestCase):
    """Verify old and new payload shapes remain readable."""

    def test_frozen_compressed_trace_fixture_still_deserializes(self) -> None:
        fixture = _load_compatibility_fixture("compressed_trace_compatibility.json")
        trace = CompressedTrace.from_dict(fixture["legacy_input"])

        self.assertEqual(trace.task_id, "legacy-trace-fixture")
        self.assertEqual(trace.tokens, ("@lookup", "@emit"))
        self.assertEqual(trace.expanded_preview, ())
        self.assertEqual(trace.macros_used, ("@emit",))
        self.assertEqual(trace.confidence, 0.72)
        self.assertEqual(trace.reasoner_notes, "legacy trace fixture")

    def test_frozen_task_result_fixture_still_deserializes(self) -> None:
        fixture = _load_compatibility_fixture("task_result_compatibility.json")
        result = TaskResult.from_dict(fixture["legacy_input"])

        self.assertEqual(result.task_id, "legacy-task-fixture")
        self.assertEqual(result.plan.task_id, "legacy-task-fixture")
        self.assertEqual(result.reasoning.tokens, ("@lookup", "@emit"))
        self.assertEqual(result.critique.result, CritiqueResult.VALID)
        self.assertEqual(result.answer_text, "")
        self.assertEqual(result.warnings, ())
        self.assertEqual(result.metrics, ())

    def test_current_compressed_trace_projection_matches_frozen_fixture(self) -> None:
        fixture = _load_compatibility_fixture("compressed_trace_compatibility.json")
        payload = _build_frozen_compressed_trace().to_dict()

        _assert_payload_subset(self, payload, fixture["stable_projection"])

    def test_current_task_result_projection_matches_frozen_fixture(self) -> None:
        fixture = _load_compatibility_fixture("task_result_compatibility.json")
        payload = _build_frozen_task_result().to_dict()

        _assert_payload_subset(self, payload, fixture["stable_projection"])

    def test_old_serialized_payloads_still_deserialize(self) -> None:
        plan_step_payload = {
            "step_id": "legacy-step",
            "description": "Human debug step",
        }
        plan_payload = {
            "task_id": "legacy-task",
            "question": "Legacy question",
            "steps": [plan_step_payload],
            "required_evidence": ["local docs"],
            "success_criteria": ["answer coherently"],
            "planner_notes": "legacy notes",
        }
        evidence_item_payload = {
            "id": "legacy-evidence",
            "content": "Legacy evidence text",
            "source_type": "local",
            "source_ref": "local://legacy",
            "score": 0.9,
        }
        evidence_payload = {
            "task_id": "legacy-task",
            "local_results": [evidence_item_payload],
            "web_results": [],
            "used_web_fallback": False,
        }
        trace_payload = {
            "task_id": "legacy-task",
            "compressed_chain": ["@read_question", "@compose_answer"],
            "macros_used": ["@compose_answer"],
            "confidence": 0.7,
            "reasoner_notes": "legacy trace",
        }
        critique_payload = {
            "task_id": "legacy-task",
            "is_valid": True,
            "issues": [],
            "fixed_trace": trace_payload,
            "evidence_coverage": 1.0,
            "critic_notes": "legacy critique",
            "result": "valid",
        }
        proposal_payload = {
            "proposal_id": "legacy-proposal",
            "macro": {
                "macro_name": "compose_answer",
                "expansion": ["@compose_answer"],
            },
            "reason": "Legacy macro reason",
            "examples": ["@compose_answer"],
            "simulation_score": 0.5,
            "approved": False,
        }
        task_result_payload = {
            "task_id": "legacy-task",
            "plan": plan_payload,
            "evidence": evidence_payload,
            "reasoning": trace_payload,
            "critique": critique_payload,
            "compression": [proposal_payload],
        }

        rebuilt = {
            "ResourceBudget": ResourceBudget.from_dict({"retrieval_top_k": 4}),
            "PlanStep": PlanStep.from_dict(plan_step_payload),
            "Plan": Plan.from_dict(plan_payload),
            "Macro": Macro.from_dict({"macro_name": "legacy_macro", "expansion": ["@step"]}),
            "ReasoningLog": ReasoningLog.from_dict(
                {"task_id": "legacy-task", "compressed_chain": ["@step"], "macros_used": []}
            ),
            "PerformanceMetric": PerformanceMetric.from_dict(
                {"task_id": "legacy-task", "time": 0.1, "VRAM_usage": 1.2, "iterations": 2}
            ),
            "EvidenceItem": EvidenceItem.from_dict(evidence_item_payload),
            "EvidenceBundle": EvidenceBundle.from_dict(evidence_payload),
            "CompressedTrace": CompressedTrace.from_dict(trace_payload),
            "CritiqueReport": CritiqueReport.from_dict(critique_payload),
            "MacroProposal": MacroProposal.from_dict(proposal_payload),
            "TaskResult": TaskResult.from_dict(task_result_payload),
            "RuntimeEvent": RuntimeEvent.from_dict(
                {"stage": "pipeline.completed", "payload": {"task_id": "legacy-task"}}
            ),
            "AgentStatus": AgentStatus.from_dict({"component": "reasoner", "state": "running"}),
        }

        self.assertIsInstance(rebuilt["ResourceBudget"], ResourceBudget)
        self.assertIsInstance(rebuilt["PlanStep"], PlanStep)
        self.assertIsInstance(rebuilt["Plan"], Plan)
        self.assertIsInstance(rebuilt["Macro"], Macro)
        self.assertIsInstance(rebuilt["ReasoningLog"], ReasoningLog)
        self.assertIsInstance(rebuilt["PerformanceMetric"], PerformanceMetric)
        self.assertIsInstance(rebuilt["EvidenceItem"], EvidenceItem)
        self.assertIsInstance(rebuilt["EvidenceBundle"], EvidenceBundle)
        self.assertIsInstance(rebuilt["CompressedTrace"], CompressedTrace)
        self.assertIsInstance(rebuilt["CritiqueReport"], CritiqueReport)
        self.assertIsInstance(rebuilt["MacroProposal"], MacroProposal)
        self.assertIsInstance(rebuilt["TaskResult"], TaskResult)
        self.assertIsInstance(rebuilt["RuntimeEvent"], RuntimeEvent)
        self.assertIsInstance(rebuilt["AgentStatus"], AgentStatus)

    def test_old_compressed_trace_payload_without_ir_fields_is_valid(self) -> None:
        trace = CompressedTrace.from_dict(
            {
                "task_id": "legacy-trace",
                "compressed_chain": ["@read_question", "@compose_answer"],
                "confidence": 0.9,
            }
        )
        self.assertEqual(trace.tokens, ("@read_question", "@compose_answer"))
        self.assertEqual(trace.expanded_preview, ())

    def test_ir_backed_trace_payload_still_emits_legacy_projections(self) -> None:
        trace = CompressedTrace.from_dict(
            {
                "task_id": "future-trace",
                "tokens": ["@lookup", "@emit"],
                "expanded_preview": ["Lookup evidence", "Emit answer"],
                "macros_used": ["@emit"],
                "confidence": 0.8,
                "reasoner_notes": "future trace",
                "ir_version": "1",
                "operation_stream": [{"op": "lookup", "args": ["ev-1"]}],
                "symbol_table_refs": ["sym-1"],
                "context_frames": [{"frame_id": "ctx-1"}],
                "proof_hash": "abc123",
            }
        )
        payload = trace.to_dict()
        self.assertEqual(payload["tokens"], ["@lookup", "@emit"])
        self.assertEqual(payload["expanded_preview"], ["Lookup evidence", "Emit answer"])
        self.assertEqual(payload["ir_version"], "1")
        self.assertEqual(payload["operation_stream"][0]["opcode"], "lookup")
        self.assertEqual(payload["context_frames"][0]["frame_id"], "ctx-1")
        self.assertEqual(payload["proof_hash"], "abc123")


class CompatibilityRuntimeTests(unittest.IsolatedAsyncioTestCase):
    """Verify runtime compatibility surfaces keep working."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_compatibility.sqlite3")
        self.test_logs = Path("test_compatibility_logs")
        self.orchestrator: Orchestrator | None = None
        self.storage: StorageManager | None = None
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

    async def asyncTearDown(self) -> None:
        if self.orchestrator is not None:
            await self.orchestrator.stop()
            self.orchestrator = None
        if self.storage is not None:
            await self.storage.stop()
            self.storage = None
        if self.test_db.exists():
            _unlink_with_retries(self.test_db)
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_run_task_and_run_pipeline_remain_compatible(self) -> None:
        self.orchestrator = Orchestrator(config=self.test_config)
        await self.orchestrator.start()

        full_result = await self.orchestrator.run_task("Compatibility question", thinking_minutes=30)
        compat_result = await self.orchestrator.run_pipeline("Compatibility wrapper question")

        self.assertIsInstance(full_result, TaskResult)
        self.assertIsInstance(compat_result, TaskResult)
        self.assertEqual(
            compat_result.plan.budget,
            BudgetPolicy.from_minutes(1),
        )

    async def test_storage_manager_legacy_log_and_kv_api_still_work(self) -> None:
        self.storage = StorageManager(config=self.test_config)
        await self.storage.start()

        await self.storage.log_event("compatibility.event", {"value": 1})
        await self.storage.set_kv("compatibility-key", {"value": "ok"})

        stored = await self.storage.get_kv("compatibility-key")

        self.assertEqual(stored, {"value": "ok"})
        self.assertTrue((self.test_logs / APP_CONFIG.storage.events_log_name).exists())


class CompatibilityDashboardTests(unittest.TestCase):
    """Verify dashboard input compatibility remains intact."""

    def test_publish_event_accepts_current_event_dict_shape(self) -> None:
        dashboard = DashboardService(config=replace(APP_CONFIG, dashboard=replace(APP_CONFIG.dashboard, enable_ui=False)))
        dashboard.publish_event({"stage": "pipeline.completed", "task_id": "task-1"})
        event = dashboard._events.get_nowait()

        self.assertEqual(event["stage"], "pipeline.completed")
        self.assertEqual(event["task_id"], "task-1")
        self.assertIn("timestamp", event)


if __name__ == "__main__":
    unittest.main()
