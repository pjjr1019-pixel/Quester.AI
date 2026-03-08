"""Phase 17 long-horizon control and throttling regressions."""

from __future__ import annotations

import asyncio
import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from config import APP_CONFIG, BudgetPolicy
from dashboard import DashboardService
from data_structures import (
    CandidateTrace,
    CompressedTrace,
    ContextFrame,
    CritiqueReport,
    CritiqueResult,
    EvidenceItem,
    EvidenceBundle,
    LongHorizonSessionState,
    OptimizerSuggestionKind,
    OptimizerSuggestionRecord,
    PerformanceMetric,
    Plan,
    PlanStep,
    SourceType,
    TaskResult,
    TaskState,
)
from model_manager import ModelHealthSnapshot
from orchestrator import Orchestrator
from storage import StorageManager


class _Var:
    def __init__(self, value) -> None:
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


def _build_test_config(*, sqlite_name: str, logs_name: str):
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
    return replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)


def _build_stub_task_result(
    *,
    task_id: str,
    question: str,
    answer_text: str,
    metric_time: float = 0.05,
    candidate_count: int = 1,
    candidate_score: float = 0.85,
    critique_result: CritiqueResult = CritiqueResult.VALID,
    repair_actions: tuple[str, ...] = (),
    local_evidence_ids: tuple[str, ...] = (),
    web_evidence_ids: tuple[str, ...] = (),
) -> TaskResult:
    plan = Plan(
        task_id=task_id,
        question=question,
        steps=(
            PlanStep(
                step_id="step_1",
                description="Return a verified answer.",
                status=TaskState.COMPLETED,
            ),
        ),
        required_evidence=("ev-1",),
        success_criteria=("Return a supported answer.",),
        budget=BudgetPolicy.from_minutes(5),
    )
    local_results = tuple(
        EvidenceItem(
            id=evidence_id,
            content=f"Local evidence for {evidence_id}",
            source_type=SourceType.LOCAL,
            source_ref=f"local:{evidence_id}",
            score=0.85,
        )
        for evidence_id in local_evidence_ids
    )
    web_results = tuple(
        EvidenceItem(
            id=evidence_id,
            content=f"Web evidence for {evidence_id}",
            source_type=SourceType.WEB,
            source_ref=f"web:{evidence_id}",
            score=0.8,
        )
        for evidence_id in web_evidence_ids
    )
    evidence = EvidenceBundle(
        task_id=task_id,
        local_results=local_results,
        web_results=web_results,
        used_web_fallback=bool(web_results),
    )
    supporting_evidence_ids = tuple((*local_evidence_ids, *web_evidence_ids))
    candidates = tuple(
        CandidateTrace(
            candidate_id=f"{task_id}_candidate_{index}",
            answer_text=answer_text,
            strategy="stub_cycle",
            verifier_type="tool.evidence_grounding",
            verified=index == 1,
            total_score=max(0.0, min(1.0, candidate_score - (0.03 * (index - 1)))),
            agreement_score=0.8,
            evidence_support_score=0.9,
            proof_hash_stability=1.0,
            supporting_evidence_ids=supporting_evidence_ids,
            proof_hash=f"proof-{task_id}-{index}",
        )
        for index in range(1, candidate_count + 1)
    )
    reasoning = CompressedTrace(
        task_id=task_id,
        tokens=("@lookup", "@emit"),
        expanded_preview=("lookup", "emit"),
        macros_used=(),
        confidence=0.9,
        candidate_traces=candidates,
        proof_hash=f"trace-{task_id}",
    )
    critique = CritiqueReport(
        task_id=task_id,
        is_valid=critique_result == CritiqueResult.VALID,
        issues=(),
        fixed_trace=None,
        evidence_coverage=1.0,
        result=critique_result,
        verifier_type="tool.evidence_grounding",
        proof_hash_match=True,
        candidate_score=candidate_score,
        repair_actions=repair_actions,
    )
    return TaskResult(
        task_id=task_id,
        plan=plan,
        evidence=evidence,
        reasoning=reasoning,
        critique=critique,
        compression=(),
        answer_text=answer_text,
        warnings=(),
        metrics=(
            PerformanceMetric(
                task_id=task_id,
                time=metric_time,
                vram_usage=1.25,
                iterations=1,
            ),
        ),
    )


def _build_suggestion(
    *,
    suggestion_id: str,
    kind: OptimizerSuggestionKind,
    summary: str,
    budget_delta: dict[str, int] | None = None,
    advisory_only_reason: str = "",
) -> OptimizerSuggestionRecord:
    metadata: dict[str, object] = {"budget_delta": dict(budget_delta or {})}
    if advisory_only_reason:
        metadata["advisory_only_reason"] = advisory_only_reason
    return OptimizerSuggestionRecord(
        suggestion_id=suggestion_id,
        cycle_id="phase17:test-cycle",
        kind=kind,
        summary=summary,
        rationale="Phase 17 regression suggestion.",
        target_components=("reasoner",),
        source_task_ids=("phase17_source_task",),
        confidence=0.72,
        advisory_only=True,
        metadata=metadata,
    )


class Phase17LongHorizonTests(unittest.IsolatedAsyncioTestCase):
    """Lock the Phase 17 control behavior to bounded, resumable sessions."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase17_long_horizon.sqlite3")
        self.test_logs = Path("test_phase17_long_horizon_logs")
        self.config = _build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def _wait_for_session(self, session_id: str) -> object:
        for _ in range(100):
            session = await self.orchestrator.storage.load_long_horizon_session(session_id)
            if session is not None:
                return session
            await asyncio.sleep(0.01)
        self.fail(f"Timed out waiting for long-horizon session '{session_id}'.")

    async def test_pause_resume_persists_checkpoints_and_dashboard_state(self) -> None:
        question = "Pause and resume the long-horizon session."
        session_id = "lh_phase17_pause_resume"
        cycle_started = asyncio.Event()
        release_first_cycle = asyncio.Event()
        call_count = 0

        async def fake_run_bounded_task(
            *,
            question: str,
            thinking_minutes: int,
            budget,
            persist_task_result: bool = True,
            publish_history: bool = True,
            emit_completion_event: bool = True,
        ) -> TaskResult:
            nonlocal call_count
            _ = (thinking_minutes, budget, persist_task_result, publish_history, emit_completion_event)
            call_count += 1
            cycle_started.set()
            if call_count == 1:
                await release_first_cycle.wait()
            return _build_stub_task_result(
                task_id=f"phase17_pause_cycle_{call_count}",
                question=question,
                answer_text=f"cycle {call_count} answer",
                metric_time=0.1 * call_count,
                local_evidence_ids=("ev-1",),
            )

        with (
            mock.patch.object(self.orchestrator, "_long_horizon_session_id", return_value=session_id),
            mock.patch.object(self.orchestrator, "_run_bounded_task", side_effect=fake_run_bounded_task),
        ):
            task = asyncio.create_task(self.orchestrator.run_task(question, thinking_minutes=121))
            await cycle_started.wait()
            session = await self._wait_for_session(session_id)
            self.assertEqual(session.status, LongHorizonSessionState.RUNNING)

            self.assertTrue(
                await self.orchestrator._request_long_horizon_pause(session_id, reason="test_pause_requested")
            )
            pending = await self.orchestrator.storage.load_long_horizon_session(session_id)
            assert pending is not None
            self.assertTrue(pending.pause_requested)
            self.assertEqual(pending.last_control_reason, "test_pause_requested")

            release_first_cycle.set()
            paused_result = await asyncio.wait_for(task, timeout=1.0)

        self.assertIn("long_horizon_paused", paused_result.warnings)
        self.assertIn("long_horizon_resume_available", paused_result.warnings)
        self.assertIn("long_horizon_cycles_completed:1", paused_result.warnings)
        self.assertEqual(len(paused_result.metrics), 1)

        paused_session = await self.orchestrator.storage.load_long_horizon_session(session_id)
        assert paused_session is not None
        self.assertEqual(paused_session.status, LongHorizonSessionState.PAUSED)
        self.assertEqual(paused_session.completed_cycles, 1)
        self.assertEqual(paused_session.resume_count, 0)
        self.assertFalse(paused_session.pause_requested)
        self.assertEqual(paused_session.last_control_reason, "test_pause_requested")

        checkpoints = await self.orchestrator.storage.list_long_horizon_checkpoints(session_id)
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].cycle_index, 1)
        self.assertEqual(checkpoints[0].resume_count, 0)

        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertEqual(state.active_task.long_horizon_session_id, session_id)
        self.assertEqual(state.active_task.long_horizon_status, "paused")
        self.assertEqual(state.active_task.long_horizon_completed_cycles, 1)
        self.assertEqual(state.active_task.long_horizon_total_cycles, 2)

        resumed_result = await self.orchestrator._resume_long_horizon_session(session_id)

        self.assertIn("long_horizon_checkpointed_run", resumed_result.warnings)
        self.assertIn("long_horizon_cycles_completed:2", resumed_result.warnings)
        self.assertIn("long_horizon_resume_count:1", resumed_result.warnings)
        self.assertEqual(len(resumed_result.metrics), 2)

        completed_session = await self.orchestrator.storage.load_long_horizon_session(session_id)
        assert completed_session is not None
        self.assertEqual(completed_session.status, LongHorizonSessionState.COMPLETED)
        self.assertEqual(completed_session.completed_cycles, 2)
        self.assertEqual(completed_session.resume_count, 1)

        checkpoints = await self.orchestrator.storage.list_long_horizon_checkpoints(session_id)
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(checkpoints[1].cycle_index, 2)
        self.assertEqual(checkpoints[1].resume_count, 1)

        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertEqual(state.active_task.long_horizon_status, "completed")
        self.assertEqual(state.active_task.long_horizon_resume_count, 1)
        self.assertEqual(state.active_task.long_horizon_completed_cycles, 2)

    async def test_cancel_marks_active_session_cancelled(self) -> None:
        question = "Cancel the long-horizon session."
        session_id = "lh_phase17_cancel"
        cycle_started = asyncio.Event()
        release_cycle = asyncio.Event()

        async def fake_run_bounded_task(
            *,
            question: str,
            thinking_minutes: int,
            budget,
            persist_task_result: bool = True,
            publish_history: bool = True,
            emit_completion_event: bool = True,
        ) -> TaskResult:
            _ = (question, thinking_minutes, budget, persist_task_result, publish_history, emit_completion_event)
            cycle_started.set()
            await release_cycle.wait()
            return _build_stub_task_result(
                task_id="phase17_cancel_cycle_1",
                question="Cancel the long-horizon session.",
                answer_text="cancel should prevent this result",
                local_evidence_ids=("ev-1",),
            )

        with (
            mock.patch.object(self.orchestrator, "_long_horizon_session_id", return_value=session_id),
            mock.patch.object(self.orchestrator, "_run_bounded_task", side_effect=fake_run_bounded_task),
        ):
            task = asyncio.create_task(self.orchestrator.run_task(question, thinking_minutes=121))
            await cycle_started.wait()
            await self._wait_for_session(session_id)

            self.assertTrue(
                await self.orchestrator._request_long_horizon_cancel(session_id, reason="test_cancel_requested")
            )
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)

        cancelled_session = await self.orchestrator.storage.load_long_horizon_session(session_id)
        assert cancelled_session is not None
        self.assertEqual(cancelled_session.status, LongHorizonSessionState.CANCELLED)
        self.assertFalse(cancelled_session.cancel_requested)
        self.assertEqual(cancelled_session.last_control_reason, "test_cancel_requested")

        checkpoints = await self.orchestrator.storage.list_long_horizon_checkpoints(session_id)
        self.assertEqual(checkpoints, ())

        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertEqual(state.active_task.long_horizon_status, "cancelled")
        self.assertEqual(state.active_task.long_horizon_session_id, session_id)

    async def test_pressure_throttling_reduces_cycle_budget_and_persists_reason(self) -> None:
        question = "Throttle the long-horizon session under pressure."
        session_id = "lh_phase17_throttle"
        baseline_budget = BudgetPolicy.from_minutes(121)
        observed_budgets = []
        self.orchestrator.dashboard._dropped_events = 3
        pressured_snapshot = ModelHealthSnapshot(
            started=True,
            generation_backend="stub_generation",
            embedding_backend="stub_embedding",
            active_generation_jobs=0,
            active_embedding_jobs=0,
            last_used_at=None,
            fallback_active=True,
            fallback_reason="phase17_test_fallback",
            available_ram_gb=1.0,
            total_ram_gb=8.0,
            generation_backend_vram_gb=5.1,
            embedding_backend_vram_gb=5.0,
            telemetry_enabled=True,
            last_error="phase17_test_error",
        )
        call_count = 0

        async def fake_run_bounded_task(
            *,
            question: str,
            thinking_minutes: int,
            budget,
            persist_task_result: bool = True,
            publish_history: bool = True,
            emit_completion_event: bool = True,
        ) -> TaskResult:
            nonlocal call_count
            _ = (thinking_minutes, persist_task_result, publish_history, emit_completion_event)
            call_count += 1
            observed_budgets.append(budget)
            return _build_stub_task_result(
                task_id=f"phase17_throttle_cycle_{call_count}",
                question=question,
                answer_text=f"throttled cycle {call_count}",
                local_evidence_ids=("ev-1", "ev-2"),
                web_evidence_ids=("web-1",),
            )

        with (
            mock.patch.object(self.orchestrator, "_long_horizon_session_id", return_value=session_id),
            mock.patch.object(self.orchestrator.model_manager, "health_snapshot", return_value=pressured_snapshot),
            mock.patch.object(self.orchestrator, "_run_bounded_task", side_effect=fake_run_bounded_task),
        ):
            result = await self.orchestrator.run_task(question, thinking_minutes=121)

        self.assertEqual(len(observed_budgets), baseline_budget.planned_cycles)
        throttled_budget = observed_budgets[0]
        self.assertLess(throttled_budget.retrieval_top_k, baseline_budget.retrieval_top_k)
        self.assertLess(throttled_budget.max_web_queries, baseline_budget.max_web_queries)
        self.assertLess(throttled_budget.reasoner_passes, baseline_budget.reasoner_passes)
        self.assertLess(throttled_budget.critic_passes, baseline_budget.critic_passes)
        self.assertLess(throttled_budget.macro_depth, baseline_budget.macro_depth)
        self.assertLessEqual(throttled_budget.duty_cycle_ratio, 0.5)
        self.assertGreater(throttled_budget.cooldown_seconds, baseline_budget.cooldown_seconds)
        self.assertIn("long_horizon_throttled:", " ".join(result.warnings))

        throttled_session = await self.orchestrator.storage.load_long_horizon_session(session_id)
        assert throttled_session is not None
        self.assertEqual(throttled_session.status, LongHorizonSessionState.COMPLETED)
        self.assertTrue(throttled_session.throttled)
        self.assertIn("model_fallback_active", throttled_session.throttle_reason)
        self.assertIn("dashboard_backpressure", throttled_session.throttle_reason)

        checkpoints = await self.orchestrator.storage.list_long_horizon_checkpoints(session_id)
        self.assertEqual(len(checkpoints), 2)
        self.assertTrue(all(checkpoint.throttled for checkpoint in checkpoints))
        self.assertTrue(all(checkpoint.throttle_reason for checkpoint in checkpoints))

        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertTrue(state.active_task.long_horizon_throttled)
        self.assertIn("model_fallback_active", state.active_task.long_horizon_throttle_reason)
        self.assertTrue(any("dashboard_backpressure" in condition.reason for condition in state.recent_conditions))

    async def test_stop_pauses_active_session_for_safe_shutdown(self) -> None:
        question = "Pause the active long-horizon session during shutdown."
        session_id = "lh_phase17_shutdown"
        cycle_started = asyncio.Event()
        release_cycle = asyncio.Event()

        async def fake_run_bounded_task(
            *,
            question: str,
            thinking_minutes: int,
            budget,
            persist_task_result: bool = True,
            publish_history: bool = True,
            emit_completion_event: bool = True,
        ) -> TaskResult:
            _ = (question, thinking_minutes, budget, persist_task_result, publish_history, emit_completion_event)
            cycle_started.set()
            await release_cycle.wait()
            return _build_stub_task_result(
                task_id="phase17_shutdown_cycle_1",
                question="Pause the active long-horizon session during shutdown.",
                answer_text="shutdown should prevent this result",
                local_evidence_ids=("ev-1",),
            )

        with (
            mock.patch.object(self.orchestrator, "_long_horizon_session_id", return_value=session_id),
            mock.patch.object(self.orchestrator, "_run_bounded_task", side_effect=fake_run_bounded_task),
        ):
            task = asyncio.create_task(self.orchestrator.run_task(question, thinking_minutes=121))
            await cycle_started.wait()
            await self._wait_for_session(session_id)
            await self.orchestrator.stop()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)

        reopened_storage = StorageManager(config=self.config)
        await reopened_storage.start()
        try:
            paused_session = await reopened_storage.load_long_horizon_session(session_id)
        finally:
            await reopened_storage.stop()

        assert paused_session is not None
        self.assertEqual(paused_session.status, LongHorizonSessionState.PAUSED)
        self.assertEqual(paused_session.last_control_reason, "shutdown_requested")
        self.assertEqual(paused_session.completed_cycles, 0)

    async def test_dashboard_projects_time_progress_and_extra_time_gains(self) -> None:
        question = "Show what extra time bought."
        session_id = "lh_phase17_dashboard_gains"
        call_count = 0

        async def fake_run_bounded_task(
            *,
            question: str,
            thinking_minutes: int,
            budget,
            persist_task_result: bool = True,
            publish_history: bool = True,
            emit_completion_event: bool = True,
        ) -> TaskResult:
            nonlocal call_count
            _ = (thinking_minutes, budget, persist_task_result, publish_history, emit_completion_event)
            call_count += 1
            if call_count == 1:
                return _build_stub_task_result(
                    task_id="phase17_dashboard_cycle_1",
                    question=question,
                    answer_text="first cycle answer",
                    candidate_count=1,
                    candidate_score=0.45,
                    critique_result=CritiqueResult.DEGRADED,
                    repair_actions=("tighten_selection",),
                    local_evidence_ids=("ev-1",),
                )
            return _build_stub_task_result(
                task_id="phase17_dashboard_cycle_2",
                question=question,
                answer_text="second cycle answer",
                candidate_count=3,
                candidate_score=0.82,
                critique_result=CritiqueResult.VALID,
                repair_actions=(),
                local_evidence_ids=("ev-1", "ev-2"),
                web_evidence_ids=("web-1",),
            )

        with (
            mock.patch.object(self.orchestrator, "_long_horizon_session_id", return_value=session_id),
            mock.patch.object(self.orchestrator, "_run_bounded_task", side_effect=fake_run_bounded_task),
        ):
            result = await self.orchestrator.run_task(question, thinking_minutes=121)

        self.assertIn("long_horizon_checkpointed_run", result.warnings)
        state = self.orchestrator.dashboard.app_state_snapshot()

        self.assertEqual(state.active_task.execution_mode, "long_horizon")
        self.assertEqual(state.active_task.requested_thinking_minutes, 121)
        self.assertEqual(state.active_task.long_horizon_session_id, session_id)
        self.assertEqual(state.active_task.long_horizon_status, "completed")
        self.assertEqual(state.active_task.long_horizon_current_phase, "completed")
        self.assertEqual(state.active_task.long_horizon_completed_cycles, 2)
        self.assertEqual(state.active_task.long_horizon_total_cycles, 2)
        self.assertEqual(state.active_task.long_horizon_cycle_budget_minutes, 120)
        self.assertEqual(state.active_task.long_horizon_checkpoint_interval_minutes, 120)
        self.assertGreaterEqual(state.active_task.long_horizon_elapsed_seconds, 0.0)
        self.assertEqual(state.active_task.long_horizon_eta_seconds, 0.0)
        self.assertEqual(state.active_task.long_horizon_initial_candidate_count, 1)
        self.assertEqual(state.active_task.long_horizon_peak_candidate_count, 3)
        self.assertEqual(state.active_task.long_horizon_additional_candidate_count, 2)
        self.assertEqual(state.active_task.long_horizon_initial_supporting_evidence_count, 1)
        self.assertEqual(state.active_task.long_horizon_additional_supporting_evidence_count, 2)
        self.assertEqual(state.active_task.long_horizon_total_verification_passes, 6)
        self.assertEqual(state.active_task.long_horizon_total_repairs, 1)
        self.assertAlmostEqual(state.active_task.long_horizon_first_candidate_score, 0.45, places=3)
        self.assertAlmostEqual(state.active_task.long_horizon_confidence_gain, 0.37, places=3)
        self.assertEqual(state.active_task.long_horizon_first_critique_result, CritiqueResult.DEGRADED.value)
        self.assertTrue(state.active_task.long_horizon_validity_improved)
        self.assertEqual(state.active_task.candidate_trace_count, 3)
        self.assertEqual(len(state.active_task.supporting_evidence_ids), 3)

    async def test_advisory_usage_is_persisted_and_explained_in_dashboard(self) -> None:
        question = "Show advisory decisions between long-horizon cycles."
        session_id = "lh_phase17_advisory"
        pressured_snapshot = ModelHealthSnapshot(
            started=True,
            generation_backend="stub_generation",
            embedding_backend="stub_embedding",
            active_generation_jobs=0,
            active_embedding_jobs=0,
            last_used_at=None,
            fallback_active=True,
            fallback_reason="phase17_advisory_pressure",
            available_ram_gb=1.0,
            total_ram_gb=8.0,
            generation_backend_vram_gb=5.0,
            embedding_backend_vram_gb=5.0,
            telemetry_enabled=True,
            last_error="phase17_advisory_error",
        )
        call_count = 0

        async def fake_run_bounded_task(
            *,
            question: str,
            thinking_minutes: int,
            budget,
            persist_task_result: bool = True,
            publish_history: bool = True,
            emit_completion_event: bool = True,
        ) -> TaskResult:
            nonlocal call_count
            _ = (thinking_minutes, budget, persist_task_result, publish_history, emit_completion_event)
            call_count += 1
            return _build_stub_task_result(
                task_id=f"phase17_advisory_cycle_{call_count}",
                question=question,
                answer_text=f"advisory cycle {call_count}",
                candidate_count=2,
                candidate_score=0.6 + (0.05 * call_count),
                critique_result=CritiqueResult.VALID,
                local_evidence_ids=("ev-1",),
            )

        async def fake_suggest_for_long_horizon(**kwargs):
            _ = kwargs
            return (
                _build_suggestion(
                    suggestion_id="suggestion:accepted",
                    kind=OptimizerSuggestionKind.RETRIEVAL_STRATEGY,
                    summary="Recover some bounded retrieval depth.",
                    budget_delta={
                        "retrieval_top_k": 4,
                        "max_web_queries": 2,
                        "reasoner_passes": 1,
                        "critic_passes": 1,
                        "macro_depth": 1,
                    },
                ),
                _build_suggestion(
                    suggestion_id="suggestion:rejected",
                    kind=OptimizerSuggestionKind.RETRIEVAL_STRATEGY,
                    summary="Try to push the already restored budget again.",
                    budget_delta={
                        "retrieval_top_k": 4,
                        "max_web_queries": 2,
                        "reasoner_passes": 1,
                        "critic_passes": 1,
                        "macro_depth": 1,
                    },
                ),
                _build_suggestion(
                    suggestion_id="suggestion:deferred",
                    kind=OptimizerSuggestionKind.MACRO_ADVICE,
                    summary="Keep macro guidance deferred under proposal-only policy.",
                    advisory_only_reason="proposal_only_policy",
                ),
            )

        with (
            mock.patch.object(self.orchestrator, "_long_horizon_session_id", return_value=session_id),
            mock.patch.object(self.orchestrator.model_manager, "health_snapshot", return_value=pressured_snapshot),
            mock.patch.object(self.orchestrator, "_run_bounded_task", side_effect=fake_run_bounded_task),
            mock.patch.object(
                self.orchestrator.self_optimizer,
                "suggest_for_long_horizon",
                side_effect=fake_suggest_for_long_horizon,
            ),
        ):
            result = await self.orchestrator.run_task(question, thinking_minutes=121)

        self.assertIn("long_horizon_checkpointed_run", result.warnings)
        suggestion_records = await self.orchestrator.storage.list_optimizer_suggestion_records()
        usage_records = await self.orchestrator.storage.list_optimizer_suggestion_usage_records(session_id=session_id)
        self.assertEqual(len(suggestion_records), 3)
        self.assertEqual(sum(record.disposition.value == "requested" for record in usage_records), 3)
        self.assertEqual(sum(record.disposition.value == "accepted" for record in usage_records), 1)
        self.assertEqual(sum(record.disposition.value == "rejected" for record in usage_records), 1)
        self.assertEqual(sum(record.disposition.value == "deferred" for record in usage_records), 1)

        checkpoints = await self.orchestrator.storage.list_long_horizon_checkpoints(session_id)
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(len(checkpoints[0].metadata["advisory_suggestions"]), 3)
        self.assertEqual(len(checkpoints[0].metadata["advisory_usage_records"]), 6)

        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertEqual(state.active_task.long_horizon_advisory_requested_count, 3)
        self.assertEqual(state.active_task.long_horizon_advisory_accepted_count, 1)
        self.assertEqual(state.active_task.long_horizon_advisory_rejected_count, 1)
        self.assertEqual(state.active_task.long_horizon_advisory_deferred_count, 1)
        advisory_text = "\n".join(state.active_task.long_horizon_advisory_entries)
        self.assertIn("accepted Recover some bounded retrieval depth.", advisory_text)
        self.assertIn("rejected Try to push the already restored budget again.", advisory_text)
        self.assertIn("deferred Keep macro guidance deferred under proposal-only policy.", advisory_text)

    async def test_twelve_hour_mode_stays_bounded_and_longer_budget_can_improve(self) -> None:
        question = "Prove the 12-hour schedule stays bounded."

        async def run_with_minutes(minutes: int, *, session_id: str) -> tuple[TaskResult, list]:
            observed_budgets = []
            call_count = 0

            async def fake_run_bounded_task(
                *,
                question: str,
                thinking_minutes: int,
                budget,
                persist_task_result: bool = True,
                publish_history: bool = True,
                emit_completion_event: bool = True,
            ) -> TaskResult:
                nonlocal call_count
                _ = (thinking_minutes, persist_task_result, publish_history, emit_completion_event)
                call_count += 1
                observed_budgets.append(budget)
                evidence_ids = tuple(f"ev-{index}" for index in range(1, min(call_count, 4) + 1))
                return _build_stub_task_result(
                    task_id=f"{session_id}_cycle_{call_count}",
                    question=question,
                    answer_text=f"{minutes} minute cycle {call_count}",
                    candidate_count=min(call_count, 5),
                    candidate_score=0.45 + (0.05 * call_count),
                    critique_result=CritiqueResult.VALID,
                    local_evidence_ids=evidence_ids,
                )

            with (
                mock.patch.object(self.orchestrator, "_long_horizon_session_id", return_value=session_id),
                mock.patch.object(self.orchestrator, "_run_bounded_task", side_effect=fake_run_bounded_task),
            ):
                result = await self.orchestrator.run_task(question, thinking_minutes=minutes)
            return result, observed_budgets

        result_121, budgets_121 = await run_with_minutes(121, session_id="lh_phase17_121m")
        result_720, budgets_720 = await run_with_minutes(720, session_id="lh_phase17_720m")

        self.assertEqual(len(budgets_121), 2)
        self.assertEqual(len(budgets_720), 6)
        for budget in (*budgets_121, *budgets_720):
            self.assertLessEqual(budget.retrieval_top_k, self.config.budget_calibration.max_retrieval_top_k)
            self.assertLessEqual(budget.max_web_queries, self.config.budget_calibration.max_web_queries)
            self.assertLessEqual(budget.reasoner_passes, self.config.budget_calibration.max_reasoner_passes)
            self.assertLessEqual(budget.critic_passes, self.config.budget_calibration.max_critic_passes)
            self.assertLessEqual(budget.macro_depth, self.config.budget_calibration.max_macro_depth)
            self.assertLessEqual(budget.cycle_budget_minutes, self.config.budget_calibration.max_cycle_budget_minutes)
            self.assertLessEqual(
                budget.retrieval_top_k,
                self.config.budget_calibration.max_retrieval_top_k,
            )
        self.assertGreater(result_720.critique.candidate_score, result_121.critique.candidate_score)
        self.assertGreater(len(result_720.metrics), len(result_121.metrics))

    async def test_early_stop_returns_structured_reason_after_flat_cycles(self) -> None:
        question = "Stop early when no improvement is found."
        session_id = "lh_phase17_early_stop"
        call_count = 0

        async def fake_run_bounded_task(
            *,
            question: str,
            thinking_minutes: int,
            budget,
            persist_task_result: bool = True,
            publish_history: bool = True,
            emit_completion_event: bool = True,
        ) -> TaskResult:
            nonlocal call_count
            _ = (thinking_minutes, budget, persist_task_result, publish_history, emit_completion_event)
            call_count += 1
            return _build_stub_task_result(
                task_id=f"phase17_early_stop_cycle_{call_count}",
                question=question,
                answer_text="flat result",
                candidate_count=1,
                candidate_score=0.41,
                critique_result=CritiqueResult.DEGRADED,
                local_evidence_ids=("ev-1",),
            )

        with (
            mock.patch.object(self.orchestrator, "_long_horizon_session_id", return_value=session_id),
            mock.patch.object(self.orchestrator, "_run_bounded_task", side_effect=fake_run_bounded_task),
        ):
            result = await self.orchestrator.run_task(question, thinking_minutes=720)

        self.assertIn("long_horizon_early_stop", result.warnings)
        self.assertTrue(any(item.startswith("long_horizon_early_stop_reason:") for item in result.warnings))
        stopped_session = await self.orchestrator.storage.load_long_horizon_session(session_id)
        assert stopped_session is not None
        self.assertEqual(stopped_session.status, LongHorizonSessionState.COMPLETED)
        self.assertLess(stopped_session.completed_cycles, stopped_session.total_cycles)
        self.assertEqual(stopped_session.last_control_reason, "no_measurable_improvement")

        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertEqual(state.active_task.long_horizon_current_phase, "early_stopped")
        self.assertIn("No measurable improvement", state.active_task.long_horizon_early_stop_reason)

    async def test_export_bundle_only_emits_verified_final_trace(self) -> None:
        valid_session_id = "lh_phase17_export_verified"
        invalid_session_id = "lh_phase17_export_unverified"
        question = "Export machine-readable long-horizon artifacts."

        async def run_with_result(session_id: str, *, verified: bool) -> TaskResult:
            async def fake_run_bounded_task(
                *,
                question: str,
                thinking_minutes: int,
                budget,
                persist_task_result: bool = True,
                publish_history: bool = True,
                emit_completion_event: bool = True,
            ) -> TaskResult:
                _ = (thinking_minutes, budget, persist_task_result, publish_history, emit_completion_event)
                result = _build_stub_task_result(
                    task_id=f"{session_id}_task",
                    question=question,
                    answer_text="verified export result" if verified else "non-verified export result",
                    candidate_count=2,
                    candidate_score=0.8,
                    critique_result=CritiqueResult.VALID if verified else CritiqueResult.DEGRADED,
                    local_evidence_ids=("ev-1",),
                )
                if verified:
                    result = replace(
                        result,
                        reasoning=replace(
                            result.reasoning,
                            context_frames=(
                                ContextFrame(
                                    frame_id="frame-deep",
                                    scope="task",
                                    confidence=0.9,
                                    metadata={"rm": "deep"},
                                ),
                            ),
                        ),
                    )
                return result

            with (
                mock.patch.object(self.orchestrator, "_long_horizon_session_id", return_value=session_id),
                mock.patch.object(self.orchestrator, "_run_bounded_task", side_effect=fake_run_bounded_task),
            ):
                return await self.orchestrator.run_task(question, thinking_minutes=121)

        await run_with_result(valid_session_id, verified=True)
        await run_with_result(invalid_session_id, verified=False)

        valid_bundle = await self.orchestrator.storage.export_long_horizon_session_bundle(
            valid_session_id,
            export_dir=self.test_logs / "export_verified",
        )
        invalid_bundle = await self.orchestrator.storage.export_long_horizon_session_bundle(
            invalid_session_id,
            export_dir=self.test_logs / "export_unverified",
        )

        self.assertTrue(Path(valid_bundle.session_path).exists())
        self.assertTrue(Path(valid_bundle.checkpoints_path).exists())
        self.assertTrue(Path(valid_bundle.verified_trace_export_path).exists())
        self.assertTrue(Path(invalid_bundle.session_path).exists())
        self.assertTrue(Path(invalid_bundle.checkpoints_path).exists())
        self.assertEqual(invalid_bundle.verified_trace_export_path, "")


class Phase17DashboardTimeControlTests(unittest.TestCase):
    def test_time_presets_switch_between_interactive_and_long_horizon_modes(self) -> None:
        dashboard = DashboardService(config=_build_test_config(sqlite_name="preset.sqlite3", logs_name="preset_logs"))
        dashboard._thinking_minutes_var = _Var(30)
        dashboard._thinking_label_var = _Var("30 minutes")
        dashboard._long_horizon_enabled_var = _Var(False)
        dashboard._long_horizon_minutes_var = _Var("120")
        dashboard._time_summary_var = _Var("")
        dashboard._run_status_var = _Var("Ready.")

        dashboard._refresh_time_control_summary()
        self.assertIn("Interactive", dashboard._time_summary_var.get())

        dashboard._on_apply_time_preset(121)
        self.assertEqual(dashboard._thinking_minutes_var.get(), 121)
        self.assertTrue(dashboard._long_horizon_enabled_var.get())
        self.assertEqual(dashboard._long_horizon_minutes_var.get(), "121")
        self.assertIn("long-horizon", dashboard._run_status_var.get())
        self.assertIn("Long-horizon", dashboard._time_summary_var.get())

        dashboard._on_apply_time_preset(30)
        self.assertEqual(dashboard._thinking_minutes_var.get(), 30)
        self.assertFalse(dashboard._long_horizon_enabled_var.get())
        self.assertEqual(dashboard._long_horizon_minutes_var.get(), "120")
        self.assertIn("Interactive", dashboard._time_summary_var.get())


if __name__ == "__main__":
    unittest.main()
