"""Phase 21 explicit local task session lifecycle and gating regressions."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from capability_runtime import CapabilityExecutor, WindowSnapshot
from data_structures import (
    CapabilityRequest,
    CapabilityType,
    DesktopInputSpec,
    FileOperationSpec,
    LocalTaskSessionState,
)
from orchestrator import Orchestrator
from tests.test_phase20_capability_foundation import _build_test_config, _desktop_profile


class Phase21LocalTaskSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase21_local_task_sessions.sqlite3")
        self.test_logs = Path("test_phase21_local_task_sessions_logs")
        self.config = _build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_session_lifecycle_updates_storage_and_dashboard(self) -> None:
        profile = _desktop_profile(enabled_capabilities=("file_operation",))
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)

        session = await self.orchestrator.start_local_task_session(
            "Review workspace files",
            active_profile=profile,
        )

        persisted = await self.orchestrator.storage.load_local_task_session(session.session_id)
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.status, LocalTaskSessionState.RUNNING)
        self.assertEqual(self.orchestrator.dashboard.app_state_snapshot().local_task_session.session_id, session.session_id)

        self.assertTrue(await self.orchestrator.pause_local_task_session(session.session_id, reason="test_pause"))
        paused = await self.orchestrator.storage.load_local_task_session(session.session_id)
        self.assertEqual(paused.status, LocalTaskSessionState.PAUSED)
        self.assertEqual(self.orchestrator.dashboard.app_state_snapshot().local_task_session.status, "paused")

        self.assertTrue(await self.orchestrator.resume_local_task_session(session.session_id, reason="test_resume"))
        resumed = await self.orchestrator.storage.load_local_task_session(session.session_id)
        self.assertEqual(resumed.status, LocalTaskSessionState.RUNNING)
        self.assertEqual(self.orchestrator.dashboard.app_state_snapshot().local_task_session.status, "running")

        self.assertTrue(await self.orchestrator.stop_local_task_session(session.session_id, reason="test_stop"))
        stopped = await self.orchestrator.storage.load_local_task_session(session.session_id)
        self.assertEqual(stopped.status, LocalTaskSessionState.STOPPED)
        self.assertIsNone(await self.orchestrator.storage.load_active_local_task_session())
        self.assertEqual(self.orchestrator.dashboard.app_state_snapshot().local_task_session.status, "stopped")

    async def test_capability_request_requires_explicit_local_session(self) -> None:
        profile = _desktop_profile(enabled_capabilities=("file_operation",))
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        request = CapabilityRequest(
            request_id="cap-no-session",
            capability_type=CapabilityType.FILE_OPERATION,
            summary="Read README without a session",
            file_operation=FileOperationSpec(operation="read", source_path="README.md"),
        )

        result = await self.orchestrator.run_capability_request(request)
        audits = await self.orchestrator.storage.list_capability_audits(request_id="cap-no-session")
        registry = await self.orchestrator.storage.load_capability_registry_view()

        self.assertEqual(result.status.value, "blocked")
        self.assertEqual(audits[1].reason_codes, ("session_not_active",))
        self.assertEqual(registry.recent_decisions[-1].reason_codes, ("session_not_active",))

    async def test_pending_approval_is_queued_until_approval_and_kill_switch_ends_session(self) -> None:
        profile = _desktop_profile(enabled_capabilities=("desktop_input", "file_operation"))
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        session = await self.orchestrator.start_local_task_session(
            "Desktop validation",
            active_profile=profile,
        )
        approval_request = CapabilityRequest(
            request_id="cap-approval-pending",
            capability_type=CapabilityType.DESKTOP_INPUT,
            summary="Type into Notepad",
            desktop_input=DesktopInputSpec(action="type_text", text="hello", target="notepad"),
        )

        pending_result = await self.orchestrator.run_capability_request(approval_request)

        pending_session = await self.orchestrator.storage.load_local_task_session(session.session_id)
        self.assertEqual(pending_result.status.value, "blocked")
        self.assertEqual(len(pending_session.pending_approvals), 1)
        self.assertEqual(pending_session.pending_approvals[0].request_id, approval_request.request_id)
        self.assertTrue(self.orchestrator.dashboard.app_state_snapshot().local_task_session.pending_approval_summaries)

        matched_window = WindowSnapshot(
            hwnd=707,
            title="Desktop validation - Notepad",
            process_name="notepad",
            pid=1111,
        )
        with (
            patch.object(CapabilityExecutor, "_wait_for_window_match", return_value=matched_window),
            patch.object(CapabilityExecutor, "_focus_window", return_value=True),
            patch.object(CapabilityExecutor, "_foreground_window_title", return_value="Desktop validation - Notepad"),
            patch.object(CapabilityExecutor, "_send_text_input"),
        ):
            granted_result = await self.orchestrator.run_capability_request(approval_request, approval_granted=True)

        cleared_session = await self.orchestrator.storage.load_local_task_session(session.session_id)
        self.assertEqual(granted_result.status.value, "succeeded")
        self.assertEqual(cleared_session.pending_approvals, ())

        self.assertTrue(await self.orchestrator.kill_local_task_session(session.session_id, reason="test_kill"))
        killed_session = await self.orchestrator.storage.load_local_task_session(session.session_id)
        post_kill_request = CapabilityRequest(
            request_id="cap-after-kill",
            capability_type=CapabilityType.FILE_OPERATION,
            summary="Read README after kill-switch",
            file_operation=FileOperationSpec(operation="read", source_path="README.md"),
        )

        blocked_after_kill = await self.orchestrator.run_capability_request(post_kill_request)
        audits = await self.orchestrator.storage.list_capability_audits(request_id="cap-after-kill")

        self.assertEqual(killed_session.status, LocalTaskSessionState.KILLED)
        self.assertTrue(self.orchestrator.dashboard.app_state_snapshot().local_task_session.kill_switch_engaged)
        self.assertEqual(blocked_after_kill.status.value, "blocked")
        self.assertIn(audits[1].reason_codes[0], {"session_not_active", "session_not_running", "kill_switch_engaged"})


if __name__ == "__main__":
    unittest.main()
