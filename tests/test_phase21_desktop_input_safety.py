"""Phase 21 desktop-input and session-safety regressions."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from capability_runtime import CapabilityExecutor, WindowSnapshot
from data_structures import CapabilityRequest, CapabilityType, DesktopInputSpec, FileOperationSpec, LocalTaskSessionState
from orchestrator import Orchestrator
from tests.test_phase20_capability_foundation import _build_test_config, _desktop_profile


class Phase21DesktopInputSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase21_desktop_input.sqlite3")
        self.test_logs = Path("test_phase21_desktop_input_logs")
        self.config = _build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()
        self.workspace = (self.test_logs / "workspace").resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_loop_guard_pauses_session_after_repeated_identical_requests(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("file_operation",),
            allowlisted_roots=(str(self.workspace),),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        session = await self.orchestrator.start_local_task_session(
            "Loop guard",
            active_profile=profile,
        )
        target_path = self.workspace / "loop.txt"
        target_path.write_text("loop guard", encoding="utf-8")

        last_result = None
        for index in range(4):
            last_result = await self.orchestrator.run_capability_request(
                CapabilityRequest(
                    request_id=f"cap-loop-{index}",
                    capability_type=CapabilityType.FILE_OPERATION,
                    summary="Read the same file repeatedly",
                    file_operation=FileOperationSpec(operation="read", source_path=str(target_path)),
                )
            )

        paused_session = await self.orchestrator.storage.load_local_task_session(session.session_id)

        assert last_result is not None
        self.assertEqual(last_result.status.value, "blocked")
        self.assertIn("loop_guard_triggered", last_result.detail)
        self.assertEqual(paused_session.status, LocalTaskSessionState.PAUSED)
        self.assertEqual(paused_session.last_control_reason, "loop_guard_triggered")
        self.assertGreaterEqual(paused_session.repeated_request_count, 4)

    async def test_desktop_input_runs_after_approval_with_validated_target(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("desktop_input",),
            allowlisted_apps=("notepad",),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        session = await self.orchestrator.start_local_task_session(
            "Desktop input",
            active_profile=profile,
        )
        matched_window = WindowSnapshot(
            hwnd=404,
            title="notes.txt - Notepad",
            process_name="notepad",
            pid=2222,
        )
        request = CapabilityRequest(
            request_id="cap-desktop-approved",
            capability_type=CapabilityType.DESKTOP_INPUT,
            summary="Type into Notepad",
            desktop_input=DesktopInputSpec(action="type_text", text="hello", target="notepad"),
            metadata={"expected_window_title": "notes.txt"},
        )
        with (
            patch.object(CapabilityExecutor, "_desktop_input_supported", return_value=True),
            patch.object(CapabilityExecutor, "_wait_for_window_match", return_value=matched_window),
            patch.object(CapabilityExecutor, "_focus_window", return_value=True),
            patch.object(CapabilityExecutor, "_foreground_window_title", return_value="notes.txt - Notepad"),
            patch.object(CapabilityExecutor, "_send_text_input") as send_text,
        ):
            pending_result = await self.orchestrator.run_capability_request(request)
            result = await self.orchestrator.run_capability_request(request, approval_granted=True)

        final_session = await self.orchestrator.storage.load_local_task_session(session.session_id)

        self.assertEqual(pending_result.status.value, "blocked")
        self.assertIn("approval_pending", pending_result.warnings)
        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(result.executor_kind, "live_input")
        self.assertEqual(send_text.call_count, 5)
        self.assertEqual(final_session.status, LocalTaskSessionState.RUNNING)
        self.assertEqual(final_session.last_request_id, "cap-desktop-approved")
        self.assertEqual(final_session.pending_approvals, ())

    async def test_desktop_input_focus_loss_pauses_session_for_recovery(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("desktop_input",),
            allowlisted_apps=("notepad",),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        session = await self.orchestrator.start_local_task_session(
            "Desktop input focus loss",
            active_profile=profile,
        )
        matched_window = WindowSnapshot(
            hwnd=505,
            title="notes.txt - Notepad",
            process_name="notepad",
            pid=3333,
        )
        with (
            patch.object(CapabilityExecutor, "_desktop_input_supported", return_value=True),
            patch.object(CapabilityExecutor, "_wait_for_window_match", return_value=matched_window),
            patch.object(CapabilityExecutor, "_focus_window", return_value=True),
            patch.object(
                CapabilityExecutor,
                "_foreground_window_title",
                side_effect=["notes.txt - Notepad", "Unexpected Window"],
            ),
            patch.object(CapabilityExecutor, "_send_text_input"),
        ):
            result = await self.orchestrator.run_capability_request(
                CapabilityRequest(
                    request_id="cap-desktop-focus-loss",
                    capability_type=CapabilityType.DESKTOP_INPUT,
                    summary="Type into Notepad with focus loss",
                    desktop_input=DesktopInputSpec(action="type_text", text="ok", target="notepad"),
                    metadata={"expected_window_title": "notes.txt"},
                ),
                approval_granted=True,
            )

        paused_session = await self.orchestrator.storage.load_local_task_session(session.session_id)

        self.assertEqual(result.status.value, "failed")
        self.assertIn("foreground_target_mismatch", result.warnings)
        self.assertEqual(paused_session.status, LocalTaskSessionState.PAUSED)
        self.assertEqual(paused_session.last_control_reason, "target_validation_failed")
        self.assertIn("foreground target changed unexpectedly", paused_session.last_error.lower())

    async def test_desktop_input_emergency_stop_pauses_session(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("desktop_input",),
            allowlisted_apps=("notepad",),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        session = await self.orchestrator.start_local_task_session(
            "Desktop input emergency stop",
            active_profile=profile,
        )
        matched_window = WindowSnapshot(
            hwnd=606,
            title="notes.txt - Notepad",
            process_name="notepad",
            pid=4444,
        )

        def trigger_stop(_character: str) -> None:
            self.orchestrator._local_task_emergency_stop.set()

        with (
            patch.object(CapabilityExecutor, "_desktop_input_supported", return_value=True),
            patch.object(CapabilityExecutor, "_wait_for_window_match", return_value=matched_window),
            patch.object(CapabilityExecutor, "_focus_window", return_value=True),
            patch.object(CapabilityExecutor, "_foreground_window_title", return_value="notes.txt - Notepad"),
            patch.object(CapabilityExecutor, "_send_text_input", side_effect=trigger_stop),
        ):
            result = await self.orchestrator.run_capability_request(
                CapabilityRequest(
                    request_id="cap-desktop-emergency-stop",
                    capability_type=CapabilityType.DESKTOP_INPUT,
                    summary="Type into Notepad but stop immediately",
                    desktop_input=DesktopInputSpec(action="type_text", text="halt", target="notepad"),
                    metadata={"expected_window_title": "notes.txt"},
                ),
                approval_granted=True,
            )

        paused_session = await self.orchestrator.storage.load_local_task_session(session.session_id)

        self.assertEqual(result.status.value, "failed")
        self.assertIn("emergency_stop_requested", result.warnings)
        self.assertEqual(paused_session.status, LocalTaskSessionState.PAUSED)
        self.assertEqual(paused_session.last_control_reason, "emergency_stop_requested")


if __name__ == "__main__":
    unittest.main()
