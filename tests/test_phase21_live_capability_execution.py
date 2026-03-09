"""Phase 21 live file and shell capability execution regressions."""

from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path

from config import APP_CONFIG
from data_structures import CapabilityRequest, CapabilityType, FileOperationSpec, ShellCommandSpec
from orchestrator import Orchestrator
from tests.test_phase20_capability_foundation import _build_test_config, _desktop_profile


class Phase21LiveCapabilityExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase21_live_capability.sqlite3")
        self.test_logs = Path("test_phase21_live_capability_logs")
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

    async def test_live_file_operations_cover_write_read_copy_move_archive_and_delete(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("file_operation",),
            allowlisted_roots=(str(self.workspace),),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        session = await self.orchestrator.start_local_task_session(
            "Live file execution",
            active_profile=profile,
        )
        source_path = self.workspace / "note.txt"
        copy_path = self.workspace / "note-copy.txt"
        moved_path = self.workspace / "moved" / "note-copy.txt"
        archived_path = self.workspace / "archive" / "note-copy.txt"

        write_result = await self.orchestrator.run_capability_request(
            CapabilityRequest(
                request_id="cap-live-file-write",
                capability_type=CapabilityType.FILE_OPERATION,
                summary="Write a bounded local file",
                file_operation=FileOperationSpec(operation="write", source_path=str(source_path)),
                metadata={"content": "phase 21 live file path\n"},
            )
        )
        read_result = await self.orchestrator.run_capability_request(
            CapabilityRequest(
                request_id="cap-live-file-read",
                capability_type=CapabilityType.FILE_OPERATION,
                summary="Read the bounded local file",
                file_operation=FileOperationSpec(operation="read", source_path=str(source_path)),
            )
        )
        copy_result = await self.orchestrator.run_capability_request(
            CapabilityRequest(
                request_id="cap-live-file-copy",
                capability_type=CapabilityType.FILE_OPERATION,
                summary="Copy the local file",
                file_operation=FileOperationSpec(
                    operation="copy",
                    source_path=str(source_path),
                    destination_path=str(copy_path),
                ),
            )
        )
        move_result = await self.orchestrator.run_capability_request(
            CapabilityRequest(
                request_id="cap-live-file-move",
                capability_type=CapabilityType.FILE_OPERATION,
                summary="Move the copied local file",
                file_operation=FileOperationSpec(
                    operation="move",
                    source_path=str(copy_path),
                    destination_path=str(moved_path),
                ),
            )
        )
        archive_result = await self.orchestrator.run_capability_request(
            CapabilityRequest(
                request_id="cap-live-file-archive",
                capability_type=CapabilityType.FILE_OPERATION,
                summary="Archive the moved local file",
                file_operation=FileOperationSpec(
                    operation="archive",
                    source_path=str(moved_path),
                    destination_path=str(archived_path),
                ),
            ),
            approval_granted=True,
        )
        delete_result = await self.orchestrator.run_capability_request(
            CapabilityRequest(
                request_id="cap-live-file-delete",
                capability_type=CapabilityType.FILE_OPERATION,
                summary="Delete the archived local file",
                file_operation=FileOperationSpec(operation="delete", source_path=str(archived_path)),
                destructive=True,
            ),
            approval_granted=True,
        )

        final_session = await self.orchestrator.storage.load_local_task_session(session.session_id)

        self.assertEqual(write_result.status.value, "succeeded")
        self.assertEqual(write_result.executor_kind, "live_file")
        self.assertTrue(source_path.exists())
        self.assertEqual(source_path.read_text(encoding="utf-8"), "phase 21 live file path\n")
        self.assertEqual(read_result.status.value, "succeeded")
        self.assertIn("phase 21 live file path", read_result.metadata["content_preview"])
        self.assertEqual(copy_result.status.value, "succeeded")
        self.assertEqual(Path(copy_result.output_ref), copy_path)
        self.assertEqual(move_result.status.value, "succeeded")
        self.assertEqual(Path(move_result.output_ref), moved_path)
        self.assertFalse(copy_path.exists())
        self.assertEqual(archive_result.status.value, "succeeded")
        self.assertEqual(Path(archive_result.output_ref), archived_path)
        self.assertFalse(moved_path.exists())
        self.assertEqual(delete_result.status.value, "succeeded")
        self.assertEqual(Path(delete_result.output_ref), archived_path)
        self.assertFalse(archived_path.exists())
        self.assertEqual(final_session.last_request_id, "cap-live-file-delete")
        self.assertIn("Deleted", final_session.last_action_summary)

    async def test_live_shell_command_runs_inside_allowlisted_workspace(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("shell_command",),
            allowlisted_roots=(str(self.workspace),),
            allowlisted_shell_commands=("python",),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        session = await self.orchestrator.start_local_task_session(
            "Live shell execution",
            active_profile=profile,
        )

        result = await self.orchestrator.run_capability_request(
            CapabilityRequest(
                request_id="cap-live-shell-success",
                capability_type=CapabilityType.SHELL_COMMAND,
                summary="Run a bounded python command",
                shell_command=ShellCommandSpec(
                    command="python",
                    args=("-c", "print('live shell ok')"),
                    working_directory=str(self.workspace),
                ),
            )
        )

        final_session = await self.orchestrator.storage.load_local_task_session(session.session_id)

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(result.executor_kind, "live_shell")
        self.assertEqual(result.metadata["exit_code"], 0)
        self.assertIn("live shell ok", result.metadata["stdout_preview"])
        self.assertEqual(final_session.last_request_id, "cap-live-shell-success")
        self.assertIn("completed successfully", final_session.last_action_summary)

    async def test_live_shell_timeout_returns_failed_result_and_updates_session_error(self) -> None:
        timeout_db = Path("test_phase21_live_capability_timeout.sqlite3")
        timeout_logs = Path("test_phase21_live_capability_timeout_logs")
        timeout_workspace = (timeout_logs / "workspace").resolve()
        timeout_workspace.mkdir(parents=True, exist_ok=True)
        timeout_config = replace(
            _build_test_config(sqlite_name=str(timeout_db), logs_name=str(timeout_logs)),
            backend_runtime=replace(APP_CONFIG.backend_runtime, request_timeout_s=0.2),
        )
        timeout_orchestrator = Orchestrator(config=timeout_config)
        await timeout_orchestrator.start()
        try:
            profile = _desktop_profile(
                enabled_capabilities=("shell_command",),
                allowlisted_roots=(str(timeout_workspace),),
                allowlisted_shell_commands=("python",),
            )
            await timeout_orchestrator.storage.save_user_settings_profile(profile)
            await timeout_orchestrator._apply_runtime_settings_profile(profile)
            session = await timeout_orchestrator.start_local_task_session(
                "Timed shell execution",
                active_profile=profile,
            )

            result = await timeout_orchestrator.run_capability_request(
                CapabilityRequest(
                    request_id="cap-live-shell-timeout",
                    capability_type=CapabilityType.SHELL_COMMAND,
                    summary="Sleep past the bounded shell timeout",
                    shell_command=ShellCommandSpec(
                        command="python",
                        args=("-c", "import time; time.sleep(1.0)"),
                        working_directory=str(timeout_workspace),
                    ),
                )
            )

            final_session = await timeout_orchestrator.storage.load_local_task_session(session.session_id)

            self.assertEqual(result.status.value, "failed")
            self.assertIn("shell_timeout", result.warnings)
            self.assertIn("timeout", result.detail.lower())
            self.assertEqual(final_session.last_request_id, "cap-live-shell-timeout")
            self.assertIn("timeout", final_session.last_error.lower())
        finally:
            await timeout_orchestrator.stop()
            if timeout_db.exists():
                timeout_db.unlink()
            if timeout_logs.exists():
                shutil.rmtree(timeout_logs)


if __name__ == "__main__":
    unittest.main()
