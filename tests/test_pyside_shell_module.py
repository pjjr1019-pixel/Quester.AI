"""Dependency-gated regressions for the optional PySide6 shell runtime."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from data_structures import DashboardAppState, ShellState
from pyside_shell import (
    PySideShellHost,
    PySideShellUnavailableError,
    PySideShellWindow,
    pyside6_available,
)


class PySideShellModuleTests(unittest.TestCase):
    def test_pyside6_dependency_is_available_in_the_active_desktop_env(self) -> None:
        self.assertTrue(pyside6_available())

    def test_shell_window_renders_live_shell_state_and_accepts_submission_callbacks(self) -> None:
        if not pyside6_available():
            self.skipTest("PySide6 is not installed in this environment.")

        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        submissions: list[tuple[str, int]] = []
        window = PySideShellWindow(
            shell_state=ShellState(
                status_text="Deep Thought",
                sub_status_text="Ranking options",
                current_task_summary="Analyze multiple candidates.",
            ),
            app_state=DashboardAppState(),
            submit_task=lambda question, minutes: submissions.append((question, minutes)) or True,
        )
        window._input.setText("Test the shell")
        window._slider.setValue(120)
        window._submit_from_input()

        self.assertEqual(window.windowTitle(), "Quester.AI")
        self.assertEqual(window._status.text(), "Deep Thought")
        self.assertEqual(submissions, [("Test the shell", 120)])
        window.close()
        app.quit()
        app.processEvents()

    def test_shell_host_starts_and_stops_cleanly(self) -> None:
        if not pyside6_available():
            with self.assertRaises(PySideShellUnavailableError):
                PySideShellHost(
                    shell_state_provider=ShellState,
                    app_state_provider=DashboardAppState,
                )
            return

        host = PySideShellHost(
            shell_state_provider=ShellState,
            app_state_provider=DashboardAppState,
            startup_timeout_s=5.0,
        )
        host.start()
        self.assertTrue(host.is_running)
        host.stop(timeout_s=5.0)
        self.assertFalse(host.is_running)
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.quit()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
