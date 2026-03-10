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


def _cleanup_qt() -> None:
    if not pyside6_available():
        return
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    app.closeAllWindows()
    QtCore.QCoreApplication.sendPostedEvents(None, 0)
    app.processEvents()
    app.quit()
    QtCore.QCoreApplication.sendPostedEvents(None, 0)
    app.processEvents()


class PySideShellModuleTests(unittest.TestCase):
    def tearDown(self) -> None:
        _cleanup_qt()

    def test_pyside6_availability_flag_is_boolean(self) -> None:
        self.assertIsInstance(pyside6_available(), bool)

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
                hero_metric_strip=("Candidates 3", "Evidence 6", "Verifier verified"),
                active_route_summary=("generation:qwen", "embedding:e5-small"),
                current_project="sandbox",
                current_file="sandbox:solution.py",
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
        self.assertIn("Candidates 3", window._hero_metrics.text())
        self.assertIn("generation:qwen", window._route_summary.text())
        self.assertEqual(submissions, [("Test the shell", 120)])
        window.close()
        _cleanup_qt()

    def test_shell_host_starts_and_stops_cleanly(self) -> None:
        if not pyside6_available():
            with self.assertRaises(PySideShellUnavailableError):
                PySideShellHost(
                    shell_state_provider=ShellState,
                    app_state_provider=DashboardAppState,
                )
            return

        from PySide6 import QtWidgets

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        host = PySideShellHost(
            shell_state_provider=ShellState,
            app_state_provider=DashboardAppState,
            startup_timeout_s=5.0,
        )
        host.start()
        self.assertTrue(host.is_running)
        host.stop(timeout_s=5.0)
        self.assertFalse(host.is_running)
        _cleanup_qt()


if __name__ == "__main__":
    unittest.main()
