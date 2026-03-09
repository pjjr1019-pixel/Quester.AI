"""Phase 21 browser and app-window control regressions."""

from __future__ import annotations

import shutil
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from capability_runtime import CapabilityExecutor, WindowSnapshot
from data_structures import AppFocusSpec, BrowserActionSpec, CapabilityRequest, CapabilityType
from orchestrator import Orchestrator
from tests.test_phase20_capability_foundation import _build_test_config, _desktop_profile


class _BrowserReadHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        payload = (
            "<html><head><title>Quester Browser Read</title></head>"
            "<body><h1>browser read ok</h1></body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class Phase21BrowserAppControlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase21_browser_app.sqlite3")
        self.test_logs = Path("test_phase21_browser_app_logs")
        self.config = _build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_live_browser_read_fetches_allowlisted_local_content(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("browser_action",),
            allowlisted_browser_domains=("127.0.0.1",),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        await self.orchestrator.start_local_task_session(
            "Browser read",
            active_profile=profile,
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), _BrowserReadHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/"
            result = await self.orchestrator.run_capability_request(
                CapabilityRequest(
                    request_id="cap-live-browser-read",
                    capability_type=CapabilityType.BROWSER_ACTION,
                    summary="Read a local browser target",
                    browser_action=BrowserActionSpec(action="read", url=url, domain="127.0.0.1"),
                    metadata={"expected_title": "Quester Browser Read"},
                )
            )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2.0)

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(result.executor_kind, "live_browser")
        self.assertEqual(result.metadata["page_title"], "Quester Browser Read")
        self.assertIn("browser read ok", result.metadata["content_preview"])

    async def test_live_browser_navigate_validates_visible_window(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("browser_action",),
            allowlisted_browser_domains=("localhost",),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        session = await self.orchestrator.start_local_task_session(
            "Browser navigate",
            active_profile=profile,
        )
        matched_window = WindowSnapshot(
            hwnd=101,
            title="Quester Browser Test - Edge",
            process_name="msedge",
            pid=4321,
        )
        with (
            patch.object(CapabilityExecutor, "_open_browser_url") as open_browser,
            patch.object(
                CapabilityExecutor,
                "_wait_for_window_match",
                return_value=matched_window,
            ),
            patch.object(CapabilityExecutor, "_focus_window", return_value=True),
            patch.object(
                CapabilityExecutor,
                "_foreground_window_title",
                return_value="Quester Browser Test - Edge",
            ),
        ):
            result = await self.orchestrator.run_capability_request(
                CapabilityRequest(
                    request_id="cap-live-browser-navigate",
                    capability_type=CapabilityType.BROWSER_ACTION,
                    summary="Open a local browser target",
                    browser_action=BrowserActionSpec(
                        action="navigate",
                        url="http://localhost:8080/",
                        domain="localhost",
                    ),
                    metadata={"expected_title": "Quester Browser Test"},
                )
            )

        final_session = await self.orchestrator.storage.load_local_task_session(session.session_id)

        open_browser.assert_called_once_with("http://localhost:8080/")
        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(result.executor_kind, "live_browser")
        self.assertEqual(result.metadata["matched_process_name"], "msedge")
        self.assertEqual(final_session.last_request_id, "cap-live-browser-navigate")

    async def test_live_app_focus_validates_foreground_window_title(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("app_window_focus",),
            allowlisted_apps=("notepad",),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        session = await self.orchestrator.start_local_task_session(
            "App focus",
            active_profile=profile,
        )
        matched_window = WindowSnapshot(
            hwnd=202,
            title="notes.txt - Notepad",
            process_name="notepad",
            pid=9876,
        )
        with (
            patch.object(
                CapabilityExecutor,
                "_wait_for_window_match",
                return_value=matched_window,
            ),
            patch.object(CapabilityExecutor, "_focus_window", return_value=True),
            patch.object(
                CapabilityExecutor,
                "_foreground_window_title",
                return_value="notes.txt - Notepad",
            ),
        ):
            result = await self.orchestrator.run_capability_request(
                CapabilityRequest(
                    request_id="cap-live-app-focus",
                    capability_type=CapabilityType.APP_WINDOW_FOCUS,
                    summary="Focus the allowlisted editor",
                    app_focus=AppFocusSpec(
                        app_name="notepad",
                        window_title="notes.txt",
                        require_visible_match=True,
                    ),
                )
            )

        final_session = await self.orchestrator.storage.load_local_task_session(session.session_id)

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(result.executor_kind, "live_window")
        self.assertEqual(result.metadata["matched_process_name"], "notepad")
        self.assertEqual(final_session.last_request_id, "cap-live-app-focus")
        self.assertIn("Focused", final_session.last_action_summary)


if __name__ == "__main__":
    unittest.main()
