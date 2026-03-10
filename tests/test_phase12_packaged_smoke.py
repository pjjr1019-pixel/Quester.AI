"""Phase 12 packaged-app smoke tests for launch fallback and support bundles."""

from __future__ import annotations

import json
import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from config import APP_CONFIG
from data_structures import UserSettingsProfile
from model_manager import ModelHealthSnapshot
from orchestrator import Orchestrator, run_packaged_dashboard_app
from storage import StorageManager


def _build_test_config(
    *,
    stub_mode: bool,
    sqlite_name: str,
    logs_name: str,
    bundle_name: str,
    models_dir: str,
    generation_backend: str = "ollama",
    generation_fallback_backend: str = "llama_cpp",
    embedding_backend: str = "sentence_transformers",
):
    backends = replace(
        APP_CONFIG.preflight.backends,
        generation_backend=generation_backend,
        generation_fallback_backend=generation_fallback_backend,
        generation_fallback_model="missing-fallback.gguf",
        embedding_backend=embedding_backend,
        vector_store_backend="simple_inmemory",
        vector_store_fallback_backend="simple_inmemory",
    )
    preflight = replace(
        APP_CONFIG.preflight,
        backends=backends,
        flags=replace(
            APP_CONFIG.preflight.flags,
            stub_mode=stub_mode,
            enable_self_optimizer=False,
            allow_web_fallback=True,
        ),
    )
    storage_cfg = replace(APP_CONFIG.storage, sqlite_path=Path(sqlite_name), logs_dir=Path(logs_name))
    dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
    backend_runtime = replace(APP_CONFIG.backend_runtime, models_dir=Path(models_dir))
    return (
        replace(
            APP_CONFIG,
            preflight=preflight,
            storage=storage_cfg,
            dashboard=dashboard,
            backend_runtime=backend_runtime,
        ),
        Path(bundle_name),
    )


def _snapshot(
    *,
    started: bool,
    generation_backend: str,
    embedding_backend: str,
    fallback_active: bool = False,
    fallback_reason: str | None = None,
) -> ModelHealthSnapshot:
    return ModelHealthSnapshot(
        started=started,
        generation_backend=generation_backend,
        embedding_backend=embedding_backend,
        active_generation_jobs=0,
        active_embedding_jobs=0,
        last_used_at=None,
        fallback_active=fallback_active,
        fallback_reason=fallback_reason,
        available_ram_gb=4.0,
        total_ram_gb=8.0,
        generation_backend_vram_gb=0.0,
        embedding_backend_vram_gb=0.0,
        telemetry_enabled=False,
        last_error=None,
    )


class Phase12PackagedSmokeTests(unittest.IsolatedAsyncioTestCase):
    """Validate the packaged entrypoint, stub fallback, and support-bundle export."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase12_packaged.sqlite3")
        self.test_logs = Path("test_phase12_packaged_logs")
        self.test_bundle = Path("test_phase12_packaged_bundle")
        self.models_dir = Path("test_phase12_packaged_models")

    async def asyncTearDown(self) -> None:
        if self.test_db.exists():
            self.test_db.unlink()
        for path in (self.test_logs, self.test_bundle, self.models_dir):
            if path.exists():
                shutil.rmtree(path)

    async def test_packaged_headless_stub_launch_runs_first_task_and_exports_bundle(self) -> None:
        config, bundle_dir = _build_test_config(
            stub_mode=True,
            sqlite_name=str(self.test_db),
            logs_name=str(self.test_logs),
            bundle_name=str(self.test_bundle),
            models_dir=str(self.models_dir),
        )

        result = await run_packaged_dashboard_app(config=config, support_bundle_dir=bundle_dir)

        assert result is not None
        self.assertEqual(result.plan.question, "What should I build first?")
        self.assertTrue((bundle_dir / "support_bundle_manifest.json").exists())
        self.assertTrue((bundle_dir / "launch_report.json").exists())
        self.assertTrue((bundle_dir / "readiness_report.json").exists())
        self.assertTrue((bundle_dir / "packaged_preflight_report.json").exists())
        self.assertTrue((bundle_dir / "packaged_onboarding.txt").exists())
        self.assertTrue((bundle_dir / "LOCAL_MODEL_SETUP.md").exists())
        self.assertTrue((bundle_dir / "user_settings_profile.json").exists())
        self.assertTrue((bundle_dir / "app_state.json").exists())
        self.assertTrue((bundle_dir / "support_bundle.txt").exists())
        self.assertTrue((bundle_dir / "events.jsonl").exists())
        self.assertTrue((bundle_dir / "traces.jsonl").exists())
        self.assertTrue((bundle_dir / "status.jsonl").exists())
        launch_report = json.loads((bundle_dir / "launch_report.json").read_text(encoding="utf-8"))
        preflight_report = json.loads((bundle_dir / "packaged_preflight_report.json").read_text(encoding="utf-8"))
        manifest = json.loads((bundle_dir / "support_bundle_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(launch_report["requested_mode"], "stub")
        self.assertEqual(launch_report["effective_mode"], "stub")
        self.assertTrue(launch_report["launch_ready"])
        self.assertFalse(launch_report["used_stub_fallback"])
        self.assertEqual(
            preflight_report["default_model_bundle"]["generation"],
            "ollama:qwen2.5:3b-instruct-q4_K_M",
        )
        self.assertTrue(str(preflight_report["setup_guide_path"]).endswith("LOCAL_MODEL_SETUP.md"))
        self.assertEqual(Path(manifest["bundle_dir"]).name, bundle_dir.name)
        self.assertTrue(str(manifest["preflight_report_path"]).endswith("packaged_preflight_report.json"))
        self.assertGreater((bundle_dir / "events.jsonl").stat().st_size, 0)
        self.assertGreater((bundle_dir / "traces.jsonl").stat().st_size, 0)
        self.assertGreater((bundle_dir / "status.jsonl").stat().st_size, 0)
        onboarding_text = (bundle_dir / "packaged_onboarding.txt").read_text(encoding="utf-8")
        self.assertIn("PySide6 orb cockpit", onboarding_text)
        self.assertIn("Coding Mode remains local-first", onboarding_text)

    async def test_packaged_first_run_uses_stub_mode_even_when_real_mode_is_requested_in_config(self) -> None:
        config, bundle_dir = _build_test_config(
            stub_mode=False,
            sqlite_name=str(self.test_db),
            logs_name=str(self.test_logs),
            bundle_name=str(self.test_bundle),
            models_dir=str(self.models_dir),
            generation_backend="ollama",
            generation_fallback_backend="llama_cpp",
            embedding_backend="sentence_transformers",
        )

        result = await run_packaged_dashboard_app(config=config, support_bundle_dir=bundle_dir)

        assert result is not None
        launch_report = json.loads((bundle_dir / "launch_report.json").read_text(encoding="utf-8"))
        app_state = json.loads((bundle_dir / "app_state.json").read_text(encoding="utf-8"))

        self.assertEqual(launch_report["requested_mode"], "stub")
        self.assertEqual(launch_report["effective_mode"], "stub")
        self.assertTrue(launch_report["launch_ready"])
        self.assertFalse(launch_report["used_stub_fallback"])
        self.assertEqual(launch_report["blocking_reason"], "")
        self.assertIn("first-run local validation", launch_report["summary"])
        self.assertTrue(app_state["user_settings"]["runtime"]["stub_mode"])
        self.assertIn("first-run validation", app_state["last_notice"])

    async def test_packaged_launch_falls_back_to_stub_for_saved_real_profile_and_records_readable_failure(self) -> None:
        config, bundle_dir = _build_test_config(
            stub_mode=False,
            sqlite_name=str(self.test_db),
            logs_name=str(self.test_logs),
            bundle_name=str(self.test_bundle),
            models_dir=str(self.models_dir),
            generation_backend="ollama",
            generation_fallback_backend="llama_cpp",
            embedding_backend="sentence_transformers",
        )
        storage = StorageManager(config=config)
        await storage.start()
        try:
            await storage.save_user_settings_profile(
                UserSettingsProfile(
                    profile_name="default",
                    runtime={
                        "stub_mode": False,
                        "allow_web_fallback": True,
                        "enable_self_optimizer": False,
                        "generation_backend": "ollama",
                        "embedding_backend": "sentence_transformers",
                        "vector_store_backend": "simple_inmemory",
                    },
                )
            )
        finally:
            await storage.stop()

        dependency_map = {
            "chromadb": False,
            "sentence_transformers": True,
            "llama_cpp": True,
        }
        with patch.object(
            Orchestrator,
            "_dependency_available",
            autospec=True,
            side_effect=lambda _self, module_name: dependency_map.get(module_name, True),
        ):
            with patch.object(
                Orchestrator,
                "_probe_ollama_service",
                autospec=True,
                return_value=(False, "Ollama service probe failed at http://localhost:11434/api/tags: refused"),
            ):
                result = await run_packaged_dashboard_app(config=config, support_bundle_dir=bundle_dir)

        assert result is not None
        launch_report = json.loads((bundle_dir / "launch_report.json").read_text(encoding="utf-8"))
        app_state = json.loads((bundle_dir / "app_state.json").read_text(encoding="utf-8"))
        support_readme = (bundle_dir / "support_bundle.txt").read_text(encoding="utf-8")

        self.assertEqual(launch_report["requested_mode"], "real")
        self.assertEqual(launch_report["effective_mode"], "stub")
        self.assertTrue(launch_report["launch_ready"])
        self.assertTrue(launch_report["used_stub_fallback"])
        self.assertEqual(launch_report["blocking_reason"], "missing_ollama_service")
        self.assertIn("probe failed", launch_report["blocking_detail"])
        self.assertIn("fall back to stub mode", launch_report["summary"])
        self.assertTrue(app_state["user_settings"]["runtime"]["stub_mode"])
        self.assertIn("missing_ollama_service", support_readme)
        self.assertIn("fall back to stub mode", app_state["last_notice"])

    async def test_packaged_launch_report_can_stay_in_real_mode_when_readiness_is_satisfied(self) -> None:
        config, _bundle_dir = _build_test_config(
            stub_mode=False,
            sqlite_name=str(self.test_db),
            logs_name=str(self.test_logs),
            bundle_name=str(self.test_bundle),
            models_dir=str(self.models_dir),
            generation_backend="ollama",
            generation_fallback_backend="llama_cpp",
            embedding_backend="sentence_transformers",
        )
        orchestrator = Orchestrator(config=config)

        with patch.object(orchestrator, "_dependency_available", return_value=True):
            with patch.object(orchestrator, "_probe_ollama_service", return_value=(True, "reachable")):
                orchestrator.model_manager.health_snapshot = lambda: _snapshot(
                    started=True,
                    generation_backend="ollama",
                    embedding_backend="sentence_transformers",
                )
                launch_report = orchestrator.build_packaged_launch_report()

        self.assertEqual(launch_report.requested_mode, "real")
        self.assertEqual(launch_report.effective_mode, "real")
        self.assertTrue(launch_report.launch_ready)
        self.assertFalse(launch_report.used_stub_fallback)
        self.assertEqual(launch_report.blocking_reason, "")
        self.assertIn("ready for real mode", launch_report.summary.lower())


if __name__ == "__main__":
    unittest.main()
