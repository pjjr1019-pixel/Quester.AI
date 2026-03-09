"""Phase 24 packaged-startup contract regressions."""

from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from config import APP_CONFIG
from data_structures import UserSettingsProfile
from orchestrator import Orchestrator, run_dashboard_app, run_packaged_dashboard_app
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


class Phase24PackagedStartupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase24_packaged.sqlite3")
        self.test_logs = Path("test_phase24_packaged_logs")
        self.test_bundle = Path("test_phase24_packaged_bundle")
        self.models_dir = Path("test_phase24_packaged_models")

    async def asyncTearDown(self) -> None:
        if self.test_db.exists():
            self.test_db.unlink()
        for path in (self.test_logs, self.test_bundle, self.models_dir):
            if path.exists():
                shutil.rmtree(path)

    async def test_source_runner_does_not_use_packaged_startup_loading(self) -> None:
        config, _bundle_dir = _build_test_config(
            stub_mode=True,
            sqlite_name=str(self.test_db),
            logs_name=str(self.test_logs),
            bundle_name=str(self.test_bundle),
            models_dir=str(self.models_dir),
        )

        with patch.object(Orchestrator, "build_packaged_startup_plan", side_effect=AssertionError("packaged only")):
            result = await run_dashboard_app(config=config)

        assert result is not None
        self.assertEqual(result.plan.question, "What should I build first?")

    def test_packaged_startup_plan_forces_first_run_into_stub_and_reuses_readiness_contract(self) -> None:
        config, _bundle_dir = _build_test_config(
            stub_mode=False,
            sqlite_name=str(self.test_db),
            logs_name=str(self.test_logs),
            bundle_name=str(self.test_bundle),
            models_dir=str(self.models_dir),
        )
        orchestrator = Orchestrator(config=config)

        plan = orchestrator.build_packaged_startup_plan(startup_profile=None)
        direct_readiness = orchestrator._build_dashboard_readiness_report(active_profile=plan.requested_profile)

        self.assertTrue(plan.first_run)
        self.assertTrue(plan.persist_effective_profile)
        self.assertTrue(plan.requested_profile.runtime["stub_mode"])
        self.assertTrue(plan.effective_profile.runtime["stub_mode"])
        self.assertEqual(plan.launch_report.requested_mode, "stub")
        self.assertEqual(plan.launch_report.effective_mode, "stub")
        self.assertFalse(plan.launch_report.used_stub_fallback)
        self.assertEqual(plan.launch_report.readiness_report.stub_mode_ready, direct_readiness.stub_mode_ready)
        self.assertEqual(plan.launch_report.readiness_report.real_mode_ready, direct_readiness.real_mode_ready)
        self.assertEqual(plan.launch_report.readiness_report.checks, direct_readiness.checks)
        self.assertEqual(plan.launch_report.readiness_report.capabilities, direct_readiness.capabilities)
        self.assertEqual(plan.launch_report.readiness_report.guidance, direct_readiness.guidance)

    async def test_packaged_runner_uses_effective_stub_profile_when_saved_real_profile_is_not_ready(self) -> None:
        config, bundle_dir = _build_test_config(
            stub_mode=False,
            sqlite_name=str(self.test_db),
            logs_name=str(self.test_logs),
            bundle_name=str(self.test_bundle),
            models_dir=str(self.models_dir),
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
        app_state = (bundle_dir / "app_state.json").read_text(encoding="utf-8")
        launch_report = (bundle_dir / "launch_report.json").read_text(encoding="utf-8")

        self.assertIn('"stub_mode": true', app_state.lower())
        self.assertIn('"requested_mode": "real"', launch_report)
        self.assertIn('"effective_mode": "stub"', launch_report)

    async def test_packaged_runner_recovers_to_stub_and_writes_diagnostics_after_real_mode_start_exception(self) -> None:
        config, bundle_dir = _build_test_config(
            stub_mode=False,
            sqlite_name=str(self.test_db),
            logs_name=str(self.test_logs),
            bundle_name=str(self.test_bundle),
            models_dir=str(self.models_dir),
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

        original_start = Orchestrator.start

        async def _patched_start(self):
            if not self.config.preflight.flags.stub_mode:
                raise RuntimeError("simulated packaged real-mode startup failure")
            return await original_start(self)

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
                return_value=(True, "reachable"),
            ):
                with patch.object(Orchestrator, "start", new=_patched_start):
                    result = await run_packaged_dashboard_app(config=config, support_bundle_dir=bundle_dir)

        assert result is not None
        launch_report = (bundle_dir / "launch_report.json").read_text(encoding="utf-8")
        app_state = (bundle_dir / "app_state.json").read_text(encoding="utf-8")
        diagnostics = (bundle_dir / "packaged_startup_diagnostics.json").read_text(encoding="utf-8")

        self.assertIn('"requested_mode": "real"', launch_report)
        self.assertIn('"effective_mode": "stub"', launch_report)
        self.assertIn('"blocking_reason": "startup_exception"', launch_report)
        self.assertIn('"stub_mode": true', app_state.lower())
        self.assertIn("simulated packaged real-mode startup failure", diagnostics)
        self.assertIn("RuntimeError", diagnostics)


if __name__ == "__main__":
    unittest.main()
