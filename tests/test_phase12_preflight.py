"""Phase 12 acceptance tests for readiness and preflight behavior."""

from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from config import APP_CONFIG
from data_structures import UserSettingsProfile
from model_manager import ModelHealthSnapshot
from orchestrator import Orchestrator


def _build_test_config(
    *,
    stub_mode: bool,
    sqlite_name: str,
    logs_name: str,
    models_dir: str,
    generation_backend: str = "ollama",
    generation_model: str | None = None,
    generation_fallback_backend: str = "llama_cpp",
    generation_fallback_model: str = "missing-fallback.gguf",
    embedding_backend: str = "sentence_transformers",
) :
    backends = replace(
        APP_CONFIG.preflight.backends,
        generation_backend=generation_backend,
        generation_model=generation_model or APP_CONFIG.preflight.backends.generation_model,
        generation_fallback_backend=generation_fallback_backend,
        generation_fallback_model=generation_fallback_model,
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
        ),
    )
    storage_cfg = replace(APP_CONFIG.storage, sqlite_path=Path(sqlite_name), logs_dir=Path(logs_name))
    dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
    backend_runtime = replace(APP_CONFIG.backend_runtime, models_dir=Path(models_dir))
    return replace(
        APP_CONFIG,
        preflight=preflight,
        storage=storage_cfg,
        dashboard=dashboard,
        backend_runtime=backend_runtime,
    )


def _snapshot(
    *,
    started: bool,
    generation_backend: str,
    embedding_backend: str,
    fallback_active: bool = False,
    fallback_reason: str = "",
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


class Phase12PreflightAcceptanceTests(unittest.TestCase):
    """Validate deterministic readiness reporting for real-mode prerequisites."""

    def setUp(self) -> None:
        self.test_db = Path("test_phase12_preflight.sqlite3")
        self.test_logs = Path("test_phase12_preflight_logs")
        self.models_dir = Path("test_phase12_models")

    def tearDown(self) -> None:
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)
        if self.models_dir.exists():
            shutil.rmtree(self.models_dir)

    def test_readiness_reports_missing_dependencies_while_stub_mode_blocks_real_mode(self) -> None:
        config = _build_test_config(
            stub_mode=True,
            sqlite_name=str(self.test_db),
            logs_name=str(self.test_logs),
            models_dir=str(self.models_dir),
        )
        orchestrator = Orchestrator(config=config)
        dependency_map = {
            "chromadb": False,
            "sentence_transformers": False,
            "llama_cpp": False,
        }

        with patch.object(
            orchestrator,
            "_dependency_available",
            side_effect=lambda module_name: dependency_map.get(module_name, True),
        ):
            orchestrator.model_manager.health_snapshot = lambda: _snapshot(
                started=True,
                generation_backend="stub_generation",
                embedding_backend="stub_embedding",
            )
            report = orchestrator._build_dashboard_readiness_report()

        checks = {check.check_id: check for check in report.checks}
        capabilities = {item.capability_name: item for item in report.capabilities}

        self.assertTrue(report.stub_mode_ready)
        self.assertFalse(report.real_mode_ready)
        self.assertEqual(checks["vector_backend"].status, "degraded")
        self.assertEqual(checks["embedding_dependency"].status, "degraded")
        self.assertEqual(checks["llama_cpp_dependency"].status, "degraded")
        self.assertEqual(checks["ollama_service"].status, "disabled")
        self.assertEqual(capabilities["real_mode"].reason, "stub_mode_enabled")

    def test_readiness_reports_missing_ollama_service_and_missing_fallback_model_file(self) -> None:
        config = _build_test_config(
            stub_mode=False,
            sqlite_name=str(self.test_db),
            logs_name=str(self.test_logs),
            models_dir=str(self.models_dir),
            generation_backend="ollama",
            generation_fallback_backend="llama_cpp",
            generation_fallback_model="missing-fallback.gguf",
        )
        orchestrator = Orchestrator(config=config)

        with patch.object(orchestrator, "_dependency_available", return_value=True):
            with patch.object(
                orchestrator,
                "_probe_ollama_service",
                return_value=(False, "Ollama service probe failed at http://localhost:11434/api/tags: refused"),
            ):
                orchestrator.model_manager.health_snapshot = lambda: _snapshot(
                    started=True,
                    generation_backend="ollama",
                    embedding_backend="sentence_transformers",
                )
                report = orchestrator._build_dashboard_readiness_report()

        checks = {check.check_id: check for check in report.checks}
        capabilities = {item.capability_name: item for item in report.capabilities}

        self.assertFalse(report.real_mode_ready)
        self.assertEqual(checks["ollama_service"].status, "blocked")
        self.assertIn("probe failed", checks["ollama_service"].detail)
        self.assertEqual(checks["llama_cpp_model_file"].status, "degraded")
        self.assertIn("missing-fallback.gguf", checks["llama_cpp_model_file"].detail)
        self.assertEqual(capabilities["real_mode"].reason, "missing_ollama_service")

    def test_readiness_reports_missing_primary_llama_cpp_model_file_as_blocking(self) -> None:
        config = _build_test_config(
            stub_mode=False,
            sqlite_name=str(self.test_db),
            logs_name=str(self.test_logs),
            models_dir=str(self.models_dir),
            generation_backend="llama_cpp",
            generation_model="missing-primary.gguf",
            generation_fallback_backend="llama_cpp",
            generation_fallback_model="missing-fallback.gguf",
        )
        orchestrator = Orchestrator(config=config)

        with patch.object(orchestrator, "_dependency_available", return_value=True):
            orchestrator.model_manager.health_snapshot = lambda: _snapshot(
                started=True,
                generation_backend="llama_cpp",
                embedding_backend="sentence_transformers",
            )
            report = orchestrator._build_dashboard_readiness_report()

        checks = {check.check_id: check for check in report.checks}

        self.assertFalse(report.real_mode_ready)
        self.assertEqual(checks["llama_cpp_model_file"].status, "blocked")
        self.assertIn("primary_generation", checks["llama_cpp_model_file"].detail)

    def test_readiness_capabilities_flag_unimplemented_optional_toggles(self) -> None:
        config = _build_test_config(
            stub_mode=True,
            sqlite_name=str(self.test_db),
            logs_name=str(self.test_logs),
            models_dir=str(self.models_dir),
        )
        orchestrator = Orchestrator(config=config)
        profile = UserSettingsProfile(
            profile_name="future-toggles",
            desktop={"enabled": True, "approval_policy": "approve_risky_only"},
            observation={
                "tier": "vision_on_step",
                "continuous_capture": False,
                "ocr_on_step": True,
                "vision_on_step": True,
            },
            cloud={"enabled": True, "mode": "auxiliary_only"},
        )

        with patch.object(orchestrator, "_dependency_available", return_value=True):
            orchestrator.model_manager.health_snapshot = lambda: _snapshot(
                started=True,
                generation_backend="stub_generation",
                embedding_backend="stub_embedding",
            )
            report = orchestrator._build_dashboard_readiness_report(active_profile=profile)

        capabilities = {item.capability_name: item for item in report.capabilities}

        self.assertEqual(capabilities["desktop_control"].status, "visible_not_enabled")
        self.assertEqual(capabilities["desktop_control"].reason, "desktop_capabilities_not_enabled")
        self.assertEqual(capabilities["observation_tiers"].status, "degraded")
        self.assertEqual(capabilities["observation_tiers"].reason, "vision_on_step_cpu_fallback")
        self.assertEqual(capabilities["cloud_offload"].status, "visible_not_enabled")
        self.assertEqual(capabilities["cloud_offload"].reason, "cloud_capabilities_not_enabled")


if __name__ == "__main__":
    unittest.main()
