"""Phase 20 capability contracts, policy, audit, and stub-execution regressions."""

from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path

from capability_runtime import CapabilityPolicyEngine
from config import APP_CONFIG
from data_structures import (
    CapabilityPolicyOutcome,
    CapabilityRequest,
    CapabilityType,
    DesktopInputSpec,
    FileOperationSpec,
    ShellCommandSpec,
    UserSettingsProfile,
)
from model_manager import ModelHealthSnapshot
from orchestrator import Orchestrator


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
    backend_runtime = replace(APP_CONFIG.backend_runtime, low_ram_headroom_gb=0.01)
    return replace(
        APP_CONFIG,
        preflight=preflight,
        storage=storage_cfg,
        dashboard=dashboard,
        backend_runtime=backend_runtime,
    )


def _snapshot(*, available_ram_gb: float = 4.0, fallback_active: bool = False) -> ModelHealthSnapshot:
    governor_reasons: tuple[str, ...] = ()
    if fallback_active:
        governor_reasons = ("model_fallback_active",)
    elif available_ram_gb <= 1.0:
        governor_reasons = ("low_available_ram",)
    return ModelHealthSnapshot(
        started=True,
        generation_backend="stub_generation",
        embedding_backend="stub_embedding",
        active_generation_jobs=0,
        active_embedding_jobs=0,
        active_heavy_roles=("generation", "embedding"),
        heavy_slot_limit=2,
        last_used_at=None,
        fallback_active=fallback_active,
        fallback_reason="pressure" if fallback_active else "",
        available_ram_gb=available_ram_gb,
        total_ram_gb=8.0,
        generation_backend_vram_gb=0.0,
        embedding_backend_vram_gb=0.0,
        governor_active=bool(governor_reasons),
        governor_pressure_reasons=governor_reasons,
        governor_degraded_features=(
            ("continuous_capture", "ocr_on_step", "vision_on_step", "optional_heavy_residency", "background_work")
            if governor_reasons
            else ()
        ),
        queue_pressure=False,
        backend_health_degraded=fallback_active,
        allow_continuous_capture=not bool(governor_reasons),
        allow_ocr_on_step=not bool(governor_reasons),
        allow_vision_on_step=not bool(governor_reasons),
        allow_optional_heavy_residency=not bool(governor_reasons),
        allow_background_work=not bool(governor_reasons),
        governor_summary=",".join(governor_reasons),
        telemetry_enabled=False,
        last_error=None,
    )


def _desktop_profile(
    *,
    enabled_capabilities: tuple[str, ...],
    approval_policy: str = "approve_risky_only",
    observation_tier: str = "screenshot_on_demand",
    observation_overrides: dict[str, object] | None = None,
    allowlisted_roots: tuple[str, ...] | None = None,
    allowlisted_shell_commands: tuple[str, ...] | None = None,
    allowlisted_browser_domains: tuple[str, ...] | None = None,
    allowlisted_apps: tuple[str, ...] | None = None,
) -> UserSettingsProfile:
    profile = UserSettingsProfile(
        profile_name="desktop",
        desktop={
            "enabled": True,
            "approval_policy": approval_policy,
            "enabled_capabilities": enabled_capabilities,
            "allowlisted_roots": allowlisted_roots or (".", "logs", "examples", "models"),
            "allowlisted_shell_commands": allowlisted_shell_commands or ("python", "git", "rg", "pytest"),
            "allowlisted_browser_domains": allowlisted_browser_domains or ("localhost", "127.0.0.1"),
            "allowlisted_apps": allowlisted_apps or ("notepad",),
            "allowlisted_background_services": (),
        },
        observation={
            "tier": observation_tier,
            "continuous_capture": observation_tier == "continuous_capture",
            "ocr_on_step": observation_tier in {"ocr_on_step", "vision_on_step", "continuous_capture"},
            "vision_on_step": observation_tier in {"vision_on_step", "continuous_capture"},
            **(observation_overrides or {}),
        },
    )
    profile.validate()
    return profile


class Phase20CapabilityPolicyTests(unittest.TestCase):
    def test_capability_request_round_trip_preserves_nested_payload(self) -> None:
        request = CapabilityRequest(
            request_id="cap-1",
            capability_type=CapabilityType.FILE_OPERATION,
            summary="Read project README",
            file_operation=FileOperationSpec(operation="read", source_path="README.md"),
        )

        restored = CapabilityRequest.from_dict(request.to_dict())

        self.assertEqual(restored, request)
        self.assertEqual(restored.action_name(), "read")
        self.assertEqual(restored.target_summary(), "README.md")

    def test_policy_denies_dangerous_and_non_allowlisted_shell_requests(self) -> None:
        engine = CapabilityPolicyEngine()
        profile = _desktop_profile(enabled_capabilities=("shell_command",))
        disallowed_shell = CapabilityRequest(
            request_id="cap-shell-1",
            capability_type=CapabilityType.SHELL_COMMAND,
            summary="Run non-allowlisted shell command",
            shell_command=ShellCommandSpec(command="powershell", args=("-Command", "Get-Date")),
        )
        dangerous_request = CapabilityRequest(
            request_id="cap-shell-2",
            capability_type=CapabilityType.SHELL_COMMAND,
            summary="Request elevation for shell command",
            shell_command=ShellCommandSpec(command="python", args=("--version",)),
            requires_elevation=True,
        )

        disallowed_decision = engine.evaluate(disallowed_shell, profile=profile, snapshot=_snapshot())
        dangerous_decision = engine.evaluate(dangerous_request, profile=profile, snapshot=_snapshot())

        self.assertEqual(disallowed_decision.outcome, CapabilityPolicyOutcome.DENIED)
        self.assertIn("command_not_allowlisted", disallowed_decision.reason_codes)
        self.assertEqual(dangerous_decision.outcome, CapabilityPolicyOutcome.DENIED)
        self.assertIn("admin_elevation_blocked", dangerous_decision.reason_codes)

    def test_policy_requires_approval_for_destructive_file_and_desktop_input_requests(self) -> None:
        engine = CapabilityPolicyEngine()
        profile = _desktop_profile(enabled_capabilities=("file_operation", "desktop_input"))
        file_request = CapabilityRequest(
            request_id="cap-file-1",
            capability_type=CapabilityType.FILE_OPERATION,
            summary="Delete log file",
            file_operation=FileOperationSpec(operation="delete", source_path="logs/old.log"),
            destructive=True,
        )
        input_request = CapabilityRequest(
            request_id="cap-input-1",
            capability_type=CapabilityType.DESKTOP_INPUT,
            summary="Type into Notepad",
            desktop_input=DesktopInputSpec(action="type_text", text="hello", target="notepad"),
        )

        file_decision = engine.evaluate(file_request, profile=profile, snapshot=_snapshot())
        input_decision = engine.evaluate(input_request, profile=profile, snapshot=_snapshot())

        self.assertEqual(file_decision.outcome, CapabilityPolicyOutcome.REQUIRES_APPROVAL)
        self.assertIn("destructive_file_operation", file_decision.reason_codes)
        self.assertEqual(input_decision.outcome, CapabilityPolicyOutcome.REQUIRES_APPROVAL)
        self.assertIn("manual_input_requires_approval", input_decision.reason_codes)

    def test_registry_view_tracks_enabled_and_degraded_capabilities(self) -> None:
        engine = CapabilityPolicyEngine()
        profile = _desktop_profile(
            enabled_capabilities=("file_operation", "screenshot"),
            observation_tier="ocr_on_step",
        )

        view = engine.build_registry_view(profile=profile, snapshot=_snapshot(available_ram_gb=0.25))
        registrations = {item.capability_type.value: item for item in view.registrations}

        self.assertTrue(registrations["file_operation"].enabled)
        self.assertEqual(registrations["file_operation"].default_policy_outcome, CapabilityPolicyOutcome.ALLOWED)
        self.assertEqual(registrations["file_operation"].executor_kind, "live")
        self.assertEqual(registrations["screenshot"].executor_kind, "live")
        self.assertEqual(registrations["screenshot"].status.value, "degraded")
        self.assertIn(".", registrations["file_operation"].allowlisted_targets)


class Phase20CapabilityIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase20_capability.sqlite3")
        self.test_logs = Path("test_phase20_capability_logs")
        self.config = _build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_allowed_live_request_persists_audits_and_updates_dashboard_registry(self) -> None:
        profile = _desktop_profile(enabled_capabilities=("file_operation",))
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        await self.orchestrator.start_local_task_session(
            "Phase20 capability test",
            active_profile=profile,
        )
        request = CapabilityRequest(
            request_id="cap-run-1",
            capability_type=CapabilityType.FILE_OPERATION,
            summary="Read README in stub mode",
            file_operation=FileOperationSpec(operation="read", source_path="README.md"),
        )

        result = await self.orchestrator.run_capability_request(request)

        audits = await self.orchestrator.storage.list_capability_audits(request_id="cap-run-1")
        persisted_view = await self.orchestrator.storage.load_capability_registry_view()
        state = self.orchestrator.dashboard.app_state_snapshot()

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(result.executor_kind, "live_file")
        self.assertEqual(len(audits), 3)
        self.assertEqual(audits[0].event_type.value, "requested")
        self.assertEqual(audits[-1].event_type.value, "executor_result")
        self.assertEqual(persisted_view.recent_decisions[-1].request_id, "cap-run-1")
        self.assertEqual(state.capability_registry_view.recent_decisions[-1].request_id, "cap-run-1")

    async def test_approval_denied_request_records_denial_and_blocked_stub_result(self) -> None:
        profile = _desktop_profile(enabled_capabilities=("desktop_input",))
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        await self.orchestrator.start_local_task_session(
            "Phase20 approval test",
            active_profile=profile,
        )
        request = CapabilityRequest(
            request_id="cap-run-2",
            capability_type=CapabilityType.DESKTOP_INPUT,
            summary="Type into Notepad",
            desktop_input=DesktopInputSpec(action="type_text", text="hello", target="notepad"),
        )

        result = await self.orchestrator.run_capability_request(request, approval_granted=False)

        audits = await self.orchestrator.storage.list_capability_audits(request_id="cap-run-2")

        self.assertEqual(result.status.value, "blocked")
        self.assertEqual(audits[2].event_type.value, "approval_denied")
        self.assertEqual(audits[-1].event_type.value, "executor_result")

    def test_readiness_report_surfaces_phase20_capability_entries(self) -> None:
        profile = _desktop_profile(enabled_capabilities=("file_operation",))

        report = self.orchestrator._build_dashboard_readiness_report(active_profile=profile)
        capabilities = {item.capability_name: item for item in report.capabilities}

        self.assertIn("desktop_control", capabilities)
        self.assertIn("file_operation", capabilities)
        self.assertEqual(capabilities["file_operation"].status, "ready")
        self.assertEqual(capabilities["desktop_control"].reason, "live_control_tier_available")


if __name__ == "__main__":
    unittest.main()
