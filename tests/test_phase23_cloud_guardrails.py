"""Phase 23 auxiliary cloud adapter, policy, and fallback regressions."""

from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path

from config import APP_CONFIG
from data_structures import (
    CloudFallbackBehavior,
    CloudJobContract,
    CloudJobPayloadClass,
    CloudJobPrivacyClass,
    CloudOffloadCapability,
    CloudOffloadMode,
    CloudOffloadOutcome,
    UserSettingsProfile,
)
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
    return replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)


class _FailingCloudAdapter:
    provider_family = "provider_agnostic"

    def is_available(self) -> bool:
        return True

    def supports(self, capability: CloudOffloadCapability) -> bool:
        return True

    async def dispatch(self, contract: CloudJobContract, payload: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("simulated provider outage")


class Phase23CloudContractTests(unittest.TestCase):
    def test_cloud_job_contract_requires_auxiliary_mode_and_approved_content_flag(self) -> None:
        contract = CloudJobContract(
            job_id="cloud-job-1",
            capability=CloudOffloadCapability.BACKGROUND_MAINTENANCE,
            payload_class=CloudJobPayloadClass.METADATA_ONLY,
            privacy_class=CloudJobPrivacyClass.METADATA_ONLY,
            max_payload_bytes=1024 * 64,
            max_retries=1,
            fallback_behavior=CloudFallbackBehavior.RETRY_THEN_LOCAL,
            dispatch_mode=CloudOffloadMode.AUXILIARY_ONLY,
            provider_family="provider_agnostic",
            content_approved=False,
        )

        restored = CloudJobContract.from_dict(contract.to_dict())

        self.assertEqual(restored, contract)
        with self.assertRaises(ValueError):
            CloudJobContract(
                job_id="cloud-job-2",
                capability=CloudOffloadCapability.VISION_HELPER,
                payload_class=CloudJobPayloadClass.IMAGE_REGION,
                privacy_class=CloudJobPrivacyClass.APPROVED_CONTENT,
                max_payload_bytes=1024 * 64,
                dispatch_mode=CloudOffloadMode.AUXILIARY_ONLY,
                provider_family="provider_agnostic",
                content_approved=False,
            )

    def test_user_settings_profile_validates_per_capability_cloud_modes(self) -> None:
        profile = UserSettingsProfile(
            profile_name="cloud-per-capability",
            cloud={
                "mode": CloudOffloadMode.AUXILIARY_ONLY.value,
                "provider": "stub_cloud",
                "capability_modes": {
                    CloudOffloadCapability.VISION_HELPER.value: CloudOffloadMode.AUXILIARY_ONLY.value,
                    CloudOffloadCapability.BROWSER_HELPER.value: CloudOffloadMode.DISABLED.value,
                },
            },
        )

        profile.validate()

        self.assertTrue(profile.cloud["enabled"])
        self.assertEqual(profile.cloud["provider"], "stub_cloud")
        self.assertEqual(
            profile.cloud["capability_modes"][CloudOffloadCapability.VISION_HELPER.value],
            CloudOffloadMode.AUXILIARY_ONLY.value,
        )
        with self.assertRaises(ValueError):
            UserSettingsProfile(
                profile_name="cloud-invalid-capability",
                cloud={
                    "capability_modes": {
                        "unknown_helper": CloudOffloadMode.AUXILIARY_ONLY.value,
                    }
                },
            ).validate()


class Phase23CloudIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase23_cloud_guardrails.sqlite3")
        self.test_logs = Path("test_phase23_cloud_guardrails_logs")
        self.bundle_dir = Path("test_phase23_support_bundle")
        self.config = _build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)
        if self.bundle_dir.exists():
            shutil.rmtree(self.bundle_dir)

    def test_readiness_report_surfaces_live_auxiliary_cloud_state(self) -> None:
        profile = UserSettingsProfile(
            profile_name="cloud-readiness",
            cloud={
                "mode": CloudOffloadMode.AUXILIARY_ONLY.value,
                "provider": "stub_cloud",
                "provider_family": "provider_agnostic",
                "max_payload_bytes": 1024 * 64,
                "max_retries": 1,
                "fallback_behavior": CloudFallbackBehavior.RETRY_THEN_LOCAL.value,
                "capability_modes": {
                    CloudOffloadCapability.VISION_HELPER.value: CloudOffloadMode.AUXILIARY_ONLY.value,
                    CloudOffloadCapability.BACKGROUND_MAINTENANCE.value: CloudOffloadMode.AUXILIARY_ONLY.value,
                    CloudOffloadCapability.BROWSER_HELPER.value: CloudOffloadMode.DISABLED.value,
                },
            },
            privacy={
                "log_runtime_events": True,
                "allow_cloud_content": False,
                "log_level": "INFO",
            },
        )

        report = self.orchestrator._build_dashboard_readiness_report(active_profile=profile)
        capabilities = {item.capability_name: item for item in report.capabilities}
        maintenance_contract = self.orchestrator._cloud_job_contract_for_capability(
            profile=profile,
            capability=CloudOffloadCapability.BACKGROUND_MAINTENANCE,
        )
        vision_contract = self.orchestrator._cloud_job_contract_for_capability(
            profile=profile,
            capability=CloudOffloadCapability.VISION_HELPER,
        )

        self.assertIn("cloud_offload", capabilities)
        self.assertIn("cloud_vision_helper", capabilities)
        self.assertIn("cloud_background_maintenance", capabilities)
        self.assertEqual(capabilities["cloud_offload"].status, "ready")
        self.assertEqual(capabilities["cloud_offload"].reason, "cloud_auxiliary_available")
        self.assertEqual(capabilities["cloud_vision_helper"].reason, "cloud_content_not_approved")
        self.assertEqual(capabilities["cloud_background_maintenance"].reason, "cloud_auxiliary_ready")
        self.assertEqual(capabilities["cloud_browser_helper"].reason, "cloud_capability_disabled")
        self.assertIsNotNone(maintenance_contract)
        self.assertIsNone(vision_contract)
        self.assertIn("provider=stub_cloud", capabilities["cloud_background_maintenance"].detail)

    def test_readiness_report_degrades_when_provider_is_missing(self) -> None:
        profile = UserSettingsProfile(
            profile_name="cloud-provider-missing",
            cloud={
                "mode": CloudOffloadMode.AUXILIARY_ONLY.value,
                "provider": "missing_cloud",
                "capability_modes": {
                    CloudOffloadCapability.BACKGROUND_MAINTENANCE.value: CloudOffloadMode.AUXILIARY_ONLY.value,
                },
            },
        )

        report = self.orchestrator._build_dashboard_readiness_report(active_profile=profile)
        capabilities = {item.capability_name: item for item in report.capabilities}

        self.assertEqual(capabilities["cloud_offload"].status, "degraded")
        self.assertEqual(capabilities["cloud_offload"].reason, "cloud_provider_unavailable")
        self.assertEqual(capabilities["cloud_background_maintenance"].reason, "cloud_provider_unavailable")

    async def test_dispatch_persists_successful_auxiliary_cloud_record(self) -> None:
        profile = UserSettingsProfile(
            profile_name="cloud-success",
            cloud={
                "mode": CloudOffloadMode.AUXILIARY_ONLY.value,
                "provider": "stub_cloud",
                "capability_modes": {
                    CloudOffloadCapability.BACKGROUND_MAINTENANCE.value: CloudOffloadMode.AUXILIARY_ONLY.value,
                },
            },
        )

        record = await self.orchestrator.dispatch_auxiliary_cloud_job(
            capability=CloudOffloadCapability.BACKGROUND_MAINTENANCE,
            payload={"operation": "rebuild_indexes", "count": 3},
            active_profile=profile,
            job_id="cloud-maint-1",
            metadata={"source": "phase23_test"},
        )
        records = await self.orchestrator.storage.list_cloud_offload_records(job_id="cloud-maint-1")

        self.assertEqual(record.outcome, CloudOffloadOutcome.SUCCEEDED)
        self.assertEqual(record.provider_name, "stub_cloud")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].outcome, CloudOffloadOutcome.SUCCEEDED)
        self.assertEqual(records[0].metadata["source"], "phase23_test")

    async def test_dispatch_runs_local_fallback_when_provider_fails(self) -> None:
        self.orchestrator.cloud_offload.register_adapter("failing_cloud", _FailingCloudAdapter())
        fallback_runs: list[str] = []
        profile = UserSettingsProfile(
            profile_name="cloud-failure-fallback",
            cloud={
                "mode": CloudOffloadMode.AUXILIARY_ONLY.value,
                "provider": "failing_cloud",
                "max_retries": 2,
                "capability_modes": {
                    CloudOffloadCapability.BACKGROUND_MAINTENANCE.value: CloudOffloadMode.AUXILIARY_ONLY.value,
                },
            },
        )

        async def _local_fallback() -> None:
            fallback_runs.append("ran")

        record = await self.orchestrator.dispatch_auxiliary_cloud_job(
            capability=CloudOffloadCapability.BACKGROUND_MAINTENANCE,
            payload={"operation": "compact_cache"},
            active_profile=profile,
            local_fallback=_local_fallback,
            job_id="cloud-fallback-1",
        )
        records = await self.orchestrator.storage.list_cloud_offload_records(job_id="cloud-fallback-1")

        self.assertEqual(record.outcome, CloudOffloadOutcome.LOCAL_FALLBACK)
        self.assertTrue(record.local_fallback_used)
        self.assertEqual(record.fallback_reason, "cloud_dispatch_failed")
        self.assertEqual(record.retry_count, 2)
        self.assertEqual(fallback_runs, ["ran"])
        self.assertEqual(records[-1].outcome, CloudOffloadOutcome.LOCAL_FALLBACK)

    async def test_dispatch_enforces_content_approval_before_cloud_attempt(self) -> None:
        fallback_runs: list[str] = []
        profile = UserSettingsProfile(
            profile_name="cloud-content-blocked",
            cloud={
                "mode": CloudOffloadMode.AUXILIARY_ONLY.value,
                "provider": "stub_cloud",
                "capability_modes": {
                    CloudOffloadCapability.VISION_HELPER.value: CloudOffloadMode.AUXILIARY_ONLY.value,
                },
            },
            privacy={
                "log_runtime_events": True,
                "allow_cloud_content": False,
                "log_level": "INFO",
            },
        )

        async def _local_fallback() -> None:
            fallback_runs.append("ran")

        record = await self.orchestrator.dispatch_auxiliary_cloud_job(
            capability=CloudOffloadCapability.VISION_HELPER,
            payload={"image_ref": "capture_01.png"},
            active_profile=profile,
            local_fallback=_local_fallback,
            job_id="cloud-vision-blocked",
        )

        self.assertEqual(record.outcome, CloudOffloadOutcome.LOCAL_FALLBACK)
        self.assertEqual(record.fallback_reason, "cloud_content_not_approved")
        self.assertEqual(fallback_runs, ["ran"])

    async def test_support_bundle_export_stays_local_successful_when_cloud_export_fails(self) -> None:
        self.orchestrator.cloud_offload.register_adapter("failing_cloud", _FailingCloudAdapter())
        profile = UserSettingsProfile(
            profile_name="cloud-export",
            cloud={
                "mode": CloudOffloadMode.AUXILIARY_ONLY.value,
                "provider": "failing_cloud",
                "max_retries": 1,
                "capability_modes": {
                    CloudOffloadCapability.EXPORT.value: CloudOffloadMode.AUXILIARY_ONLY.value,
                },
            },
            privacy={
                "log_runtime_events": True,
                "allow_cloud_content": True,
                "log_level": "INFO",
            },
        )

        bundle = await self.orchestrator.export_packaged_support_bundle(self.bundle_dir, active_profile=profile)
        records = await self.orchestrator.storage.list_cloud_offload_records(
            capability=CloudOffloadCapability.EXPORT.value
        )
        cloud_log_path = self.bundle_dir / self.config.storage.cloud_offload_log_name

        self.assertTrue(Path(bundle.manifest_path).exists())
        self.assertTrue(cloud_log_path.exists())
        self.assertTrue(any(path.endswith(self.config.storage.cloud_offload_log_name) for path in bundle.copied_artifact_paths))
        self.assertEqual(records[-1].outcome, CloudOffloadOutcome.FAILED)
        self.assertEqual(records[-1].provider_name, "failing_cloud")


if __name__ == "__main__":
    unittest.main()
