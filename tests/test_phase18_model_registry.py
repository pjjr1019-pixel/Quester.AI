"""Phase 18 registry, routing, control-plane, and bounded-cache regressions."""

from __future__ import annotations

import shutil
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from config import APP_CONFIG
from data_structures import (
    ModelLoadPolicy,
    ModelRegistration,
    ModelResourceClass,
    ModelRole,
    OptimizerSuggestionKind,
    OptimizerSuggestionRecord,
    ReasoningLog,
    UserSettingsProfile,
)
from model_manager import ModelManager
from orchestrator import Orchestrator
from tests.test_phase17_long_horizon import _build_stub_task_result


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


class _AsyncComponentStub:
    def __init__(self, config) -> None:
        self.config = config

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class _StorageStub(_AsyncComponentStub):
    def __init__(self, config, profile: UserSettingsProfile) -> None:
        super().__init__(config)
        self.profile = profile
        self.saved_profile = None

    def add_runtime_event_listener(self, listener) -> None:
        _ = listener

    def add_agent_status_listener(self, listener) -> None:
        _ = listener

    async def load_user_settings_profile(self, profile_name: str = "default") -> UserSettingsProfile | None:
        _ = profile_name
        return self.profile

    async def save_user_settings_profile(self, profile: UserSettingsProfile) -> None:
        self.saved_profile = profile


class _ModelManagerStartupStub(_AsyncComponentStub):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.started_with_stub_mode = None
        self.applied_profile = None

    async def start(self) -> None:
        self.started_with_stub_mode = self.config.preflight.flags.stub_mode

    def apply_user_settings_profile(self, profile: UserSettingsProfile) -> None:
        self.applied_profile = profile

    def health_snapshot(self):
        return SimpleNamespace(
            started=True,
            generation_backend="ollama",
            embedding_backend="sentence_transformers",
            active_generation_jobs=0,
            active_embedding_jobs=0,
            active_heavy_roles=("generation", "embedding"),
            heavy_slot_limit=2,
            last_used_at=None,
            fallback_active=False,
            fallback_reason=None,
            available_ram_gb=None,
            total_ram_gb=None,
            generation_backend_vram_gb=None,
            embedding_backend_vram_gb=None,
            governor_active=False,
            governor_pressure_reasons=(),
            governor_degraded_features=(),
            queue_pressure=False,
            backend_health_degraded=False,
            allow_continuous_capture=True,
            allow_ocr_on_step=True,
            allow_vision_on_step=True,
            allow_optional_heavy_residency=True,
            allow_background_work=True,
            governor_summary="",
            telemetry_enabled=False,
            last_error=None,
        )


class _DashboardStartupStub(_AsyncComponentStub):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.ui_running = False
        self.applied_profile = None

    def apply_user_settings(self, profile: UserSettingsProfile) -> None:
        self.applied_profile = profile

    def attach_controller(self, **kwargs) -> None:
        _ = kwargs

    def app_state_snapshot(self):
        return SimpleNamespace(user_settings=self.applied_profile or UserSettingsProfile())


class _Phase11ContentStub:
    def __init__(self, config) -> None:
        self.config = config


class Phase18ModelRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase18_model_registry.sqlite3")
        self.test_logs = Path("test_phase18_model_registry_logs")
        self.config = _build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))
        self.manager = ModelManager(config=self.config)
        await self.manager.start()

    async def asyncTearDown(self) -> None:
        await self.manager.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_default_registry_covers_required_roles_and_wrappers_still_work(self) -> None:
        roles = {registration.role.value for registration in self.manager.list_registered_models()}
        self.assertEqual(
            roles,
            {
                "generation",
                "embedding",
                "reranker",
                "speech_to_text",
                "text_to_speech",
                "vad",
                "translation",
                "code_specialist",
                "vision",
                "specialist_perception",
            },
        )

        response = await self.manager.generate("hello world", max_tokens=32)
        vector = await self.manager.embed("hello world")
        view = self.manager.registry_view(advisory_available=True, optimizer_subscriptions=("dashboard",))

        self.assertTrue(response.startswith("[stub]"))
        self.assertTrue(vector)
        self.assertEqual(view.active_heavy_roles, ("generation", "embedding"))
        routed_roles = {decision.requested_role.value for decision in view.last_route_decisions}
        self.assertIn("generation", routed_roles)
        self.assertIn("embedding", routed_roles)
        embedding_cache = next(snapshot for snapshot in view.cache_snapshots if snapshot.namespace == "embeddings")
        self.assertGreaterEqual(embedding_cache.entry_count, 1)

    async def test_optional_heavy_roles_can_swap_in_without_exceeding_heavy_cap_and_sidecars_stay_free(self) -> None:
        self.manager.register_model(
            ModelRegistration(
                registration_id="code_specialist:test-heavy",
                role=ModelRole.CODE_SPECIALIST,
                backend="mock_heavy",
                model_identifier="mock-code-specialist",
                resource_class=ModelResourceClass.HEAVY,
                enabled=True,
                load_policy=ModelLoadPolicy.ON_DEMAND,
                supported_capabilities=("code_assist",),
            )
        )
        self.manager.register_model(
            ModelRegistration(
                registration_id="translation:test-sidecar",
                role=ModelRole.TRANSLATION,
                backend="cpu_helper",
                model_identifier="mock-translation-sidecar",
                resource_class=ModelResourceClass.SIDECAR,
                enabled=True,
                preferred_device="cpu",
                load_policy=ModelLoadPolicy.ON_DEMAND,
                supported_capabilities=("translate",),
            )
        )

        heavy_decision = self.manager.route_role(ModelRole.CODE_SPECIALIST, capability="code_assist")
        sidecar_decision = self.manager.route_role(ModelRole.TRANSLATION, capability="translate")
        view = self.manager.registry_view()

        self.assertTrue(heavy_decision.allowed)
        self.assertIn("embedding", heavy_decision.metadata["swapped_out_roles"])
        self.assertTrue(sidecar_decision.allowed)
        self.assertEqual(len(view.active_heavy_roles), 2)
        self.assertIn("code_specialist", view.active_heavy_roles)

    async def test_visual_role_guidance_pins_future_lightweight_recommendations(self) -> None:
        vision_guidance = " ".join(self.manager.install_guidance_for_role(ModelRole.VISION))
        specialist_guidance = " ".join(self.manager.install_guidance_for_role(ModelRole.SPECIALIST_PERCEPTION))

        self.assertIn("HuggingFaceTB/SmolVLM-256M-Instruct", vision_guidance)
        self.assertIn("PaddleOCR", specialist_guidance)

    async def test_bounded_strategy_cache_caps_entries_and_records_evictions(self) -> None:
        for index in range(80):
            self.manager.warm_cache("strategy_artifacts", f"strategy:{index}", f"artifact {index}")
        view = self.manager.registry_view()
        strategy_cache = next(snapshot for snapshot in view.cache_snapshots if snapshot.namespace == "strategy_artifacts")

        self.assertEqual(strategy_cache.entry_count, strategy_cache.max_entries)
        self.assertGreater(strategy_cache.evictions, 0)

    async def test_optional_heavy_roles_unload_after_idle_when_default_pair_is_inactive(self) -> None:
        self.manager.register_model(
            ModelRegistration(
                registration_id="code_specialist:test-heavy-idle",
                role=ModelRole.CODE_SPECIALIST,
                backend="mock_heavy",
                model_identifier="mock-code-specialist-idle",
                resource_class=ModelResourceClass.HEAVY,
                enabled=True,
                load_policy=ModelLoadPolicy.ON_DEMAND,
                supported_capabilities=("code_assist",),
            )
        )
        self.manager._generation_health = replace(self.manager._generation_health, started=False, available=False)
        self.manager._embedding_health = replace(self.manager._embedding_health, started=False, available=False)
        self.manager._sync_active_heavy_roles()

        decision = self.manager.route_role(ModelRole.CODE_SPECIALIST, capability="code_assist")

        self.assertTrue(decision.allowed)
        self.assertEqual(self.manager.registry_view().active_heavy_roles, ("code_specialist",))

        self.manager._last_used_monotonic = time.monotonic() - (
            self.config.backend_runtime.idle_unload_after_s + 1.0
        )
        await self.manager._maybe_unload_idle_backends()

        self.assertNotIn("code_specialist", self.manager.registry_view().active_heavy_roles)

    async def test_hardware_governor_blocks_new_optional_heavy_roles_under_queue_pressure(self) -> None:
        self.manager.register_model(
            ModelRegistration(
                registration_id="code_specialist:test-heavy-pressure",
                role=ModelRole.CODE_SPECIALIST,
                backend="mock_heavy",
                model_identifier="mock-code-specialist-pressure",
                resource_class=ModelResourceClass.HEAVY,
                enabled=True,
                load_policy=ModelLoadPolicy.ON_DEMAND,
                supported_capabilities=("code_assist",),
            )
        )
        self.manager._active_generation_jobs = self.config.concurrency.generation_slots

        decision = self.manager.route_role(ModelRole.CODE_SPECIALIST, capability="code_assist")
        view = self.manager.registry_view()

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.fallback_reason, "hardware_governor_optional_heavy_residency_disabled")
        self.assertIn("generation_queue_pressure", decision.metadata["pressure_reasons"])
        self.assertTrue(view.governor_active)
        self.assertIn("optional_heavy_residency", view.governor_degraded_features)
        self.assertIn("generation_queue_pressure", view.governor_pressure_reasons)

    async def test_governor_model_loading_advisory_retains_optional_heavy_role_until_ttl_expires(self) -> None:
        self.manager.register_model(
            ModelRegistration(
                registration_id="code_specialist:test-heavy-advisory",
                role=ModelRole.CODE_SPECIALIST,
                backend="mock_heavy",
                model_identifier="mock-code-specialist-advisory",
                resource_class=ModelResourceClass.HEAVY,
                enabled=True,
                load_policy=ModelLoadPolicy.ON_DEMAND,
                supported_capabilities=("code_assist",),
            )
        )
        self.manager._generation_health = replace(self.manager._generation_health, started=False, available=False)
        self.manager._embedding_health = replace(self.manager._embedding_health, started=False, available=False)
        self.manager._sync_active_heavy_roles()

        decision = self.manager.route_role(ModelRole.CODE_SPECIALIST, capability="code_assist")
        self.assertTrue(decision.allowed)

        self.manager.apply_governor_advisory_inputs(
            (
                OptimizerSuggestionRecord(
                    suggestion_id="phase18_governor_model_loading",
                    cycle_id="phase18_cycle",
                    kind=OptimizerSuggestionKind.MODEL_LOADING,
                    summary="Retain code specialist for the next cycle",
                    rationale="Bounded demand forecast",
                    target_components=("model_manager",),
                    advisory_only=True,
                    metadata={"role": "code_specialist", "retention_seconds": 60},
                ),
            )
        )
        self.assertIn("retain(code_specialist)", self.manager.registry_view().governor_summary)

        self.manager._last_used_monotonic = time.monotonic() - (
            self.config.backend_runtime.idle_unload_after_s + 1.0
        )
        await self.manager._maybe_unload_idle_backends()
        self.assertIn("code_specialist", self.manager.registry_view().active_heavy_roles)

        advisory_id, advisory_payload = next(iter(self.manager._governor_advisory_inputs.items()))
        self.manager._governor_advisory_inputs[advisory_id] = replace(
            advisory_payload,
            expires_at_monotonic=time.monotonic() - 1.0,
        )
        await self.manager._maybe_unload_idle_backends()
        self.assertNotIn("code_specialist", self.manager.registry_view().active_heavy_roles)


class Phase18ControlPlaneIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase18_control_plane.sqlite3")
        self.test_logs = Path("test_phase18_control_plane_logs")
        self.config = _build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_dashboard_and_storage_receive_model_registry_view(self) -> None:
        await self.orchestrator.model_manager.embed("phase18 control plane")
        await self.orchestrator.run_task("What is 2 + 2?", thinking_minutes=30)

        state = self.orchestrator.dashboard.app_state_snapshot()
        persisted = await self.orchestrator.storage.load_model_registry_view()

        self.assertTrue(state.model_registry_view.registrations)
        self.assertTrue(persisted.registrations)
        self.assertEqual(state.model_registry_view.active_heavy_roles, ("generation", "embedding"))
        cache_names = {snapshot.namespace for snapshot in state.model_registry_view.cache_snapshots}
        self.assertIn("embeddings", cache_names)
        retrieval_cache = next(
            snapshot for snapshot in state.model_registry_view.cache_snapshots if snapshot.namespace == "retrieval_candidates"
        )
        runtime_subset_cache = next(
            snapshot for snapshot in state.model_registry_view.cache_snapshots if snapshot.namespace == "runtime_subsets"
        )
        self.assertGreaterEqual(retrieval_cache.entry_count, 1)
        self.assertGreaterEqual(runtime_subset_cache.entry_count, 1)

    async def test_advisory_runtime_events_feed_strategy_caches_without_hidden_bus(self) -> None:
        session_id = "lh_phase18_advisory_control"

        async def fake_run_bounded_task(
            *,
            question: str,
            thinking_minutes: int,
            budget,
            persist_task_result: bool = True,
            publish_history: bool = True,
            emit_completion_event: bool = True,
        ):
            _ = (thinking_minutes, budget, persist_task_result, publish_history, emit_completion_event)
            return _build_stub_task_result(
                task_id="phase18_advisory_task",
                question=question,
                answer_text="phase18 advisory",
                candidate_count=2,
                candidate_score=0.8,
                local_evidence_ids=("ev-1",),
            )

        async def fake_suggest_for_long_horizon(**kwargs):
            _ = kwargs
            return (
                OptimizerSuggestionRecord(
                    suggestion_id="phase18_macro_advice",
                    cycle_id="phase18_cycle",
                    kind=OptimizerSuggestionKind.MACRO_ADVICE,
                    summary="Phase 18 macro advice",
                    rationale="Phase 18 test",
                    target_components=("reasoner", "critic"),
                    source_task_ids=("phase18_advisory_task",),
                    confidence=0.7,
                    advisory_only=True,
                    metadata={"budget_delta": {}, "advisory_only_reason": "proposal_only_policy"},
                ),
            )

        with (
            mock.patch.object(self.orchestrator, "_long_horizon_session_id", return_value=session_id),
            mock.patch.object(self.orchestrator, "_run_bounded_task", side_effect=fake_run_bounded_task),
            mock.patch.object(
                self.orchestrator.self_optimizer,
                "suggest_for_long_horizon",
                side_effect=fake_suggest_for_long_horizon,
            ),
        ):
            result = await self.orchestrator.run_task("phase18 advisory routing", thinking_minutes=121)

        self.assertIn("long_horizon_checkpointed_run", result.warnings)
        events = await self.orchestrator.storage.list_runtime_events(stage="pipeline.long_horizon_advisory_planned")
        self.assertTrue(events)
        await self.orchestrator._publish_dashboard_model_registry_view()
        state = self.orchestrator.dashboard.app_state_snapshot()
        strategy_cache = next(
            snapshot for snapshot in state.model_registry_view.cache_snapshots if snapshot.namespace == "strategy_artifacts"
        )
        compression_cache = next(
            snapshot for snapshot in state.model_registry_view.cache_snapshots if snapshot.namespace == "compression_artifacts"
        )
        self.assertGreaterEqual(strategy_cache.entry_count, 1)
        self.assertGreaterEqual(compression_cache.entry_count, 1)

    async def test_governor_advisories_inform_summary_but_pressure_still_blocks_optional_heavy_roles(self) -> None:
        self.orchestrator.model_manager.register_model(
            ModelRegistration(
                registration_id="code_specialist:phase18-governor-advisory",
                role=ModelRole.CODE_SPECIALIST,
                backend="mock_heavy",
                model_identifier="mock-code-specialist-governor-advisory",
                resource_class=ModelResourceClass.HEAVY,
                enabled=True,
                load_policy=ModelLoadPolicy.ON_DEMAND,
                supported_capabilities=("code_assist",),
            )
        )
        session_id = "lh_phase18_governor_advisory"

        async def fake_run_bounded_task(
            *,
            question: str,
            thinking_minutes: int,
            budget,
            persist_task_result: bool = True,
            publish_history: bool = True,
            emit_completion_event: bool = True,
        ):
            _ = (thinking_minutes, budget, persist_task_result, publish_history, emit_completion_event)
            return _build_stub_task_result(
                task_id="phase18_governor_advisory_task",
                question=question,
                answer_text="phase18 governor advisory",
                candidate_count=2,
                candidate_score=0.82,
                local_evidence_ids=("ev-1",),
            )

        async def fake_suggest_for_long_horizon(**kwargs):
            _ = kwargs
            return (
                OptimizerSuggestionRecord(
                    suggestion_id="phase18_model_loading_hint",
                    cycle_id="phase18_governor_cycle",
                    kind=OptimizerSuggestionKind.MODEL_LOADING,
                    summary="Keep the code specialist warm for the next foreground cycle",
                    rationale="The next cycle is expected to revisit code work.",
                    target_components=("model_manager",),
                    source_task_ids=("phase18_governor_advisory_task",),
                    confidence=0.65,
                    advisory_only=True,
                    metadata={"role": "code_specialist", "retention_seconds": 300},
                ),
                OptimizerSuggestionRecord(
                    suggestion_id="phase18_cache_prefetch_hint",
                    cycle_id="phase18_governor_cycle",
                    kind=OptimizerSuggestionKind.CACHE_PREFETCH,
                    summary="Warm retrieval candidates likely to be reused next cycle",
                    rationale="The next cycle may revisit the same evidence cluster.",
                    target_components=("model_manager",),
                    source_task_ids=("phase18_governor_advisory_task",),
                    confidence=0.6,
                    advisory_only=True,
                    metadata={
                        "cache_namespace": "retrieval_candidates",
                        "warm_keys": (
                            "task:phase18:1",
                            "task:phase18:2",
                            "task:phase18:3",
                            "task:phase18:4",
                            "task:phase18:5",
                        ),
                        "retention_seconds": 300,
                    },
                ),
            )

        with (
            mock.patch.object(self.orchestrator, "_long_horizon_session_id", return_value=session_id),
            mock.patch.object(self.orchestrator, "_run_bounded_task", side_effect=fake_run_bounded_task),
            mock.patch.object(
                self.orchestrator.self_optimizer,
                "suggest_for_long_horizon",
                side_effect=fake_suggest_for_long_horizon,
            ),
        ):
            result = await self.orchestrator.run_task("phase18 governor advisory routing", thinking_minutes=121)

        self.assertIn("long_horizon_checkpointed_run", result.warnings)
        self.orchestrator.model_manager._active_generation_jobs = self.config.concurrency.generation_slots
        decision = self.orchestrator.model_manager.route_role(ModelRole.CODE_SPECIALIST, capability="code_assist")
        await self.orchestrator._publish_dashboard_model_registry_view()
        state = self.orchestrator.dashboard.app_state_snapshot()

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.fallback_reason, "hardware_governor_optional_heavy_residency_disabled")
        self.assertIn("generation_queue_pressure", decision.metadata["pressure_reasons"])
        self.assertIn("retain(code_specialist)", state.model_registry_view.governor_summary)
        self.assertIn("prefetch(retrieval_candidates:4)", state.model_registry_view.governor_summary)

    async def test_compressor_reuses_shared_compression_artifact_scores(self) -> None:
        result = _build_stub_task_result(
            task_id="phase18_compression_task",
            question="phase18 compression scoring",
            answer_text="phase18 compression scoring",
            candidate_count=2,
            candidate_score=0.8,
            local_evidence_ids=("ev-1",),
        )
        logs = [
            ReasoningLog(
                task_id="phase18_log_1",
                compressed_chain=("lookup", "compare", "emit"),
                macros_used=(),
            ),
            ReasoningLog(
                task_id="phase18_log_2",
                compressed_chain=("lookup", "compare", "emit"),
                macros_used=(),
            ),
        ]
        proposals_before = await self.orchestrator.compressor.propose(result.reasoning, logs=logs)
        top_before = proposals_before[0]
        self.orchestrator.model_manager.warm_cache(
            "compression_artifacts",
            top_before.proof_fingerprint or top_before.proposal_id,
            {
                "proposal_id": top_before.proposal_id,
                "proof_fingerprint": top_before.proof_fingerprint,
                "effectiveness_score": 1.0,
                "validation_pass_rate": 1.0,
                "source": "self_optimizer",
            },
        )

        proposals_after = await self.orchestrator.compressor.propose(result.reasoning, logs=logs)
        rescored = next(
            item
            for item in proposals_after
            if item.proof_fingerprint == top_before.proof_fingerprint
        )

        self.assertGreater(rescored.simulation_score, top_before.simulation_score)


class Phase18StartupProfileTests(unittest.IsolatedAsyncioTestCase):
    def test_profile_preferences_can_override_startup_model_identifiers(self) -> None:
        config = _build_test_config(
            sqlite_name="test_phase18_profile_override.sqlite3",
            logs_name="test_phase18_profile_override_logs",
        )
        orchestrator = Orchestrator(config=config)
        profile = UserSettingsProfile(
            profile_name="default",
            runtime={
                "stub_mode": False,
                "generation_backend": "ollama",
                "embedding_backend": "ollama_embeddings",
                "vector_store_backend": "chromadb",
            },
            models={
                "preferred_by_role": {
                    "generation": "ollama:qwen2.5:3b-instruct-q4_K_M",
                    "embedding": "ollama_embeddings:nomic-embed-text",
                },
                "enabled_roles": ("generation", "embedding"),
            },
        )

        resolved = orchestrator._config_for_user_settings_profile(profile)

        self.assertFalse(resolved.preflight.flags.stub_mode)
        self.assertEqual(resolved.preflight.backends.generation_backend, "ollama")
        self.assertEqual(resolved.preflight.backends.generation_model, "qwen2.5:3b-instruct-q4_K_M")
        self.assertEqual(resolved.preflight.backends.embedding_backend, "ollama_embeddings")
        self.assertEqual(resolved.preflight.backends.embedding_model, "nomic-embed-text")

    async def test_start_uses_persisted_profile_before_model_manager_start(self) -> None:
        config = _build_test_config(
            sqlite_name="test_phase18_startup_profile.sqlite3",
            logs_name="test_phase18_startup_profile_logs",
        )
        profile = UserSettingsProfile(
            profile_name="default",
            runtime={
                "stub_mode": False,
                "allow_web_fallback": True,
                "enable_self_optimizer": False,
                "generation_backend": "ollama",
                "embedding_backend": "sentence_transformers",
                "vector_store_backend": "simple_inmemory",
            },
            retrieval={
                "allow_web_fallback": True,
                "provider": "wikipedia",
                "reranking": True,
            },
        )
        storage = _StorageStub(config, profile)
        model_manager = _ModelManagerStartupStub(config)
        dashboard = _DashboardStartupStub(config)
        component = _AsyncComponentStub(config)
        orchestrator = Orchestrator(
            config=config,
            storage=storage,
            model_manager=model_manager,
            dashboard=dashboard,
            planner=component,
            researcher=component,
            reasoner=component,
            critic=component,
            compressor=component,
            self_optimizer=component,
            phase11_content=_Phase11ContentStub(config),
        )
        internal_async_methods = (
            "_publish_dashboard_capability_registry_view",
            "_publish_dashboard_model_registry_view",
            "_emit_event",
            "_emit_health_snapshot",
            "_record_status",
            "_publish_dashboard_settings_profiles",
            "_publish_dashboard_task_history",
            "_publish_dashboard_knowledge_library",
            "_publish_dashboard_readiness_report",
            "_recover_local_task_session",
            "_publish_dashboard_examples",
        )
        with mock.patch.multiple(
            orchestrator,
            **{name: mock.AsyncMock() for name in internal_async_methods},
        ):
            await orchestrator.start()

        self.assertFalse(model_manager.started_with_stub_mode)
        self.assertFalse(orchestrator.config.preflight.flags.stub_mode)
        self.assertIs(model_manager.applied_profile, profile)
        self.assertIs(dashboard.applied_profile, profile)

    async def test_real_mode_can_auto_start_user_space_ollama(self) -> None:
        config = _build_test_config(
            sqlite_name="test_phase18_ollama_autostart.sqlite3",
            logs_name="test_phase18_ollama_autostart_logs",
        )
        config = replace(
            config,
            preflight=replace(
                config.preflight,
                flags=replace(config.preflight.flags, stub_mode=False),
            ),
        )
        orchestrator = Orchestrator(
            config=config,
            storage=_StorageStub(config, UserSettingsProfile()),
            model_manager=_ModelManagerStartupStub(config),
            dashboard=_DashboardStartupStub(config),
            planner=_AsyncComponentStub(config),
            researcher=_AsyncComponentStub(config),
            reasoner=_AsyncComponentStub(config),
            critic=_AsyncComponentStub(config),
            compressor=_AsyncComponentStub(config),
            self_optimizer=_AsyncComponentStub(config),
            phase11_content=_Phase11ContentStub(config),
        )

        with (
            mock.patch.object(
                orchestrator,
                "_probe_ollama_service",
                side_effect=[(False, "down"), (True, "ready")],
            ) as probe,
            mock.patch.object(
                orchestrator,
                "_discover_ollama_binary",
                return_value=Path(r"C:\Users\Pgiov\AppData\Local\Programs\OllamaPortable\ollama.exe"),
            ),
            mock.patch("orchestrator.subprocess.Popen") as popen,
        ):
            ready, detail = await orchestrator._ensure_local_ollama_service()

        self.assertTrue(ready)
        self.assertIn("Auto-started Ollama service", detail)
        self.assertEqual(probe.call_count, 2)
        popen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
