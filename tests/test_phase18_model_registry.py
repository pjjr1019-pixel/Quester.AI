"""Phase 18 registry, routing, control-plane, and bounded-cache regressions."""

from __future__ import annotations

import shutil
import time
import unittest
from dataclasses import replace
from pathlib import Path
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

    async def test_heavy_slot_cap_blocks_optional_heavy_roles_but_sidecars_do_not_consume_slots(self) -> None:
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

        self.assertFalse(heavy_decision.allowed)
        self.assertEqual(heavy_decision.fallback_reason, "heavy_slot_cap_reached")
        self.assertTrue(sidecar_decision.allowed)
        self.assertEqual(len(view.active_heavy_roles), 2)

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


if __name__ == "__main__":
    unittest.main()
