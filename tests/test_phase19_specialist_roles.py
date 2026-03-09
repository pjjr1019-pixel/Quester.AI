"""Phase 19 specialist-role routing, reranking, and control-plane regressions."""

from __future__ import annotations

import math
import shutil
import struct
import unittest
import wave
from dataclasses import replace
from pathlib import Path

from config import APP_CONFIG
from data_structures import (
    ModelRole,
    OptimizerSuggestionKind,
    OptimizerSuggestionRecord,
    ResourceBudget,
    UserSettingsProfile,
)
from model_manager import ModelManager
from orchestrator import Orchestrator
from research_service import ResearchService
from retrieval import DocumentChunkRecord, SearchResult


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


def _specialist_profile() -> UserSettingsProfile:
    return UserSettingsProfile.from_dict(
        {
            "profile_name": "phase19-specialists",
            "models": {
                "preferred_by_role": {
                    "reranker": "stub_reranker:stub-reranker",
                    "speech_to_text": "stub_speech_to_text:stub-whisper-tiny",
                    "text_to_speech": "stub_text_to_speech:stub-piper",
                    "vad": "stub_vad:stub-silero-vad",
                    "translation": "stub_translation:stub-argos",
                    "code_specialist": "stub_code_specialist:stub-qwen-coder-1.5b",
                },
                "enabled_roles": (
                    "generation",
                    "embedding",
                    "reranker",
                    "speech_to_text",
                    "text_to_speech",
                    "vad",
                    "translation",
                    "code_specialist",
                ),
            },
        }
    )


def _chunk(
    *,
    chunk_id: str,
    title: str,
    content: str,
) -> DocumentChunkRecord:
    return DocumentChunkRecord(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        source_ref=f"local://{chunk_id}",
        chunk_index=0,
        content=content,
        content_hash=f"hash-{chunk_id}",
        metadata={"title": title},
        embedding_model="stub-embedding",
        vector=(0.1, 0.2, 0.3, 0.4),
        created_at="2026-03-08T00:00:00Z",
    )


class _FakeResearchStorage:
    def __init__(self, results: tuple[SearchResult, ...]) -> None:
        self._results = results

    async def log_event(self, stage: str, payload: dict[str, object]) -> None:
        _ = (stage, payload)

    async def search_local_chunks(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        metadata_exclusions: dict[str, object] | None = None,
        allow_rerank: bool = True,
    ):
        _ = (query_text, query_vector, top_k, metadata_exclusions, allow_rerank)
        return self._results

    async def record_web_evidence_batch(self, records) -> None:
        _ = records

    async def count_documents(self) -> int:
        return 1

    async def ingest_document(
        self,
        *,
        source_ref: str,
        title: str,
        content: str,
        metadata: dict[str, object] | None,
        embed_document,
        embedding_model_name: str | None = None,
    ) -> None:
        _ = (source_ref, title, content, metadata, embed_document, embedding_model_name)


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


def _write_voice_like_wav(path: Path, *, sample_rate_hz: int = 16000) -> None:
    frame_count = int(sample_rate_hz * 1.2)
    silence_count = int(sample_rate_hz * 0.2)
    frames: list[int] = []
    for index in range(frame_count):
        sample = int(12000 * math.sin(2.0 * math.pi * 220.0 * (index / sample_rate_hz)))
        frames.append(sample)
    frames.extend([0] * silence_count)
    for index in range(frame_count):
        sample = int(10000 * math.sin(2.0 * math.pi * 330.0 * (index / sample_rate_hz)))
        frames.append(sample)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(struct.pack(f"<{len(frames)}h", *frames))


class Phase19SpecialistRoleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase19_specialists.sqlite3")
        self.test_logs = Path("test_phase19_specialists_logs")
        self.config = _build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))

    async def asyncTearDown(self) -> None:
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_specialist_roles_enable_route_unload_and_preserve_base_runtime(self) -> None:
        manager = ModelManager(config=self.config)
        await manager.start()
        try:
            manager.apply_user_settings_profile(_specialist_profile())

            reranker = manager.route_role(ModelRole.RERANKER, capability="rerank")
            speech_to_text = manager.route_role(ModelRole.SPEECH_TO_TEXT, capability="transcribe")
            vad = manager.route_role(ModelRole.VAD, capability="voice_activity_detection")
            translation = manager.route_role(ModelRole.TRANSLATION, capability="translate")
            code_specialist = manager.route_role(ModelRole.CODE_SPECIALIST, capability="code_assist")

            self.assertTrue(reranker.allowed)
            self.assertTrue(speech_to_text.allowed)
            self.assertTrue(vad.allowed)
            self.assertTrue(translation.allowed)
            self.assertTrue(code_specialist.allowed)
            self.assertTrue(manager.unload_optional_role(ModelRole.RERANKER))
            self.assertTrue(manager.unload_optional_role(ModelRole.SPEECH_TO_TEXT))
            self.assertTrue(manager.unload_optional_role(ModelRole.VAD))
            self.assertTrue(manager.unload_optional_role(ModelRole.TRANSLATION))
            self.assertTrue(manager.unload_optional_role(ModelRole.CODE_SPECIALIST))

            generation = await manager.generate("phase19 baseline", max_tokens=24)
            embedding = await manager.embed("phase19 baseline")

            self.assertTrue(generation.startswith("[stub]"))
            self.assertTrue(embedding)
        finally:
            await manager.stop()

    async def test_vad_and_stub_transcription_produce_bounded_voice_input_result(self) -> None:
        manager = ModelManager(config=self.config)
        audio_path = self.test_logs / "what_is_two_plus_two.wav"
        self.test_logs.mkdir(parents=True, exist_ok=True)
        _write_voice_like_wav(audio_path)
        await manager.start()
        try:
            manager.apply_user_settings_profile(_specialist_profile())

            vad_report = await manager.detect_voice_activity(audio_path)
            transcription = await manager.transcribe_audio(audio_path)

            self.assertGreaterEqual(vad_report.segment_count, 1)
            self.assertGreater(vad_report.speech_ratio, 0.0)
            self.assertEqual(transcription.status, "transcribed")
            self.assertEqual(transcription.transcript_text, "what is two plus two")
            self.assertEqual(transcription.normalized_question, "what is two plus two?")
            self.assertTrue(transcription.used_vad)
            self.assertEqual(transcription.transcription_backend, "stub_speech_to_text")
        finally:
            await manager.stop()

    async def test_speech_to_text_can_run_when_vad_role_is_disabled(self) -> None:
        manager = ModelManager(config=self.config)
        audio_path = self.test_logs / "explain_runtime_limits.wav"
        self.test_logs.mkdir(parents=True, exist_ok=True)
        _write_voice_like_wav(audio_path)
        await manager.start()
        try:
            profile = UserSettingsProfile.from_dict(
                {
                    "profile_name": "phase19-stt-only",
                    "models": {
                        "preferred_by_role": {
                            "speech_to_text": "stub_speech_to_text:stub-whisper-tiny",
                        },
                        "enabled_roles": ("generation", "embedding", "speech_to_text"),
                    },
                }
            )
            manager.apply_user_settings_profile(profile)

            transcription = await manager.transcribe_audio(audio_path)

            self.assertEqual(transcription.status, "transcribed")
            self.assertFalse(transcription.used_vad)
            self.assertIn("vad_role_disabled", transcription.warnings)
        finally:
            await manager.stop()

    async def test_text_to_speech_stub_synthesis_produces_bounded_local_wav(self) -> None:
        manager = ModelManager(config=self.config)
        output_path = self.test_logs / "phase19_answer.wav"
        self.test_logs.mkdir(parents=True, exist_ok=True)
        await manager.start()
        try:
            manager.apply_user_settings_profile(_specialist_profile())

            synthesis = await manager.synthesize_text(
                "This is a bounded local speech test.",
                output_path=output_path,
            )

            self.assertEqual(synthesis.status, "synthesized")
            self.assertEqual(synthesis.synthesis_backend, "stub_text_to_speech")
            self.assertEqual(synthesis.target_path, str(output_path))
            self.assertTrue(output_path.exists())
            self.assertGreater(synthesis.duration_seconds, 0.0)
            self.assertEqual(synthesis.sample_rate_hz, self.config.audio.tts_sample_rate_hz)
        finally:
            await manager.stop()

    async def test_translation_role_translates_bounded_local_text(self) -> None:
        manager = ModelManager(config=self.config)
        await manager.start()
        try:
            manager.apply_user_settings_profile(_specialist_profile())

            result = await manager.translate_text(
                "hello local models",
                source_language="en",
                target_language="es",
                source_scope="free_text",
            )

            self.assertEqual(result.status, "translated")
            self.assertEqual(result.translation_backend, "stub_translation")
            self.assertEqual(result.source_language, "en")
            self.assertEqual(result.target_language, "es")
            self.assertIn("hola", result.translated_text.lower())
        finally:
            await manager.stop()

    async def test_code_specialist_stub_analyzes_bounded_local_code(self) -> None:
        manager = ModelManager(config=self.config)
        await manager.start()
        try:
            manager.apply_user_settings_profile(_specialist_profile())

            result = await manager.analyze_code(
                text="import asyncio\n\nasync def run_task():\n    await asyncio.sleep(0)\n",
                request_text="Summarize maintenance risks.",
                source_scope="snippet",
            )

            self.assertEqual(result.status, "analyzed")
            self.assertEqual(result.code_backend, "stub_code_specialist")
            self.assertEqual(result.detected_language, "python")
            self.assertIn("maintenance", result.summary.lower())
            self.assertTrue(result.suggested_actions)
        finally:
            await manager.stop()

    async def test_translation_and_code_roles_remain_optional_when_disabled(self) -> None:
        manager = ModelManager(config=self.config)
        await manager.start()
        try:
            profile = UserSettingsProfile.from_dict(
                {
                    "profile_name": "phase19-optional-baseline",
                    "models": {
                        "enabled_roles": ("generation", "embedding"),
                    },
                }
            )
            manager.apply_user_settings_profile(profile)

            translation = await manager.translate_text(
                "hello",
                source_language="en",
                target_language="es",
            )
            code_result = await manager.analyze_code(
                text="def noop():\n    return None\n",
                request_text="Summarize this helper.",
            )
            generation = await manager.generate("phase19 optional baseline", max_tokens=24)

            self.assertEqual(translation.status, "blocked")
            self.assertEqual(code_result.status, "blocked")
            self.assertTrue(generation.startswith("[stub]"))
        finally:
            await manager.stop()

    async def test_reranker_role_reorders_bounded_local_results(self) -> None:
        manager = ModelManager(config=self.config)
        await manager.start()
        try:
            manager.apply_user_settings_profile(_specialist_profile())
            exact_match = SearchResult(
                chunk=_chunk(
                    chunk_id="chunk-exact",
                    title="Local Retrieval Foundation",
                    content="Local retrieval foundation notes describe stable chunk IDs.",
                ),
                lexical_score=0.60,
                vector_score=0.58,
                combined_score=0.62,
            )
            high_baseline = SearchResult(
                chunk=_chunk(
                    chunk_id="chunk-high",
                    title="Runtime Notes",
                    content="Resource budgeting and dashboards stay bounded under local load.",
                ),
                lexical_score=0.75,
                vector_score=0.82,
                combined_score=0.85,
            )

            reranked, decision = await manager.rerank_local_results(
                query_text="local retrieval foundation",
                results=(high_baseline, exact_match),
                top_k=2,
            )

            self.assertTrue(decision.allowed)
            self.assertEqual(reranked[0].chunk.chunk_id, "chunk-exact")
            self.assertGreater(reranked[0].rerank_score, reranked[1].rerank_score)
        finally:
            await manager.stop()

    async def test_research_service_uses_specialist_reranker_metadata(self) -> None:
        manager = ModelManager(config=self.config)
        await manager.start()
        try:
            manager.apply_user_settings_profile(_specialist_profile())
            exact_match = SearchResult(
                chunk=_chunk(
                    chunk_id="chunk-exact",
                    title="Local Retrieval Foundation",
                    content="Local retrieval foundation notes describe stable chunk IDs.",
                ),
                lexical_score=0.60,
                vector_score=0.58,
                combined_score=0.62,
            )
            high_baseline = SearchResult(
                chunk=_chunk(
                    chunk_id="chunk-high",
                    title="Runtime Notes",
                    content="Resource budgeting and dashboards stay bounded under local load.",
                ),
                lexical_score=0.75,
                vector_score=0.82,
                combined_score=0.85,
            )
            storage = _FakeResearchStorage(results=(high_baseline, exact_match))
            service = ResearchService(model_manager=manager, storage=storage, config=self.config)
            budget = ResourceBudget(
                retrieval_top_k=self.config.retrieval.rerank_min_budget_top_k,
                max_web_queries=1,
                reasoner_passes=1,
                critic_passes=1,
                macro_depth=2,
            )

            local_results, decision = await service._build_local_results(
                "local retrieval foundation",
                budget,
                [0.1, 0.2, 0.3, 0.4],
            )

            self.assertTrue(decision is not None and decision.allowed)
            self.assertEqual(local_results[0].id, "chunk-exact")
            self.assertTrue(local_results[0].metadata["specialist_reranked"])
            self.assertEqual(local_results[0].metadata["specialist_reranker_backend"], "stub_reranker")
        finally:
            await manager.stop()

    async def test_research_service_skips_specialist_reranker_when_not_enabled_or_needed(self) -> None:
        manager = ModelManager(config=self.config)
        await manager.start()
        try:
            baseline = SearchResult(
                chunk=_chunk(
                    chunk_id="chunk-baseline",
                    title="Runtime Notes",
                    content="Resource budgeting and dashboards stay bounded under local load.",
                ),
                lexical_score=0.75,
                vector_score=0.82,
                combined_score=0.85,
            )
            storage = _FakeResearchStorage(results=(baseline,))
            service = ResearchService(model_manager=manager, storage=storage, config=self.config)
            budget = ResourceBudget(
                retrieval_top_k=2,
                max_web_queries=1,
                reasoner_passes=1,
                critic_passes=1,
                macro_depth=2,
            )

            local_results, decision = await service._build_local_results(
                "runtime notes",
                budget,
                [0.1, 0.2, 0.3, 0.4],
            )

            self.assertIsNone(decision)
            self.assertEqual(local_results[0].id, "chunk-baseline")
            self.assertFalse(local_results[0].metadata["specialist_reranked"])
            self.assertEqual(local_results[0].metadata["specialist_reranker_backend"], "")
        finally:
            await manager.stop()


class Phase19SpecialistControlPlaneTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase19_control_plane.sqlite3")
        self.test_logs = Path("test_phase19_control_plane_logs")
        self.config = _build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_dashboard_and_readiness_surface_specialist_roles(self) -> None:
        profile = _specialist_profile()

        await self.orchestrator._save_dashboard_settings_and_refresh(profile)

        state = self.orchestrator.dashboard.app_state_snapshot()
        enabled_roles = {
            registration.role.value
            for registration in state.model_registry_view.registrations
            if registration.enabled and registration.backend != "unconfigured"
        }
        checks = {check.check_id: check for check in state.readiness_report.checks}
        formatted = self.orchestrator.dashboard._format_model_registry_view(state.model_registry_view)

        self.assertIn("reranker", enabled_roles)
        self.assertIn("speech_to_text", enabled_roles)
        self.assertIn("text_to_speech", enabled_roles)
        self.assertIn("vad", enabled_roles)
        self.assertIn("translation", enabled_roles)
        self.assertIn("code_specialist", enabled_roles)
        self.assertEqual(checks["specialist_roles"].status, "ready")
        self.assertIn("reranker: stub_reranker / stub-reranker", formatted)
        self.assertIn("speech_to_text: stub_speech_to_text / stub-whisper-tiny", formatted)
        self.assertIn("text_to_speech: stub_text_to_speech / stub-piper", formatted)
        self.assertIn("vad: stub_vad / stub-silero-vad", formatted)
        self.assertIn("translation: stub_translation / stub-argos", formatted)
        self.assertIn("code_specialist: stub_code_specialist / stub-qwen-coder-1.5b", formatted)

    async def test_unified_model_registry_panel_surfaces_optimizer_suggestions(self) -> None:
        await self.orchestrator.storage.record_optimizer_suggestion_records(
            (
                OptimizerSuggestionRecord(
                    suggestion_id="phase19_dashboard_hint",
                    cycle_id="phase19_cycle",
                    kind=OptimizerSuggestionKind.DASHBOARD_HINT,
                    summary="Surface routed local-AI decisions in the control plane.",
                    rationale="Phase 19 control-plane regression.",
                    target_components=("dashboard",),
                    source_task_ids=("phase19_task",),
                    confidence=0.81,
                ),
            )
        )

        await self.orchestrator._publish_dashboard_model_registry_view()

        state = self.orchestrator.dashboard.app_state_snapshot()
        formatted = self.orchestrator.dashboard._format_model_registry_view(state.model_registry_view)

        self.assertEqual(len(state.model_registry_view.recent_optimizer_suggestions), 1)
        self.assertIn("Recent optimizer suggestions:", formatted)
        self.assertIn("dashboard_hint: Surface routed local-AI decisions in the control plane.", formatted)
        self.assertIn("text_to_speech: 0/", formatted)
        self.assertIn("translation: 0/", formatted)

    async def test_control_plane_quick_actions_manage_roles_and_publish_typed_reports(self) -> None:
        await self.orchestrator._run_dashboard_action(
            action="model.enable_role",
            payload={"role": "reranker"},
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        reranker_registration = next(
            registration
            for registration in state.model_registry_view.registrations
            if registration.role.value == "reranker" and registration.backend == "stub_reranker"
        )
        self.assertIn("reranker", state.user_settings.models["enabled_roles"])
        self.assertTrue(reranker_registration.enabled)
        self.assertEqual(state.model_role_action.action, "enable_role")
        self.assertTrue(state.model_role_action.ok)

        await self.orchestrator._run_dashboard_action(
            action="model.warm_role",
            payload={"role": "reranker"},
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertEqual(state.model_role_action.action, "warm_role")
        self.assertTrue(state.model_role_action.ok)
        self.assertIsNotNone(state.model_role_action.route_decision)
        assert state.model_role_action.route_decision is not None
        self.assertTrue(state.model_role_action.route_decision.allowed)

        await self.orchestrator._run_dashboard_action(
            action="model.test_ping",
            payload={"role": "reranker"},
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertEqual(state.model_role_action.action, "test_ping")
        self.assertTrue(state.model_role_action.ok)
        self.assertIn("Reranker route is ready", state.model_role_action.detail)

        await self.orchestrator._run_dashboard_action(
            action="model.disable_role",
            payload={"role": "reranker"},
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        reranker_registration = next(
            registration
            for registration in state.model_registry_view.registrations
            if registration.role.value == "reranker" and registration.backend == "stub_reranker"
        )
        self.assertNotIn("reranker", state.user_settings.models["enabled_roles"])
        self.assertFalse(reranker_registration.enabled)
        self.assertEqual(state.model_role_action.action, "disable_role")

    async def test_install_guidance_and_fallback_inspection_stay_explainable(self) -> None:
        await self.orchestrator._run_dashboard_action(
            action="model.install_guidance",
            payload={"role": "text_to_speech"},
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertEqual(state.model_role_action.action, "install_guidance")
        joined_guidance = " ".join(state.model_role_action.guidance)
        self.assertTrue(
            "System.Speech" in joined_guidance or "stub-piper" in joined_guidance or "Piper" in joined_guidance
        )

        await self.orchestrator._run_dashboard_action(
            action="model.install_guidance",
            payload={"role": "translation"},
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertIn("Argos Translate", " ".join(state.model_role_action.guidance))

        await self.orchestrator._run_dashboard_action(
            action="model.install_guidance",
            payload={"role": "code_specialist"},
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertIn("Qwen/Qwen2.5-Coder-1.5B-Instruct", " ".join(state.model_role_action.guidance))

        await self.orchestrator._run_dashboard_action(
            action="model.install_guidance",
            payload={"role": "vision"},
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertIn("HuggingFaceTB/SmolVLM-256M-Instruct", " ".join(state.model_role_action.guidance))

        await self.orchestrator._run_dashboard_action(
            action="model.install_guidance",
            payload={"role": "specialist_perception"},
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertIn("PaddleOCR", " ".join(state.model_role_action.guidance))

        await self.orchestrator._run_dashboard_action(
            action="model.test_ping",
            payload={"role": "reranker"},
        )
        await self.orchestrator._run_dashboard_action(
            action="model.inspect_fallback",
            payload={"role": "reranker"},
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        detail = self.orchestrator.dashboard._format_model_role_detail(
            "reranker",
            state.model_registry_view,
            state.model_role_action,
        )
        self.assertEqual(state.model_role_action.action, "inspect_fallback")
        self.assertIn("role_disabled", state.model_role_action.detail)
        self.assertIn("Last quick action:", detail)
        self.assertIn("Fallback reasons:", detail)

    async def test_audio_transcript_can_fill_dashboard_question_box(self) -> None:
        profile = _specialist_profile()
        audio_path = self.test_logs / "how_long_can_you_think.wav"
        self.test_logs.mkdir(parents=True, exist_ok=True)
        _write_voice_like_wav(audio_path)
        self.orchestrator.dashboard._question_var = _FakeVar("")

        await self.orchestrator._save_dashboard_settings_and_refresh(profile)
        await self.orchestrator._run_dashboard_action(
            action="audio.transcribe_file",
            payload={"path": str(audio_path)},
        )
        await self.orchestrator._run_dashboard_action(
            action="audio.use_transcript_as_question",
            payload={},
        )

        state = self.orchestrator.dashboard.app_state_snapshot()

        self.assertEqual(state.audio_input.status, "transcribed")
        self.assertEqual(state.audio_input.transcript_text, "how long can you think")
        self.assertTrue(state.audio_input.imported_into_question)
        self.assertEqual(self.orchestrator.dashboard._question_var.get(), "how long can you think?")

    async def test_audio_output_can_synthesize_the_current_answer(self) -> None:
        profile = _specialist_profile()
        output_path = self.test_logs / "phase19_current_answer.wav"
        self.test_logs.mkdir(parents=True, exist_ok=True)

        await self.orchestrator._save_dashboard_settings_and_refresh(profile)
        await self.orchestrator.run_task("What is 2 + 2?", thinking_minutes=30)
        await self.orchestrator._run_dashboard_action(
            action="audio.speak_answer",
            payload={"path": str(output_path)},
        )

        state = self.orchestrator.dashboard.app_state_snapshot()

        self.assertEqual(state.audio_output.status, "synthesized")
        self.assertEqual(state.audio_output.synthesis_backend, "stub_text_to_speech")
        self.assertEqual(state.audio_output.target_path, str(output_path))
        self.assertTrue(output_path.exists())
        self.assertTrue(state.audio_output.clipped_text)

    async def test_translation_output_can_translate_answer_and_fill_question_box(self) -> None:
        profile = _specialist_profile()
        self.orchestrator.dashboard._question_var = _FakeVar("")

        await self.orchestrator._save_dashboard_settings_and_refresh(profile)
        await self.orchestrator.run_task("What is 2 + 2?", thinking_minutes=30)
        await self.orchestrator._run_dashboard_action(
            action="translation.translate_answer",
            payload={"source_language": "en", "target_language": "es"},
        )
        await self.orchestrator._run_dashboard_action(
            action="translation.use_as_question",
            payload={},
        )

        state = self.orchestrator.dashboard.app_state_snapshot()

        self.assertEqual(state.translation_output.status, "translated")
        self.assertEqual(state.translation_output.translation_backend, "stub_translation")
        self.assertEqual(state.translation_output.target_language, "es")
        self.assertTrue(state.translation_output.imported_into_question)
        self.assertEqual(
            self.orchestrator.dashboard._question_var.get(),
            state.translation_output.translated_text,
        )

    async def test_code_specialist_can_analyze_local_file_from_dashboard(self) -> None:
        profile = _specialist_profile()
        code_path = self.test_logs / "phase19_dashboard_code.py"
        self.test_logs.mkdir(parents=True, exist_ok=True)
        code_path.write_text(
            "import asyncio\n\nasync def run_task():\n    await asyncio.sleep(0)\n",
            encoding="utf-8",
        )

        await self.orchestrator._save_dashboard_settings_and_refresh(profile)
        await self.orchestrator._run_dashboard_action(
            action="code.analyze_file",
            payload={"path": str(code_path), "request_text": "Summarize maintenance risks."},
        )

        state = self.orchestrator.dashboard.app_state_snapshot()

        self.assertEqual(state.code_output.status, "analyzed")
        self.assertEqual(state.code_output.code_backend, "stub_code_specialist")
        self.assertEqual(state.code_output.source_path, str(code_path))
        self.assertEqual(state.code_output.detected_language, "python")
        self.assertTrue(state.code_output.suggested_actions)

    async def test_task_inspector_explains_specialist_role_usage_and_advisor_summaries(self) -> None:
        profile = _specialist_profile()

        await self.orchestrator._save_dashboard_settings_and_refresh(profile)
        await self.orchestrator.run_task("local retrieval foundation", thinking_minutes=30)
        task_id = self.orchestrator.dashboard.app_state_snapshot().active_task.task_id
        self.assertTrue(task_id)

        await self.orchestrator.storage.record_optimizer_suggestion_records(
            (
                OptimizerSuggestionRecord(
                    suggestion_id="phase19_task_specific_hint",
                    cycle_id="phase19_cycle",
                    kind=OptimizerSuggestionKind.RETRIEVAL_STRATEGY,
                    summary="Keep the reranked local evidence ordering for this retrieval-heavy task.",
                    rationale="Phase 19 task inspector regression.",
                    target_components=("researcher",),
                    source_task_ids=(task_id,),
                    confidence=0.78,
                ),
            )
        )
        await self.orchestrator._publish_dashboard_task_detail(task_id)

        selected = self.orchestrator.dashboard.app_state_snapshot().selected_task

        self.assertIn("reranker", selected.specialist_roles_used)
        self.assertTrue(
            any("reranker used" in item for item in selected.specialist_role_explanations)
        )
        self.assertTrue(
            any("retrieval_strategy" in item for item in selected.advisor_summaries)
        )

    async def test_control_plane_surfaces_typed_bounded_compression_insights(self) -> None:
        profile = _specialist_profile()

        await self.orchestrator._save_dashboard_settings_and_refresh(profile)
        for index in range(6):
            self.orchestrator.model_manager.warm_cache(
                "compression_artifacts",
                f"proof-{index}",
                {
                    "proposal_id": f"proposal-{index}",
                    "proof_fingerprint": f"proof-{index}",
                    "macro_name": f"macro_{index}",
                    "compression_gain": 0.2 + (index * 0.05),
                    "validation_pass_rate": 1.0 if index % 2 == 0 else 0.0,
                    "validation_state": "validated" if index % 2 == 0 else "blocked",
                    "blocked_reason": "" if index % 2 == 0 else "validation_failed",
                    "accepted": index % 2 == 0,
                    "evidence_basis": "deterministic_analysis" if index < 3 else "replay_evidence",
                    "origin_component": "compression_service" if index < 3 else "self_optimizer",
                    "source": "compression_service" if index < 3 else "self_optimizer",
                },
            )
        await self.orchestrator._publish_dashboard_model_registry_view()

        state = self.orchestrator.dashboard.app_state_snapshot()
        formatted = self.orchestrator.dashboard._format_model_registry_view(state.model_registry_view)

        self.assertLessEqual(len(state.model_registry_view.compression_insights), 4)
        first_insight = state.model_registry_view.compression_insights[0]
        self.assertTrue(first_insight.proposal_id)
        self.assertIn(first_insight.validation_state, {"validated", "blocked"})
        self.assertTrue(first_insight.evidence_basis)
        self.assertIn("Compression insights:", formatted)
        self.assertIn("basis", formatted)


if __name__ == "__main__":
    unittest.main()
