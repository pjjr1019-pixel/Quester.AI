"""Phase 3 runtime/backend tests."""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace

from config import APP_CONFIG
from model_backends import BackendHealth
from model_manager import ModelManager
from runtime_errors import BackendStartupError, BackendUnavailableError, ResourcePressureError


class FakeGenerationBackend:
    """Simple async generation backend for runtime tests."""

    def __init__(
        self,
        name: str,
        *,
        start_error: Exception | None = None,
        generate_error: Exception | None = None,
        estimated_vram_gb: float | None = None,
        unloadable: bool = False,
        delay_s: float = 0.02,
    ) -> None:
        self.backend_name = name
        self.model_name = f"{name}-model"
        self.start_error = start_error
        self.generate_error = generate_error
        self.estimated_vram_gb = estimated_vram_gb
        self.unloadable = unloadable
        self.delay_s = delay_s
        self.started = False
        self.unloaded = False
        self.calls = 0
        self.max_inflight = 0
        self._inflight = 0

    async def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started = True
        self.unloaded = False

    async def stop(self) -> None:
        self.started = False

    async def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        _ = (prompt, max_tokens, temperature)
        if self.generate_error is not None:
            raise self.generate_error
        self.calls += 1
        self._inflight += 1
        self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            await asyncio.sleep(self.delay_s)
            return f"{self.backend_name}-ok"
        finally:
            self._inflight -= 1

    async def health(self) -> BackendHealth:
        return BackendHealth(
            backend_name=self.backend_name,
            model_name=self.model_name,
            started=self.started,
            available=self.started,
            mode="test",
            estimated_vram_gb=self.estimated_vram_gb,
            metadata={"unloaded": self.unloaded},
        )

    async def unload(self) -> bool:
        if not self.unloadable or not self.started:
            return False
        self.unloaded = True
        self.started = False
        return True


class FakeEmbeddingBackend:
    """Simple async embedding backend for runtime tests."""

    def __init__(
        self,
        name: str,
        *,
        start_error: Exception | None = None,
        embed_error: Exception | None = None,
        unloadable: bool = False,
        delay_s: float = 0.01,
    ) -> None:
        self.backend_name = name
        self.model_name = f"{name}-model"
        self.start_error = start_error
        self.embed_error = embed_error
        self.unloadable = unloadable
        self.delay_s = delay_s
        self.started = False
        self.unloaded = False
        self.calls = 0

    async def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started = True
        self.unloaded = False

    async def stop(self) -> None:
        self.started = False

    async def embed(self, text: str) -> list[float]:
        _ = text
        if self.embed_error is not None:
            raise self.embed_error
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return [0.1, 0.2, 0.3]

    async def health(self) -> BackendHealth:
        return BackendHealth(
            backend_name=self.backend_name,
            model_name=self.model_name,
            started=self.started,
            available=self.started,
            mode="test",
            metadata={"unloaded": self.unloaded},
        )

    async def unload(self) -> bool:
        if not self.unloadable or not self.started:
            return False
        self.unloaded = True
        self.started = False
        return True


class Phase3ModelManagerTests(unittest.IsolatedAsyncioTestCase):
    """Validate backend abstraction, fallback, telemetry, and idle unload."""

    def _build_config(self, **overrides):
        preflight = replace(APP_CONFIG.preflight, flags=replace(APP_CONFIG.preflight.flags, stub_mode=False))
        backend_runtime = replace(
            APP_CONFIG.backend_runtime,
            idle_unload_after_s=0.05,
            idle_check_interval_s=0.01,
            telemetry_enable_psutil=False,
            telemetry_enable_backend_stats=False,
        )
        config = replace(APP_CONFIG, preflight=preflight, backend_runtime=backend_runtime)
        if not overrides:
            return config
        return replace(config, **overrides)

    async def test_health_snapshot_reports_active_backends(self) -> None:
        manager = ModelManager(
            config=self._build_config(),
            generation_backend=FakeGenerationBackend("primary_gen"),
            embedding_backend=FakeEmbeddingBackend("primary_embed"),
        )
        await manager.start()
        self.addAsyncCleanup(manager.stop)

        await manager.generate("hello")
        await manager.embed("world")
        snapshot = manager.health_snapshot()

        self.assertTrue(snapshot.started)
        self.assertEqual(snapshot.generation_backend, "primary_gen")
        self.assertEqual(snapshot.embedding_backend, "primary_embed")
        self.assertFalse(snapshot.fallback_active)
        self.assertFalse(snapshot.telemetry_enabled)

    async def test_generation_fallback_activates_on_start_failure(self) -> None:
        manager = ModelManager(
            config=self._build_config(),
            generation_backend=FakeGenerationBackend(
                "broken_gen",
                start_error=BackendStartupError("primary failed"),
            ),
            embedding_backend=FakeEmbeddingBackend("primary_embed"),
            generation_fallback_backend=FakeGenerationBackend("fallback_gen"),
        )
        await manager.start()
        self.addAsyncCleanup(manager.stop)

        result = await manager.generate("hello")
        snapshot = manager.health_snapshot()

        self.assertEqual(result, "fallback_gen-ok")
        self.assertTrue(snapshot.fallback_active)
        self.assertEqual(snapshot.generation_backend, "fallback_gen")

    async def test_embedding_fallback_activates_on_runtime_failure(self) -> None:
        manager = ModelManager(
            config=self._build_config(),
            generation_backend=FakeGenerationBackend("primary_gen"),
            embedding_backend=FakeEmbeddingBackend(
                "broken_embed",
                embed_error=BackendUnavailableError("embedding failed"),
            ),
            embedding_fallback_backend=FakeEmbeddingBackend("fallback_embed"),
        )
        await manager.start()
        self.addAsyncCleanup(manager.stop)

        vector = await manager.embed("hello")
        snapshot = manager.health_snapshot()

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        self.assertTrue(snapshot.fallback_active)
        self.assertEqual(snapshot.embedding_backend, "fallback_embed")

    async def test_generation_semaphore_limits_concurrency(self) -> None:
        backend = FakeGenerationBackend("slow_gen", delay_s=0.03)
        manager = ModelManager(
            config=self._build_config(),
            generation_backend=backend,
            embedding_backend=FakeEmbeddingBackend("primary_embed"),
        )
        await manager.start()
        self.addAsyncCleanup(manager.stop)

        await asyncio.gather(manager.generate("a"), manager.generate("b"))

        self.assertEqual(backend.max_inflight, 1)

    async def test_idle_unload_releases_unloadable_backends(self) -> None:
        generation_backend = FakeGenerationBackend("primary_gen", unloadable=True)
        embedding_backend = FakeEmbeddingBackend("primary_embed", unloadable=True)
        manager = ModelManager(
            config=self._build_config(),
            generation_backend=generation_backend,
            embedding_backend=embedding_backend,
        )
        await manager.start()
        self.addAsyncCleanup(manager.stop)

        await manager.generate("hello")
        await manager.embed("world")
        await asyncio.sleep(0.12)

        self.assertTrue(generation_backend.unloaded)
        self.assertTrue(embedding_backend.unloaded)

    async def test_low_memory_pressure_uses_generation_fallback(self) -> None:
        config = self._build_config(
            backend_runtime=replace(
                APP_CONFIG.backend_runtime,
                idle_unload_after_s=0.05,
                idle_check_interval_s=0.01,
                telemetry_enable_psutil=False,
                enable_fallback_on_low_memory=True,
                low_vram_headroom_gb=0.5,
            )
        )
        primary = FakeGenerationBackend("high_vram_gen", estimated_vram_gb=5.8)
        fallback = FakeGenerationBackend("cpu_fallback_gen")
        manager = ModelManager(
            config=replace(config, preflight=replace(APP_CONFIG.preflight, flags=replace(APP_CONFIG.preflight.flags, stub_mode=False))),
            generation_backend=primary,
            embedding_backend=FakeEmbeddingBackend("primary_embed"),
            generation_fallback_backend=fallback,
        )
        await manager.start()
        self.addAsyncCleanup(manager.stop)

        result = await manager.generate("hello")
        snapshot = manager.health_snapshot()

        self.assertEqual(result, "cpu_fallback_gen-ok")
        self.assertEqual(primary.calls, 0)
        self.assertTrue(snapshot.fallback_active)
        self.assertEqual(snapshot.generation_backend, "cpu_fallback_gen")

    async def test_low_memory_pressure_raises_without_fallback(self) -> None:
        config = self._build_config(
            backend_runtime=replace(
                APP_CONFIG.backend_runtime,
                idle_unload_after_s=0.05,
                idle_check_interval_s=0.01,
                telemetry_enable_psutil=False,
                enable_fallback_on_low_memory=True,
                low_vram_headroom_gb=0.5,
            )
        )
        manager = ModelManager(
            config=replace(config, preflight=replace(APP_CONFIG.preflight, flags=replace(APP_CONFIG.preflight.flags, stub_mode=False))),
            generation_backend=FakeGenerationBackend("high_vram_gen", estimated_vram_gb=5.8),
            embedding_backend=FakeEmbeddingBackend("primary_embed"),
        )
        await manager.start()
        self.addAsyncCleanup(manager.stop)

        with self.assertRaises(ResourcePressureError):
            await manager.generate("hello")


if __name__ == "__main__":
    unittest.main()
