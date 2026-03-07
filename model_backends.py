"""Backend adapters for local generation and embedding runtimes."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

from runtime_errors import BackendStartupError, BackendUnavailableError


@dataclass(slots=True)
class BackendHealth:
    """Health details reported by an individual backend adapter."""

    backend_name: str
    model_name: str
    started: bool
    available: bool
    mode: str
    estimated_ram_gb: float | None = None
    estimated_vram_gb: float | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GenerationBackendAdapter(Protocol):
    """Behavior required by generation runtimes."""

    backend_name: str

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str: ...

    async def health(self) -> BackendHealth: ...

    async def unload(self) -> bool: ...


class EmbeddingBackendAdapter(Protocol):
    """Behavior required by embedding runtimes."""

    backend_name: str

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def embed(self, text: str) -> list[float]: ...

    async def health(self) -> BackendHealth: ...

    async def unload(self) -> bool: ...


class StubGenerationBackend:
    """Deterministic lightweight generation backend for tests and stub mode."""

    backend_name = "stub_generation"

    def __init__(self, model_name: str = "stub-generation") -> None:
        self.model_name = model_name
        self._started = False

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        _ = temperature
        compressed = " ".join(prompt.split())
        return f"[stub] {compressed[:max_tokens]}"

    async def health(self) -> BackendHealth:
        return BackendHealth(
            backend_name=self.backend_name,
            model_name=self.model_name,
            started=self._started,
            available=True,
            mode="stub",
        )

    async def unload(self) -> bool:
        return False


class StubEmbeddingBackend:
    """Deterministic lightweight embedding backend for tests and stub mode."""

    backend_name = "stub_embedding"

    def __init__(self, model_name: str = "stub-embedding", dimensions: int = 32) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self._started = False

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[index % len(digest)] / 255.0 for index in range(self.dimensions)]

    async def health(self) -> BackendHealth:
        return BackendHealth(
            backend_name=self.backend_name,
            model_name=self.model_name,
            started=self._started,
            available=True,
            mode="stub",
        )

    async def unload(self) -> bool:
        return False


class OllamaGenerationBackend:
    """Generation adapter backed by a local Ollama server."""

    backend_name = "ollama"

    def __init__(self, base_url: str, model_name: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_s = timeout_s
        self._started = False
        self._last_error: str | None = None

    async def start(self) -> None:
        try:
            await self._request_json("/api/tags", payload=None)
        except BackendUnavailableError as exc:
            raise BackendStartupError(str(exc)) from exc
        self._started = True
        self._last_error = None

    async def stop(self) -> None:
        self._started = False

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        self._require_started()
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        response = await self._request_json("/api/generate", payload=payload)
        return str(response.get("response", ""))

    async def health(self) -> BackendHealth:
        return BackendHealth(
            backend_name=self.backend_name,
            model_name=self.model_name,
            started=self._started,
            available=self._started,
            mode="service",
            last_error=self._last_error,
            metadata={"base_url": self.base_url},
        )

    async def unload(self) -> bool:
        return False

    async def _request_json(self, route: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_json_blocking, route, payload)

    def _request_json_blocking(
        self,
        route: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        request_url = f"{self.base_url}{route}"
        data_bytes = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(request_url, data=data_bytes, headers=headers)
        try:
            with urllib_request.urlopen(req, timeout=self.timeout_s) as response:
                response_body = response.read().decode("utf-8")
        except (urllib_error.URLError, TimeoutError) as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise BackendUnavailableError(
                f"Ollama backend unavailable at {request_url}: {exc}"
            ) from exc
        try:
            return json.loads(response_body or "{}")
        except json.JSONDecodeError as exc:
            self._last_error = f"JSONDecodeError: {exc}"
            raise BackendUnavailableError(
                f"Ollama backend returned invalid JSON for {request_url}: {exc}"
            ) from exc

    def _require_started(self) -> None:
        if not self._started:
            raise BackendUnavailableError("Ollama generation backend must be started before use.")


class OllamaEmbeddingBackend:
    """Embedding adapter backed by a local Ollama server."""

    backend_name = "ollama_embeddings"

    def __init__(self, base_url: str, model_name: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_s = timeout_s
        self._started = False
        self._last_error: str | None = None

    async def start(self) -> None:
        try:
            await self._request_json("/api/tags", payload=None)
        except BackendUnavailableError as exc:
            raise BackendStartupError(str(exc)) from exc
        self._started = True
        self._last_error = None

    async def stop(self) -> None:
        self._started = False

    async def embed(self, text: str) -> list[float]:
        self._require_started()
        payload = {"model": self.model_name, "input": text}
        try:
            response = await self._request_json("/api/embed", payload=payload)
            embeddings = response.get("embeddings", [])
            if embeddings:
                return [float(value) for value in embeddings[0]]
        except BackendUnavailableError:
            pass
        payload = {"model": self.model_name, "prompt": text}
        response = await self._request_json("/api/embeddings", payload=payload)
        raw_embedding = response.get("embedding", [])
        return [float(value) for value in raw_embedding]

    async def health(self) -> BackendHealth:
        return BackendHealth(
            backend_name=self.backend_name,
            model_name=self.model_name,
            started=self._started,
            available=self._started,
            mode="service",
            last_error=self._last_error,
            metadata={"base_url": self.base_url},
        )

    async def unload(self) -> bool:
        return False

    async def _request_json(self, route: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_json_blocking, route, payload)

    def _request_json_blocking(
        self,
        route: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        request_url = f"{self.base_url}{route}"
        data_bytes = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(request_url, data=data_bytes, headers=headers)
        try:
            with urllib_request.urlopen(req, timeout=self.timeout_s) as response:
                response_body = response.read().decode("utf-8")
        except (urllib_error.URLError, TimeoutError) as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise BackendUnavailableError(
                f"Ollama embedding backend unavailable at {request_url}: {exc}"
            ) from exc
        try:
            return json.loads(response_body or "{}")
        except json.JSONDecodeError as exc:
            self._last_error = f"JSONDecodeError: {exc}"
            raise BackendUnavailableError(
                f"Ollama embedding backend returned invalid JSON for {request_url}: {exc}"
            ) from exc

    def _require_started(self) -> None:
        if not self._started:
            raise BackendUnavailableError("Ollama embedding backend must be started before use.")


class LlamaCppGenerationBackend:
    """Generation adapter backed by `llama-cpp-python`."""

    backend_name = "llama_cpp"

    def __init__(
        self,
        model_name: str,
        models_dir: Path,
        context_window: int,
        gpu_layers: int,
        timeout_s: float,
    ) -> None:
        self.model_name = model_name
        self.models_dir = models_dir
        self.context_window = context_window
        self.gpu_layers = gpu_layers
        self.timeout_s = timeout_s
        self._started = False
        self._llm: Any | None = None
        self._last_error: str | None = None

    async def start(self) -> None:
        await asyncio.to_thread(self._load_model)
        self._started = True
        self._last_error = None

    async def stop(self) -> None:
        await self.unload()
        self._started = False

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        self._require_started()
        return await asyncio.wait_for(
            asyncio.to_thread(self._generate_blocking, prompt, max_tokens, temperature),
            timeout=self.timeout_s,
        )

    async def health(self) -> BackendHealth:
        return BackendHealth(
            backend_name=self.backend_name,
            model_name=self.model_name,
            started=self._started,
            available=self._llm is not None,
            mode="cpu_fallback" if self.gpu_layers == 0 else "hybrid",
            last_error=self._last_error,
            metadata={"model_path": str(self._resolve_model_path())},
        )

    async def unload(self) -> bool:
        unloaded = self._llm is not None
        self._llm = None
        self._started = False
        return unloaded

    def _load_model(self) -> None:
        model_path = self._resolve_model_path()
        if not model_path.exists():
            raise BackendStartupError(f"llama.cpp model file not found: {model_path}")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise BackendStartupError(
                "llama-cpp-python is not installed for the configured fallback backend."
            ) from exc
        try:
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=self.context_window,
                n_gpu_layers=self.gpu_layers,
            )
        except Exception as exc:  # pragma: no cover - backend-specific
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise BackendStartupError(f"Failed to load llama.cpp model: {exc}") from exc

    def _generate_blocking(self, prompt: str, max_tokens: int, temperature: float) -> str:
        if self._llm is None:
            raise BackendUnavailableError("llama.cpp backend is not loaded.")
        try:
            response = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=[],
            )
        except Exception as exc:  # pragma: no cover - backend-specific
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise BackendUnavailableError(f"llama.cpp generation failed: {exc}") from exc
        choices = response.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("text", ""))

    def _resolve_model_path(self) -> Path:
        model_path = Path(self.model_name)
        if model_path.is_absolute():
            return model_path
        return self.models_dir / self.model_name

    def _require_started(self) -> None:
        if not self._started or self._llm is None:
            raise BackendUnavailableError("llama.cpp generation backend must be started before use.")


class SentenceTransformersEmbeddingBackend:
    """Embedding adapter backed by `sentence-transformers`."""

    backend_name = "sentence_transformers"

    def __init__(self, model_name: str, timeout_s: float) -> None:
        self.model_name = model_name
        self.timeout_s = timeout_s
        self._started = False
        self._model: Any | None = None
        self._last_error: str | None = None

    async def start(self) -> None:
        await asyncio.to_thread(self._load_model)
        self._started = True
        self._last_error = None

    async def stop(self) -> None:
        await self.unload()
        self._started = False

    async def embed(self, text: str) -> list[float]:
        self._require_started()
        return await asyncio.wait_for(asyncio.to_thread(self._embed_blocking, text), timeout=self.timeout_s)

    async def health(self) -> BackendHealth:
        return BackendHealth(
            backend_name=self.backend_name,
            model_name=self.model_name,
            started=self._started,
            available=self._model is not None,
            mode="cpu",
            estimated_vram_gb=0.0,
            last_error=self._last_error,
        )

    async def unload(self) -> bool:
        unloaded = self._model is not None
        self._model = None
        self._started = False
        return unloaded

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise BackendStartupError(
                "sentence-transformers is not installed for the configured embedding backend."
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name, device="cpu")
        except Exception as exc:  # pragma: no cover - backend-specific
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise BackendStartupError(
                f"Failed to load sentence-transformers model '{self.model_name}': {exc}"
            ) from exc

    def _embed_blocking(self, text: str) -> list[float]:
        if self._model is None:
            raise BackendUnavailableError("sentence-transformers backend is not loaded.")
        try:
            vector = self._model.encode(text)
        except Exception as exc:  # pragma: no cover - backend-specific
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise BackendUnavailableError(f"sentence-transformers encode failed: {exc}") from exc
        return [float(value) for value in vector]

    def _require_started(self) -> None:
        if not self._started or self._model is None:
            raise BackendUnavailableError(
                "sentence-transformers embedding backend must be started before use."
            )
