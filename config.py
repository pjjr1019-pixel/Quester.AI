"""Centralized runtime configuration for Quester.AI.

Phase 1 goal:
- Keep all tunable values in one place.
- Enforce hard design targets (6GB VRAM, 8GB RAM) for consumer hardware.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from data_structures import ResourceBudget

GenerationBackend = Literal["ollama", "llama_cpp"]
EmbeddingBackend = Literal["sentence_transformers", "ollama_embeddings"]
VectorStoreBackend = Literal["chromadb", "simple_inmemory"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@dataclass(frozen=True)
class HardwareBudget:
    """Hard runtime targets for the baseline deployment machine."""

    max_vram_gb: float = 6.0
    max_ram_gb: float = 8.0


@dataclass(frozen=True)
class BackendSelection:
    """Pinned defaults selected in Phase 0.5."""

    generation_backend: GenerationBackend = "ollama"
    generation_model: str = "qwen2.5:3b-instruct-q4_K_M"
    generation_fallback_backend: GenerationBackend = "llama_cpp"
    generation_fallback_model: str = "qwen2.5-3b-instruct-q4_k_m.gguf"

    embedding_backend: EmbeddingBackend = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_fallback_backend: EmbeddingBackend = "ollama_embeddings"
    embedding_fallback_model: str = "nomic-embed-text"

    vector_store_backend: VectorStoreBackend = "chromadb"
    vector_store_fallback_backend: VectorStoreBackend = "simple_inmemory"
    vector_collection_name: str = "quester_knowledge"


@dataclass(frozen=True)
class RuntimeFlags:
    """Runtime mode switches and lifecycle settings."""

    stub_mode: bool = True
    allow_web_fallback: bool = True
    enable_self_optimizer: bool = True
    startup_timeout_s: float = 20.0
    shutdown_timeout_s: float = 20.0


@dataclass(frozen=True)
class RunGoals:
    """Locked first-run goals from Phase 0.5."""

    first_stub_goal: str = (
        "One question completes the full pipeline in stub mode and returns a "
        "TaskResult-shaped payload without errors."
    )
    first_real_backend_goal: str = (
        "One question completes with real model calls under bounded concurrency, "
        "stays within practical 6GB VRAM / 8GB RAM limits, and logs stage status."
    )


@dataclass(frozen=True)
class PreflightConfig:
    """Phase 0.5 decisions consumed by Phase 1 runtime components."""

    hardware: HardwareBudget = field(default_factory=HardwareBudget)
    backends: BackendSelection = field(default_factory=BackendSelection)
    flags: RuntimeFlags = field(default_factory=RuntimeFlags)
    goals: RunGoals = field(default_factory=RunGoals)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ConcurrencySettings:
    """Bounded concurrency helps preserve 6GB VRAM / 8GB RAM stability."""

    generation_slots: int = 1
    embedding_slots: int = 1
    dashboard_queue_maxsize: int = 256
    optimizer_queue_maxsize: int = 128


@dataclass(frozen=True)
class ModelTuningSettings:
    """Model and simulation tuning knobs used across agents."""

    default_max_tokens: int = 256
    default_temperature: float = 0.2
    embedding_dimensions: int = 32
    simulation_latency_s: float = 0.01


@dataclass(frozen=True)
class BackendRuntimeSettings:
    """Runtime settings for local backend integrations and fallbacks."""

    ollama_base_url: str = "http://localhost:11434"
    request_timeout_s: float = 30.0
    idle_unload_after_s: float = 60.0
    idle_check_interval_s: float = 5.0
    enable_fallback_on_error: bool = True
    enable_fallback_on_low_memory: bool = True
    telemetry_enable_psutil: bool = True
    telemetry_enable_backend_stats: bool = True
    low_ram_headroom_gb: float = 1.0
    low_vram_headroom_gb: float = 0.5
    models_dir: Path = Path("models")
    llama_cpp_context_window: int = 2048
    llama_cpp_gpu_layers: int = 0


class BudgetPolicy:
    """Convert thinking-time input into a bounded task budget."""

    @staticmethod
    def from_minutes(minutes: int) -> ResourceBudget:
        """Return a bounded budget profile for the requested think time."""
        safe_minutes = max(1, int(minutes))
        if safe_minutes <= 5:
            return ResourceBudget(
                retrieval_top_k=4,
                max_web_queries=1,
                reasoner_passes=1,
                critic_passes=1,
                macro_depth=2,
            )
        if safe_minutes <= 30:
            return ResourceBudget(
                retrieval_top_k=6,
                max_web_queries=2,
                reasoner_passes=2,
                critic_passes=2,
                macro_depth=3,
            )
        if safe_minutes <= 120:
            return ResourceBudget(
                retrieval_top_k=8,
                max_web_queries=3,
                reasoner_passes=3,
                critic_passes=2,
                macro_depth=4,
            )
        return ResourceBudget(
            retrieval_top_k=10,
            max_web_queries=5,
            reasoner_passes=4,
            critic_passes=3,
            macro_depth=4,
        )


@dataclass(frozen=True)
class StorageSettings:
    """Local persistence paths and table/logging options."""

    sqlite_path: Path = Path("quester.sqlite3")
    logs_dir: Path = Path("logs")
    events_log_name: str = "events.jsonl"


@dataclass(frozen=True)
class DashboardSettings:
    """Local GUI controls."""

    enable_ui: bool = True
    window_title: str = "Quester.AI Dashboard"
    refresh_interval_ms: int = 250


@dataclass(frozen=True)
class SelfOptimizerSettings:
    """Background optimizer behavior."""

    cycle_interval_s: float = 30.0


@dataclass(frozen=True)
class LoggingSettings:
    """Structured logging defaults."""

    level: LogLevel = "INFO"
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration object used across all modules."""

    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    concurrency: ConcurrencySettings = field(default_factory=ConcurrencySettings)
    model_tuning: ModelTuningSettings = field(default_factory=ModelTuningSettings)
    backend_runtime: BackendRuntimeSettings = field(default_factory=BackendRuntimeSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    dashboard: DashboardSettings = field(default_factory=DashboardSettings)
    self_optimizer: SelfOptimizerSettings = field(default_factory=SelfOptimizerSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        """Validate required runtime constraints before startup."""
        if self.preflight.hardware.max_vram_gb != 6.0:
            raise ValueError("max_vram_gb must stay pinned to 6.0 for baseline target.")
        if self.preflight.hardware.max_ram_gb != 8.0:
            raise ValueError("max_ram_gb must stay pinned to 8.0 for baseline target.")
        if self.concurrency.generation_slots < 1 or self.concurrency.generation_slots > 2:
            raise ValueError("generation_slots must be between 1 and 2.")
        if self.concurrency.embedding_slots < 1 or self.concurrency.embedding_slots > 2:
            raise ValueError("embedding_slots must be between 1 and 2.")
        if self.concurrency.dashboard_queue_maxsize < 1:
            raise ValueError("dashboard_queue_maxsize must be positive.")
        if self.concurrency.optimizer_queue_maxsize < 1:
            raise ValueError("optimizer_queue_maxsize must be positive.")
        if self.model_tuning.default_max_tokens < 1:
            raise ValueError("default_max_tokens must be positive.")
        if self.model_tuning.embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be positive.")
        if self.preflight.flags.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive.")
        if self.preflight.flags.shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive.")
        if self.backend_runtime.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive.")
        if self.backend_runtime.idle_unload_after_s <= 0:
            raise ValueError("idle_unload_after_s must be positive.")
        if self.backend_runtime.idle_check_interval_s <= 0:
            raise ValueError("idle_check_interval_s must be positive.")
        if self.backend_runtime.low_ram_headroom_gb < 0:
            raise ValueError("low_ram_headroom_gb must be >= 0.")
        if self.backend_runtime.low_vram_headroom_gb < 0:
            raise ValueError("low_vram_headroom_gb must be >= 0.")
        if self.backend_runtime.llama_cpp_context_window < 256:
            raise ValueError("llama_cpp_context_window must be at least 256.")


PREFLIGHT = PreflightConfig()
APP_CONFIG = AppConfig()
APP_CONFIG.validate()
