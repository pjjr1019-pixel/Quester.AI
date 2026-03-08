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
WebSearchProvider = Literal["wikipedia", "stub"]
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
    embedding_model: str = "intfloat/e5-small-v2"
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
    enable_self_optimizer: bool = False
    startup_timeout_s: float = 20.0
    shutdown_timeout_s: float = 20.0
    max_component_retries: int = 1
    retry_backoff_s: float = 0.05


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


@dataclass(frozen=True)
class RetrievalSettings:
    """Local-first retrieval and chunking settings."""

    chunk_size_chars: int = 400
    chunk_overlap_chars: int = 80
    max_chunks_per_document: int = 64
    lexical_weight: float = 0.7
    vector_weight: float = 0.3
    max_vector_candidates: int = 32
    minimum_combined_score: float = 0.05
    vector_only_score_threshold: float = 0.97
    prefer_asymmetric_embeddings: bool = True
    query_prefix: str = "query: "
    document_prefix: str = "passage: "
    enable_reranking: bool = True
    max_rerank_candidates: int = 6
    rerank_min_budget_top_k: int = 6
    rerank_combined_weight: float = 0.5
    rerank_lexical_weight: float = 0.2
    rerank_order_weight: float = 0.15
    rerank_exact_phrase_weight: float = 0.1
    rerank_title_weight: float = 0.05
    seed_default_corpus: bool = True
    seed_corpus_mode: str = "stub_only"
    seed_corpus_tier: str = "seed_demo"
    exclude_seed_corpus_from_live_queries: bool = True


@dataclass(frozen=True)
class WebSearchSettings:
    """Optional bounded web lookup settings for freshness and missing-evidence fallback."""

    provider: WebSearchProvider = "wikipedia"
    api_base_url: str = "https://en.wikipedia.org/w/api.php"
    user_agent: str = "QuesterAI/0.1 (local-first-runtime)"
    request_timeout_s: float = 5.0
    max_retries: int = 1
    retry_backoff_s: float = 0.25
    max_results_per_query: int = 5
    max_extract_chars: int = 1200
    snippet_chars: int = 320
    live_web_in_stub_mode: bool = False


@dataclass(frozen=True)
class AudioSettings:
    """Bounded local audio-input settings for optional speech features."""

    max_input_duration_s: float = 30.0
    vad_frame_ms: int = 30
    vad_min_speech_ms: int = 180
    vad_merge_silence_ms: int = 150
    vad_energy_threshold: float = 0.02
    system_speech_timeout_s: float = 20.0
    max_transcript_chars: int = 512
    max_tts_chars: int = 800
    tts_sample_rate_hz: int = 16000
    system_speech_synthesis_timeout_s: float = 20.0


@dataclass(frozen=True)
class TranslationSettings:
    """Bounded local text-translation settings for optional multilingual features."""

    max_input_chars: int = 2400
    default_source_language: str = "auto"
    default_target_language: str = "en"


@dataclass(frozen=True)
class CodeSpecialistSettings:
    """Bounded local code-specialist settings for optional maintenance analysis."""

    max_input_chars: int = 6000
    max_input_lines: int = 240
    default_request: str = "Summarize maintenance risks and the next bounded changes."


@dataclass(frozen=True)
class BudgetCalibrationSettings:
    """Budget caps calibrated for the baseline target and the current dev machine."""

    development_vram_gb: float = 4.0
    development_ram_gb: float = 8.0
    max_thinking_minutes: int = 720
    max_cycle_budget_minutes: int = 120
    max_checkpoint_interval_minutes: int = 120
    min_duty_cycle_ratio: float = 0.25
    max_cooldown_seconds: float = 1.0
    max_resume_count: int = 8
    max_retrieval_top_k: int = 10
    max_web_queries: int = 5
    max_reasoner_passes: int = 4
    max_critic_passes: int = 3
    max_macro_depth: int = 4

    def clamp_budget(self, budget: ResourceBudget) -> ResourceBudget:
        """Return a bounded budget that is safe for both baseline and dev profiles."""
        return ResourceBudget(
            retrieval_top_k=max(1, min(budget.retrieval_top_k, self.max_retrieval_top_k)),
            max_web_queries=max(0, min(budget.max_web_queries, self.max_web_queries)),
            reasoner_passes=max(1, min(budget.reasoner_passes, self.max_reasoner_passes)),
            critic_passes=max(1, min(budget.critic_passes, self.max_critic_passes)),
            macro_depth=max(1, min(budget.macro_depth, self.max_macro_depth)),
            wall_clock_minutes=max(1, min(budget.wall_clock_minutes, self.max_thinking_minutes)),
            cycle_budget_minutes=max(1, min(budget.cycle_budget_minutes, self.max_cycle_budget_minutes)),
            checkpoint_interval_minutes=max(
                1,
                min(
                    budget.checkpoint_interval_minutes,
                    self.max_checkpoint_interval_minutes,
                    budget.cycle_budget_minutes,
                ),
            ),
            duty_cycle_ratio=max(self.min_duty_cycle_ratio, min(budget.duty_cycle_ratio, 1.0)),
            cooldown_seconds=max(0.0, min(budget.cooldown_seconds, self.max_cooldown_seconds)),
            max_resume_count=max(0, min(budget.max_resume_count, self.max_resume_count)),
            planned_cycles=max(
                1,
                min(
                    budget.planned_cycles,
                    max(1, (self.max_thinking_minutes + max(1, self.max_cycle_budget_minutes) - 1) // max(1, self.max_cycle_budget_minutes)),
                ),
            ),
        )


class BudgetPolicy:
    """Convert thinking-time input into a bounded task budget."""

    @staticmethod
    def from_minutes(
        minutes: int,
        calibration: BudgetCalibrationSettings | None = None,
    ) -> ResourceBudget:
        """Return a bounded budget profile for the requested think time."""
        calibration = calibration or APP_CONFIG.budget_calibration
        safe_minutes = max(1, min(int(minutes), calibration.max_thinking_minutes))
        if safe_minutes <= 5:
            budget = ResourceBudget(
                retrieval_top_k=4,
                max_web_queries=1,
                reasoner_passes=1,
                critic_passes=1,
                macro_depth=2,
                wall_clock_minutes=safe_minutes,
                cycle_budget_minutes=safe_minutes,
                checkpoint_interval_minutes=safe_minutes,
            )
        elif safe_minutes <= 30:
            budget = ResourceBudget(
                retrieval_top_k=6,
                max_web_queries=2,
                reasoner_passes=2,
                critic_passes=2,
                macro_depth=3,
                wall_clock_minutes=safe_minutes,
                cycle_budget_minutes=safe_minutes,
                checkpoint_interval_minutes=safe_minutes,
            )
        elif safe_minutes <= 120:
            budget = ResourceBudget(
                retrieval_top_k=8,
                max_web_queries=3,
                reasoner_passes=3,
                critic_passes=2,
                macro_depth=4,
                wall_clock_minutes=safe_minutes,
                cycle_budget_minutes=safe_minutes,
                checkpoint_interval_minutes=safe_minutes,
            )
        else:
            cycle_budget_minutes = min(calibration.max_cycle_budget_minutes, 120)
            planned_cycles = max(2, (safe_minutes + cycle_budget_minutes - 1) // cycle_budget_minutes)
            budget = ResourceBudget(
                retrieval_top_k=10,
                max_web_queries=5,
                reasoner_passes=4,
                critic_passes=3,
                macro_depth=4,
                wall_clock_minutes=safe_minutes,
                cycle_budget_minutes=cycle_budget_minutes,
                checkpoint_interval_minutes=min(cycle_budget_minutes, calibration.max_checkpoint_interval_minutes),
                duty_cycle_ratio=0.75,
                cooldown_seconds=0.05,
                max_resume_count=max(1, planned_cycles - 1),
                planned_cycles=planned_cycles,
            )
        return calibration.clamp_budget(budget)


@dataclass(frozen=True)
class StorageSettings:
    """Local persistence paths and table/logging options."""

    sqlite_path: Path = Path("quester.sqlite3")
    logs_dir: Path = Path("logs")
    events_log_name: str = "events.jsonl"
    trace_log_name: str = "traces.jsonl"
    web_log_name: str = "web.jsonl"
    status_log_name: str = "status.jsonl"
    schema_version: int = 1
    integrity_check_on_start: bool = True


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
    replay_history_limit: int = 64
    proposal_limit: int = 5
    compression_gain_weight: float = 0.30
    proof_hash_stability_weight: float = 0.25
    critique_validity_weight: float = 0.25
    latency_weight: float = 0.10
    memory_weight: float = 0.10
    minimum_simulation_score: float = 0.55
    max_latency_ratio: float = 1.15
    max_memory_ratio: float = 1.15


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
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    web: WebSearchSettings = field(default_factory=WebSearchSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    translation: TranslationSettings = field(default_factory=TranslationSettings)
    code_specialist: CodeSpecialistSettings = field(default_factory=CodeSpecialistSettings)
    budget_calibration: BudgetCalibrationSettings = field(default_factory=BudgetCalibrationSettings)
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
        if self.preflight.flags.max_component_retries < 0:
            raise ValueError("max_component_retries must be zero or positive.")
        if self.preflight.flags.retry_backoff_s < 0:
            raise ValueError("retry_backoff_s must be zero or positive.")
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
        if self.retrieval.chunk_size_chars < 64:
            raise ValueError("chunk_size_chars must be at least 64.")
        if self.retrieval.chunk_overlap_chars < 0:
            raise ValueError("chunk_overlap_chars must be zero or positive.")
        if self.retrieval.chunk_overlap_chars >= self.retrieval.chunk_size_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars.")
        if self.retrieval.max_chunks_per_document < 1:
            raise ValueError("max_chunks_per_document must be positive.")
        if self.retrieval.seed_corpus_mode not in {"stub_only", "always", "disabled"}:
            raise ValueError("seed_corpus_mode must be one of: stub_only, always, disabled.")
        if not self.retrieval.seed_corpus_tier.strip():
            raise ValueError("seed_corpus_tier must not be empty.")
        if self.retrieval.lexical_weight < 0 or self.retrieval.vector_weight < 0:
            raise ValueError("retrieval weights must be zero or positive.")
        if self.retrieval.lexical_weight + self.retrieval.vector_weight <= 0:
            raise ValueError("at least one retrieval weight must be positive.")
        if self.retrieval.max_vector_candidates < 1:
            raise ValueError("max_vector_candidates must be positive.")
        if self.retrieval.minimum_combined_score < 0:
            raise ValueError("minimum_combined_score must be zero or positive.")
        if not 0.0 <= self.retrieval.vector_only_score_threshold <= 1.0:
            raise ValueError("vector_only_score_threshold must be between 0 and 1.")
        if self.retrieval.max_rerank_candidates < 1:
            raise ValueError("max_rerank_candidates must be positive.")
        if self.retrieval.rerank_min_budget_top_k < 1:
            raise ValueError("rerank_min_budget_top_k must be positive.")
        if self.retrieval.rerank_combined_weight < 0:
            raise ValueError("rerank_combined_weight must be zero or positive.")
        if self.retrieval.rerank_lexical_weight < 0:
            raise ValueError("rerank_lexical_weight must be zero or positive.")
        if self.retrieval.rerank_order_weight < 0:
            raise ValueError("rerank_order_weight must be zero or positive.")
        if self.retrieval.rerank_exact_phrase_weight < 0:
            raise ValueError("rerank_exact_phrase_weight must be zero or positive.")
        if self.retrieval.rerank_title_weight < 0:
            raise ValueError("rerank_title_weight must be zero or positive.")
        if (
            self.retrieval.rerank_combined_weight
            + self.retrieval.rerank_lexical_weight
            + self.retrieval.rerank_order_weight
            + self.retrieval.rerank_exact_phrase_weight
            + self.retrieval.rerank_title_weight
            <= 0
        ):
            raise ValueError("at least one rerank weight must be positive.")
        if self.web.request_timeout_s <= 0:
            raise ValueError("web.request_timeout_s must be positive.")
        if self.web.max_retries < 0:
            raise ValueError("web.max_retries must be zero or positive.")
        if self.web.retry_backoff_s < 0:
            raise ValueError("web.retry_backoff_s must be zero or positive.")
        if self.web.max_results_per_query < 1:
            raise ValueError("web.max_results_per_query must be positive.")
        if self.web.max_extract_chars < 128:
            raise ValueError("web.max_extract_chars must be at least 128.")
        if self.web.snippet_chars < 64:
            raise ValueError("web.snippet_chars must be at least 64.")
        if self.audio.max_input_duration_s <= 0:
            raise ValueError("audio.max_input_duration_s must be positive.")
        if self.audio.vad_frame_ms < 10:
            raise ValueError("audio.vad_frame_ms must be at least 10.")
        if self.audio.vad_min_speech_ms < self.audio.vad_frame_ms:
            raise ValueError("audio.vad_min_speech_ms must be at least one frame.")
        if self.audio.vad_merge_silence_ms < 0:
            raise ValueError("audio.vad_merge_silence_ms must be zero or positive.")
        if not 0.0 <= self.audio.vad_energy_threshold <= 1.0:
            raise ValueError("audio.vad_energy_threshold must be between 0 and 1.")
        if self.audio.system_speech_timeout_s <= 0:
            raise ValueError("audio.system_speech_timeout_s must be positive.")
        if self.audio.max_transcript_chars < 64:
            raise ValueError("audio.max_transcript_chars must be at least 64.")
        if self.budget_calibration.development_vram_gb <= 0:
            raise ValueError("development_vram_gb must be positive.")
        if self.budget_calibration.development_ram_gb <= 0:
            raise ValueError("development_ram_gb must be positive.")
        if self.budget_calibration.development_vram_gb > self.preflight.hardware.max_vram_gb:
            raise ValueError("development_vram_gb must not exceed the locked baseline VRAM target.")
        if self.budget_calibration.development_ram_gb > self.preflight.hardware.max_ram_gb:
            raise ValueError("development_ram_gb must not exceed the locked baseline RAM target.")
        if self.budget_calibration.max_thinking_minutes < 720:
            raise ValueError("max_thinking_minutes must support the 121-720 minute preset band.")
        if self.budget_calibration.max_cycle_budget_minutes < 120:
            raise ValueError("max_cycle_budget_minutes must support the bounded long-horizon cycle size.")
        if self.budget_calibration.max_checkpoint_interval_minutes < 1:
            raise ValueError("max_checkpoint_interval_minutes must be positive.")
        if not 0.0 < self.budget_calibration.min_duty_cycle_ratio <= 1.0:
            raise ValueError("min_duty_cycle_ratio must be between 0 and 1.")
        if self.budget_calibration.max_cooldown_seconds < 0.0:
            raise ValueError("max_cooldown_seconds must be zero or positive.")
        if self.budget_calibration.max_resume_count < 1:
            raise ValueError("max_resume_count must be positive.")
        if self.budget_calibration.max_retrieval_top_k < 10:
            raise ValueError("max_retrieval_top_k must support the maximum preset.")
        if self.budget_calibration.max_web_queries < 5:
            raise ValueError("max_web_queries must support the maximum preset.")
        if self.budget_calibration.max_reasoner_passes < 4:
            raise ValueError("max_reasoner_passes must support the maximum preset.")
        if self.budget_calibration.max_critic_passes < 3:
            raise ValueError("max_critic_passes must support the maximum preset.")
        if self.budget_calibration.max_macro_depth < 4:
            raise ValueError("max_macro_depth must support the maximum preset.")
        if not self.storage.events_log_name.strip():
            raise ValueError("events_log_name must not be empty.")
        if not self.storage.trace_log_name.strip():
            raise ValueError("trace_log_name must not be empty.")
        if not self.storage.web_log_name.strip():
            raise ValueError("web_log_name must not be empty.")
        if not self.storage.status_log_name.strip():
            raise ValueError("status_log_name must not be empty.")
        if self.storage.schema_version < 1:
            raise ValueError("storage.schema_version must be positive.")
        if self.self_optimizer.cycle_interval_s <= 0:
            raise ValueError("self_optimizer.cycle_interval_s must be positive.")
        if self.self_optimizer.replay_history_limit < 1:
            raise ValueError("self_optimizer.replay_history_limit must be positive.")
        if self.self_optimizer.proposal_limit < 1:
            raise ValueError("self_optimizer.proposal_limit must be positive.")
        if not 0.0 <= self.self_optimizer.minimum_simulation_score <= 1.0:
            raise ValueError("self_optimizer.minimum_simulation_score must be between 0 and 1.")
        if self.self_optimizer.max_latency_ratio <= 0:
            raise ValueError("self_optimizer.max_latency_ratio must be positive.")
        if self.self_optimizer.max_memory_ratio <= 0:
            raise ValueError("self_optimizer.max_memory_ratio must be positive.")
        optimizer_weights = (
            self.self_optimizer.compression_gain_weight,
            self.self_optimizer.proof_hash_stability_weight,
            self.self_optimizer.critique_validity_weight,
            self.self_optimizer.latency_weight,
            self.self_optimizer.memory_weight,
        )
        if any(weight < 0 for weight in optimizer_weights):
            raise ValueError("self_optimizer metric weights must be zero or positive.")
        if abs(sum(optimizer_weights) - 1.0) > 1e-6:
            raise ValueError("self_optimizer metric weights must sum to 1.0.")


PREFLIGHT = PreflightConfig()
APP_CONFIG = AppConfig()
APP_CONFIG.validate()
