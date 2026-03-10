"""Typed runtime contracts used by agents, orchestrator, and tests."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Mapping, Sequence, TypeVar


def utc_now() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(UTC)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


EnumType = TypeVar("EnumType", bound=Enum)


def _parse_enum(enum_type: type[EnumType], value: EnumType | str) -> EnumType:
    if isinstance(value, enum_type):
        return value
    return enum_type(value)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {meta_field.name: _serialize_value(getattr(value, meta_field.name)) for meta_field in fields(value)}
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def _compact_payload(
    payload: dict[str, Any],
    *,
    drop_empty: tuple[str, ...] = (),
    drop_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compact = dict(payload)
    for field_name in drop_empty:
        if compact.get(field_name) in ("", [], {}, None):
            compact.pop(field_name, None)
    for field_name, default_value in (drop_defaults or {}).items():
        if compact.get(field_name) == default_value:
            compact.pop(field_name, None)
    return compact


class DictSerializable:
    """Small helper mixin for dict conversion."""

    def to_dict(self) -> dict[str, Any]:
        serialized = _serialize_value(self)
        _require(isinstance(serialized, dict), "Serialized value must be a dictionary.")
        return serialized


class AgentState(str, Enum):
    """Lifecycle state values for services and agents."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class SourceType(str, Enum):
    """Evidence source category."""

    LOCAL = "local"
    WEB = "web"


class TaskState(str, Enum):
    """Generic task progress state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class LongHorizonSessionState(str, Enum):
    """Lifecycle state for long-horizon checkpointed task sessions."""

    PENDING = "pending"
    RUNNING = "running"
    CHECKPOINTED = "checkpointed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class LocalTaskSessionState(str, Enum):
    """Lifecycle state for explicit local task execution sessions."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    KILLED = "killed"
    COMPLETED = "completed"
    FAILED = "failed"


class SeverityLevel(str, Enum):
    """Severity values for issues and diagnostics."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CritiqueResult(str, Enum):
    """Validation outcome category."""

    VALID = "valid"
    INVALID = "invalid"
    DEGRADED = "degraded"


class OptimizerLifecycleStage(str, Enum):
    """Lifecycle stage for one optimizer proposal."""

    PROPOSED = "proposed"
    SIMULATED = "simulated"
    VALIDATED = "validated"
    ACTIVATION_BLOCKED = "activation_blocked"
    ACTIVATED = "activated"
    REJECTED = "rejected"
    ROLLBACK_PREPARED = "rollback_prepared"
    ROLLED_BACK = "rolled_back"


class OptimizerActivationDecision(str, Enum):
    """Activation decision emitted after replay and validation."""

    BLOCKED = "blocked"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    ACTIVATED = "activated"
    ROLLED_BACK = "rolled_back"


class OptimizerSuggestionKind(str, Enum):
    """Typed advisory categories emitted by the optimizer."""

    MACRO_ADVICE = "macro_advice"
    RETRIEVAL_STRATEGY = "retrieval_strategy"
    PLANNING_TEMPLATE = "planning_template"
    CRITIQUE_HEURISTIC = "critique_heuristic"
    DASHBOARD_HINT = "dashboard_hint"
    MODEL_LOADING = "model_loading"
    CACHE_PREFETCH = "cache_prefetch"


class CompressionHintType(str, Enum):
    """Typed upstream hint kinds used by the compressor to seed deterministic proposals."""

    SHARED_SUBPROOF = "shared_subproof"
    STABLE_SYMBOL_BUNDLE = "stable_symbol_bundle"
    REPEATED_EVIDENCE_PATH = "repeated_evidence_path"
    VERIFIER_STABLE_MOTIF = "verifier_stable_motif"
    GRAPH_PATH = "graph_path"


class OptimizerSuggestionDisposition(str, Enum):
    """Lifecycle disposition for one advisory suggestion in a foreground run."""

    REQUESTED = "requested"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class ModelRole(str, Enum):
    """Typed local-model roles supported by the registry and router."""

    GENERATION = "generation"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    VAD = "vad"
    TRANSLATION = "translation"
    CODE_SPECIALIST = "code_specialist"
    VISION = "vision"
    SPECIALIST_PERCEPTION = "specialist_perception"


class ModelResourceClass(str, Enum):
    """Resource classification used by the model scheduler."""

    HEAVY = "heavy"
    SIDECAR = "sidecar"


class ModelLoadPolicy(str, Enum):
    """Preferred runtime loading strategy for one registered model."""

    ALWAYS_ON = "always_on"
    ON_DEMAND = "on_demand"
    PREFER_IDLE_UNLOAD = "prefer_idle_unload"


class CodingTaskType(str, Enum):
    """Supported bounded coding-task workflows."""

    FEATURE_GENERATION = "feature_generation"
    BUG_FIXING = "bug_fixing"
    REFACTORING = "refactoring"
    TEST_GENERATION = "test_generation"
    CODE_REVIEW = "code_review"
    EXPLANATION = "explanation"
    PROJECT_SCAFFOLDING = "project_scaffolding"
    ARCHITECTURE_PLANNING = "architecture_planning"
    PRACTICE = "practice"


class CodingRole(str, Enum):
    """Specialized coding-role assignments used by Coding Mode."""

    PLANNER = "planner"
    GENERATOR = "generator"
    DEBUGGER = "debugger"
    REVIEWER = "reviewer"
    TEST_WRITER = "test_writer"
    SUMMARIZER = "summarizer"
    REFACTORER = "refactorer"


class CodingPatternTier(str, Enum):
    """Promotion tier for learned coding patterns."""

    VERIFIED = "verified"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


@dataclass(slots=True, frozen=True)
class ResourceBudget(DictSerializable):
    """Budget envelope attached to a single task."""

    retrieval_top_k: int = 4
    max_web_queries: int = 1
    reasoner_passes: int = 1
    critic_passes: int = 1
    macro_depth: int = 2
    wall_clock_minutes: int = 1
    cycle_budget_minutes: int = 1
    checkpoint_interval_minutes: int = 1
    duty_cycle_ratio: float = 1.0
    cooldown_seconds: float = 0.0
    max_resume_count: int = 0
    planned_cycles: int = 1

    def __post_init__(self) -> None:
        _require(self.retrieval_top_k > 0, "retrieval_top_k must be positive.")
        _require(self.max_web_queries >= 0, "max_web_queries must be zero or positive.")
        _require(self.reasoner_passes > 0, "reasoner_passes must be positive.")
        _require(self.critic_passes > 0, "critic_passes must be positive.")
        _require(self.macro_depth > 0, "macro_depth must be positive.")
        _require(self.wall_clock_minutes > 0, "wall_clock_minutes must be positive.")
        _require(self.cycle_budget_minutes > 0, "cycle_budget_minutes must be positive.")
        _require(self.checkpoint_interval_minutes > 0, "checkpoint_interval_minutes must be positive.")
        _require(0.0 < self.duty_cycle_ratio <= 1.0, "duty_cycle_ratio must be between 0 and 1.")
        _require(self.cooldown_seconds >= 0.0, "cooldown_seconds must be zero or positive.")
        _require(self.max_resume_count >= 0, "max_resume_count must be zero or positive.")
        _require(self.planned_cycles > 0, "planned_cycles must be positive.")
        _require(
            self.cycle_budget_minutes <= self.wall_clock_minutes,
            "cycle_budget_minutes must not exceed wall_clock_minutes.",
        )
        _require(
            self.checkpoint_interval_minutes <= self.cycle_budget_minutes,
            "checkpoint_interval_minutes must not exceed cycle_budget_minutes.",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResourceBudget:
        return cls(
            retrieval_top_k=int(data.get("retrieval_top_k", 4)),
            max_web_queries=int(data.get("max_web_queries", 1)),
            reasoner_passes=int(data.get("reasoner_passes", 1)),
            critic_passes=int(data.get("critic_passes", 1)),
            macro_depth=int(data.get("macro_depth", 2)),
            wall_clock_minutes=int(data.get("wall_clock_minutes", 1)),
            cycle_budget_minutes=int(data.get("cycle_budget_minutes", 1)),
            checkpoint_interval_minutes=int(data.get("checkpoint_interval_minutes", 1)),
            duty_cycle_ratio=float(data.get("duty_cycle_ratio", 1.0)),
            cooldown_seconds=float(data.get("cooldown_seconds", 0.0)),
            max_resume_count=int(data.get("max_resume_count", 0)),
            planned_cycles=int(data.get("planned_cycles", 1)),
        )


@dataclass(slots=True, frozen=True)
class PlanStep(DictSerializable):
    """Single step in a planner-generated task plan."""

    step_id: str
    description: str
    depends_on: tuple[str, ...] = ()
    status: TaskState = TaskState.PENDING
    notes: str = ""

    def __post_init__(self) -> None:
        _require(bool(self.step_id.strip()), "step_id must not be empty.")
        _require(bool(self.description.strip()), "description must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlanStep:
        return cls(
            step_id=str(data["step_id"]),
            description=str(data["description"]),
            depends_on=tuple(data.get("depends_on", [])),
            status=_parse_enum(TaskState, data.get("status", TaskState.PENDING)),
            notes=str(data.get("notes", "")),
        )


@dataclass(slots=True, frozen=True)
class Plan(DictSerializable):
    """Structured plan passed from planner to downstream stages."""

    task_id: str
    question: str
    steps: tuple[PlanStep, ...]
    required_evidence: tuple[str, ...]
    success_criteria: tuple[str, ...]
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    planner_notes: str = ""
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "task_id must not be empty.")
        _require(bool(self.question.strip()), "question must not be empty.")
        _require(len(self.steps) > 0, "Plan must include at least one step.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Plan:
        return cls(
            task_id=str(data["task_id"]),
            question=str(data["question"]),
            steps=tuple(PlanStep.from_dict(item) for item in data.get("steps", [])),
            required_evidence=tuple(str(item) for item in data.get("required_evidence", [])),
            success_criteria=tuple(str(item) for item in data.get("success_criteria", [])),
            budget=ResourceBudget.from_dict(data.get("budget", {})),
            planner_notes=str(data.get("planner_notes", "")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class Macro(DictSerializable):
    """Reusable macro token and expansion steps."""

    macro_name: str
    expansion: tuple[str, ...]
    version: int = 1
    parameters: tuple[str, ...] = ()
    opcode_pattern: tuple[str, ...] = ()
    invariants: tuple[str, ...] = ()
    proof_fingerprint: str = ""
    semantic_kind: str = "token_macro"
    is_active: bool = True
    decoder_template: str = ""

    def __post_init__(self) -> None:
        _require(bool(self.macro_name.strip()), "macro_name must not be empty.")
        _require(len(self.expansion) > 0, "expansion must include at least one step.")
        _require(self.version > 0, "version must be positive.")
        _require(
            len(set(self.parameters)) == len(self.parameters),
            "Macro.parameters must not contain duplicates.",
        )
        _require(bool(self.semantic_kind.strip()), "semantic_kind must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Macro:
        return cls(
            macro_name=str(data["macro_name"]),
            expansion=tuple(str(item) for item in data["expansion"]),
            version=int(data.get("version", 1)),
            parameters=tuple(str(item) for item in data.get("parameters", [])),
            opcode_pattern=tuple(str(item) for item in data.get("opcode_pattern", [])),
            invariants=tuple(str(item) for item in data.get("invariants", [])),
            proof_fingerprint=str(data.get("proof_fingerprint", "")),
            semantic_kind=str(data.get("semantic_kind", "token_macro")),
            is_active=bool(data.get("is_active", True)),
            decoder_template=str(data.get("decoder_template", "")),
        )


@dataclass(slots=True, frozen=True)
class OpcodeEntry(DictSerializable):
    """Machine-readable opcode registry entry."""

    opcode_name: str
    description: str
    category: str = "core"
    version: int = 1
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(bool(self.opcode_name.strip()), "opcode_name must not be empty.")
        _require(bool(self.description.strip()), "description must not be empty.")
        _require(self.version > 0, "version must be positive.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OpcodeEntry:
        return cls(
            opcode_name=str(data["opcode_name"]),
            description=str(data["description"]),
            category=str(data.get("category", "core")),
            version=int(data.get("version", 1)),
            is_active=bool(data.get("is_active", True)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True, frozen=True)
class DecoderEntry(DictSerializable):
    """Machine-readable decoder lexicon entry."""

    decoder_name: str
    template: str
    version: int = 1
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(bool(self.decoder_name.strip()), "decoder_name must not be empty.")
        _require(bool(self.template.strip()), "template must not be empty.")
        _require(self.version > 0, "version must be positive.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecoderEntry:
        return cls(
            decoder_name=str(data["decoder_name"]),
            template=str(data["template"]),
            version=int(data.get("version", 1)),
            is_active=bool(data.get("is_active", True)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True, frozen=True)
class SymbolTableSnapshot(DictSerializable):
    """Persisted local symbol-table state for a task."""

    task_id: str
    symbols: dict[str, str]
    snapshot_name: str = "active"
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "task_id must not be empty.")
        _require(bool(self.snapshot_name.strip()), "snapshot_name must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SymbolTableSnapshot:
        raw_symbols = dict(data.get("symbols", {}))
        return cls(
            task_id=str(data["task_id"]),
            symbols={str(key): str(value) for key, value in raw_symbols.items()},
            snapshot_name=str(data.get("snapshot_name", "active")),
            is_active=bool(data.get("is_active", True)),
            metadata=dict(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class ProofHashRecord(DictSerializable):
    """Persisted proof-hash history entry for runtime artifacts."""

    task_id: str
    artifact_id: str
    artifact_type: str
    proof_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "task_id must not be empty.")
        _require(bool(self.artifact_id.strip()), "artifact_id must not be empty.")
        _require(bool(self.artifact_type.strip()), "artifact_type must not be empty.")
        _require(bool(self.proof_hash.strip()), "proof_hash must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProofHashRecord:
        return cls(
            task_id=str(data["task_id"]),
            artifact_id=str(data["artifact_id"]),
            artifact_type=str(data["artifact_type"]),
            proof_hash=str(data["proof_hash"]),
            metadata=dict(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class KnowledgeVector(DictSerializable):
    """Vectorized document chunk metadata."""

    id: str
    vector: tuple[float, ...]
    source: SourceType
    timestamp: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.id.strip()), "KnowledgeVector.id must not be empty.")
        _require(len(self.vector) > 0, "KnowledgeVector.vector must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KnowledgeVector:
        return cls(
            id=str(data["id"]),
            vector=tuple(float(item) for item in data["vector"]),
            source=_parse_enum(SourceType, data["source"]),
            timestamp=_parse_datetime(data.get("timestamp", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class ReasoningLog(DictSerializable):
    """Persisted reasoning snapshot for later optimization."""

    task_id: str
    compressed_chain: tuple[str, ...]
    macros_used: tuple[str, ...]
    timestamp: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "ReasoningLog.task_id must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReasoningLog:
        return cls(
            task_id=str(data["task_id"]),
            compressed_chain=tuple(str(item) for item in data.get("compressed_chain", [])),
            macros_used=tuple(str(item) for item in data.get("macros_used", [])),
            timestamp=_parse_datetime(data.get("timestamp", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class PerformanceMetric(DictSerializable):
    """Performance tracking for pipeline runs."""

    task_id: str
    time: float
    vram_usage: float
    iterations: int

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "PerformanceMetric.task_id must not be empty.")
        _require(self.time >= 0.0, "PerformanceMetric.time must be >= 0.")
        _require(self.vram_usage >= 0.0, "PerformanceMetric.vram_usage must be >= 0.")
        _require(self.iterations >= 0, "PerformanceMetric.iterations must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        payload = DictSerializable.to_dict(self)
        payload["VRAM_usage"] = payload.pop("vram_usage")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PerformanceMetric:
        raw_vram = data.get("vram_usage", data.get("VRAM_usage", 0.0))
        return cls(
            task_id=str(data["task_id"]),
            time=float(data.get("time", 0.0)),
            vram_usage=float(raw_vram),
            iterations=int(data.get("iterations", 0)),
        )


@dataclass(slots=True, frozen=True)
class OptimizerReplaySample(DictSerializable):
    """Optimizer-facing summary of one completed foreground task."""

    task_id: str
    trace_proof_hash: str
    selected_candidate_id: str = ""
    candidate_ids: tuple[str, ...] = ()
    candidate_trace_count: int = 0
    selected_strategy: str = ""
    selected_verifier: str = ""
    selected_candidate_score: float = 0.0
    selected_agreement_score: float = 0.0
    selected_evidence_support_score: float = 0.0
    provenance_coverage: float = 1.0
    evidence_coverage: float = 1.0
    final_adjudication: CritiqueResult = CritiqueResult.VALID
    is_valid: bool = False
    proof_hash_match: bool = True
    failure_categories: tuple[str, ...] = ()
    suggested_repair_actions: tuple[str, ...] = ()
    applied_repair_actions: tuple[str, ...] = ()
    answer_text: str = ""
    latency_s: float = 0.0
    vram_usage_gb: float = 0.0
    iterations: int = 0
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "OptimizerReplaySample.task_id must not be empty.")
        _require(
            0.0 <= self.selected_candidate_score <= 1.0,
            "OptimizerReplaySample.selected_candidate_score must be between 0 and 1.",
        )
        _require(
            0.0 <= self.selected_agreement_score <= 1.0,
            "OptimizerReplaySample.selected_agreement_score must be between 0 and 1.",
        )
        _require(
            0.0 <= self.selected_evidence_support_score <= 1.0,
            "OptimizerReplaySample.selected_evidence_support_score must be between 0 and 1.",
        )
        _require(
            0.0 <= self.provenance_coverage <= 1.0,
            "OptimizerReplaySample.provenance_coverage must be between 0 and 1.",
        )
        _require(
            0.0 <= self.evidence_coverage <= 1.0,
            "OptimizerReplaySample.evidence_coverage must be between 0 and 1.",
        )
        _require(self.candidate_trace_count >= 0, "OptimizerReplaySample.candidate_trace_count must be >= 0.")
        _require(self.latency_s >= 0.0, "OptimizerReplaySample.latency_s must be >= 0.")
        _require(self.vram_usage_gb >= 0.0, "OptimizerReplaySample.vram_usage_gb must be >= 0.")
        _require(self.iterations >= 0, "OptimizerReplaySample.iterations must be >= 0.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizerReplaySample:
        return cls(
            task_id=str(data["task_id"]),
            trace_proof_hash=str(data.get("trace_proof_hash", "")),
            selected_candidate_id=str(data.get("selected_candidate_id", "")),
            candidate_ids=tuple(str(item) for item in data.get("candidate_ids", [])),
            candidate_trace_count=int(data.get("candidate_trace_count", 0)),
            selected_strategy=str(data.get("selected_strategy", "")),
            selected_verifier=str(data.get("selected_verifier", "")),
            selected_candidate_score=float(data.get("selected_candidate_score", 0.0)),
            selected_agreement_score=float(data.get("selected_agreement_score", 0.0)),
            selected_evidence_support_score=float(data.get("selected_evidence_support_score", 0.0)),
            provenance_coverage=float(data.get("provenance_coverage", 1.0)),
            evidence_coverage=float(data.get("evidence_coverage", 1.0)),
            final_adjudication=_parse_enum(CritiqueResult, data.get("final_adjudication", CritiqueResult.VALID)),
            is_valid=bool(data.get("is_valid", False)),
            proof_hash_match=bool(data.get("proof_hash_match", True)),
            failure_categories=tuple(str(item) for item in data.get("failure_categories", [])),
            suggested_repair_actions=tuple(str(item) for item in data.get("suggested_repair_actions", [])),
            applied_repair_actions=tuple(str(item) for item in data.get("applied_repair_actions", [])),
            answer_text=str(data.get("answer_text", "")),
            latency_s=float(data.get("latency_s", 0.0)),
            vram_usage_gb=float(data.get("vram_usage_gb", 0.0)),
            iterations=int(data.get("iterations", 0)),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class OptimizerReplayEvaluation(DictSerializable):
    """Offline replay evaluation for one proposal against one persisted sample."""

    proposal_id: str
    task_id: str
    trace_proof_hash: str
    compression_gain: float
    proof_hash_stability: float
    critique_validity: float
    latency_ratio: float
    memory_ratio: float
    aggregate_score: float
    accepted: bool
    rejection_reason: str = ""
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.proposal_id.strip()), "OptimizerReplayEvaluation.proposal_id must not be empty.")
        _require(bool(self.task_id.strip()), "OptimizerReplayEvaluation.task_id must not be empty.")
        _require(0.0 <= self.compression_gain <= 1.0, "OptimizerReplayEvaluation.compression_gain must be between 0 and 1.")
        _require(
            0.0 <= self.proof_hash_stability <= 1.0,
            "OptimizerReplayEvaluation.proof_hash_stability must be between 0 and 1.",
        )
        _require(
            0.0 <= self.critique_validity <= 1.0,
            "OptimizerReplayEvaluation.critique_validity must be between 0 and 1.",
        )
        _require(self.latency_ratio >= 0.0, "OptimizerReplayEvaluation.latency_ratio must be >= 0.")
        _require(self.memory_ratio >= 0.0, "OptimizerReplayEvaluation.memory_ratio must be >= 0.")
        _require(
            0.0 <= self.aggregate_score <= 1.0,
            "OptimizerReplayEvaluation.aggregate_score must be between 0 and 1.",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizerReplayEvaluation:
        return cls(
            proposal_id=str(data["proposal_id"]),
            task_id=str(data["task_id"]),
            trace_proof_hash=str(data.get("trace_proof_hash", "")),
            compression_gain=float(data.get("compression_gain", 0.0)),
            proof_hash_stability=float(data.get("proof_hash_stability", 0.0)),
            critique_validity=float(data.get("critique_validity", 0.0)),
            latency_ratio=float(data.get("latency_ratio", 0.0)),
            memory_ratio=float(data.get("memory_ratio", 0.0)),
            aggregate_score=float(data.get("aggregate_score", 0.0)),
            accepted=bool(data.get("accepted", False)),
            rejection_reason=str(data.get("rejection_reason", "")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class OptimizerProposalRecord(DictSerializable):
    """Persisted lifecycle summary for one optimizer proposal inside a cycle."""

    cycle_id: str
    proposal_id: str
    proposal: MacroProposal
    lifecycle_stage: OptimizerLifecycleStage = OptimizerLifecycleStage.PROPOSED
    source_task_ids: tuple[str, ...] = ()
    replay_sample_count: int = 0
    accepted_simulation_count: int = 0
    mean_simulation_score: float = 0.0
    pass_rate: float = 0.0
    contradiction_risk: float = 1.0
    activation_eligible: bool = False
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.cycle_id.strip()), "OptimizerProposalRecord.cycle_id must not be empty.")
        _require(bool(self.proposal_id.strip()), "OptimizerProposalRecord.proposal_id must not be empty.")
        _require(self.replay_sample_count >= 0, "OptimizerProposalRecord.replay_sample_count must be >= 0.")
        _require(
            self.accepted_simulation_count >= 0,
            "OptimizerProposalRecord.accepted_simulation_count must be >= 0.",
        )
        _require(
            0.0 <= self.mean_simulation_score <= 1.0,
            "OptimizerProposalRecord.mean_simulation_score must be between 0 and 1.",
        )
        _require(0.0 <= self.pass_rate <= 1.0, "OptimizerProposalRecord.pass_rate must be between 0 and 1.")
        _require(
            0.0 <= self.contradiction_risk <= 1.0,
            "OptimizerProposalRecord.contradiction_risk must be between 0 and 1.",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizerProposalRecord:
        return cls(
            cycle_id=str(data["cycle_id"]),
            proposal_id=str(data["proposal_id"]),
            proposal=MacroProposal.from_dict(data["proposal"]),
            lifecycle_stage=_parse_enum(
                OptimizerLifecycleStage,
                data.get("lifecycle_stage", OptimizerLifecycleStage.PROPOSED),
            ),
            source_task_ids=tuple(str(item) for item in data.get("source_task_ids", [])),
            replay_sample_count=int(data.get("replay_sample_count", 0)),
            accepted_simulation_count=int(data.get("accepted_simulation_count", 0)),
            mean_simulation_score=float(data.get("mean_simulation_score", 0.0)),
            pass_rate=float(data.get("pass_rate", 0.0)),
            contradiction_risk=float(data.get("contradiction_risk", 1.0)),
            activation_eligible=bool(data.get("activation_eligible", False)),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class MacroEffectivenessRecord(DictSerializable):
    """Bounded persisted effectiveness summary for one macro proposal/context combination."""

    record_id: str
    proposal_id: str
    proof_fingerprint: str = ""
    macro_name: str = ""
    context_tags: tuple[str, ...] = ()
    seen_count: int = 0
    replay_pass_rate: float = 0.0
    proof_hash_stability: float = 0.0
    critic_validity_rate: float = 0.0
    realized_compression_gain: float = 0.0
    last_used_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.record_id.strip()), "MacroEffectivenessRecord.record_id must not be empty.")
        _require(bool(self.proposal_id.strip()), "MacroEffectivenessRecord.proposal_id must not be empty.")
        _require(self.seen_count >= 0, "MacroEffectivenessRecord.seen_count must be >= 0.")
        _require(0.0 <= self.replay_pass_rate <= 1.0, "MacroEffectivenessRecord.replay_pass_rate must be between 0 and 1.")
        _require(
            0.0 <= self.proof_hash_stability <= 1.0,
            "MacroEffectivenessRecord.proof_hash_stability must be between 0 and 1.",
        )
        _require(
            0.0 <= self.critic_validity_rate <= 1.0,
            "MacroEffectivenessRecord.critic_validity_rate must be between 0 and 1.",
        )
        _require(
            0.0 <= self.realized_compression_gain <= 1.0,
            "MacroEffectivenessRecord.realized_compression_gain must be between 0 and 1.",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MacroEffectivenessRecord:
        return cls(
            record_id=str(data["record_id"]),
            proposal_id=str(data["proposal_id"]),
            proof_fingerprint=str(data.get("proof_fingerprint", "")),
            macro_name=str(data.get("macro_name", "")),
            context_tags=tuple(str(item) for item in data.get("context_tags", [])),
            seen_count=int(data.get("seen_count", 0)),
            replay_pass_rate=float(data.get("replay_pass_rate", 0.0)),
            proof_hash_stability=float(data.get("proof_hash_stability", 0.0)),
            critic_validity_rate=float(data.get("critic_validity_rate", 0.0)),
            realized_compression_gain=float(data.get("realized_compression_gain", 0.0)),
            last_used_at=_parse_datetime(data.get("last_used_at", utc_now())),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class OptimizerActivationRecord(DictSerializable):
    """Activation decision record for one proposal after replay and validation."""

    cycle_id: str
    proposal_id: str
    decision: OptimizerActivationDecision
    lifecycle_stage: OptimizerLifecycleStage
    reason: str = ""
    validation_passed: bool = False
    mean_simulation_score: float = 0.0
    pass_rate: float = 0.0
    contradiction_risk: float = 1.0
    activation_applied: bool = False
    rollback_record_id: str = ""
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.cycle_id.strip()), "OptimizerActivationRecord.cycle_id must not be empty.")
        _require(bool(self.proposal_id.strip()), "OptimizerActivationRecord.proposal_id must not be empty.")
        _require(
            0.0 <= self.mean_simulation_score <= 1.0,
            "OptimizerActivationRecord.mean_simulation_score must be between 0 and 1.",
        )
        _require(0.0 <= self.pass_rate <= 1.0, "OptimizerActivationRecord.pass_rate must be between 0 and 1.")
        _require(
            0.0 <= self.contradiction_risk <= 1.0,
            "OptimizerActivationRecord.contradiction_risk must be between 0 and 1.",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizerActivationRecord:
        return cls(
            cycle_id=str(data["cycle_id"]),
            proposal_id=str(data["proposal_id"]),
            decision=_parse_enum(
                OptimizerActivationDecision,
                data.get("decision", OptimizerActivationDecision.BLOCKED),
            ),
            lifecycle_stage=_parse_enum(
                OptimizerLifecycleStage,
                data.get("lifecycle_stage", OptimizerLifecycleStage.ACTIVATION_BLOCKED),
            ),
            reason=str(data.get("reason", "")),
            validation_passed=bool(data.get("validation_passed", False)),
            mean_simulation_score=float(data.get("mean_simulation_score", 0.0)),
            pass_rate=float(data.get("pass_rate", 0.0)),
            contradiction_risk=float(data.get("contradiction_risk", 1.0)),
            activation_applied=bool(data.get("activation_applied", False)),
            rollback_record_id=str(data.get("rollback_record_id", "")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class OptimizerRollbackRecord(DictSerializable):
    """Prepared or applied rollback snapshot for an optimizer proposal."""

    rollback_record_id: str
    cycle_id: str
    proposal_id: str
    proposal_macro_name: str
    active_macro_versions: dict[str, int] = field(default_factory=dict)
    reason: str = ""
    applied: bool = False
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.rollback_record_id.strip()), "OptimizerRollbackRecord.rollback_record_id must not be empty.")
        _require(bool(self.cycle_id.strip()), "OptimizerRollbackRecord.cycle_id must not be empty.")
        _require(bool(self.proposal_id.strip()), "OptimizerRollbackRecord.proposal_id must not be empty.")
        _require(
            bool(self.proposal_macro_name.strip()),
            "OptimizerRollbackRecord.proposal_macro_name must not be empty.",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizerRollbackRecord:
        return cls(
            rollback_record_id=str(data["rollback_record_id"]),
            cycle_id=str(data["cycle_id"]),
            proposal_id=str(data["proposal_id"]),
            proposal_macro_name=str(data["proposal_macro_name"]),
            active_macro_versions={
                str(key): int(value)
                for key, value in dict(data.get("active_macro_versions", {})).items()
            },
            reason=str(data.get("reason", "")),
            applied=bool(data.get("applied", False)),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class OptimizerSuggestionRecord(DictSerializable):
    """Typed advisory suggestion persisted for foreground explainability and reuse."""

    suggestion_id: str
    cycle_id: str
    kind: OptimizerSuggestionKind
    summary: str
    rationale: str = ""
    target_components: tuple[str, ...] = ()
    source_task_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    advisory_only: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.suggestion_id.strip()), "OptimizerSuggestionRecord.suggestion_id must not be empty.")
        _require(bool(self.cycle_id.strip()), "OptimizerSuggestionRecord.cycle_id must not be empty.")
        _require(bool(self.summary.strip()), "OptimizerSuggestionRecord.summary must not be empty.")
        _require(0.0 <= self.confidence <= 1.0, "OptimizerSuggestionRecord.confidence must be between 0 and 1.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizerSuggestionRecord:
        return cls(
            suggestion_id=str(data["suggestion_id"]),
            cycle_id=str(data["cycle_id"]),
            kind=_parse_enum(OptimizerSuggestionKind, data.get("kind", OptimizerSuggestionKind.MACRO_ADVICE)),
            summary=str(data.get("summary", "")),
            rationale=str(data.get("rationale", "")),
            target_components=tuple(str(item) for item in data.get("target_components", ())),
            source_task_ids=tuple(str(item) for item in data.get("source_task_ids", ())),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            advisory_only=bool(data.get("advisory_only", True)),
            metadata=dict(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class OptimizerSuggestionUsageRecord(DictSerializable):
    """Append-only usage record for one advisory suggestion during a live task."""

    usage_id: str
    suggestion_id: str
    session_id: str
    disposition: OptimizerSuggestionDisposition
    cycle_index: int = 0
    task_id: str = ""
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.usage_id.strip()), "OptimizerSuggestionUsageRecord.usage_id must not be empty.")
        _require(bool(self.suggestion_id.strip()), "OptimizerSuggestionUsageRecord.suggestion_id must not be empty.")
        _require(bool(self.session_id.strip()), "OptimizerSuggestionUsageRecord.session_id must not be empty.")
        _require(self.cycle_index >= 0, "OptimizerSuggestionUsageRecord.cycle_index must be zero or positive.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizerSuggestionUsageRecord:
        return cls(
            usage_id=str(data["usage_id"]),
            suggestion_id=str(data["suggestion_id"]),
            session_id=str(data["session_id"]),
            disposition=_parse_enum(
                OptimizerSuggestionDisposition,
                data.get("disposition", OptimizerSuggestionDisposition.REQUESTED),
            ),
            cycle_index=int(data.get("cycle_index", 0)),
            task_id=str(data.get("task_id", "")),
            reason=str(data.get("reason", "")),
            metadata=dict(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class VerifiedDeepTraceExport(DictSerializable):
    """Machine-readable dataset row for one verified deep-mode task result."""

    export_id: str
    task_id: str
    question: str
    answer_text: str
    trace_proof_hash: str
    reasoning: CompressedTrace
    critique: CritiqueReport
    evidence_source_refs: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.export_id.strip()), "VerifiedDeepTraceExport.export_id must not be empty.")
        _require(bool(self.task_id.strip()), "VerifiedDeepTraceExport.task_id must not be empty.")
        _require(bool(self.question.strip()), "VerifiedDeepTraceExport.question must not be empty.")
        _require(bool(self.answer_text.strip()), "VerifiedDeepTraceExport.answer_text must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VerifiedDeepTraceExport:
        return cls(
            export_id=str(data["export_id"]),
            task_id=str(data["task_id"]),
            question=str(data["question"]),
            answer_text=str(data["answer_text"]),
            trace_proof_hash=str(data.get("trace_proof_hash", "")),
            reasoning=CompressedTrace.from_dict(data["reasoning"]),
            critique=CritiqueReport.from_dict(data["critique"]),
            evidence_source_refs=tuple(str(item) for item in data.get("evidence_source_refs", [])),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class EvidenceItem(DictSerializable):
    """Single evidence object retrieved from local/web sources."""

    id: str
    content: str
    source_type: SourceType
    source_ref: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_preview: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _require(bool(self.id.strip()), "EvidenceItem.id must not be empty.")
        _require(bool(self.content.strip()), "EvidenceItem.content must not be empty.")
        _require(0.0 <= self.score <= 1.0, "EvidenceItem.score must be between 0 and 1.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceItem:
        vector_raw = data.get("vector_preview")
        vector_preview = None
        if vector_raw is not None:
            vector_preview = tuple(float(item) for item in vector_raw)
        return cls(
            id=str(data["id"]),
            content=str(data["content"]),
            source_type=_parse_enum(SourceType, data["source_type"]),
            source_ref=str(data["source_ref"]),
            score=float(data.get("score", 0.0)),
            metadata=dict(data.get("metadata", {})),
            vector_preview=vector_preview,
        )


@dataclass(slots=True, frozen=True)
class EvidenceBundle(DictSerializable):
    """Combined evidence payload passed into reasoning."""

    task_id: str
    local_results: tuple[EvidenceItem, ...]
    web_results: tuple[EvidenceItem, ...]
    used_web_fallback: bool
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "EvidenceBundle.task_id must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceBundle:
        return cls(
            task_id=str(data["task_id"]),
            local_results=tuple(
                EvidenceItem.from_dict(item) for item in data.get("local_results", [])
            ),
            web_results=tuple(EvidenceItem.from_dict(item) for item in data.get("web_results", [])),
            used_web_fallback=bool(data.get("used_web_fallback", False)),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class WebEvidenceRecord(DictSerializable):
    """Persisted fetched web evidence plus lookup provenance."""

    task_id: str
    query: str
    provider: str
    reason: str
    evidence: EvidenceItem
    degraded: bool = False
    warnings: tuple[str, ...] = ()
    lookup_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "WebEvidenceRecord.task_id must not be empty.")
        _require(bool(self.query.strip()), "WebEvidenceRecord.query must not be empty.")
        _require(bool(self.provider.strip()), "WebEvidenceRecord.provider must not be empty.")
        _require(bool(self.reason.strip()), "WebEvidenceRecord.reason must not be empty.")
        _require(
            self.evidence.source_type == SourceType.WEB,
            "WebEvidenceRecord.evidence must use SourceType.WEB.",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WebEvidenceRecord:
        return cls(
            task_id=str(data["task_id"]),
            query=str(data["query"]),
            provider=str(data["provider"]),
            reason=str(data["reason"]),
            evidence=EvidenceItem.from_dict(data["evidence"]),
            degraded=bool(data.get("degraded", False)),
            warnings=tuple(str(item) for item in data.get("warnings", [])),
            lookup_metadata=dict(data.get("lookup_metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CompressionRuntimeSubset(DictSerializable):
    """Active subset of runtime compression registries for one task."""

    task_id: str
    macros: tuple[Macro, ...] = ()
    opcodes: tuple[OpcodeEntry, ...] = ()
    decoders: tuple[DecoderEntry, ...] = ()
    symbol_table: SymbolTableSnapshot | None = None
    proof_hashes: tuple[ProofHashRecord, ...] = ()
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "CompressionRuntimeSubset.task_id must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompressionRuntimeSubset:
        raw_symbol_table = data.get("symbol_table")
        symbol_table = None
        if isinstance(raw_symbol_table, Mapping):
            symbol_table = SymbolTableSnapshot.from_dict(raw_symbol_table)
        return cls(
            task_id=str(data["task_id"]),
            macros=tuple(Macro.from_dict(item) for item in data.get("macros", [])),
            opcodes=tuple(OpcodeEntry.from_dict(item) for item in data.get("opcodes", [])),
            decoders=tuple(DecoderEntry.from_dict(item) for item in data.get("decoders", [])),
            symbol_table=symbol_table,
            proof_hashes=tuple(
                ProofHashRecord.from_dict(item) for item in data.get("proof_hashes", [])
            ),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class SemanticEntity(DictSerializable):
    """Typed canonical entity carried inside the reasoning graph."""

    entity_id: str
    entity_type: str
    value: str
    evidence_handles: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    uncertainty: str = ""
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.entity_id.strip()), "SemanticEntity.entity_id must not be empty.")
        _require(bool(self.entity_type.strip()), "SemanticEntity.entity_type must not be empty.")
        _require(bool(self.value.strip()), "SemanticEntity.value must not be empty.")
        _require(0.0 <= self.confidence <= 1.0, "SemanticEntity.confidence must be between 0 and 1.")

    def to_dict(self) -> dict[str, Any]:
        return _compact_payload(
            DictSerializable.to_dict(self),
            drop_empty=("evidence_handles", "attributes", "uncertainty"),
            drop_defaults={"confidence": 1.0},
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SemanticEntity:
        return cls(
            entity_id=str(data["entity_id"]),
            entity_type=str(data["entity_type"]),
            value=str(data["value"]),
            evidence_handles=tuple(str(item) for item in data.get("evidence_handles", [])),
            attributes=dict(data.get("attributes", {})),
            confidence=float(data.get("confidence", 1.0)),
            uncertainty=str(data.get("uncertainty", "")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class SemanticActivity(DictSerializable):
    """Typed canonical activity linking input and output entities."""

    activity_id: str
    activity_type: str
    input_entity_ids: tuple[str, ...] = ()
    output_entity_ids: tuple[str, ...] = ()
    agent_id: str = ""
    evidence_handles: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.activity_id.strip()), "SemanticActivity.activity_id must not be empty.")
        _require(bool(self.activity_type.strip()), "SemanticActivity.activity_type must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return _compact_payload(
            DictSerializable.to_dict(self),
            drop_empty=("input_entity_ids", "output_entity_ids", "agent_id", "evidence_handles", "metadata"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SemanticActivity:
        return cls(
            activity_id=str(data["activity_id"]),
            activity_type=str(data["activity_type"]),
            input_entity_ids=tuple(str(item) for item in data.get("input_entity_ids", [])),
            output_entity_ids=tuple(str(item) for item in data.get("output_entity_ids", [])),
            agent_id=str(data.get("agent_id", "")),
            evidence_handles=tuple(str(item) for item in data.get("evidence_handles", [])),
            metadata=dict(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class SemanticAgent(DictSerializable):
    """Typed producer/owner identity for graph artifacts."""

    agent_id: str
    component: str
    backend: str = ""
    role: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.agent_id.strip()), "SemanticAgent.agent_id must not be empty.")
        _require(bool(self.component.strip()), "SemanticAgent.component must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return _compact_payload(
            DictSerializable.to_dict(self),
            drop_empty=("backend", "role", "metadata"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SemanticAgent:
        return cls(
            agent_id=str(data["agent_id"]),
            component=str(data["component"]),
            backend=str(data.get("backend", "")),
            role=str(data.get("role", "")),
            metadata=dict(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class ProvenanceBundle(DictSerializable):
    """Explicit provenance bundle linking entities, activities, and agents."""

    bundle_id: str
    entity_ids: tuple[str, ...] = ()
    activity_ids: tuple[str, ...] = ()
    agent_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.bundle_id.strip()), "ProvenanceBundle.bundle_id must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return _compact_payload(
            DictSerializable.to_dict(self),
            drop_empty=("entity_ids", "activity_ids", "agent_ids", "metadata"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProvenanceBundle:
        return cls(
            bundle_id=str(data["bundle_id"]),
            entity_ids=tuple(str(item) for item in data.get("entity_ids", [])),
            activity_ids=tuple(str(item) for item in data.get("activity_ids", [])),
            agent_ids=tuple(str(item) for item in data.get("agent_ids", [])),
            metadata=dict(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class ContextFrame(DictSerializable):
    """Inherited scope and confidence applied to a set of operations."""

    frame_id: str
    scope: str
    confidence: float
    provenance_bundle_id: str = ""
    assumptions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.frame_id.strip()), "ContextFrame.frame_id must not be empty.")
        _require(bool(self.scope.strip()), "ContextFrame.scope must not be empty.")
        _require(0.0 <= self.confidence <= 1.0, "ContextFrame.confidence must be between 0 and 1.")

    def to_dict(self) -> dict[str, Any]:
        return _compact_payload(
            DictSerializable.to_dict(self),
            drop_empty=("provenance_bundle_id", "assumptions", "metadata"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContextFrame:
        return cls(
            frame_id=str(data["frame_id"]),
            scope=str(data.get("scope", "task")),
            confidence=float(data.get("confidence", 0.0)),
            provenance_bundle_id=str(data.get("provenance_bundle_id", "")),
            assumptions=tuple(str(item) for item in data.get("assumptions", [])),
            metadata=dict(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class OperationStep(DictSerializable):
    """One compact operation in the canonical reasoning operation stream."""

    op_id: str
    opcode: str
    args: tuple[str, ...] = ()
    output_ref: str = ""
    context_frame_id: str = ""
    evidence_handles: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(bool(self.op_id.strip()), "OperationStep.op_id must not be empty.")
        _require(bool(self.opcode.strip()), "OperationStep.opcode must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return _compact_payload(
            DictSerializable.to_dict(self),
            drop_empty=("args", "output_ref", "context_frame_id", "evidence_handles", "metadata"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OperationStep:
        opcode = str(data.get("opcode", data.get("op", "")))
        return cls(
            op_id=str(data.get("op_id", f"op_{opcode or 'unknown'}")),
            opcode=opcode,
            args=tuple(str(item) for item in data.get("args", [])),
            output_ref=str(data.get("output_ref", "")),
            context_frame_id=str(data.get("context_frame_id", "")),
            evidence_handles=tuple(str(item) for item in data.get("evidence_handles", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True, frozen=True)
class DecodeHint(DictSerializable):
    """Verified decode hint kept separate from natural-language rendering."""

    hint_id: str
    template: str
    entity_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(bool(self.hint_id.strip()), "DecodeHint.hint_id must not be empty.")
        _require(bool(self.template.strip()), "DecodeHint.template must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return _compact_payload(
            DictSerializable.to_dict(self),
            drop_empty=("entity_ids", "metadata"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecodeHint:
        return cls(
            hint_id=str(data["hint_id"]),
            template=str(data["template"]),
            entity_ids=tuple(str(item) for item in data.get("entity_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True, frozen=True)
class CanonicalReasoningGraph(DictSerializable):
    """Canonical typed reasoning/provenance graph for compressed traces."""

    entities: tuple[SemanticEntity, ...] = ()
    activities: tuple[SemanticActivity, ...] = ()
    agents: tuple[SemanticAgent, ...] = ()
    bundles: tuple[ProvenanceBundle, ...] = ()
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        entity_ids = tuple(entity.entity_id for entity in self.entities)
        activity_ids = tuple(activity.activity_id for activity in self.activities)
        agent_ids = tuple(agent.agent_id for agent in self.agents)
        bundle_ids = tuple(bundle.bundle_id for bundle in self.bundles)
        _require(len(set(entity_ids)) == len(entity_ids), "CanonicalReasoningGraph entity IDs must be unique.")
        _require(
            len(set(activity_ids)) == len(activity_ids),
            "CanonicalReasoningGraph activity IDs must be unique.",
        )
        _require(len(set(agent_ids)) == len(agent_ids), "CanonicalReasoningGraph agent IDs must be unique.")
        _require(len(set(bundle_ids)) == len(bundle_ids), "CanonicalReasoningGraph bundle IDs must be unique.")

    def to_dict(self) -> dict[str, Any]:
        return _compact_payload(
            DictSerializable.to_dict(self),
            drop_empty=("entities", "activities", "agents", "bundles"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CanonicalReasoningGraph:
        return cls(
            entities=tuple(SemanticEntity.from_dict(item) for item in data.get("entities", [])),
            activities=tuple(
                SemanticActivity.from_dict(item) for item in data.get("activities", [])
            ),
            agents=tuple(SemanticAgent.from_dict(item) for item in data.get("agents", [])),
            bundles=tuple(ProvenanceBundle.from_dict(item) for item in data.get("bundles", [])),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CandidateTrace(DictSerializable):
    """One bounded deep-mode candidate kept in canonical IR form."""

    candidate_id: str
    answer_text: str
    strategy: str
    verifier_type: str = ""
    verified: bool = False
    total_score: float = 0.0
    agreement_score: float = 0.0
    evidence_support_score: float = 0.0
    proof_hash_stability: float = 1.0
    degraded_reason: str = ""
    supporting_evidence_ids: tuple[str, ...] = ()
    tokens: tuple[str, ...] = ()
    expanded_preview: tuple[str, ...] = ()
    operation_stream: tuple[OperationStep, ...] = ()
    decode_hints: tuple[DecodeHint, ...] = ()
    proof_hash: str = ""
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.candidate_id.strip()), "CandidateTrace.candidate_id must not be empty.")
        _require(bool(self.answer_text.strip()), "CandidateTrace.answer_text must not be empty.")
        _require(0.0 <= self.total_score <= 1.0, "CandidateTrace.total_score must be between 0 and 1.")
        _require(0.0 <= self.agreement_score <= 1.0, "CandidateTrace.agreement_score must be between 0 and 1.")
        _require(
            0.0 <= self.evidence_support_score <= 1.0,
            "CandidateTrace.evidence_support_score must be between 0 and 1.",
        )
        _require(
            0.0 <= self.proof_hash_stability <= 1.0,
            "CandidateTrace.proof_hash_stability must be between 0 and 1.",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateTrace:
        return cls(
            candidate_id=str(data["candidate_id"]),
            answer_text=str(data["answer_text"]),
            strategy=str(data.get("strategy", "")),
            verifier_type=str(data.get("verifier_type", "")),
            verified=bool(data.get("verified", False)),
            total_score=float(data.get("total_score", 0.0)),
            agreement_score=float(data.get("agreement_score", 0.0)),
            evidence_support_score=float(data.get("evidence_support_score", 0.0)),
            proof_hash_stability=float(data.get("proof_hash_stability", 1.0)),
            degraded_reason=str(data.get("degraded_reason", "")),
            supporting_evidence_ids=tuple(str(item) for item in data.get("supporting_evidence_ids", [])),
            tokens=tuple(str(item) for item in data.get("tokens", [])),
            expanded_preview=tuple(str(item) for item in data.get("expanded_preview", [])),
            operation_stream=tuple(OperationStep.from_dict(item) for item in data.get("operation_stream", [])),
            decode_hints=tuple(DecodeHint.from_dict(item) for item in data.get("decode_hints", [])),
            proof_hash=str(data.get("proof_hash", "")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CompressionHint(DictSerializable):
    """Typed upstream structure hint that can seed deterministic macro proposals."""

    hint_id: str
    hint_type: CompressionHintType
    signature: tuple[str, ...]
    source_component: str
    reason: str
    weight: float = 0.0
    supporting_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.hint_id.strip()), "CompressionHint.hint_id must not be empty.")
        _require(bool(self.source_component.strip()), "CompressionHint.source_component must not be empty.")
        _require(bool(self.reason.strip()), "CompressionHint.reason must not be empty.")
        _require(len(self.signature) > 0, "CompressionHint.signature must not be empty.")
        _require(0.0 <= self.weight <= 1.0, "CompressionHint.weight must be between 0 and 1.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompressionHint:
        return cls(
            hint_id=str(data["hint_id"]),
            hint_type=_parse_enum(CompressionHintType, data.get("hint_type", CompressionHintType.SHARED_SUBPROOF)),
            signature=tuple(str(item) for item in data.get("signature", [])),
            source_component=str(data.get("source_component", "")),
            reason=str(data.get("reason", "")),
            weight=float(data.get("weight", 0.0)),
            supporting_refs=tuple(str(item) for item in data.get("supporting_refs", [])),
            metadata={str(key): value for key, value in dict(data.get("metadata", {})).items()},
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CompressedTrace(DictSerializable):
    """Compressed reasoning chain used by critic and compressor."""

    task_id: str
    tokens: tuple[str, ...]
    expanded_preview: tuple[str, ...]
    macros_used: tuple[str, ...]
    confidence: float
    reasoner_notes: str = ""
    ir_version: str = ""
    canonical_graph: CanonicalReasoningGraph | None = None
    canonical_graph_builder: str = ""
    operation_stream: tuple[OperationStep, ...] = ()
    symbol_table_refs: tuple[str, ...] = ()
    evidence_handles: tuple[str, ...] = ()
    context_frames: tuple[ContextFrame, ...] = ()
    candidate_traces: tuple[CandidateTrace, ...] = ()
    compression_hints: tuple[CompressionHint, ...] = ()
    proof_hash: str = ""
    decode_hints: tuple[DecodeHint, ...] = ()
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "CompressedTrace.task_id must not be empty.")
        _require(len(self.tokens) > 0, "CompressedTrace.tokens must not be empty.")
        _require(0.0 <= self.confidence <= 1.0, "CompressedTrace.confidence must be between 0 and 1.")

    def to_dict(self) -> dict[str, Any]:
        payload = _compact_payload(
            DictSerializable.to_dict(self),
            drop_empty=(
                "expanded_preview",
                "macros_used",
                "reasoner_notes",
                "ir_version",
                "canonical_graph_builder",
                "canonical_graph",
                "operation_stream",
                "symbol_table_refs",
                "evidence_handles",
                "context_frames",
                "candidate_traces",
                "compression_hints",
                "proof_hash",
                "decode_hints",
            ),
        )
        if self.canonical_graph_builder and "canonical_graph" in payload:
            payload.pop("canonical_graph", None)
        if self.canonical_graph_builder and "symbol_table_refs" in payload:
            payload.pop("symbol_table_refs", None)
        if self.canonical_graph_builder and "decode_hints" in payload:
            payload.pop("decode_hints", None)
        return payload

    def to_storage_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        if self.canonical_graph_builder == "reasoner_stub_v1":
            for field_name in (
                "expanded_preview",
                "reasoner_notes",
                "evidence_handles",
            ):
                payload.pop(field_name, None)
            raw_steps = payload.get("operation_stream")
            if isinstance(raw_steps, list):
                compact_steps: list[dict[str, Any]] = []
                for index, step in enumerate(raw_steps):
                    if not isinstance(step, dict):
                        compact_steps.append(step)
                        continue
                    compact_step = dict(step)
                    compact_step.pop("metadata", None)
                    if index > 0:
                        compact_step.pop("evidence_handles", None)
                    compact_steps.append(compact_step)
                payload["operation_stream"] = compact_steps
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompressedTrace:
        tokens = tuple(str(item) for item in data.get("tokens", data.get("compressed_chain", [])))
        macros_used = tuple(str(item) for item in data.get("macros_used", []))
        raw_graph = data.get("canonical_graph", data.get("graph"))
        canonical_graph = None
        if isinstance(raw_graph, Mapping):
            canonical_graph = CanonicalReasoningGraph.from_dict(raw_graph)
        canonical_graph_builder = str(data.get("canonical_graph_builder", ""))
        operation_stream = tuple(
            OperationStep.from_dict(item) for item in data.get("operation_stream", [])
        )
        context_frames = tuple(
            ContextFrame.from_dict(item) for item in data.get("context_frames", [])
        )
        candidate_traces = tuple(
            CandidateTrace.from_dict(item) for item in data.get("candidate_traces", [])
        )
        compression_hints = tuple(
            CompressionHint.from_dict(item) for item in data.get("compression_hints", [])
        )
        decode_hints = tuple(DecodeHint.from_dict(item) for item in data.get("decode_hints", []))
        created_at = _parse_datetime(data.get("created_at", utc_now()))
        symbol_table_refs = tuple(str(item) for item in data.get("symbol_table_refs", []))
        if not symbol_table_refs and canonical_graph_builder:
            symbol_table_refs = _derive_symbol_table_refs(
                builder=canonical_graph_builder,
                operation_stream=operation_stream,
            )
        evidence_handles = tuple(str(item) for item in data.get("evidence_handles", []))
        if not evidence_handles and canonical_graph_builder:
            evidence_handles = _derive_evidence_handles(
                builder=canonical_graph_builder,
                operation_stream=operation_stream,
            )
        if canonical_graph_builder == "reasoner_stub_v1":
            operation_stream = _restore_reasoner_stub_operation_stream(
                operation_stream=operation_stream,
                evidence_handles=evidence_handles,
                context_frames=context_frames,
            )
        expanded_preview = tuple(str(item) for item in data.get("expanded_preview", []))
        if not expanded_preview and canonical_graph_builder:
            expanded_preview = _derive_expanded_preview(
                builder=canonical_graph_builder,
                tokens=tokens,
            )
        if not decode_hints and canonical_graph_builder:
            decode_hints = _derive_decode_hints(
                builder=canonical_graph_builder,
                operation_stream=operation_stream,
            )
        reasoner_notes = str(data.get("reasoner_notes", ""))
        if not reasoner_notes and canonical_graph_builder:
            reasoner_notes = _derive_reasoner_notes(
                builder=canonical_graph_builder,
                context_frames=context_frames,
                symbol_table_refs=symbol_table_refs,
            )
        if canonical_graph is None and canonical_graph_builder:
            canonical_graph = _rebuild_derived_canonical_graph(
                builder=canonical_graph_builder,
                task_id=str(data["task_id"]),
                confidence=float(data.get("confidence", 0.0)),
                tokens=tokens,
                macros_used=macros_used,
                operation_stream=operation_stream,
                symbol_table_refs=symbol_table_refs,
                evidence_handles=evidence_handles,
                context_frames=context_frames,
                decode_hints=decode_hints,
                created_at=created_at,
            )
        return cls(
            task_id=str(data["task_id"]),
            tokens=tokens,
            expanded_preview=expanded_preview,
            macros_used=macros_used,
            confidence=float(data.get("confidence", 0.0)),
            reasoner_notes=reasoner_notes,
            ir_version=str(data.get("ir_version", "")),
            canonical_graph=canonical_graph,
            canonical_graph_builder=canonical_graph_builder,
            operation_stream=operation_stream,
            symbol_table_refs=symbol_table_refs,
            evidence_handles=evidence_handles,
            context_frames=context_frames,
            candidate_traces=candidate_traces,
            compression_hints=compression_hints,
            proof_hash=str(data.get("proof_hash", "")),
            decode_hints=decode_hints,
            created_at=created_at,
        )


@dataclass(slots=True, frozen=True)
class ResearchReasonerHandoff(DictSerializable):
    """Locked handoff contract between Researcher and Reasoner."""

    plan: Plan
    evidence: EvidenceBundle
    budget: ResourceBudget
    evidence_handles: tuple[str, ...]
    reasoning_mode: str = "fast"
    final_text_policy: str = "post_verification"
    output_contract: str = "compressed_trace_v1"
    implementation_mode: str = "deterministic_stub"
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(
            self.plan.task_id == self.evidence.task_id,
            "ResearchReasonerHandoff plan and evidence task IDs must match.",
        )
        _require(bool(self.reasoning_mode.strip()), "ResearchReasonerHandoff.reasoning_mode must not be empty.")
        _require(
            bool(self.final_text_policy.strip()),
            "ResearchReasonerHandoff.final_text_policy must not be empty.",
        )
        _require(
            bool(self.output_contract.strip()),
            "ResearchReasonerHandoff.output_contract must not be empty.",
        )
        _require(
            bool(self.implementation_mode.strip()),
            "ResearchReasonerHandoff.implementation_mode must not be empty.",
        )

    @classmethod
    def from_inputs(
        cls,
        *,
        plan: Plan,
        evidence: EvidenceBundle,
        budget: ResourceBudget,
        reasoning_mode: str = "fast",
        final_text_policy: str = "post_verification",
        output_contract: str = "compressed_trace_v1",
        implementation_mode: str = "deterministic_stub",
    ) -> ResearchReasonerHandoff:
        evidence_handles = tuple(
            dict.fromkeys(item.id for item in evidence.local_results + evidence.web_results)
        )
        return cls(
            plan=plan,
            evidence=evidence,
            budget=budget,
            evidence_handles=evidence_handles,
            reasoning_mode=reasoning_mode,
            final_text_policy=final_text_policy,
            output_contract=output_contract,
            implementation_mode=implementation_mode,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResearchReasonerHandoff:
        plan = Plan.from_dict(data["plan"])
        evidence = EvidenceBundle.from_dict(data["evidence"])
        default_handles = tuple(
            dict.fromkeys(item.id for item in evidence.local_results + evidence.web_results)
        )
        return cls(
            plan=plan,
            evidence=evidence,
            budget=ResourceBudget.from_dict(data["budget"]),
            evidence_handles=tuple(str(item) for item in data.get("evidence_handles", default_handles)),
            reasoning_mode=str(data.get("reasoning_mode", "fast")),
            final_text_policy=str(data.get("final_text_policy", "post_verification")),
            output_contract=str(data.get("output_contract", "compressed_trace_v1")),
            implementation_mode=str(data.get("implementation_mode", "deterministic_stub")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class ReasonerCriticHandoff(DictSerializable):
    """Locked handoff contract between Reasoner and Critic."""

    plan: Plan
    evidence: EvidenceBundle
    trace: CompressedTrace
    budget: ResourceBudget
    evidence_handles: tuple[str, ...]
    proof_hash: str
    required_opcode_names: tuple[str, ...]
    required_macro_names: tuple[str, ...]
    required_decoder_names: tuple[str, ...]
    repair_attempt_count: int = 0
    repair_history: tuple[str, ...] = ()
    final_text_policy: str = "post_verification"
    output_contract: str = "critique_report_v1"
    implementation_mode: str = "deterministic_stub"
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(
            self.plan.task_id == self.evidence.task_id == self.trace.task_id,
            "ReasonerCriticHandoff task IDs must match across plan, evidence, and trace.",
        )
        _require(
            self.repair_attempt_count >= 0,
            "ReasonerCriticHandoff.repair_attempt_count must be non-negative.",
        )
        _require(
            bool(self.final_text_policy.strip()),
            "ReasonerCriticHandoff.final_text_policy must not be empty.",
        )
        _require(
            bool(self.output_contract.strip()),
            "ReasonerCriticHandoff.output_contract must not be empty.",
        )
        _require(
            bool(self.implementation_mode.strip()),
            "ReasonerCriticHandoff.implementation_mode must not be empty.",
        )

    @classmethod
    def from_inputs(
        cls,
        *,
        plan: Plan,
        evidence: EvidenceBundle,
        trace: CompressedTrace,
        budget: ResourceBudget,
        repair_attempt_count: int = 0,
        repair_history: tuple[str, ...] = (),
        final_text_policy: str = "post_verification",
        output_contract: str = "critique_report_v1",
        implementation_mode: str = "deterministic_stub",
    ) -> ReasonerCriticHandoff:
        evidence_handles = trace.evidence_handles or tuple(
            dict.fromkeys(item.id for item in evidence.local_results + evidence.web_results)
        )
        return cls(
            plan=plan,
            evidence=evidence,
            trace=trace,
            budget=budget,
            evidence_handles=evidence_handles,
            proof_hash=trace.proof_hash,
            required_opcode_names=tuple(dict.fromkeys(step.opcode for step in trace.operation_stream if step.opcode)),
            required_macro_names=tuple(dict.fromkeys(trace.macros_used)),
            required_decoder_names=tuple(
                dict.fromkeys(hint.template for hint in trace.decode_hints if hint.template)
            ),
            repair_attempt_count=repair_attempt_count,
            repair_history=repair_history,
            final_text_policy=final_text_policy,
            output_contract=output_contract,
            implementation_mode=implementation_mode,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReasonerCriticHandoff:
        plan = Plan.from_dict(data["plan"])
        evidence = EvidenceBundle.from_dict(data["evidence"])
        trace = CompressedTrace.from_dict(data["trace"])
        default_handles = trace.evidence_handles or tuple(
            dict.fromkeys(item.id for item in evidence.local_results + evidence.web_results)
        )
        default_opcodes = tuple(dict.fromkeys(step.opcode for step in trace.operation_stream if step.opcode))
        default_macros = tuple(dict.fromkeys(trace.macros_used))
        default_decoders = tuple(
            dict.fromkeys(hint.template for hint in trace.decode_hints if hint.template)
        )
        return cls(
            plan=plan,
            evidence=evidence,
            trace=trace,
            budget=ResourceBudget.from_dict(data["budget"]),
            evidence_handles=tuple(str(item) for item in data.get("evidence_handles", default_handles)),
            proof_hash=str(data.get("proof_hash", trace.proof_hash)),
            required_opcode_names=tuple(
                str(item) for item in data.get("required_opcode_names", default_opcodes)
            ),
            required_macro_names=tuple(
                str(item) for item in data.get("required_macro_names", default_macros)
            ),
            required_decoder_names=tuple(
                str(item) for item in data.get("required_decoder_names", default_decoders)
            ),
            repair_attempt_count=int(data.get("repair_attempt_count", 0)),
            repair_history=tuple(str(item) for item in data.get("repair_history", [])),
            final_text_policy=str(data.get("final_text_policy", "post_verification")),
            output_contract=str(data.get("output_contract", "critique_report_v1")),
            implementation_mode=str(data.get("implementation_mode", "deterministic_stub")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


def _derive_symbol_table_refs(
    *,
    builder: str,
    operation_stream: tuple[OperationStep, ...],
) -> tuple[str, ...]:
    if builder == "reasoner_stub_v1":
        bind_step = next((step for step in operation_stream if step.opcode == "bind"), None)
        if bind_step is not None:
            refs = [arg for arg in bind_step.args if arg]
            if bind_step.output_ref:
                refs.append(bind_step.output_ref)
            return tuple(refs)
    if builder == "macro_engine_v1":
        return tuple(step.output_ref for step in operation_stream if step.output_ref)
    refs: list[str] = []
    for step in operation_stream:
        refs.extend(arg for arg in step.args if arg)
        if step.output_ref:
            refs.append(step.output_ref)
    return tuple(dict.fromkeys(refs))


def _derive_decode_hints(
    *,
    builder: str,
    operation_stream: tuple[OperationStep, ...],
) -> tuple[DecodeHint, ...]:
    if builder == "reasoner_stub_v1":
        emit_step = next((step for step in reversed(operation_stream) if step.opcode == "emit"), None)
        metadata = {}
        if emit_step is not None:
            for key in (
                "candidate_id",
                "answer_text",
                "selected_strategy",
                "selected_verifier",
                "candidate_score",
                "verified",
                "degraded_reason",
                "candidate_count",
                "supporting_evidence_ids",
            ):
                if key in emit_step.metadata:
                    metadata[key] = emit_step.metadata[key]
        return (
            DecodeHint(
                hint_id="d0",
                template="verified_answer",
                entity_ids=("a",),
                metadata=metadata,
            ),
        )
    if builder == "macro_engine_v1":
        emit_refs = tuple(
            sorted(
                f"ent_{step.output_ref}"
                for step in operation_stream
                if step.opcode == "emit" and step.output_ref
            )
        )
        if emit_refs:
            return (
                DecodeHint(
                    hint_id="hint_verified_answer",
                    template="verified_answer",
                    entity_ids=emit_refs,
                    metadata={"kind": "emit_projection"},
                ),
            )
        if operation_stream and operation_stream[-1].output_ref:
            return (
                DecodeHint(
                    hint_id="hint_trace_summary",
                    template="compressed_trace_summary",
                    entity_ids=(f"ent_{operation_stream[-1].output_ref}",),
                    metadata={"kind": "summary_projection"},
                ),
            )
    return ()


def _derive_evidence_handles(
    *,
    builder: str,
    operation_stream: tuple[OperationStep, ...],
) -> tuple[str, ...]:
    if builder in {"reasoner_stub_v1", "macro_engine_v1"}:
        handles: list[str] = []
        for step in operation_stream:
            handles.extend(handle for handle in step.evidence_handles if handle)
        return tuple(dict.fromkeys(handles))
    return ()


def _restore_reasoner_stub_operation_stream(
    *,
    operation_stream: tuple[OperationStep, ...],
    evidence_handles: tuple[str, ...],
    context_frames: tuple[ContextFrame, ...],
) -> tuple[OperationStep, ...]:
    context_metadata = dict(context_frames[0].metadata) if context_frames else {}
    selected_candidate_id = str(context_metadata.get("cid", "")).strip()
    selected_answer = str(context_metadata.get("ta", "")).strip()
    selected_strategy = str(context_metadata.get("sa", "")).strip()
    selected_verifier = str(context_metadata.get("sv", "")).strip()
    degraded_reason = str(context_metadata.get("dr", "")).strip()
    supporting_evidence_ids = tuple(
        str(item) for item in context_metadata.get("si", ()) if str(item).strip()
    )
    try:
        candidate_score = float(context_metadata.get("ss", 0.0))
    except (TypeError, ValueError):
        candidate_score = 0.0
    try:
        candidate_count = max(1, int(context_metadata.get("cc", 1)))
    except (TypeError, ValueError):
        candidate_count = 1
    verified = bool(context_metadata.get("vv", False))
    restored: list[OperationStep] = []
    for index, step in enumerate(operation_stream):
        metadata = dict(step.metadata)
        if "source_token" not in metadata:
            if step.opcode == "lookup":
                metadata["source_token"] = "@match_local_evidence"
            elif step.opcode == "compare":
                metadata["source_token"] = "@compare_candidates"
            elif step.opcode in {"infer", "check"}:
                metadata["source_token"] = "@select_verified_candidate"
            else:
                metadata["source_token"] = "@compose_answer"
        if step.opcode == "bind":
            if selected_candidate_id and "candidate_id" not in metadata:
                metadata["candidate_id"] = selected_candidate_id
            if selected_strategy and "selected_strategy" not in metadata:
                metadata["selected_strategy"] = selected_strategy
            if selected_verifier and "selected_verifier" not in metadata:
                metadata["selected_verifier"] = selected_verifier
            if "candidate_score" not in metadata:
                metadata["candidate_score"] = candidate_score
            if "verified" not in metadata:
                metadata["verified"] = verified
        if step.opcode == "check":
            if selected_candidate_id and "candidate_id" not in metadata:
                metadata["candidate_id"] = selected_candidate_id
            if selected_verifier and "tool_check" not in metadata:
                metadata["tool_check"] = selected_verifier
            if "candidate_count" not in metadata:
                metadata["candidate_count"] = candidate_count
        if step.opcode == "cite":
            if "evidence_count" not in metadata:
                metadata["evidence_count"] = len(evidence_handles)
        if step.opcode == "confidence_update" and "confidence" not in metadata and context_frames:
            metadata["confidence"] = round(context_frames[0].confidence, 2)
        if step.opcode == "emit":
            if selected_candidate_id and "candidate_id" not in metadata:
                metadata["candidate_id"] = selected_candidate_id
            if selected_answer and "answer_text" not in metadata:
                metadata["answer_text"] = selected_answer
            if selected_strategy and "selected_strategy" not in metadata:
                metadata["selected_strategy"] = selected_strategy
            if selected_verifier and "selected_verifier" not in metadata:
                metadata["selected_verifier"] = selected_verifier
            if "candidate_score" not in metadata:
                metadata["candidate_score"] = candidate_score
            if "verified" not in metadata:
                metadata["verified"] = verified
            if degraded_reason and "degraded_reason" not in metadata:
                metadata["degraded_reason"] = degraded_reason
            if "candidate_count" not in metadata:
                metadata["candidate_count"] = candidate_count
            if supporting_evidence_ids and "supporting_evidence_ids" not in metadata:
                metadata["supporting_evidence_ids"] = list(supporting_evidence_ids)
        restored.append(
            OperationStep(
                op_id=step.op_id,
                opcode=step.opcode,
                args=step.args,
                output_ref=step.output_ref,
                context_frame_id=step.context_frame_id,
                evidence_handles=step.evidence_handles or evidence_handles,
                metadata=metadata,
            )
        )
    return tuple(restored)


def _derive_expanded_preview(
    *,
    builder: str,
    tokens: tuple[str, ...],
) -> tuple[str, ...]:
    if builder == "reasoner_stub_v1":
        review_depth = sum(1 for token in tokens if token.startswith("@review_evidence_"))
        reasoner_passes = sum(1 for token in tokens if token.startswith("@reason_pass_"))
        preview = [
            "Read question and constraints",
            f"Review {max(1, review_depth)} evidence item(s)",
        ]
        for pass_index in range(1, reasoner_passes + 1):
            preview.append(f"Reasoning pass {pass_index} of {reasoner_passes}")
        return tuple(preview)
    return ()


def _format_reasoner_stub_notes(
    *,
    model_backend: str,
    evidence_count: int,
    review_depth: int,
    reasoner_passes: int,
    loaded_opcodes: Sequence[str],
    loaded_macros: Sequence[str],
    loaded_decoders: Sequence[str],
    symbol_table_refs: Sequence[str],
) -> str:
    return (
        "reasoner_stub=active\n"
        f"model_backend={model_backend}\n"
        f"evidence_count={evidence_count}\n"
        f"review_depth={review_depth}\n"
        f"reasoner_passes={reasoner_passes}\n"
        f"loaded_opcodes={','.join(loaded_opcodes)}\n"
        f"loaded_macros={','.join(loaded_macros)}\n"
        f"loaded_decoders={','.join(loaded_decoders)}\n"
        f"symbol_table_refs={','.join(symbol_table_refs)}"
    )


def _derive_reasoner_notes(
    *,
    builder: str,
    context_frames: tuple[ContextFrame, ...],
    symbol_table_refs: tuple[str, ...],
) -> str:
    if builder != "reasoner_stub_v1" or not context_frames:
        return ""
    metadata = dict(context_frames[0].metadata)
    return _format_reasoner_stub_notes(
        model_backend=str(metadata.get("mb", metadata.get("model_backend", ""))),
        evidence_count=int(metadata.get("ec", metadata.get("evidence_count", 0))),
        review_depth=int(metadata.get("rd", metadata.get("review_depth", 1))),
        reasoner_passes=int(metadata.get("rp", metadata.get("reasoner_passes", 1))),
        loaded_opcodes=_metadata_sequence(metadata.get("op", metadata.get("loaded_opcodes", ()))),
        loaded_macros=_metadata_sequence(metadata.get("mc", metadata.get("loaded_macros", ()))),
        loaded_decoders=_metadata_sequence(metadata.get("dc", metadata.get("loaded_decoders", ()))),
        symbol_table_refs=symbol_table_refs,
    )


def _metadata_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        if not value:
            return ()
        return tuple(item for item in value.split(",") if item)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def _rebuild_derived_canonical_graph(
    *,
    builder: str,
    task_id: str,
    confidence: float,
    tokens: tuple[str, ...],
    macros_used: tuple[str, ...],
    operation_stream: tuple[OperationStep, ...],
    symbol_table_refs: tuple[str, ...],
    evidence_handles: tuple[str, ...],
    context_frames: tuple[ContextFrame, ...],
    decode_hints: tuple[DecodeHint, ...],
    created_at: datetime,
) -> CanonicalReasoningGraph | None:
    if builder == "reasoner_stub_v1":
        return _rebuild_reasoner_stub_graph(
            confidence=confidence,
            operation_stream=operation_stream,
            symbol_table_refs=symbol_table_refs,
            evidence_handles=evidence_handles,
            context_frames=context_frames,
            created_at=created_at,
        )
    if builder == "macro_engine_v1":
        return _rebuild_macro_engine_graph(
            task_id=task_id,
            tokens=tokens,
            macros_used=macros_used,
            operation_stream=operation_stream,
            created_at=created_at,
        )
    return None


def _rebuild_reasoner_stub_graph(
    *,
    confidence: float,
    operation_stream: tuple[OperationStep, ...],
    symbol_table_refs: tuple[str, ...],
    evidence_handles: tuple[str, ...],
    context_frames: tuple[ContextFrame, ...],
    created_at: datetime,
) -> CanonicalReasoningGraph:
    context_metadata = dict(context_frames[0].metadata) if context_frames else {}
    emit_step = next((step for step in reversed(operation_stream) if step.opcode == "emit"), None)
    bind_step = next((step for step in operation_stream if step.opcode == "bind"), None)
    selected_answer = ""
    selected_strategy = ""
    selected_verifier = ""
    degraded_reason = ""
    selected_verified = False
    candidate_score = 0.0
    candidate_count = 1
    if emit_step is not None:
        selected_answer = str(emit_step.metadata.get("answer_text", "")).strip()
        selected_strategy = str(emit_step.metadata.get("selected_strategy", "")).strip()
        selected_verifier = str(emit_step.metadata.get("selected_verifier", "")).strip()
        degraded_reason = str(emit_step.metadata.get("degraded_reason", "")).strip()
        selected_verified = bool(emit_step.metadata.get("verified", False))
        try:
            candidate_score = float(emit_step.metadata.get("candidate_score", 0.0))
        except (TypeError, ValueError):
            candidate_score = 0.0
        raw_candidate_count = emit_step.metadata.get("candidate_count", context_metadata.get("cc", 1))
        try:
            candidate_count = max(1, int(raw_candidate_count))
        except (TypeError, ValueError):
            candidate_count = 1
    if not selected_answer:
        selected_answer = str(context_metadata.get("ta", "")).strip()
    if not selected_strategy:
        selected_strategy = str(context_metadata.get("sa", "")).strip()
    if not selected_verifier:
        selected_verifier = str(context_metadata.get("sv", "")).strip()
    if not degraded_reason:
        degraded_reason = str(context_metadata.get("dr", "")).strip()
    if not selected_verified:
        selected_verified = bool(context_metadata.get("vv", False))
    if candidate_score <= 0.0:
        try:
            candidate_score = float(context_metadata.get("ss", 0.0))
        except (TypeError, ValueError):
            candidate_score = 0.0
    agent = SemanticAgent(
        agent_id="ag0",
        component="reasoner",
        backend=str(context_metadata.get("mb", context_metadata.get("model_backend", ""))),
        role="foreground_reasoning",
        metadata={"builder": "reasoner_stub_v1"},
        created_at=created_at,
    )
    entities: list[SemanticEntity] = [
        SemanticEntity(
            entity_id="q",
            entity_type="question",
            value="sym_question",
            created_at=created_at,
        ),
    ]
    for index, handle in enumerate(evidence_handles, start=1):
        symbol_ref = f"sym_evidence_{index}"
        entities.append(
            SemanticEntity(
                entity_id=f"ev{index}",
                entity_type="evidence_item",
                value=handle,
                evidence_handles=(handle,),
                attributes={"symbol_ref": symbol_ref},
                created_at=created_at,
            )
        )
    entities.extend(
        [
            SemanticEntity(
            entity_id="es",
            entity_type="evidence_set",
            value="sym_evidence_set",
            evidence_handles=evidence_handles,
            attributes={"count": len(evidence_handles)},
            created_at=created_at,
        ),
        SemanticEntity(
            entity_id="b0",
            entity_type="intermediate_binding",
            value=next(
                (step.output_ref for step in operation_stream if step.opcode == "bind" and step.output_ref),
                "sym_answer",
            ),
            evidence_handles=evidence_handles,
            attributes=_compact_payload(
                {
                    "symbol_refs": [
                        ref
                        for ref in symbol_table_refs
                        if ref == "sym_question" or (ref.startswith("sym_evidence_") and ref != "sym_evidence_set")
                    ],
                    "candidate_count": candidate_count,
                    "selected_strategy": selected_strategy,
                    "selected_verifier": selected_verifier,
                    "candidate_score": candidate_score,
                    "verified": selected_verified,
                },
                drop_empty=("selected_strategy", "selected_verifier"),
            ),
            created_at=created_at,
        ),
        SemanticEntity(
            entity_id="a",
            entity_type="answer_fragment",
            value=selected_answer or "sym_answer",
            evidence_handles=evidence_handles,
            confidence=confidence,
            attributes=_compact_payload(
                {
                    "symbol_ref": bind_step.output_ref if bind_step is not None and bind_step.output_ref else "sym_answer",
                    "strategy": selected_strategy,
                    "verifier": selected_verifier,
                    "candidate_score": candidate_score,
                    "verified": selected_verified,
                    "degraded_reason": degraded_reason,
                    "candidate_count": candidate_count,
                },
                drop_empty=("strategy", "verifier", "degraded_reason"),
            ),
            created_at=created_at,
        ),
        ]
    )
    retrieve_outputs = tuple(entity.entity_id for entity in entities if entity.entity_type == "evidence_item") + ("es",)
    activities = (
        SemanticActivity(
            activity_id="ac0",
            activity_type="retrieve",
            input_entity_ids=("q",),
            output_entity_ids=retrieve_outputs,
            agent_id=agent.agent_id,
            evidence_handles=evidence_handles,
            created_at=created_at,
        ),
        SemanticActivity(
            activity_id="ac1",
            activity_type="bind",
            input_entity_ids=("q", "es"),
            output_entity_ids=("b0",),
            agent_id=agent.agent_id,
            evidence_handles=evidence_handles,
            created_at=created_at,
        ),
        SemanticActivity(
            activity_id="ac2",
            activity_type="emit",
            input_entity_ids=("b0",),
            output_entity_ids=("a",),
            agent_id=agent.agent_id,
            evidence_handles=evidence_handles,
            created_at=created_at,
        ),
    )
    bundle = ProvenanceBundle(
        bundle_id="pb0",
        entity_ids=tuple(entity.entity_id for entity in entities),
        activity_ids=tuple(activity.activity_id for activity in activities),
        agent_ids=(agent.agent_id,),
        created_at=created_at,
    )
    return CanonicalReasoningGraph(
        entities=tuple(entities),
        activities=activities,
        agents=(agent,),
        bundles=(bundle,),
        created_at=created_at,
    )


def _rebuild_macro_engine_graph(
    *,
    task_id: str,
    tokens: tuple[str, ...],
    macros_used: tuple[str, ...],
    operation_stream: tuple[OperationStep, ...],
    created_at: datetime,
) -> CanonicalReasoningGraph:
    agent = SemanticAgent(
        agent_id="agent_macro_engine",
        component="macro_engine",
        backend="deterministic_ir",
        role="compression_runtime",
        metadata={"task_id": task_id},
        created_at=created_at,
    )
    entities: list[SemanticEntity] = [
        SemanticEntity(
            entity_id="ent_input",
            entity_type="trace_input",
            value=task_id,
            attributes={"kind": "macro_engine_input"},
            created_at=created_at,
        )
    ]
    activities: list[SemanticActivity] = []
    macro_entity_ids: list[str] = []
    for index, macro_name in enumerate(macros_used, start=1):
        entity_id = f"macro_{index:03d}"
        macro_entity_ids.append(entity_id)
        entities.append(
            SemanticEntity(
                entity_id=entity_id,
                entity_type="macro_definition",
                value=macro_name.lstrip("@"),
                attributes={"token": macro_name},
                created_at=created_at,
            )
        )
    if macro_entity_ids:
        activities.append(
            SemanticActivity(
                activity_id="act_000_expand",
                activity_type="macro_expand",
                input_entity_ids=("ent_input", *macro_entity_ids),
                output_entity_ids=("ent_input",),
                agent_id=agent.agent_id,
                metadata={"token_count": len(tokens)},
                created_at=created_at,
            )
        )
    previous_entity_id = "ent_input"
    for index, step in enumerate(operation_stream, start=1):
        entity_id = f"ent_{step.output_ref or f'step_{index:03d}'}"
        source_token = str(step.metadata.get("source_token", step.opcode))
        entities.append(
            SemanticEntity(
                entity_id=entity_id,
                entity_type="answer_fragment" if step.opcode == "emit" else "intermediate_binding",
                value=source_token,
                evidence_handles=step.evidence_handles,
                attributes={
                    "opcode": step.opcode,
                    "args": list(step.args),
                },
                created_at=created_at,
            )
        )
        activities.append(
            SemanticActivity(
                activity_id=f"act_{index:03d}_{step.opcode}",
                activity_type=step.opcode,
                input_entity_ids=(previous_entity_id,),
                output_entity_ids=(entity_id,),
                agent_id=agent.agent_id,
                evidence_handles=step.evidence_handles,
                metadata={"op_id": step.op_id},
                created_at=created_at,
            )
        )
        previous_entity_id = entity_id
    bundle = ProvenanceBundle(
        bundle_id="bundle_macro_engine",
        entity_ids=tuple(entity.entity_id for entity in entities),
        activity_ids=tuple(activity.activity_id for activity in activities),
        agent_ids=(agent.agent_id,),
        metadata={"task_id": task_id},
        created_at=created_at,
    )
    return CanonicalReasoningGraph(
        entities=tuple(entities),
        activities=tuple(activities),
        agents=(agent,),
        bundles=(bundle,),
        created_at=created_at,
    )


@dataclass(slots=True, frozen=True)
class CritiqueReport(DictSerializable):
    """Critic output containing validation status and issues."""

    task_id: str
    is_valid: bool
    issues: tuple[str, ...]
    fixed_trace: CompressedTrace | None
    evidence_coverage: float
    critic_notes: str = ""
    result: CritiqueResult = CritiqueResult.VALID
    verifier_type: str = ""
    proof_hash_match: bool = True
    candidate_score: float = 0.0
    repair_actions: tuple[str, ...] = ()
    degraded_reason: str = ""
    failure_categories: tuple[str, ...] = ()
    provenance_coverage: float = 1.0
    macro_violations: tuple[str, ...] = ()
    drift_score: float = 0.0
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "CritiqueReport.task_id must not be empty.")
        _require(0.0 <= self.evidence_coverage <= 1.0, "evidence_coverage must be between 0 and 1.")
        _require(0.0 <= self.candidate_score <= 1.0, "candidate_score must be between 0 and 1.")
        _require(0.0 <= self.provenance_coverage <= 1.0, "provenance_coverage must be between 0 and 1.")
        _require(0.0 <= self.drift_score <= 1.0, "drift_score must be between 0 and 1.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CritiqueReport:
        raw_fixed = data.get("fixed_trace")
        fixed_trace = None
        if isinstance(raw_fixed, Mapping):
            fixed_trace = CompressedTrace.from_dict(raw_fixed)
        return cls(
            task_id=str(data["task_id"]),
            is_valid=bool(data.get("is_valid", False)),
            issues=tuple(str(item) for item in data.get("issues", [])),
            fixed_trace=fixed_trace,
            evidence_coverage=float(data.get("evidence_coverage", 0.0)),
            critic_notes=str(data.get("critic_notes", "")),
            result=_parse_enum(CritiqueResult, data.get("result", CritiqueResult.VALID)),
            verifier_type=str(data.get("verifier_type", "")),
            proof_hash_match=bool(data.get("proof_hash_match", True)),
            candidate_score=float(data.get("candidate_score", 0.0)),
            repair_actions=tuple(str(item) for item in data.get("repair_actions", [])),
            degraded_reason=str(data.get("degraded_reason", "")),
            failure_categories=tuple(str(item) for item in data.get("failure_categories", [])),
            provenance_coverage=float(data.get("provenance_coverage", 1.0)),
            macro_violations=tuple(str(item) for item in data.get("macro_violations", [])),
            drift_score=float(data.get("drift_score", 0.0)),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class MacroProposal(DictSerializable):
    """Candidate macro addition or revision."""

    proposal_id: str
    macro: Macro
    reason: str
    examples: tuple[str, ...]
    simulation_score: float
    approved: bool
    validation_passed: bool = False
    validation_issues: tuple[str, ...] = ()
    proof_fingerprint: str = ""
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.proposal_id.strip()), "MacroProposal.proposal_id must not be empty.")
        _require(bool(self.reason.strip()), "MacroProposal.reason must not be empty.")
        _require(0.0 <= self.simulation_score <= 1.0, "simulation_score must be between 0 and 1.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MacroProposal:
        return cls(
            proposal_id=str(data["proposal_id"]),
            macro=Macro.from_dict(data["macro"]),
            reason=str(data["reason"]),
            examples=tuple(str(item) for item in data.get("examples", [])),
            simulation_score=float(data.get("simulation_score", 0.0)),
            approved=bool(data.get("approved", False)),
            validation_passed=bool(data.get("validation_passed", False)),
            validation_issues=tuple(str(item) for item in data.get("validation_issues", [])),
            proof_fingerprint=str(data.get("proof_fingerprint", "")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class TaskResult(DictSerializable):
    """Final orchestrator output for one user task."""

    task_id: str
    plan: Plan
    evidence: EvidenceBundle
    reasoning: CompressedTrace
    critique: CritiqueReport
    compression: tuple[MacroProposal, ...]
    answer_text: str = ""
    warnings: tuple[str, ...] = ()
    metrics: tuple[PerformanceMetric, ...] = ()
    completed_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "TaskResult.task_id must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskResult:
        return cls(
            task_id=str(data["task_id"]),
            plan=Plan.from_dict(data["plan"]),
            evidence=EvidenceBundle.from_dict(data["evidence"]),
            reasoning=CompressedTrace.from_dict(data["reasoning"]),
            critique=CritiqueReport.from_dict(data["critique"]),
            compression=tuple(
                MacroProposal.from_dict(item) for item in data.get("compression", [])
            ),
            answer_text=str(data.get("answer_text", "")),
            warnings=tuple(str(item) for item in data.get("warnings", [])),
            metrics=tuple(
                PerformanceMetric.from_dict(item) for item in data.get("metrics", [])
            ),
            completed_at=_parse_datetime(data.get("completed_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class RuntimeEvent(DictSerializable):
    """Single structured event emitted during orchestration."""

    stage: str
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.stage.strip()), "RuntimeEvent.stage must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuntimeEvent:
        return cls(
            stage=str(data["stage"]),
            payload=dict(data.get("payload", {})),
            timestamp=_parse_datetime(data.get("timestamp", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class LifecycleStatus(DictSerializable):
    """Minimal health snapshot for components shown in logs/UI."""

    component: str
    state: AgentState
    last_error: str | None = None
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.component.strip()), "LifecycleStatus.component must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LifecycleStatus:
        return cls(
            component=str(data["component"]),
            state=_parse_enum(AgentState, data["state"]),
            last_error=data.get("last_error"),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class AgentStatus(DictSerializable):
    """Current status snapshot for each logical agent."""

    component: str
    state: AgentState
    task_id: str | None = None
    severity: SeverityLevel = SeverityLevel.LOW
    message: str = ""
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.component.strip()), "AgentStatus.component must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentStatus:
        return cls(
            component=str(data["component"]),
            state=_parse_enum(AgentState, data["state"]),
            task_id=data.get("task_id"),
            severity=_parse_enum(SeverityLevel, data.get("severity", SeverityLevel.LOW)),
            message=str(data.get("message", "")),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class RuntimeCondition(DictSerializable):
    """Machine-readable runtime condition surfaced through the event stream."""

    stage: str
    category: str
    component: str
    reason: str
    severity: SeverityLevel = SeverityLevel.MEDIUM
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    observed_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.stage.strip()), "RuntimeCondition.stage must not be empty.")
        _require(bool(self.category.strip()), "RuntimeCondition.category must not be empty.")
        _require(bool(self.component.strip()), "RuntimeCondition.component must not be empty.")
        _require(bool(self.reason.strip()), "RuntimeCondition.reason must not be empty.")

    def to_event_payload(self) -> dict[str, Any]:
        """Return the event payload shape shared across runtime surfaces."""
        return {
            "category": self.category,
            "component": self.component,
            "reason": self.reason,
            "severity": self.severity.value,
            "task_id": self.task_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_event(
        cls,
        stage: str,
        payload: Mapping[str, Any],
        *,
        timestamp: datetime | str | None = None,
    ) -> RuntimeCondition:
        return cls(
            stage=stage,
            category=str(payload.get("category", "runtime")),
            component=str(payload.get("component", "runtime")),
            reason=str(payload.get("reason", "unknown")),
            severity=_parse_enum(SeverityLevel, payload.get("severity", SeverityLevel.MEDIUM)),
            task_id=payload.get("task_id"),
            metadata=dict(payload.get("metadata", {})),
            observed_at=_parse_datetime(timestamp or payload.get("observed_at", utc_now())),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuntimeCondition:
        return cls(
            stage=str(data["stage"]),
            category=str(data["category"]),
            component=str(data["component"]),
            reason=str(data["reason"]),
            severity=_parse_enum(SeverityLevel, data.get("severity", SeverityLevel.MEDIUM)),
            task_id=data.get("task_id"),
            metadata=dict(data.get("metadata", {})),
            observed_at=_parse_datetime(data.get("observed_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class LongHorizonCandidateSnapshot(DictSerializable):
    """Compact persisted candidate summary stored in long-horizon checkpoints."""

    candidate_id: str
    strategy: str = ""
    verifier_type: str = ""
    verified: bool = False
    total_score: float = 0.0
    degraded_reason: str = ""
    supporting_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require(bool(self.candidate_id.strip()), "LongHorizonCandidateSnapshot.candidate_id must not be empty.")
        _require(0.0 <= self.total_score <= 1.0, "LongHorizonCandidateSnapshot.total_score must be between 0 and 1.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LongHorizonCandidateSnapshot:
        return cls(
            candidate_id=str(data["candidate_id"]),
            strategy=str(data.get("strategy", "")),
            verifier_type=str(data.get("verifier_type", "")),
            verified=bool(data.get("verified", False)),
            total_score=float(data.get("total_score", 0.0)),
            degraded_reason=str(data.get("degraded_reason", "")),
            supporting_evidence_ids=tuple(str(item) for item in data.get("supporting_evidence_ids", ())),
        )


@dataclass(slots=True, frozen=True)
class LongHorizonCheckpoint(DictSerializable):
    """Persisted checkpoint emitted after one bounded long-horizon work cycle."""

    session_id: str
    cycle_index: int
    total_cycles: int
    task_id: str
    question: str
    budget: ResourceBudget
    candidate_summaries: tuple[LongHorizonCandidateSnapshot, ...] = ()
    selected_candidate_id: str = ""
    supporting_evidence_ids: tuple[str, ...] = ()
    refreshed_web_source_refs: tuple[str, ...] = ()
    critique_result: str = ""
    critique_summary: tuple[str, ...] = ()
    repair_actions: tuple[str, ...] = ()
    answer_preview: str = ""
    resume_count: int = 0
    throttled: bool = False
    throttle_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.session_id.strip()), "LongHorizonCheckpoint.session_id must not be empty.")
        _require(self.cycle_index > 0, "LongHorizonCheckpoint.cycle_index must be positive.")
        _require(self.total_cycles >= self.cycle_index, "LongHorizonCheckpoint.total_cycles must cover cycle_index.")
        _require(bool(self.task_id.strip()), "LongHorizonCheckpoint.task_id must not be empty.")
        _require(bool(self.question.strip()), "LongHorizonCheckpoint.question must not be empty.")
        _require(self.resume_count >= 0, "LongHorizonCheckpoint.resume_count must be zero or positive.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LongHorizonCheckpoint:
        return cls(
            session_id=str(data["session_id"]),
            cycle_index=int(data["cycle_index"]),
            total_cycles=int(data["total_cycles"]),
            task_id=str(data["task_id"]),
            question=str(data["question"]),
            budget=ResourceBudget.from_dict(data["budget"]),
            candidate_summaries=tuple(
                LongHorizonCandidateSnapshot.from_dict(item)
                for item in data.get("candidate_summaries", ())
            ),
            selected_candidate_id=str(data.get("selected_candidate_id", "")),
            supporting_evidence_ids=tuple(str(item) for item in data.get("supporting_evidence_ids", ())),
            refreshed_web_source_refs=tuple(str(item) for item in data.get("refreshed_web_source_refs", ())),
            critique_result=str(data.get("critique_result", "")),
            critique_summary=tuple(str(item) for item in data.get("critique_summary", ())),
            repair_actions=tuple(str(item) for item in data.get("repair_actions", ())),
            answer_preview=str(data.get("answer_preview", "")),
            resume_count=int(data.get("resume_count", 0)),
            throttled=bool(data.get("throttled", False)),
            throttle_reason=str(data.get("throttle_reason", "")),
            metadata=dict(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class LongHorizonSession(DictSerializable):
    """Persisted session summary for one multi-cycle checkpointed run."""

    session_id: str
    question: str
    requested_minutes: int
    budget: ResourceBudget
    status: LongHorizonSessionState = LongHorizonSessionState.PENDING
    total_cycles: int = 1
    completed_cycles: int = 0
    last_task_id: str = ""
    last_checkpoint_cycle: int = 0
    resume_count: int = 0
    pause_requested: bool = False
    cancel_requested: bool = False
    throttled: bool = False
    throttle_reason: str = ""
    latest_answer_preview: str = ""
    last_control_reason: str = ""
    last_error: str = ""
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.session_id.strip()), "LongHorizonSession.session_id must not be empty.")
        _require(bool(self.question.strip()), "LongHorizonSession.question must not be empty.")
        _require(self.requested_minutes > 0, "LongHorizonSession.requested_minutes must be positive.")
        _require(self.total_cycles > 0, "LongHorizonSession.total_cycles must be positive.")
        _require(self.completed_cycles >= 0, "LongHorizonSession.completed_cycles must be zero or positive.")
        _require(
            self.completed_cycles <= self.total_cycles,
            "LongHorizonSession.completed_cycles must not exceed total_cycles.",
        )
        _require(
            0 <= self.last_checkpoint_cycle <= self.total_cycles,
            "LongHorizonSession.last_checkpoint_cycle must be within total_cycles.",
        )
        _require(self.resume_count >= 0, "LongHorizonSession.resume_count must be zero or positive.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LongHorizonSession:
        return cls(
            session_id=str(data["session_id"]),
            question=str(data["question"]),
            requested_minutes=int(data["requested_minutes"]),
            budget=ResourceBudget.from_dict(data["budget"]),
            status=_parse_enum(LongHorizonSessionState, data.get("status", LongHorizonSessionState.PENDING)),
            total_cycles=int(data.get("total_cycles", 1)),
            completed_cycles=int(data.get("completed_cycles", 0)),
            last_task_id=str(data.get("last_task_id", "")),
            last_checkpoint_cycle=int(data.get("last_checkpoint_cycle", 0)),
            resume_count=int(data.get("resume_count", 0)),
            pause_requested=bool(data.get("pause_requested", False)),
            cancel_requested=bool(data.get("cancel_requested", False)),
            throttled=bool(data.get("throttled", False)),
            throttle_reason=str(data.get("throttle_reason", "")),
            latest_answer_preview=str(data.get("latest_answer_preview", "")),
            last_control_reason=str(data.get("last_control_reason", "")),
            last_error=str(data.get("last_error", "")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class ModelRegistration(DictSerializable):
    """One typed local-model registration tracked by the router and dashboard."""

    registration_id: str
    role: ModelRole
    backend: str
    model_identifier: str
    resource_class: ModelResourceClass = ModelResourceClass.HEAVY
    enabled: bool = True
    preferred_device: str = "auto"
    load_policy: ModelLoadPolicy = ModelLoadPolicy.ON_DEMAND
    supported_capabilities: tuple[str, ...] = ()
    missing_dependencies: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(bool(self.registration_id.strip()), "ModelRegistration.registration_id must not be empty.")
        _require(bool(self.backend.strip()), "ModelRegistration.backend must not be empty.")
        _require(bool(self.model_identifier.strip()), "ModelRegistration.model_identifier must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelRegistration:
        return cls(
            registration_id=str(data["registration_id"]),
            role=_parse_enum(ModelRole, data.get("role", ModelRole.GENERATION)),
            backend=str(data.get("backend", "")),
            model_identifier=str(data.get("model_identifier", "")),
            resource_class=_parse_enum(
                ModelResourceClass,
                data.get("resource_class", ModelResourceClass.HEAVY),
            ),
            enabled=bool(data.get("enabled", True)),
            preferred_device=str(data.get("preferred_device", "auto")),
            load_policy=_parse_enum(
                ModelLoadPolicy,
                data.get("load_policy", ModelLoadPolicy.ON_DEMAND),
            ),
            supported_capabilities=tuple(str(item) for item in data.get("supported_capabilities", ())),
            missing_dependencies=tuple(str(item) for item in data.get("missing_dependencies", ())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True, frozen=True)
class ModelRouteDecision(DictSerializable):
    """Typed routing decision produced for one requested model role."""

    requested_role: ModelRole
    selected_registration_id: str = ""
    selected_backend: str = ""
    selected_model_identifier: str = ""
    resource_class: ModelResourceClass = ModelResourceClass.HEAVY
    capability: str = ""
    allowed: bool = False
    used_fallback: bool = False
    fallback_reason: str = ""
    active_heavy_roles: tuple[str, ...] = ()
    heavy_slot_limit: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelRouteDecision:
        return cls(
            requested_role=_parse_enum(ModelRole, data.get("requested_role", ModelRole.GENERATION)),
            selected_registration_id=str(data.get("selected_registration_id", "")),
            selected_backend=str(data.get("selected_backend", "")),
            selected_model_identifier=str(data.get("selected_model_identifier", "")),
            resource_class=_parse_enum(
                ModelResourceClass,
                data.get("resource_class", ModelResourceClass.HEAVY),
            ),
            capability=str(data.get("capability", "")),
            allowed=bool(data.get("allowed", False)),
            used_fallback=bool(data.get("used_fallback", False)),
            fallback_reason=str(data.get("fallback_reason", "")),
            active_heavy_roles=tuple(str(item) for item in data.get("active_heavy_roles", ())),
            heavy_slot_limit=int(data.get("heavy_slot_limit", 2)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True, frozen=True)
class BoundedCacheSnapshot(DictSerializable):
    """Read-only summary of one bounded cache namespace."""

    namespace: str
    max_entries: int
    entry_count: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    warm_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require(bool(self.namespace.strip()), "BoundedCacheSnapshot.namespace must not be empty.")
        _require(self.max_entries > 0, "BoundedCacheSnapshot.max_entries must be positive.")
        _require(self.entry_count >= 0, "BoundedCacheSnapshot.entry_count must be zero or positive.")
        _require(self.hits >= 0, "BoundedCacheSnapshot.hits must be zero or positive.")
        _require(self.misses >= 0, "BoundedCacheSnapshot.misses must be zero or positive.")
        _require(self.evictions >= 0, "BoundedCacheSnapshot.evictions must be zero or positive.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoundedCacheSnapshot:
        return cls(
            namespace=str(data["namespace"]),
            max_entries=int(data.get("max_entries", 1)),
            entry_count=int(data.get("entry_count", 0)),
            hits=int(data.get("hits", 0)),
            misses=int(data.get("misses", 0)),
            evictions=int(data.get("evictions", 0)),
            warm_keys=tuple(str(item) for item in data.get("warm_keys", ())),
        )


@dataclass(slots=True, frozen=True)
class ModelRegistryView(DictSerializable):
    """Persisted registry/control-plane projection for all local AI roles."""

    registrations: tuple[ModelRegistration, ...] = ()
    preferred_models: dict[str, str] = field(default_factory=dict)
    active_heavy_roles: tuple[str, ...] = ()
    heavy_slot_limit: int = 2
    fallback_reasons: dict[str, str] = field(default_factory=dict)
    governor_active: bool = False
    governor_pressure_reasons: tuple[str, ...] = ()
    governor_degraded_features: tuple[str, ...] = ()
    governor_summary: str = ""
    last_route_decisions: tuple[ModelRouteDecision, ...] = ()
    advisory_available: bool = False
    optimizer_subscriptions: tuple[str, ...] = ()
    recent_optimizer_suggestions: tuple[OptimizerSuggestionRecord, ...] = ()
    cache_snapshots: tuple[BoundedCacheSnapshot, ...] = ()
    compression_insights: tuple["CompressionInsightSummary", ...] = ()
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(self.heavy_slot_limit > 0, "ModelRegistryView.heavy_slot_limit must be positive.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelRegistryView:
        return cls(
            registrations=tuple(
                ModelRegistration.from_dict(item) for item in data.get("registrations", ())
            ),
            preferred_models={
                str(key): str(value)
                for key, value in dict(data.get("preferred_models", {})).items()
            },
            active_heavy_roles=tuple(str(item) for item in data.get("active_heavy_roles", ())),
            heavy_slot_limit=int(data.get("heavy_slot_limit", 2)),
            fallback_reasons={
                str(key): str(value)
                for key, value in dict(data.get("fallback_reasons", {})).items()
            },
            governor_active=bool(data.get("governor_active", False)),
            governor_pressure_reasons=tuple(
                str(item) for item in data.get("governor_pressure_reasons", ())
            ),
            governor_degraded_features=tuple(
                str(item) for item in data.get("governor_degraded_features", ())
            ),
            governor_summary=str(data.get("governor_summary", "")),
            last_route_decisions=tuple(
                ModelRouteDecision.from_dict(item) for item in data.get("last_route_decisions", ())
            ),
            advisory_available=bool(data.get("advisory_available", False)),
            optimizer_subscriptions=tuple(str(item) for item in data.get("optimizer_subscriptions", ())),
            recent_optimizer_suggestions=tuple(
                OptimizerSuggestionRecord.from_dict(item)
                for item in data.get("recent_optimizer_suggestions", ())
            ),
            cache_snapshots=tuple(
                BoundedCacheSnapshot.from_dict(item) for item in data.get("cache_snapshots", ())
            ),
            compression_insights=tuple(
                CompressionInsightSummary.from_dict(item)
                for item in data.get("compression_insights", ())
            ),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class ModelRoleActionReport(DictSerializable):
    """Typed result of one local-AI control-plane quick action."""

    role: ModelRole = ModelRole.GENERATION
    action: str = ""
    ok: bool = False
    summary: str = ""
    detail: str = ""
    route_decision: ModelRouteDecision | None = None
    guidance: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    generated_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelRoleActionReport:
        raw_route_decision = data.get("route_decision")
        return cls(
            role=_parse_enum(ModelRole, data.get("role", ModelRole.GENERATION)),
            action=str(data.get("action", "")),
            ok=bool(data.get("ok", False)),
            summary=str(data.get("summary", "")),
            detail=str(data.get("detail", "")),
            route_decision=(
                ModelRouteDecision.from_dict(raw_route_decision)
                if isinstance(raw_route_decision, Mapping)
                else None
            ),
            guidance=tuple(str(item) for item in data.get("guidance", ())),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
            generated_at=_parse_datetime(data.get("generated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CompressionInsightSummary(DictSerializable):
    """Typed dashboard-safe summary of one bounded compressor proposal prior."""

    proposal_id: str
    macro_name: str = ""
    proof_fingerprint: str = ""
    estimated_gain: float = 0.0
    validation_pass_rate: float = 0.0
    validation_state: str = "unknown"
    blocked_reason: str = ""
    evidence_basis: str = ""
    origin_component: str = ""
    accepted: bool = False

    def __post_init__(self) -> None:
        _require(bool(self.proposal_id.strip()), "CompressionInsightSummary.proposal_id must not be empty.")
        _require(0.0 <= self.estimated_gain <= 1.0, "CompressionInsightSummary.estimated_gain must be between 0 and 1.")
        _require(
            0.0 <= self.validation_pass_rate <= 1.0,
            "CompressionInsightSummary.validation_pass_rate must be between 0 and 1.",
        )
        _require(bool(self.validation_state.strip()), "CompressionInsightSummary.validation_state must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompressionInsightSummary:
        return cls(
            proposal_id=str(data["proposal_id"]),
            macro_name=str(data.get("macro_name", "")),
            proof_fingerprint=str(data.get("proof_fingerprint", "")),
            estimated_gain=float(data.get("estimated_gain", 0.0) or 0.0),
            validation_pass_rate=float(data.get("validation_pass_rate", 0.0) or 0.0),
            validation_state=str(data.get("validation_state", "unknown")),
            blocked_reason=str(data.get("blocked_reason", "")),
            evidence_basis=str(data.get("evidence_basis", "")),
            origin_component=str(data.get("origin_component", "")),
            accepted=bool(data.get("accepted", False)),
        )


@dataclass(slots=True, frozen=True)
class LongHorizonExportBundle(DictSerializable):
    """Machine-readable export bundle for one checkpointed long-horizon session."""

    session_id: str
    session_path: str
    checkpoints_path: str
    verified_trace_export_path: str = ""
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.session_id.strip()), "LongHorizonExportBundle.session_id must not be empty.")
        _require(bool(self.session_path.strip()), "LongHorizonExportBundle.session_path must not be empty.")
        _require(bool(self.checkpoints_path.strip()), "LongHorizonExportBundle.checkpoints_path must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LongHorizonExportBundle:
        return cls(
            session_id=str(data["session_id"]),
            session_path=str(data.get("session_path", "")),
            checkpoints_path=str(data.get("checkpoints_path", "")),
            verified_trace_export_path=str(data.get("verified_trace_export_path", "")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class DashboardRuntimeHealth(DictSerializable):
    """Latest runtime-health projection shown in the local app."""

    started: bool = False
    generation_backend: str = ""
    embedding_backend: str = ""
    active_generation_jobs: int = 0
    active_embedding_jobs: int = 0
    active_heavy_roles: tuple[str, ...] = ()
    heavy_slot_limit: int = 2
    last_used_at: str = ""
    fallback_active: bool = False
    fallback_reason: str = ""
    available_ram_gb: float | None = None
    total_ram_gb: float | None = None
    generation_backend_vram_gb: float | None = None
    embedding_backend_vram_gb: float | None = None
    governor_active: bool = False
    governor_pressure_reasons: tuple[str, ...] = ()
    governor_degraded_features: tuple[str, ...] = ()
    queue_pressure: bool = False
    backend_health_degraded: bool = False
    allow_continuous_capture: bool = True
    allow_ocr_on_step: bool = True
    allow_vision_on_step: bool = True
    allow_optional_heavy_residency: bool = True
    allow_background_work: bool = True
    governor_summary: str = ""
    telemetry_enabled: bool = False
    last_error: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DashboardRuntimeHealth:
        return cls(
            started=bool(data.get("started", False)),
            generation_backend=str(data.get("generation_backend", "")),
            embedding_backend=str(data.get("embedding_backend", "")),
            active_generation_jobs=int(data.get("active_generation_jobs", 0)),
            active_embedding_jobs=int(data.get("active_embedding_jobs", 0)),
            active_heavy_roles=tuple(str(item) for item in data.get("active_heavy_roles", ())),
            heavy_slot_limit=int(data.get("heavy_slot_limit", 2)),
            last_used_at=str(data.get("last_used_at", "")),
            fallback_active=bool(data.get("fallback_active", False)),
            fallback_reason=str(data.get("fallback_reason", "")),
            available_ram_gb=(
                float(data["available_ram_gb"]) if data.get("available_ram_gb") is not None else None
            ),
            total_ram_gb=float(data["total_ram_gb"]) if data.get("total_ram_gb") is not None else None,
            generation_backend_vram_gb=(
                float(data["generation_backend_vram_gb"])
                if data.get("generation_backend_vram_gb") is not None
                else None
            ),
            embedding_backend_vram_gb=(
                float(data["embedding_backend_vram_gb"])
                if data.get("embedding_backend_vram_gb") is not None
                else None
            ),
            governor_active=bool(data.get("governor_active", False)),
            governor_pressure_reasons=tuple(
                str(item) for item in data.get("governor_pressure_reasons", ())
            ),
            governor_degraded_features=tuple(
                str(item) for item in data.get("governor_degraded_features", ())
            ),
            queue_pressure=bool(data.get("queue_pressure", False)),
            backend_health_degraded=bool(data.get("backend_health_degraded", False)),
            allow_continuous_capture=bool(data.get("allow_continuous_capture", True)),
            allow_ocr_on_step=bool(data.get("allow_ocr_on_step", True)),
            allow_vision_on_step=bool(data.get("allow_vision_on_step", True)),
            allow_optional_heavy_residency=bool(data.get("allow_optional_heavy_residency", True)),
            allow_background_work=bool(data.get("allow_background_work", True)),
            governor_summary=str(data.get("governor_summary", "")),
            telemetry_enabled=bool(data.get("telemetry_enabled", False)),
            last_error=str(data.get("last_error", "")),
        )


@dataclass(slots=True, frozen=True)
class DashboardTaskState(DictSerializable):
    """Latest task-focused projection shown in the local app."""

    task_id: str = ""
    question: str = ""
    thinking_minutes: int = 0
    requested_thinking_minutes: int = 0
    execution_mode: str = "interactive"
    running_stage: str = ""
    answer_text: str = ""
    citation_refs: tuple[str, ...] = ()
    candidate_trace_count: int = 0
    selected_candidate_id: str = ""
    selected_strategy: str = ""
    selected_verifier: str = ""
    candidate_score: float = 0.0
    critique_result: str = ""
    degraded_reason: str = ""
    repair_actions: tuple[str, ...] = ()
    failure_categories: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    local_result_count: int = 0
    web_result_count: int = 0
    used_web_fallback: bool = False
    web_query: str = ""
    web_source_refs: tuple[str, ...] = ()
    long_horizon_session_id: str = ""
    long_horizon_status: str = ""
    long_horizon_current_phase: str = ""
    long_horizon_cycle_budget_minutes: int = 0
    long_horizon_checkpoint_interval_minutes: int = 0
    long_horizon_duty_cycle_ratio: float = 0.0
    long_horizon_cooldown_seconds: float = 0.0
    long_horizon_elapsed_seconds: float = 0.0
    long_horizon_eta_seconds: float | None = None
    long_horizon_completed_cycles: int = 0
    long_horizon_total_cycles: int = 0
    long_horizon_resume_count: int = 0
    long_horizon_pause_requested: bool = False
    long_horizon_cancel_requested: bool = False
    long_horizon_throttled: bool = False
    long_horizon_throttle_reason: str = ""
    long_horizon_initial_candidate_count: int = 0
    long_horizon_peak_candidate_count: int = 0
    long_horizon_additional_candidate_count: int = 0
    long_horizon_initial_supporting_evidence_count: int = 0
    long_horizon_additional_supporting_evidence_count: int = 0
    long_horizon_total_verification_passes: int = 0
    long_horizon_total_repairs: int = 0
    long_horizon_first_candidate_score: float = 0.0
    long_horizon_confidence_gain: float = 0.0
    long_horizon_first_critique_result: str = ""
    long_horizon_validity_improved: bool = False
    long_horizon_advisory_requested_count: int = 0
    long_horizon_advisory_accepted_count: int = 0
    long_horizon_advisory_rejected_count: int = 0
    long_horizon_advisory_deferred_count: int = 0
    long_horizon_advisory_entries: tuple[str, ...] = ()
    long_horizon_early_stop_reason: str = ""
    specialist_roles_used: tuple[str, ...] = ()
    specialist_role_explanations: tuple[str, ...] = ()
    advisor_summaries: tuple[str, ...] = ()
    warning_count: int = 0
    updated_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DashboardTaskState:
        raw_eta_seconds = data.get("long_horizon_eta_seconds")
        return cls(
            task_id=str(data.get("task_id", "")),
            question=str(data.get("question", "")),
            thinking_minutes=int(data.get("thinking_minutes", 0)),
            requested_thinking_minutes=int(data.get("requested_thinking_minutes", 0)),
            execution_mode=str(data.get("execution_mode", "interactive")),
            running_stage=str(data.get("running_stage", "")),
            answer_text=str(data.get("answer_text", "")),
            citation_refs=tuple(str(item) for item in data.get("citation_refs", ())),
            candidate_trace_count=int(data.get("candidate_trace_count", 0)),
            selected_candidate_id=str(data.get("selected_candidate_id", "")),
            selected_strategy=str(data.get("selected_strategy", "")),
            selected_verifier=str(data.get("selected_verifier", "")),
            candidate_score=float(data.get("candidate_score", 0.0) or 0.0),
            critique_result=str(data.get("critique_result", "")),
            degraded_reason=str(data.get("degraded_reason", "")),
            repair_actions=tuple(str(item) for item in data.get("repair_actions", ())),
            failure_categories=tuple(str(item) for item in data.get("failure_categories", ())),
            supporting_evidence_ids=tuple(str(item) for item in data.get("supporting_evidence_ids", ())),
            local_result_count=int(data.get("local_result_count", 0)),
            web_result_count=int(data.get("web_result_count", 0)),
            used_web_fallback=bool(data.get("used_web_fallback", False)),
            web_query=str(data.get("web_query", "")),
            web_source_refs=tuple(str(item) for item in data.get("web_source_refs", ())),
            long_horizon_session_id=str(data.get("long_horizon_session_id", "")),
            long_horizon_status=str(data.get("long_horizon_status", "")),
            long_horizon_current_phase=str(data.get("long_horizon_current_phase", "")),
            long_horizon_cycle_budget_minutes=int(data.get("long_horizon_cycle_budget_minutes", 0)),
            long_horizon_checkpoint_interval_minutes=int(data.get("long_horizon_checkpoint_interval_minutes", 0)),
            long_horizon_duty_cycle_ratio=float(data.get("long_horizon_duty_cycle_ratio", 0.0) or 0.0),
            long_horizon_cooldown_seconds=float(data.get("long_horizon_cooldown_seconds", 0.0) or 0.0),
            long_horizon_elapsed_seconds=float(data.get("long_horizon_elapsed_seconds", 0.0) or 0.0),
            long_horizon_eta_seconds=(
                None
                if raw_eta_seconds in (None, "")
                else float(raw_eta_seconds)
            ),
            long_horizon_completed_cycles=int(data.get("long_horizon_completed_cycles", 0)),
            long_horizon_total_cycles=int(data.get("long_horizon_total_cycles", 0)),
            long_horizon_resume_count=int(data.get("long_horizon_resume_count", 0)),
            long_horizon_pause_requested=bool(data.get("long_horizon_pause_requested", False)),
            long_horizon_cancel_requested=bool(data.get("long_horizon_cancel_requested", False)),
            long_horizon_throttled=bool(data.get("long_horizon_throttled", False)),
            long_horizon_throttle_reason=str(data.get("long_horizon_throttle_reason", "")),
            long_horizon_initial_candidate_count=int(data.get("long_horizon_initial_candidate_count", 0)),
            long_horizon_peak_candidate_count=int(data.get("long_horizon_peak_candidate_count", 0)),
            long_horizon_additional_candidate_count=int(data.get("long_horizon_additional_candidate_count", 0)),
            long_horizon_initial_supporting_evidence_count=int(
                data.get("long_horizon_initial_supporting_evidence_count", 0)
            ),
            long_horizon_additional_supporting_evidence_count=int(
                data.get("long_horizon_additional_supporting_evidence_count", 0)
            ),
            long_horizon_total_verification_passes=int(data.get("long_horizon_total_verification_passes", 0)),
            long_horizon_total_repairs=int(data.get("long_horizon_total_repairs", 0)),
            long_horizon_first_candidate_score=float(data.get("long_horizon_first_candidate_score", 0.0) or 0.0),
            long_horizon_confidence_gain=float(data.get("long_horizon_confidence_gain", 0.0) or 0.0),
            long_horizon_first_critique_result=str(data.get("long_horizon_first_critique_result", "")),
            long_horizon_validity_improved=bool(data.get("long_horizon_validity_improved", False)),
            long_horizon_advisory_requested_count=int(data.get("long_horizon_advisory_requested_count", 0)),
            long_horizon_advisory_accepted_count=int(data.get("long_horizon_advisory_accepted_count", 0)),
            long_horizon_advisory_rejected_count=int(data.get("long_horizon_advisory_rejected_count", 0)),
            long_horizon_advisory_deferred_count=int(data.get("long_horizon_advisory_deferred_count", 0)),
            long_horizon_advisory_entries=tuple(
                str(item) for item in data.get("long_horizon_advisory_entries", ())
            ),
            long_horizon_early_stop_reason=str(data.get("long_horizon_early_stop_reason", "")),
            specialist_roles_used=tuple(str(item) for item in data.get("specialist_roles_used", ())),
            specialist_role_explanations=tuple(
                str(item) for item in data.get("specialist_role_explanations", ())
            ),
            advisor_summaries=tuple(str(item) for item in data.get("advisor_summaries", ())),
            warning_count=int(data.get("warning_count", 0)),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class FileOperationSpec(DictSerializable):
    """Typed file-operation request payload used by the capability policy layer."""

    operation: str
    source_path: str
    destination_path: str = ""
    recursive: bool = False

    def __post_init__(self) -> None:
        _require(bool(self.operation.strip()), "FileOperationSpec.operation must not be empty.")
        _require(bool(self.source_path.strip()), "FileOperationSpec.source_path must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FileOperationSpec:
        return cls(
            operation=str(data.get("operation", "")),
            source_path=str(data.get("source_path", "")),
            destination_path=str(data.get("destination_path", "")),
            recursive=bool(data.get("recursive", False)),
        )


@dataclass(slots=True, frozen=True)
class ShellCommandSpec(DictSerializable):
    """Typed shell-command request payload used by the capability policy layer."""

    command: str
    args: tuple[str, ...] = ()
    working_directory: str = ""

    def __post_init__(self) -> None:
        _require(bool(self.command.strip()), "ShellCommandSpec.command must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShellCommandSpec:
        return cls(
            command=str(data.get("command", "")),
            args=tuple(str(item) for item in data.get("args", ())),
            working_directory=str(data.get("working_directory", "")),
        )


@dataclass(slots=True, frozen=True)
class BrowserActionSpec(DictSerializable):
    """Typed browser-action request payload used by the capability policy layer."""

    action: str
    url: str = ""
    domain: str = ""
    selector: str = ""
    text: str = ""

    def __post_init__(self) -> None:
        _require(bool(self.action.strip()), "BrowserActionSpec.action must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BrowserActionSpec:
        return cls(
            action=str(data.get("action", "")),
            url=str(data.get("url", "")),
            domain=str(data.get("domain", "")),
            selector=str(data.get("selector", "")),
            text=str(data.get("text", "")),
        )


@dataclass(slots=True, frozen=True)
class AppFocusSpec(DictSerializable):
    """Typed app or window focus request payload used by the capability policy layer."""

    app_name: str
    window_title: str = ""
    require_visible_match: bool = True

    def __post_init__(self) -> None:
        _require(bool(self.app_name.strip()), "AppFocusSpec.app_name must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AppFocusSpec:
        return cls(
            app_name=str(data.get("app_name", "")),
            window_title=str(data.get("window_title", "")),
            require_visible_match=bool(data.get("require_visible_match", True)),
        )


@dataclass(slots=True, frozen=True)
class ClipboardActionSpec(DictSerializable):
    """Typed clipboard-action request payload used by the capability policy layer."""

    action: str
    text: str = ""

    def __post_init__(self) -> None:
        _require(bool(self.action.strip()), "ClipboardActionSpec.action must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClipboardActionSpec:
        return cls(
            action=str(data.get("action", "")),
            text=str(data.get("text", "")),
        )


@dataclass(slots=True, frozen=True)
class ScreenshotSpec(DictSerializable):
    """Typed screenshot request payload used by the capability policy layer."""

    save_path: str
    region: str = "full_screen"

    def __post_init__(self) -> None:
        _require(bool(self.save_path.strip()), "ScreenshotSpec.save_path must not be empty.")
        _require(bool(self.region.strip()), "ScreenshotSpec.region must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScreenshotSpec:
        return cls(
            save_path=str(data.get("save_path", "")),
            region=str(data.get("region", "full_screen")),
        )


@dataclass(slots=True, frozen=True)
class OCRRequestSpec(DictSerializable):
    """Typed OCR request payload used by the capability policy layer."""

    source_image_path: str
    languages: tuple[str, ...] = ()
    region: str = "full_image"

    def __post_init__(self) -> None:
        _require(bool(self.source_image_path.strip()), "OCRRequestSpec.source_image_path must not be empty.")
        _require(bool(self.region.strip()), "OCRRequestSpec.region must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OCRRequestSpec:
        return cls(
            source_image_path=str(data.get("source_image_path", "")),
            languages=tuple(str(item) for item in data.get("languages", ())),
            region=str(data.get("region", "full_image")),
        )


@dataclass(slots=True, frozen=True)
class DesktopInputSpec(DictSerializable):
    """Typed desktop-input request payload used by the capability policy layer."""

    action: str
    text: str = ""
    keys: tuple[str, ...] = ()
    target: str = ""
    x: int | None = None
    y: int | None = None

    def __post_init__(self) -> None:
        _require(bool(self.action.strip()), "DesktopInputSpec.action must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DesktopInputSpec:
        raw_x = data.get("x")
        raw_y = data.get("y")
        return cls(
            action=str(data.get("action", "")),
            text=str(data.get("text", "")),
            keys=tuple(str(item) for item in data.get("keys", ())),
            target=str(data.get("target", "")),
            x=None if raw_x in (None, "") else int(raw_x),
            y=None if raw_y in (None, "") else int(raw_y),
        )


class CloudOffloadMode(str, Enum):
    """Allowed cloud-dispatch modes before provider-specific adapters exist."""

    DISABLED = "disabled"
    AUXILIARY_ONLY = "auxiliary_only"


class CloudOffloadCapability(str, Enum):
    """Per-capability cloud helper categories kept independent in the user-facing control surface."""

    OFFLINE_REPLAY = "offline_replay"
    EXPORT = "export"
    BROWSER_HELPER = "browser_helper"
    OCR_HELPER = "ocr_helper"
    VISION_HELPER = "vision_helper"
    EMBEDDING_HELPER = "embedding_helper"
    BACKGROUND_MAINTENANCE = "background_maintenance"


class CloudJobPayloadClass(str, Enum):
    """Provider-agnostic payload categories that future cloud adapters must honor."""

    METADATA_ONLY = "metadata_only"
    TEXT_SNIPPET = "text_snippet"
    IMAGE_REGION = "image_region"
    DOCUMENT_CHUNK = "document_chunk"
    EMBEDDING_BATCH = "embedding_batch"
    EXPORT_BUNDLE = "export_bundle"


class CloudJobPrivacyClass(str, Enum):
    """Privacy boundary used before any cloud helper may receive task data."""

    METADATA_ONLY = "metadata_only"
    APPROVED_CONTENT = "approved_content"
    DENIED_CONTENT = "denied_content"


class CloudFallbackBehavior(str, Enum):
    """Fallback behaviors allowed by the Phase 23 guardrails."""

    LOCAL_ONLY = "local_only"
    RETRY_THEN_LOCAL = "retry_then_local"


@dataclass(slots=True, frozen=True)
class CloudJobContract(DictSerializable):
    """Provider-agnostic cloud offload contract used before any concrete provider adapter is bound."""

    job_id: str
    capability: CloudOffloadCapability
    payload_class: CloudJobPayloadClass
    privacy_class: CloudJobPrivacyClass
    max_payload_bytes: int
    max_retries: int = 1
    fallback_behavior: CloudFallbackBehavior = CloudFallbackBehavior.RETRY_THEN_LOCAL
    dispatch_mode: CloudOffloadMode = CloudOffloadMode.AUXILIARY_ONLY
    provider_family: str = "provider_agnostic"
    content_approved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.job_id.strip()), "CloudJobContract.job_id must not be empty.")
        _require(bool(self.provider_family.strip()), "CloudJobContract.provider_family must not be empty.")
        _require(
            self.dispatch_mode == CloudOffloadMode.AUXILIARY_ONLY,
            "CloudJobContract.dispatch_mode must remain auxiliary_only in Phase 23.0.x.",
        )
        _require(
            1024 <= self.max_payload_bytes <= 10 * 1024 * 1024,
            "CloudJobContract.max_payload_bytes must stay between 1024 and 10485760.",
        )
        _require(0 <= self.max_retries <= 3, "CloudJobContract.max_retries must stay between 0 and 3.")
        if self.privacy_class == CloudJobPrivacyClass.APPROVED_CONTENT:
            _require(
                self.content_approved,
                "CloudJobContract.content_approved must be true for approved_content jobs.",
            )
        else:
            _require(
                not self.content_approved,
                "CloudJobContract.content_approved must remain false unless privacy_class is approved_content.",
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CloudJobContract:
        return cls(
            job_id=str(data.get("job_id", "")),
            capability=_parse_enum(
                CloudOffloadCapability,
                data.get("capability", CloudOffloadCapability.BACKGROUND_MAINTENANCE),
            ),
            payload_class=_parse_enum(
                CloudJobPayloadClass,
                data.get("payload_class", CloudJobPayloadClass.METADATA_ONLY),
            ),
            privacy_class=_parse_enum(
                CloudJobPrivacyClass,
                data.get("privacy_class", CloudJobPrivacyClass.METADATA_ONLY),
            ),
            max_payload_bytes=int(data.get("max_payload_bytes", 1024 * 256) or 1024 * 256),
            max_retries=int(data.get("max_retries", 1) or 0),
            fallback_behavior=_parse_enum(
                CloudFallbackBehavior,
                data.get("fallback_behavior", CloudFallbackBehavior.RETRY_THEN_LOCAL),
            ),
            dispatch_mode=_parse_enum(
                CloudOffloadMode,
                data.get("dispatch_mode", CloudOffloadMode.AUXILIARY_ONLY),
            ),
            provider_family=str(data.get("provider_family", "provider_agnostic")),
            content_approved=bool(data.get("content_approved", False)),
            metadata=dict(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


class CloudOffloadOutcome(str, Enum):
    """Final outcome recorded for one auxiliary cloud dispatch attempt."""

    BLOCKED = "blocked"
    SUCCEEDED = "succeeded"
    LOCAL_FALLBACK = "local_fallback"
    FAILED = "failed"


@dataclass(slots=True, frozen=True)
class CloudOffloadRecord(DictSerializable):
    """Append-only audit record for one auxiliary cloud helper dispatch."""

    dispatch_id: str
    job_id: str
    capability: CloudOffloadCapability
    provider_name: str
    provider_family: str
    payload_class: CloudJobPayloadClass
    privacy_class: CloudJobPrivacyClass
    outcome: CloudOffloadOutcome
    summary: str
    detail: str = ""
    fallback_behavior: CloudFallbackBehavior = CloudFallbackBehavior.RETRY_THEN_LOCAL
    bytes_sent: int = 0
    latency_ms: int = 0
    retry_count: int = 0
    local_fallback_used: bool = False
    fallback_reason: str = ""
    response_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.dispatch_id.strip()), "CloudOffloadRecord.dispatch_id must not be empty.")
        _require(bool(self.job_id.strip()), "CloudOffloadRecord.job_id must not be empty.")
        _require(bool(self.provider_name.strip()), "CloudOffloadRecord.provider_name must not be empty.")
        _require(bool(self.provider_family.strip()), "CloudOffloadRecord.provider_family must not be empty.")
        _require(bool(self.summary.strip()), "CloudOffloadRecord.summary must not be empty.")
        _require(self.bytes_sent >= 0, "CloudOffloadRecord.bytes_sent must be >= 0.")
        _require(self.latency_ms >= 0, "CloudOffloadRecord.latency_ms must be >= 0.")
        _require(self.retry_count >= 0, "CloudOffloadRecord.retry_count must be >= 0.")
        if self.local_fallback_used:
            _require(
                self.outcome == CloudOffloadOutcome.LOCAL_FALLBACK,
                "CloudOffloadRecord.local_fallback_used requires outcome=local_fallback.",
            )
        if self.outcome == CloudOffloadOutcome.LOCAL_FALLBACK:
            _require(
                self.local_fallback_used,
                "CloudOffloadRecord.outcome=local_fallback requires local_fallback_used=true.",
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CloudOffloadRecord:
        return cls(
            dispatch_id=str(data.get("dispatch_id", "")),
            job_id=str(data.get("job_id", "")),
            capability=_parse_enum(
                CloudOffloadCapability,
                data.get("capability", CloudOffloadCapability.BACKGROUND_MAINTENANCE),
            ),
            provider_name=str(data.get("provider_name", "")),
            provider_family=str(data.get("provider_family", "provider_agnostic")),
            payload_class=_parse_enum(
                CloudJobPayloadClass,
                data.get("payload_class", CloudJobPayloadClass.METADATA_ONLY),
            ),
            privacy_class=_parse_enum(
                CloudJobPrivacyClass,
                data.get("privacy_class", CloudJobPrivacyClass.METADATA_ONLY),
            ),
            outcome=_parse_enum(
                CloudOffloadOutcome,
                data.get("outcome", CloudOffloadOutcome.FAILED),
            ),
            summary=str(data.get("summary", "")),
            detail=str(data.get("detail", "")),
            fallback_behavior=_parse_enum(
                CloudFallbackBehavior,
                data.get("fallback_behavior", CloudFallbackBehavior.RETRY_THEN_LOCAL),
            ),
            bytes_sent=int(data.get("bytes_sent", 0) or 0),
            latency_ms=int(data.get("latency_ms", 0) or 0),
            retry_count=int(data.get("retry_count", 0) or 0),
            local_fallback_used=bool(data.get("local_fallback_used", False)),
            fallback_reason=str(data.get("fallback_reason", "")),
            response_ref=str(data.get("response_ref", "")),
            metadata=dict(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


class CapabilityType(str, Enum):
    """Typed capability families used by the local task-control policy layer."""

    FILE_OPERATION = "file_operation"
    SHELL_COMMAND = "shell_command"
    BROWSER_ACTION = "browser_action"
    APP_WINDOW_FOCUS = "app_window_focus"
    CLIPBOARD_ACTION = "clipboard_action"
    SCREENSHOT = "screenshot"
    OCR_REQUEST = "ocr_request"
    DESKTOP_INPUT = "desktop_input"


class CapabilityAvailabilityStatus(str, Enum):
    """Availability state tracked for one capability in the control plane."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DENIED_BY_POLICY = "denied_by_policy"
    DEGRADED = "degraded"
    REQUIRES_APPROVAL = "requires_approval"


class CapabilityPolicyOutcome(str, Enum):
    """Policy-decision outcome returned before any capability executor runs."""

    ALLOWED = "allowed"
    REQUIRES_APPROVAL = "requires_approval"
    DENIED = "denied"
    DEGRADED = "degraded"


class CapabilityAuditEventType(str, Enum):
    """Audit lifecycle events persisted for capability requests."""

    REQUESTED = "requested"
    POLICY_DECISION = "policy_decision"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    EXECUTOR_RESULT = "executor_result"
    WARNING = "warning"


class CapabilityExecutionStatus(str, Enum):
    """Execution result status for stub or future live capability executors."""

    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(slots=True, frozen=True)
class CapabilityRequest(DictSerializable):
    """Typed capability request evaluated by policy before any executor runs."""

    request_id: str
    capability_type: CapabilityType
    summary: str = ""
    file_operation: FileOperationSpec | None = None
    shell_command: ShellCommandSpec | None = None
    browser_action: BrowserActionSpec | None = None
    app_focus: AppFocusSpec | None = None
    clipboard_action: ClipboardActionSpec | None = None
    screenshot: ScreenshotSpec | None = None
    ocr_request: OCRRequestSpec | None = None
    desktop_input: DesktopInputSpec | None = None
    requires_elevation: bool = False
    persistent_background: bool = False
    hidden_execution: bool = False
    touches_credentials: bool = False
    unrestricted_scope: bool = False
    destructive: bool = False
    cross_app: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    requested_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.request_id.strip()), "CapabilityRequest.request_id must not be empty.")
        selected_specs = tuple(
            name
            for name, value in (
                ("file_operation", self.file_operation),
                ("shell_command", self.shell_command),
                ("browser_action", self.browser_action),
                ("app_focus", self.app_focus),
                ("clipboard_action", self.clipboard_action),
                ("screenshot", self.screenshot),
                ("ocr_request", self.ocr_request),
                ("desktop_input", self.desktop_input),
            )
            if value is not None
        )
        _require(len(selected_specs) == 1, "CapabilityRequest must define exactly one typed capability payload.")
        expected_spec_name = {
            CapabilityType.FILE_OPERATION: "file_operation",
            CapabilityType.SHELL_COMMAND: "shell_command",
            CapabilityType.BROWSER_ACTION: "browser_action",
            CapabilityType.APP_WINDOW_FOCUS: "app_focus",
            CapabilityType.CLIPBOARD_ACTION: "clipboard_action",
            CapabilityType.SCREENSHOT: "screenshot",
            CapabilityType.OCR_REQUEST: "ocr_request",
            CapabilityType.DESKTOP_INPUT: "desktop_input",
        }[self.capability_type]
        _require(
            selected_specs[0] == expected_spec_name,
            "CapabilityRequest payload must match capability_type.",
        )

    def action_name(self) -> str:
        """Return the canonical action or operation name for audit and policy summaries."""
        if self.file_operation is not None:
            return self.file_operation.operation
        if self.shell_command is not None:
            return self.shell_command.command
        if self.browser_action is not None:
            return self.browser_action.action
        if self.app_focus is not None:
            return "focus_window"
        if self.clipboard_action is not None:
            return self.clipboard_action.action
        if self.screenshot is not None:
            return "capture_screenshot"
        if self.ocr_request is not None:
            return "ocr_image"
        if self.desktop_input is not None:
            return self.desktop_input.action
        return ""

    def target_summary(self) -> str:
        """Return the primary target reference for audit and policy summaries."""
        if self.file_operation is not None:
            return self.file_operation.destination_path or self.file_operation.source_path
        if self.shell_command is not None:
            return self.shell_command.working_directory
        if self.browser_action is not None:
            return self.browser_action.url or self.browser_action.domain
        if self.app_focus is not None:
            return self.app_focus.window_title or self.app_focus.app_name
        if self.clipboard_action is not None:
            return "clipboard"
        if self.screenshot is not None:
            return self.screenshot.save_path
        if self.ocr_request is not None:
            return self.ocr_request.source_image_path
        if self.desktop_input is not None:
            return self.desktop_input.target
        return ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapabilityRequest:
        raw_file_operation = data.get("file_operation")
        raw_shell_command = data.get("shell_command")
        raw_browser_action = data.get("browser_action")
        raw_app_focus = data.get("app_focus")
        raw_clipboard_action = data.get("clipboard_action")
        raw_screenshot = data.get("screenshot")
        raw_ocr_request = data.get("ocr_request")
        raw_desktop_input = data.get("desktop_input")
        return cls(
            request_id=str(data.get("request_id", "")),
            capability_type=_parse_enum(CapabilityType, data.get("capability_type", CapabilityType.FILE_OPERATION)),
            summary=str(data.get("summary", "")),
            file_operation=(
                FileOperationSpec.from_dict(raw_file_operation)
                if isinstance(raw_file_operation, Mapping)
                else None
            ),
            shell_command=(
                ShellCommandSpec.from_dict(raw_shell_command)
                if isinstance(raw_shell_command, Mapping)
                else None
            ),
            browser_action=(
                BrowserActionSpec.from_dict(raw_browser_action)
                if isinstance(raw_browser_action, Mapping)
                else None
            ),
            app_focus=(
                AppFocusSpec.from_dict(raw_app_focus)
                if isinstance(raw_app_focus, Mapping)
                else None
            ),
            clipboard_action=(
                ClipboardActionSpec.from_dict(raw_clipboard_action)
                if isinstance(raw_clipboard_action, Mapping)
                else None
            ),
            screenshot=(
                ScreenshotSpec.from_dict(raw_screenshot)
                if isinstance(raw_screenshot, Mapping)
                else None
            ),
            ocr_request=(
                OCRRequestSpec.from_dict(raw_ocr_request)
                if isinstance(raw_ocr_request, Mapping)
                else None
            ),
            desktop_input=(
                DesktopInputSpec.from_dict(raw_desktop_input)
                if isinstance(raw_desktop_input, Mapping)
                else None
            ),
            requires_elevation=bool(data.get("requires_elevation", False)),
            persistent_background=bool(data.get("persistent_background", False)),
            hidden_execution=bool(data.get("hidden_execution", False)),
            touches_credentials=bool(data.get("touches_credentials", False)),
            unrestricted_scope=bool(data.get("unrestricted_scope", False)),
            destructive=bool(data.get("destructive", False)),
            cross_app=bool(data.get("cross_app", False)),
            metadata=dict(data.get("metadata", {})),
            requested_at=_parse_datetime(data.get("requested_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CapabilityPolicyDecision(DictSerializable):
    """Typed policy decision produced before any capability executor runs."""

    request_id: str
    capability_type: CapabilityType
    action_name: str
    outcome: CapabilityPolicyOutcome
    availability: CapabilityAvailabilityStatus
    requires_approval: bool = False
    reason_codes: tuple[str, ...] = ()
    detail: str = ""
    warnings: tuple[str, ...] = ()
    decided_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.request_id.strip()), "CapabilityPolicyDecision.request_id must not be empty.")
        _require(bool(self.action_name.strip()), "CapabilityPolicyDecision.action_name must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapabilityPolicyDecision:
        return cls(
            request_id=str(data.get("request_id", "")),
            capability_type=_parse_enum(CapabilityType, data.get("capability_type", CapabilityType.FILE_OPERATION)),
            action_name=str(data.get("action_name", "")),
            outcome=_parse_enum(CapabilityPolicyOutcome, data.get("outcome", CapabilityPolicyOutcome.DENIED)),
            availability=_parse_enum(
                CapabilityAvailabilityStatus,
                data.get("availability", CapabilityAvailabilityStatus.UNAVAILABLE),
            ),
            requires_approval=bool(data.get("requires_approval", False)),
            reason_codes=tuple(str(item) for item in data.get("reason_codes", ())),
            detail=str(data.get("detail", "")),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
            decided_at=_parse_datetime(data.get("decided_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CapabilityRegistration(DictSerializable):
    """Typed registry entry describing one capability's current policy and availability."""

    capability_type: CapabilityType
    summary: str
    available: bool = True
    enabled: bool = False
    status: CapabilityAvailabilityStatus = CapabilityAvailabilityStatus.AVAILABLE
    default_policy_outcome: CapabilityPolicyOutcome = CapabilityPolicyOutcome.DENIED
    reason: str = ""
    detail: str = ""
    supported_actions: tuple[str, ...] = ()
    allowlisted_targets: tuple[str, ...] = ()
    executor_kind: str = "stub"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(bool(self.summary.strip()), "CapabilityRegistration.summary must not be empty.")
        _require(bool(self.executor_kind.strip()), "CapabilityRegistration.executor_kind must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapabilityRegistration:
        return cls(
            capability_type=_parse_enum(CapabilityType, data.get("capability_type", CapabilityType.FILE_OPERATION)),
            summary=str(data.get("summary", "")),
            available=bool(data.get("available", True)),
            enabled=bool(data.get("enabled", False)),
            status=_parse_enum(
                CapabilityAvailabilityStatus,
                data.get("status", CapabilityAvailabilityStatus.AVAILABLE),
            ),
            default_policy_outcome=_parse_enum(
                CapabilityPolicyOutcome,
                data.get("default_policy_outcome", CapabilityPolicyOutcome.DENIED),
            ),
            reason=str(data.get("reason", "")),
            detail=str(data.get("detail", "")),
            supported_actions=tuple(str(item) for item in data.get("supported_actions", ())),
            allowlisted_targets=tuple(str(item) for item in data.get("allowlisted_targets", ())),
            executor_kind=str(data.get("executor_kind", "stub")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True, frozen=True)
class CapabilityAuditRecord(DictSerializable):
    """Persisted audit record for one capability request lifecycle step."""

    audit_id: str
    request_id: str
    capability_type: CapabilityType
    action_name: str
    event_type: CapabilityAuditEventType
    summary: str
    detail: str = ""
    policy_outcome: str = ""
    reason_codes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.audit_id.strip()), "CapabilityAuditRecord.audit_id must not be empty.")
        _require(bool(self.request_id.strip()), "CapabilityAuditRecord.request_id must not be empty.")
        _require(bool(self.action_name.strip()), "CapabilityAuditRecord.action_name must not be empty.")
        _require(bool(self.summary.strip()), "CapabilityAuditRecord.summary must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapabilityAuditRecord:
        return cls(
            audit_id=str(data.get("audit_id", "")),
            request_id=str(data.get("request_id", "")),
            capability_type=_parse_enum(CapabilityType, data.get("capability_type", CapabilityType.FILE_OPERATION)),
            action_name=str(data.get("action_name", "")),
            event_type=_parse_enum(CapabilityAuditEventType, data.get("event_type", CapabilityAuditEventType.REQUESTED)),
            summary=str(data.get("summary", "")),
            detail=str(data.get("detail", "")),
            policy_outcome=str(data.get("policy_outcome", "")),
            reason_codes=tuple(str(item) for item in data.get("reason_codes", ())),
            metadata=dict(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CapabilityExecutionResult(DictSerializable):
    """Typed execution result returned by the stub capability executor."""

    request_id: str
    capability_type: CapabilityType
    action_name: str
    status: CapabilityExecutionStatus
    summary: str
    detail: str = ""
    executor_kind: str = "stub"
    output_ref: str = ""
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.request_id.strip()), "CapabilityExecutionResult.request_id must not be empty.")
        _require(bool(self.action_name.strip()), "CapabilityExecutionResult.action_name must not be empty.")
        _require(bool(self.summary.strip()), "CapabilityExecutionResult.summary must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapabilityExecutionResult:
        return cls(
            request_id=str(data.get("request_id", "")),
            capability_type=_parse_enum(CapabilityType, data.get("capability_type", CapabilityType.FILE_OPERATION)),
            action_name=str(data.get("action_name", "")),
            status=_parse_enum(CapabilityExecutionStatus, data.get("status", CapabilityExecutionStatus.BLOCKED)),
            summary=str(data.get("summary", "")),
            detail=str(data.get("detail", "")),
            executor_kind=str(data.get("executor_kind", "stub")),
            output_ref=str(data.get("output_ref", "")),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
            metadata=dict(data.get("metadata", {})),
            completed_at=_parse_datetime(data.get("completed_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CapabilityRegistryView(DictSerializable):
    """Persisted capability-control-plane view for typed local task capabilities."""

    registrations: tuple[CapabilityRegistration, ...] = ()
    recent_decisions: tuple[CapabilityPolicyDecision, ...] = ()
    recent_audits: tuple[CapabilityAuditRecord, ...] = ()
    updated_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapabilityRegistryView:
        return cls(
            registrations=tuple(
                CapabilityRegistration.from_dict(item) for item in data.get("registrations", ())
            ),
            recent_decisions=tuple(
                CapabilityPolicyDecision.from_dict(item) for item in data.get("recent_decisions", ())
            ),
            recent_audits=tuple(
                CapabilityAuditRecord.from_dict(item) for item in data.get("recent_audits", ())
            ),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class LocalTaskPendingApproval(DictSerializable):
    """One approval request waiting inside an explicit local task session."""

    request_id: str
    capability_type: CapabilityType
    action_name: str
    summary: str
    target: str = ""
    requested_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.request_id.strip()), "LocalTaskPendingApproval.request_id must not be empty.")
        _require(bool(self.action_name.strip()), "LocalTaskPendingApproval.action_name must not be empty.")
        _require(bool(self.summary.strip()), "LocalTaskPendingApproval.summary must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LocalTaskPendingApproval:
        return cls(
            request_id=str(data.get("request_id", "")),
            capability_type=_parse_enum(CapabilityType, data.get("capability_type", CapabilityType.FILE_OPERATION)),
            action_name=str(data.get("action_name", "")),
            summary=str(data.get("summary", "")),
            target=str(data.get("target", "")),
            requested_at=_parse_datetime(data.get("requested_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class LocalTaskSession(DictSerializable):
    """Persisted session boundary for bounded local task execution."""

    session_id: str
    label: str
    profile_name: str = "default"
    status: LocalTaskSessionState = LocalTaskSessionState.PENDING
    control_mode: str = "local_task"
    current_target: str = ""
    last_action_summary: str = ""
    last_request_id: str = ""
    last_request_fingerprint: str = ""
    repeated_request_count: int = 0
    last_task_id: str = ""
    continuous_capture_active: bool = False
    continuous_capture_directory: str = ""
    continuous_capture_frame_count: int = 0
    continuous_capture_retained_frame_count: int = 0
    continuous_capture_last_frame_path: str = ""
    continuous_capture_region: str = "full_screen"
    continuous_capture_fps: float = 0.0
    continuous_capture_max_width: int = 0
    continuous_capture_max_height: int = 0
    continuous_capture_last_diff_ratio: float = 0.0
    continuous_capture_warnings: tuple[str, ...] = ()
    continuous_capture_last_capture_at: datetime | None = None
    requested_observation_tier: str = "screenshot_on_demand"
    effective_observation_tier: str = "screenshot_on_demand"
    observation_degraded_reason: str = ""
    observation_degraded_features: tuple[str, ...] = ()
    last_observation_tier: str = ""
    last_observation_status: str = ""
    last_observation_summary: str = ""
    last_observation_output_ref: str = ""
    last_observation_text_preview: str = ""
    last_observation_backend: str = ""
    last_observation_warnings: tuple[str, ...] = ()
    last_observation_at: datetime | None = None
    pending_approvals: tuple[LocalTaskPendingApproval, ...] = ()
    pause_requested: bool = False
    stop_requested: bool = False
    kill_switch_engaged: bool = False
    last_control_reason: str = ""
    last_error: str = ""
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    ended_at: datetime | None = None

    def __post_init__(self) -> None:
        _require(bool(self.session_id.strip()), "LocalTaskSession.session_id must not be empty.")
        _require(bool(self.label.strip()), "LocalTaskSession.label must not be empty.")
        _require(bool(self.profile_name.strip()), "LocalTaskSession.profile_name must not be empty.")
        _require(bool(self.control_mode.strip()), "LocalTaskSession.control_mode must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LocalTaskSession:
        raw_continuous_capture_last_capture_at = data.get("continuous_capture_last_capture_at")
        raw_last_observation_at = data.get("last_observation_at")
        raw_ended_at = data.get("ended_at")
        return cls(
            session_id=str(data.get("session_id", "")),
            label=str(data.get("label", "")),
            profile_name=str(data.get("profile_name", "default")),
            status=_parse_enum(LocalTaskSessionState, data.get("status", LocalTaskSessionState.PENDING)),
            control_mode=str(data.get("control_mode", "local_task")),
            current_target=str(data.get("current_target", "")),
            last_action_summary=str(data.get("last_action_summary", "")),
            last_request_id=str(data.get("last_request_id", "")),
            last_request_fingerprint=str(data.get("last_request_fingerprint", "")),
            repeated_request_count=int(data.get("repeated_request_count", 0)),
            last_task_id=str(data.get("last_task_id", "")),
            continuous_capture_active=bool(data.get("continuous_capture_active", False)),
            continuous_capture_directory=str(data.get("continuous_capture_directory", "")),
            continuous_capture_frame_count=int(data.get("continuous_capture_frame_count", 0)),
            continuous_capture_retained_frame_count=int(data.get("continuous_capture_retained_frame_count", 0)),
            continuous_capture_last_frame_path=str(data.get("continuous_capture_last_frame_path", "")),
            continuous_capture_region=str(data.get("continuous_capture_region", "full_screen")),
            continuous_capture_fps=float(data.get("continuous_capture_fps", 0.0)),
            continuous_capture_max_width=int(data.get("continuous_capture_max_width", 0)),
            continuous_capture_max_height=int(data.get("continuous_capture_max_height", 0)),
            continuous_capture_last_diff_ratio=float(data.get("continuous_capture_last_diff_ratio", 0.0)),
            continuous_capture_warnings=tuple(
                str(item) for item in data.get("continuous_capture_warnings", ())
            ),
            continuous_capture_last_capture_at=(
                None
                if raw_continuous_capture_last_capture_at in (None, "")
                else _parse_datetime(raw_continuous_capture_last_capture_at)
            ),
            requested_observation_tier=str(
                data.get("requested_observation_tier", "screenshot_on_demand")
            ),
            effective_observation_tier=str(
                data.get("effective_observation_tier", "screenshot_on_demand")
            ),
            observation_degraded_reason=str(data.get("observation_degraded_reason", "")),
            observation_degraded_features=tuple(
                str(item) for item in data.get("observation_degraded_features", ())
            ),
            last_observation_tier=str(data.get("last_observation_tier", "")),
            last_observation_status=str(data.get("last_observation_status", "")),
            last_observation_summary=str(data.get("last_observation_summary", "")),
            last_observation_output_ref=str(data.get("last_observation_output_ref", "")),
            last_observation_text_preview=str(data.get("last_observation_text_preview", "")),
            last_observation_backend=str(data.get("last_observation_backend", "")),
            last_observation_warnings=tuple(
                str(item) for item in data.get("last_observation_warnings", ())
            ),
            last_observation_at=(
                None if raw_last_observation_at in (None, "") else _parse_datetime(raw_last_observation_at)
            ),
            pending_approvals=tuple(
                LocalTaskPendingApproval.from_dict(item)
                for item in data.get("pending_approvals", ())
            ),
            pause_requested=bool(data.get("pause_requested", False)),
            stop_requested=bool(data.get("stop_requested", False)),
            kill_switch_engaged=bool(data.get("kill_switch_engaged", False)),
            last_control_reason=str(data.get("last_control_reason", "")),
            last_error=str(data.get("last_error", "")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
            ended_at=None if raw_ended_at in (None, "") else _parse_datetime(raw_ended_at),
        )


@dataclass(slots=True, frozen=True)
class UserSettingsProfile(DictSerializable):
    """Persisted user-facing settings profile for the local app shell."""

    profile_name: str = "default"
    runtime: dict[str, Any] = field(
        default_factory=lambda: {
            "stub_mode": True,
            "allow_web_fallback": True,
            "enable_self_optimizer": False,
            "generation_backend": "ollama",
            "embedding_backend": "sentence_transformers",
            "vector_store_backend": "chromadb",
        }
    )
    retrieval: dict[str, Any] = field(
        default_factory=lambda: {
            "allow_web_fallback": True,
            "provider": "wikipedia",
            "reranking": True,
        }
    )
    reasoning: dict[str, Any] = field(
        default_factory=lambda: {
            "thinking_minutes": 30,
            "mode": "auto",
        }
    )
    long_horizon: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "wall_clock_minutes": 120,
            "cycle_budget_minutes": 120,
            "checkpoint_interval_minutes": 120,
            "duty_cycle_ratio": 0.75,
            "cooldown_seconds": 0.05,
            "max_resume_count": 5,
        }
    )
    optimizer: dict[str, Any] = field(
        default_factory=lambda: {
            "activation_policy": "proposal_only",
            "replay_limit": 64,
        }
    )
    models: dict[str, Any] = field(
        default_factory=lambda: {
            "preferred_by_role": {
                "generation": "ollama:qwen2.5:3b-instruct-q4_K_M",
                "embedding": "sentence_transformers:intfloat/e5-small-v2",
            },
            "enabled_roles": ("generation", "embedding"),
        }
    )
    coding: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "mode": "assistant",
            "practice_when_idle": False,
            "default_language": "python",
            "default_framework": "",
            "sandbox_enabled": True,
            "local_only": True,
            "preferred_models_by_role": {},
            "enabled_roles": tuple(role.value for role in CodingRole),
        }
    )
    desktop: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "approval_policy": "approve_risky_only",
            "enabled_capabilities": (),
            "allowlisted_roots": (".", "logs", "examples", "models"),
            "allowlisted_shell_commands": ("python", "git", "rg", "pytest"),
            "allowlisted_browser_domains": ("localhost", "127.0.0.1"),
            "allowlisted_apps": (),
            "allowlisted_background_services": (),
        }
    )
    observation: dict[str, Any] = field(
        default_factory=lambda: {
            "tier": "screenshot_on_demand",
            "continuous_capture": False,
            "ocr_on_step": False,
            "vision_on_step": False,
            "capture_fps": 0.5,
            "capture_max_width": 960,
            "capture_max_height": 540,
            "capture_frame_history": 4,
            "capture_diff_threshold": 0.03,
            "region_of_interest": "full_screen",
        }
    )
    cloud: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "mode": CloudOffloadMode.AUXILIARY_ONLY.value,
            "provider": "stub_cloud",
            "provider_family": "provider_agnostic",
            "max_payload_bytes": 1024 * 256,
            "max_retries": 1,
            "fallback_behavior": CloudFallbackBehavior.RETRY_THEN_LOCAL.value,
            "capability_modes": {
                capability.value: CloudOffloadMode.DISABLED.value
                for capability in CloudOffloadCapability
            },
        }
    )
    privacy: dict[str, Any] = field(
        default_factory=lambda: {
            "log_runtime_events": True,
            "allow_cloud_content": False,
            "log_level": "INFO",
        }
    )
    ui: dict[str, Any] = field(
        default_factory=lambda: {
            "show_debug_pane": True,
            "app_shell": "tkinter",
            "shell_variant": "classic_dashboard",
            "lightweight_mode": False,
            "show_utility_drawer": False,
            "reduced_motion": False,
            "activity_strip_visible": True,
            "task_timeline_visible": True,
            "resource_ribbon_visible": True,
            "shell_notifications_visible": True,
            "shell_preset": "balanced",
        }
    )
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.profile_name.strip()), "UserSettingsProfile.profile_name must not be empty.")
        canonical_desktop = {
            "enabled": False,
            "approval_policy": "approve_risky_only",
            "enabled_capabilities": (),
            "allowlisted_roots": (".", "logs", "examples", "models"),
            "allowlisted_shell_commands": ("python", "git", "rg", "pytest"),
            "allowlisted_browser_domains": ("localhost", "127.0.0.1"),
            "allowlisted_apps": (),
            "allowlisted_background_services": (),
            **dict(self.desktop),
        }
        canonical_desktop["enabled_capabilities"] = tuple(
            str(item) for item in canonical_desktop.get("enabled_capabilities", ()) if str(item).strip()
        )
        for key in (
            "allowlisted_roots",
            "allowlisted_shell_commands",
            "allowlisted_browser_domains",
            "allowlisted_apps",
            "allowlisted_background_services",
        ):
            canonical_desktop[key] = tuple(
                str(item) for item in canonical_desktop.get(key, ()) if str(item).strip()
            )
        object.__setattr__(self, "desktop", canonical_desktop)
        canonical_observation = {
            "tier": "screenshot_on_demand",
            "continuous_capture": False,
            "ocr_on_step": False,
            "vision_on_step": False,
            "capture_fps": 0.5,
            "capture_max_width": 960,
            "capture_max_height": 540,
            "capture_frame_history": 4,
            "capture_diff_threshold": 0.03,
            "region_of_interest": "full_screen",
            **dict(self.observation),
        }
        object.__setattr__(self, "observation", canonical_observation)
        canonical_cloud = {
            "enabled": False,
            "mode": CloudOffloadMode.AUXILIARY_ONLY.value,
            "provider": "stub_cloud",
            "provider_family": "provider_agnostic",
            "max_payload_bytes": 1024 * 256,
            "max_retries": 1,
            "fallback_behavior": CloudFallbackBehavior.RETRY_THEN_LOCAL.value,
            "capability_modes": {
                capability.value: CloudOffloadMode.DISABLED.value
                for capability in CloudOffloadCapability
            },
            **dict(self.cloud),
        }
        raw_capability_modes = dict(canonical_cloud.get("capability_modes", {}))
        canonical_capability_modes = {
            capability.value: CloudOffloadMode.DISABLED.value
            for capability in CloudOffloadCapability
        }
        for raw_name, raw_mode in raw_capability_modes.items():
            capability_name = str(raw_name).strip()
            if not capability_name:
                continue
            canonical_capability_modes[capability_name] = (
                str(raw_mode).strip() or CloudOffloadMode.DISABLED.value
            )
        canonical_cloud["capability_modes"] = canonical_capability_modes
        canonical_cloud["enabled"] = bool(
            str(canonical_cloud.get("mode", CloudOffloadMode.AUXILIARY_ONLY.value)).strip()
            != CloudOffloadMode.DISABLED.value
            and any(mode != CloudOffloadMode.DISABLED.value for mode in canonical_capability_modes.values())
        )
        object.__setattr__(self, "cloud", canonical_cloud)
        canonical_coding = {
            "enabled": False,
            "mode": "assistant",
            "practice_when_idle": False,
            "default_language": "python",
            "default_framework": "",
            "sandbox_enabled": True,
            "local_only": True,
            "preferred_models_by_role": {},
            "enabled_roles": tuple(role.value for role in CodingRole),
            **dict(self.coding),
        }
        canonical_coding["preferred_models_by_role"] = {
            str(key): str(value)
            for key, value in dict(canonical_coding.get("preferred_models_by_role", {})).items()
            if str(key).strip() and str(value).strip()
        }
        canonical_coding["enabled_roles"] = tuple(
            str(item) for item in canonical_coding.get("enabled_roles", ()) if str(item).strip()
        )
        object.__setattr__(self, "coding", canonical_coding)
        canonical_ui = {
            "show_debug_pane": True,
            "app_shell": "tkinter",
            "shell_variant": "classic_dashboard",
            "lightweight_mode": False,
            "show_utility_drawer": False,
            "reduced_motion": False,
            "activity_strip_visible": True,
            "task_timeline_visible": True,
            "resource_ribbon_visible": True,
            "shell_notifications_visible": True,
            "shell_preset": "balanced",
            **dict(self.ui),
        }
        object.__setattr__(self, "ui", canonical_ui)

    def validate(self) -> None:
        """Validate bounded settings values used by the lightweight local app."""
        reasoning_mode = str(self.reasoning.get("mode", "auto"))
        _require(reasoning_mode in {"auto", "fast", "deep"}, "reasoning.mode must be auto, fast, or deep.")
        thinking_minutes = int(self.reasoning.get("thinking_minutes", 30) or 30)
        _require(thinking_minutes >= 1, "reasoning.thinking_minutes must be at least 1.")
        wall_clock_minutes = int(self.long_horizon.get("wall_clock_minutes", 120) or 120)
        cycle_budget_minutes = int(self.long_horizon.get("cycle_budget_minutes", 120) or 120)
        checkpoint_interval_minutes = int(
            self.long_horizon.get("checkpoint_interval_minutes", cycle_budget_minutes) or cycle_budget_minutes
        )
        duty_cycle_ratio = float(self.long_horizon.get("duty_cycle_ratio", 0.75) or 0.75)
        cooldown_seconds = float(self.long_horizon.get("cooldown_seconds", 0.05) or 0.05)
        max_resume_count = int(self.long_horizon.get("max_resume_count", 5) or 5)
        _require(wall_clock_minutes >= 1, "long_horizon.wall_clock_minutes must be at least 1.")
        _require(cycle_budget_minutes >= 1, "long_horizon.cycle_budget_minutes must be at least 1.")
        _require(
            checkpoint_interval_minutes >= 1,
            "long_horizon.checkpoint_interval_minutes must be at least 1.",
        )
        _require(
            cycle_budget_minutes <= wall_clock_minutes,
            "long_horizon.cycle_budget_minutes must not exceed wall_clock_minutes.",
        )
        _require(
            checkpoint_interval_minutes <= cycle_budget_minutes,
            "long_horizon.checkpoint_interval_minutes must not exceed cycle_budget_minutes.",
        )
        _require(
            0.0 < duty_cycle_ratio <= 1.0,
            "long_horizon.duty_cycle_ratio must be between 0 and 1.",
        )
        _require(cooldown_seconds >= 0.0, "long_horizon.cooldown_seconds must be zero or positive.")
        _require(max_resume_count >= 0, "long_horizon.max_resume_count must be zero or positive.")
        coding_mode = str(self.coding.get("mode", "assistant") or "assistant")
        _require(
            coding_mode in {"assistant", "coding_workspace"},
            "coding.mode must be assistant or coding_workspace.",
        )
        _require(
            bool(str(self.coding.get("default_language", "python")).strip()),
            "coding.default_language must not be empty.",
        )
        observation_tier = str(self.observation.get("tier", "screenshot_on_demand"))
        _require(
            observation_tier in {
                "screenshot_on_demand",
                "ocr_on_step",
                "vision_on_step",
                "continuous_capture",
            },
            "observation.tier is not recognized.",
        )
        capture_fps = float(self.observation.get("capture_fps", 0.5) or 0.5)
        _require(
            0.05 <= capture_fps <= 2.0,
            "observation.capture_fps must stay between 0.05 and 2.0.",
        )
        capture_max_width = int(self.observation.get("capture_max_width", 960) or 960)
        capture_max_height = int(self.observation.get("capture_max_height", 540) or 540)
        _require(
            64 <= capture_max_width <= 1280,
            "observation.capture_max_width must stay between 64 and 1280.",
        )
        _require(
            64 <= capture_max_height <= 720,
            "observation.capture_max_height must stay between 64 and 720.",
        )
        capture_frame_history = int(self.observation.get("capture_frame_history", 4) or 4)
        _require(
            1 <= capture_frame_history <= 8,
            "observation.capture_frame_history must stay between 1 and 8.",
        )
        capture_diff_threshold = float(self.observation.get("capture_diff_threshold", 0.03) or 0.03)
        _require(
            0.0 < capture_diff_threshold <= 1.0,
            "observation.capture_diff_threshold must stay between 0 and 1.",
        )
        region_of_interest = str(self.observation.get("region_of_interest", "full_screen"))
        if region_of_interest.strip().lower() != "full_screen":
            parts = [part.strip() for part in region_of_interest.split(",")]
            _require(
                len(parts) == 4,
                "observation.region_of_interest must be full_screen or left,top,width,height.",
            )
            left, top, width, height = (int(part) for part in parts)
            _require(left >= 0 and top >= 0, "observation.region_of_interest must start on-screen.")
            _require(width > 0 and height > 0, "observation.region_of_interest width and height must be positive.")
        cloud_mode = str(self.cloud.get("mode", CloudOffloadMode.AUXILIARY_ONLY.value))
        _require(
            cloud_mode in {item.value for item in CloudOffloadMode},
            "cloud.mode is not recognized.",
        )
        provider_name = str(self.cloud.get("provider", "stub_cloud")).strip()
        _require(bool(provider_name), "cloud.provider must not be empty.")
        provider_family = str(self.cloud.get("provider_family", "provider_agnostic")).strip()
        _require(bool(provider_family), "cloud.provider_family must not be empty.")
        max_payload_bytes = int(self.cloud.get("max_payload_bytes", 1024 * 256) or 1024 * 256)
        _require(
            1024 <= max_payload_bytes <= 10 * 1024 * 1024,
            "cloud.max_payload_bytes must stay between 1024 and 10485760.",
        )
        max_retries = int(self.cloud.get("max_retries", 1) or 0)
        _require(0 <= max_retries <= 3, "cloud.max_retries must stay between 0 and 3.")
        fallback_behavior = str(
            self.cloud.get("fallback_behavior", CloudFallbackBehavior.RETRY_THEN_LOCAL.value)
        ).strip()
        _require(
            fallback_behavior in {item.value for item in CloudFallbackBehavior},
            "cloud.fallback_behavior is not recognized.",
        )
        capability_modes = dict(self.cloud.get("capability_modes", {}))
        valid_cloud_capabilities = {item.value for item in CloudOffloadCapability}
        valid_cloud_modes = {item.value for item in CloudOffloadMode}
        _require(
            all(str(name).strip() in valid_cloud_capabilities for name in capability_modes),
            "cloud.capability_modes contains an unknown capability.",
        )
        _require(
            all(str(mode).strip() in valid_cloud_modes for mode in capability_modes.values()),
            "cloud.capability_modes contains an unknown mode.",
        )
        approval_policy = str(self.desktop.get("approval_policy", "approve_risky_only"))
        _require(
            approval_policy in {"approve_risky_only", "manual_only", "safe_auto"},
            "desktop.approval_policy is not recognized.",
        )
        enabled_capabilities = tuple(
            str(item)
            for item in self.desktop.get("enabled_capabilities", ())
            if str(item).strip()
        )
        valid_capabilities = {item.value for item in CapabilityType}
        _require(
            all(item in valid_capabilities for item in enabled_capabilities),
            "desktop.enabled_capabilities contains an unknown capability.",
        )
        for key in (
            "allowlisted_roots",
            "allowlisted_shell_commands",
            "allowlisted_browser_domains",
        ):
            values = tuple(str(item) for item in self.desktop.get(key, ()) if str(item).strip())
            _require(bool(values), f"desktop.{key} must keep at least one entry.")
        app_shell = str(self.ui.get("app_shell", "tkinter")).strip()
        _require(
            app_shell in {"tkinter", "pyside6"},
            "ui.app_shell must be tkinter or pyside6 for the local app.",
        )
        shell_preset = str(self.ui.get("shell_preset", "balanced")).strip().lower()
        _require(
            shell_preset in {"minimal", "balanced", "immersive"},
            "ui.shell_preset must be minimal, balanced, or immersive.",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UserSettingsProfile:
        defaults = cls(profile_name=str(data.get("profile_name", "default")))
        raw_models = dict(data.get("models", {}))
        raw_preferred_by_role = {
            **dict(defaults.models.get("preferred_by_role", {})),
            **dict(raw_models.get("preferred_by_role", {})),
        }
        raw_enabled_roles = tuple(
            str(item)
            for item in raw_models.get("enabled_roles", defaults.models.get("enabled_roles", ()))
            if str(item).strip()
        )
        profile = cls(
            profile_name=defaults.profile_name,
            runtime={**defaults.runtime, **dict(data.get("runtime", {}))},
            retrieval={**defaults.retrieval, **dict(data.get("retrieval", {}))},
            reasoning={**defaults.reasoning, **dict(data.get("reasoning", {}))},
            long_horizon={**defaults.long_horizon, **dict(data.get("long_horizon", {}))},
            optimizer={**defaults.optimizer, **dict(data.get("optimizer", {}))},
            models={
                **defaults.models,
                **raw_models,
                "preferred_by_role": raw_preferred_by_role,
                "enabled_roles": raw_enabled_roles or tuple(defaults.models.get("enabled_roles", ())),
            },
            coding={**defaults.coding, **dict(data.get("coding", {}))},
            desktop={
                **defaults.desktop,
                **dict(data.get("desktop", {})),
                "enabled_capabilities": tuple(
                    str(item)
                    for item in dict(data.get("desktop", {})).get(
                        "enabled_capabilities",
                        defaults.desktop.get("enabled_capabilities", ()),
                    )
                    if str(item).strip()
                ),
                "allowlisted_roots": tuple(
                    str(item)
                    for item in dict(data.get("desktop", {})).get(
                        "allowlisted_roots",
                        defaults.desktop.get("allowlisted_roots", ()),
                    )
                    if str(item).strip()
                )
                or tuple(str(item) for item in defaults.desktop.get("allowlisted_roots", ())),
                "allowlisted_shell_commands": tuple(
                    str(item)
                    for item in dict(data.get("desktop", {})).get(
                        "allowlisted_shell_commands",
                        defaults.desktop.get("allowlisted_shell_commands", ()),
                    )
                    if str(item).strip()
                )
                or tuple(str(item) for item in defaults.desktop.get("allowlisted_shell_commands", ())),
                "allowlisted_browser_domains": tuple(
                    str(item)
                    for item in dict(data.get("desktop", {})).get(
                        "allowlisted_browser_domains",
                        defaults.desktop.get("allowlisted_browser_domains", ()),
                    )
                    if str(item).strip()
                )
                or tuple(str(item) for item in defaults.desktop.get("allowlisted_browser_domains", ())),
                "allowlisted_apps": tuple(
                    str(item)
                    for item in dict(data.get("desktop", {})).get(
                        "allowlisted_apps",
                        defaults.desktop.get("allowlisted_apps", ()),
                    )
                    if str(item).strip()
                ),
                "allowlisted_background_services": tuple(
                    str(item)
                    for item in dict(data.get("desktop", {})).get(
                        "allowlisted_background_services",
                        defaults.desktop.get("allowlisted_background_services", ()),
                    )
                    if str(item).strip()
                ),
            },
            observation={**defaults.observation, **dict(data.get("observation", {}))},
            cloud={
                **defaults.cloud,
                **dict(data.get("cloud", {})),
                "capability_modes": {
                    **dict(defaults.cloud.get("capability_modes", {})),
                    **dict(dict(data.get("cloud", {})).get("capability_modes", {})),
                },
            },
            privacy={**defaults.privacy, **dict(data.get("privacy", {}))},
            ui={**defaults.ui, **dict(data.get("ui", {}))},
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )
        profile.validate()
        return profile


@dataclass(slots=True, frozen=True)
class DashboardTaskHistoryEntry(DictSerializable):
    """Compact task-history row shown in the local app shell."""

    task_id: str
    question: str
    answer_preview: str = ""
    critique_result: str = ""
    degraded_reason: str = ""
    warning_count: int = 0
    candidate_trace_count: int = 0
    citation_count: int = 0
    selected_strategy: str = ""
    selected_verifier: str = ""
    used_web_fallback: bool = False
    completed_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "DashboardTaskHistoryEntry.task_id must not be empty.")
        _require(bool(self.question.strip()), "DashboardTaskHistoryEntry.question must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DashboardTaskHistoryEntry:
        return cls(
            task_id=str(data["task_id"]),
            question=str(data["question"]),
            answer_preview=str(data.get("answer_preview", "")),
            critique_result=str(data.get("critique_result", "")),
            degraded_reason=str(data.get("degraded_reason", "")),
            warning_count=int(data.get("warning_count", 0)),
            candidate_trace_count=int(data.get("candidate_trace_count", 0)),
            citation_count=int(data.get("citation_count", 0)),
            selected_strategy=str(data.get("selected_strategy", "")),
            selected_verifier=str(data.get("selected_verifier", "")),
            used_web_fallback=bool(data.get("used_web_fallback", False)),
            completed_at=_parse_datetime(data.get("completed_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class DashboardTaskInspector(DictSerializable):
    """Detailed task/run inspector payload for one persisted task."""

    task_id: str = ""
    question: str = ""
    answer_text: str = ""
    critique_result: str = ""
    degraded_reason: str = ""
    warning_count: int = 0
    warnings: tuple[str, ...] = ()
    citation_refs: tuple[str, ...] = ()
    repair_actions: tuple[str, ...] = ()
    failure_categories: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    candidate_trace_count: int = 0
    selected_strategy: str = ""
    selected_verifier: str = ""
    used_web_fallback: bool = False
    trace_debug_export_path: str = ""
    optimizer_lifecycle: tuple[str, ...] = ()
    specialist_roles_used: tuple[str, ...] = ()
    specialist_role_explanations: tuple[str, ...] = ()
    advisor_summaries: tuple[str, ...] = ()
    completed_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DashboardTaskInspector:
        return cls(
            task_id=str(data.get("task_id", "")),
            question=str(data.get("question", "")),
            answer_text=str(data.get("answer_text", "")),
            critique_result=str(data.get("critique_result", "")),
            degraded_reason=str(data.get("degraded_reason", "")),
            warning_count=int(data.get("warning_count", 0)),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
            citation_refs=tuple(str(item) for item in data.get("citation_refs", ())),
            repair_actions=tuple(str(item) for item in data.get("repair_actions", ())),
            failure_categories=tuple(str(item) for item in data.get("failure_categories", ())),
            supporting_evidence_ids=tuple(str(item) for item in data.get("supporting_evidence_ids", ())),
            candidate_trace_count=int(data.get("candidate_trace_count", 0)),
            selected_strategy=str(data.get("selected_strategy", "")),
            selected_verifier=str(data.get("selected_verifier", "")),
            used_web_fallback=bool(data.get("used_web_fallback", False)),
            trace_debug_export_path=str(data.get("trace_debug_export_path", "")),
            optimizer_lifecycle=tuple(str(item) for item in data.get("optimizer_lifecycle", ())),
            specialist_roles_used=tuple(str(item) for item in data.get("specialist_roles_used", ())),
            specialist_role_explanations=tuple(
                str(item) for item in data.get("specialist_role_explanations", ())
            ),
            advisor_summaries=tuple(str(item) for item in data.get("advisor_summaries", ())),
            completed_at=_parse_datetime(data.get("completed_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class DashboardKnowledgeSource(DictSerializable):
    """Summary of one local knowledge source for the GUI library view."""

    document_id: str
    source_ref: str
    title: str
    chunk_count: int = 0
    embedding_model: str = ""
    archived: bool = False
    corpus_origin: str = ""
    corpus_tier: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(bool(self.document_id.strip()), "DashboardKnowledgeSource.document_id must not be empty.")
        _require(bool(self.source_ref.strip()), "DashboardKnowledgeSource.source_ref must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DashboardKnowledgeSource:
        return cls(
            document_id=str(data["document_id"]),
            source_ref=str(data["source_ref"]),
            title=str(data.get("title", "")),
            chunk_count=int(data.get("chunk_count", 0)),
            embedding_model=str(data.get("embedding_model", "")),
            archived=bool(data.get("archived", False)),
            corpus_origin=str(data.get("corpus_origin", "")),
            corpus_tier=str(data.get("corpus_tier", "")),
            updated_at=str(data.get("updated_at", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True, frozen=True)
class DashboardCapabilityAvailability(DictSerializable):
    """Capability-gating explanation shown in readiness/settings surfaces."""

    capability_name: str
    status: str
    reason: str = ""
    detail: str = ""
    recovery_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require(
            bool(self.capability_name.strip()),
            "DashboardCapabilityAvailability.capability_name must not be empty.",
        )
        _require(bool(self.status.strip()), "DashboardCapabilityAvailability.status must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DashboardCapabilityAvailability:
        return cls(
            capability_name=str(data["capability_name"]),
            status=str(data["status"]),
            reason=str(data.get("reason", "")),
            detail=str(data.get("detail", "")),
            recovery_actions=tuple(str(item) for item in data.get("recovery_actions", ())),
        )


@dataclass(slots=True, frozen=True)
class DashboardReadinessCheck(DictSerializable):
    """One readiness/preflight check rendered in the local app."""

    check_id: str
    title: str
    status: str
    detail: str = ""
    recovery_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require(bool(self.check_id.strip()), "DashboardReadinessCheck.check_id must not be empty.")
        _require(bool(self.title.strip()), "DashboardReadinessCheck.title must not be empty.")
        _require(bool(self.status.strip()), "DashboardReadinessCheck.status must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DashboardReadinessCheck:
        return cls(
            check_id=str(data["check_id"]),
            title=str(data["title"]),
            status=str(data["status"]),
            detail=str(data.get("detail", "")),
            recovery_actions=tuple(str(item) for item in data.get("recovery_actions", ())),
        )


@dataclass(slots=True, frozen=True)
class DashboardReadinessReport(DictSerializable):
    """Readiness summary for stub mode, real mode, and optional capability tiers."""

    stub_mode_ready: bool = False
    real_mode_ready: bool = False
    checks: tuple[DashboardReadinessCheck, ...] = ()
    capabilities: tuple[DashboardCapabilityAvailability, ...] = ()
    guidance: tuple[str, ...] = ()
    generated_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DashboardReadinessReport:
        return cls(
            stub_mode_ready=bool(data.get("stub_mode_ready", False)),
            real_mode_ready=bool(data.get("real_mode_ready", False)),
            checks=tuple(
                DashboardReadinessCheck.from_dict(item) for item in data.get("checks", ())
            ),
            capabilities=tuple(
                DashboardCapabilityAvailability.from_dict(item)
                for item in data.get("capabilities", ())
            ),
            guidance=tuple(str(item) for item in data.get("guidance", ())),
            generated_at=_parse_datetime(data.get("generated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class PackagedLaunchReport(DictSerializable):
    """Operator-facing packaged launch decision and fallback summary."""

    requested_mode: str = "stub"
    effective_mode: str = "stub"
    launch_ready: bool = False
    used_stub_fallback: bool = False
    summary: str = ""
    blocking_reason: str = ""
    blocking_detail: str = ""
    guidance: tuple[str, ...] = ()
    readiness_report: DashboardReadinessReport = field(default_factory=DashboardReadinessReport)
    generated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(
            self.requested_mode in {"stub", "real"},
            "PackagedLaunchReport.requested_mode must be stub or real.",
        )
        _require(
            self.effective_mode in {"stub", "real"},
            "PackagedLaunchReport.effective_mode must be stub or real.",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PackagedLaunchReport:
        return cls(
            requested_mode=str(data.get("requested_mode", "stub")),
            effective_mode=str(data.get("effective_mode", "stub")),
            launch_ready=bool(data.get("launch_ready", False)),
            used_stub_fallback=bool(data.get("used_stub_fallback", False)),
            summary=str(data.get("summary", "")),
            blocking_reason=str(data.get("blocking_reason", "")),
            blocking_detail=str(data.get("blocking_detail", "")),
            guidance=tuple(str(item) for item in data.get("guidance", ())),
            readiness_report=DashboardReadinessReport.from_dict(data.get("readiness_report", {})),
            generated_at=_parse_datetime(data.get("generated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class PackagedSupportBundle(DictSerializable):
    """Manifest describing one exported packaged-app support bundle."""

    bundle_dir: str
    manifest_path: str = ""
    launch_report_path: str = ""
    readiness_report_path: str = ""
    preflight_report_path: str = ""
    onboarding_guide_path: str = ""
    setup_guide_path: str = ""
    user_settings_path: str = ""
    app_state_path: str = ""
    support_readme_path: str = ""
    diagnostics_path: str = ""
    copied_artifact_paths: tuple[str, ...] = ()
    generated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.bundle_dir.strip()), "PackagedSupportBundle.bundle_dir must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PackagedSupportBundle:
        return cls(
            bundle_dir=str(data["bundle_dir"]),
            manifest_path=str(data.get("manifest_path", "")),
            launch_report_path=str(data.get("launch_report_path", "")),
            readiness_report_path=str(data.get("readiness_report_path", "")),
            preflight_report_path=str(data.get("preflight_report_path", "")),
            onboarding_guide_path=str(data.get("onboarding_guide_path", "")),
            setup_guide_path=str(data.get("setup_guide_path", "")),
            user_settings_path=str(data.get("user_settings_path", "")),
            app_state_path=str(data.get("app_state_path", "")),
            support_readme_path=str(data.get("support_readme_path", "")),
            diagnostics_path=str(data.get("diagnostics_path", "")),
            copied_artifact_paths=tuple(str(item) for item in data.get("copied_artifact_paths", ())),
            generated_at=_parse_datetime(data.get("generated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class DemoDocumentFixture(DictSerializable):
    """Repo-owned demo document fixture definition."""

    source_ref: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_task_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require(bool(self.source_ref.strip()), "DemoDocumentFixture.source_ref must not be empty.")
        _require(bool(self.title.strip()), "DemoDocumentFixture.title must not be empty.")
        _require(bool(self.content.strip()), "DemoDocumentFixture.content must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DemoDocumentFixture:
        return cls(
            source_ref=str(data["source_ref"]),
            title=str(data["title"]),
            content=str(data["content"]),
            metadata=dict(data.get("metadata", {})),
            sample_task_ids=tuple(str(item) for item in data.get("sample_task_ids", ())),
        )


@dataclass(slots=True, frozen=True)
class SampleTaskDefinition(DictSerializable):
    """Typed definition for one built-in demo/sample task."""

    sample_id: str = ""
    title: str = ""
    question: str = ""
    category: str = ""
    execution_profile: str = ""
    recommended_thinking_minutes: int = 1
    expected_behavior: str = ""
    expected_result: str = ""
    success_markers: tuple[str, ...] = ()
    required_source_refs: tuple[str, ...] = ()
    uses_web_fallback: bool = False
    requires_demo_pack: bool = False
    comparison_group: str = ""
    comparison_fast_minutes: int = 0
    comparison_deep_minutes: int = 0
    expected_degraded_reason: str = ""

    def __post_init__(self) -> None:
        if not any((self.sample_id, self.title, self.question, self.category, self.execution_profile)):
            return
        _require(bool(self.sample_id.strip()), "SampleTaskDefinition.sample_id must not be empty.")
        _require(bool(self.title.strip()), "SampleTaskDefinition.title must not be empty.")
        _require(bool(self.question.strip()), "SampleTaskDefinition.question must not be empty.")
        _require(bool(self.category.strip()), "SampleTaskDefinition.category must not be empty.")
        _require(
            bool(self.execution_profile.strip()),
            "SampleTaskDefinition.execution_profile must not be empty.",
        )
        _require(
            self.recommended_thinking_minutes >= 1,
            "SampleTaskDefinition.recommended_thinking_minutes must be positive.",
        )
        _require(
            self.comparison_fast_minutes >= 0 and self.comparison_deep_minutes >= 0,
            "SampleTaskDefinition comparison minutes must not be negative.",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SampleTaskDefinition:
        return cls(
            sample_id=str(data.get("sample_id", "")),
            title=str(data.get("title", "")),
            question=str(data.get("question", "")),
            category=str(data.get("category", "")),
            execution_profile=str(data.get("execution_profile", "")),
            recommended_thinking_minutes=int(data.get("recommended_thinking_minutes", 1)),
            expected_behavior=str(data.get("expected_behavior", "")),
            expected_result=str(data.get("expected_result", "")),
            success_markers=tuple(str(item) for item in data.get("success_markers", ())),
            required_source_refs=tuple(str(item) for item in data.get("required_source_refs", ())),
            uses_web_fallback=bool(data.get("uses_web_fallback", False)),
            requires_demo_pack=bool(data.get("requires_demo_pack", False)),
            comparison_group=str(data.get("comparison_group", "")),
            comparison_fast_minutes=int(data.get("comparison_fast_minutes", 0)),
            comparison_deep_minutes=int(data.get("comparison_deep_minutes", 0)),
            expected_degraded_reason=str(data.get("expected_degraded_reason", "")),
        )


@dataclass(slots=True, frozen=True)
class DemoRuntimePackSummary(DictSerializable):
    """Summary of the starter runtime pack shipped with the demo content."""

    pack_version: str = ""
    macro_names: tuple[str, ...] = ()
    opcode_names: tuple[str, ...] = ()
    decoder_names: tuple[str, ...] = ()
    loaded_macro_count: int = 0
    loaded_opcode_count: int = 0
    loaded_decoder_count: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DemoRuntimePackSummary:
        return cls(
            pack_version=str(data.get("pack_version", "")),
            macro_names=tuple(str(item) for item in data.get("macro_names", ())),
            opcode_names=tuple(str(item) for item in data.get("opcode_names", ())),
            decoder_names=tuple(str(item) for item in data.get("decoder_names", ())),
            loaded_macro_count=int(data.get("loaded_macro_count", 0)),
            loaded_opcode_count=int(data.get("loaded_opcode_count", 0)),
            loaded_decoder_count=int(data.get("loaded_decoder_count", 0)),
        )


@dataclass(slots=True, frozen=True)
class DemoPackStatus(DictSerializable):
    """Read-only summary of the built-in demo pack state."""

    pack_version: str = ""
    loaded: bool = False
    document_count: int = 0
    loaded_document_count: int = 0
    sample_task_count: int = 0
    runtime_pack: DemoRuntimePackSummary = field(default_factory=DemoRuntimePackSummary)
    verified_trace_example_path: str = ""
    loaded_at: str = ""
    status_detail: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DemoPackStatus:
        return cls(
            pack_version=str(data.get("pack_version", "")),
            loaded=bool(data.get("loaded", False)),
            document_count=int(data.get("document_count", 0)),
            loaded_document_count=int(data.get("loaded_document_count", 0)),
            sample_task_count=int(data.get("sample_task_count", 0)),
            runtime_pack=DemoRuntimePackSummary.from_dict(data.get("runtime_pack", {})),
            verified_trace_example_path=str(data.get("verified_trace_example_path", "")),
            loaded_at=str(data.get("loaded_at", "")),
            status_detail=str(data.get("status_detail", "")),
        )


@dataclass(slots=True, frozen=True)
class VoiceActivitySegment(DictSerializable):
    """Bounded speech-activity span extracted from one audio clip."""

    start_ms: int = 0
    end_ms: int = 0
    mean_abs_level: float = 0.0

    def __post_init__(self) -> None:
        _require(self.start_ms >= 0, "VoiceActivitySegment.start_ms must be zero or positive.")
        _require(self.end_ms >= self.start_ms, "VoiceActivitySegment.end_ms must be >= start_ms.")
        _require(0.0 <= self.mean_abs_level <= 1.0, "VoiceActivitySegment.mean_abs_level must be between 0 and 1.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VoiceActivitySegment:
        return cls(
            start_ms=int(data.get("start_ms", 0)),
            end_ms=int(data.get("end_ms", 0)),
            mean_abs_level=float(data.get("mean_abs_level", 0.0) or 0.0),
        )


@dataclass(slots=True, frozen=True)
class VoiceActivityReport(DictSerializable):
    """Typed local VAD summary used by optional speech workflows."""

    source_path: str = ""
    analyzer_backend: str = ""
    sample_rate_hz: int = 0
    channel_count: int = 0
    duration_seconds: float = 0.0
    analyzed_duration_seconds: float = 0.0
    speech_ratio: float = 0.0
    segment_count: int = 0
    segments: tuple[VoiceActivitySegment, ...] = ()
    warnings: tuple[str, ...] = ()
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(self.sample_rate_hz >= 0, "VoiceActivityReport.sample_rate_hz must be zero or positive.")
        _require(self.channel_count >= 0, "VoiceActivityReport.channel_count must be zero or positive.")
        _require(self.duration_seconds >= 0.0, "VoiceActivityReport.duration_seconds must be zero or positive.")
        _require(
            self.analyzed_duration_seconds >= 0.0,
            "VoiceActivityReport.analyzed_duration_seconds must be zero or positive.",
        )
        _require(0.0 <= self.speech_ratio <= 1.0, "VoiceActivityReport.speech_ratio must be between 0 and 1.")
        _require(self.segment_count >= 0, "VoiceActivityReport.segment_count must be zero or positive.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VoiceActivityReport:
        return cls(
            source_path=str(data.get("source_path", "")),
            analyzer_backend=str(data.get("analyzer_backend", "")),
            sample_rate_hz=int(data.get("sample_rate_hz", 0)),
            channel_count=int(data.get("channel_count", 0)),
            duration_seconds=float(data.get("duration_seconds", 0.0) or 0.0),
            analyzed_duration_seconds=float(data.get("analyzed_duration_seconds", 0.0) or 0.0),
            speech_ratio=float(data.get("speech_ratio", 0.0) or 0.0),
            segment_count=int(data.get("segment_count", 0)),
            segments=tuple(
                VoiceActivitySegment.from_dict(item) for item in data.get("segments", ())
            ),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class AudioTranscriptionResult(DictSerializable):
    """Typed result of one bounded local audio-input transcription request."""

    source_path: str = ""
    status: str = "idle"
    transcript_text: str = ""
    normalized_question: str = ""
    transcription_backend: str = ""
    transcription_model: str = ""
    used_vad: bool = False
    imported_into_question: bool = False
    duration_seconds: float = 0.0
    analyzed_duration_seconds: float = 0.0
    voice_activity: VoiceActivityReport = field(default_factory=VoiceActivityReport)
    warnings: tuple[str, ...] = ()
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.status.strip()), "AudioTranscriptionResult.status must not be empty.")
        _require(self.duration_seconds >= 0.0, "AudioTranscriptionResult.duration_seconds must be zero or positive.")
        _require(
            self.analyzed_duration_seconds >= 0.0,
            "AudioTranscriptionResult.analyzed_duration_seconds must be zero or positive.",
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AudioTranscriptionResult:
        return cls(
            source_path=str(data.get("source_path", "")),
            status=str(data.get("status", "idle")),
            transcript_text=str(data.get("transcript_text", "")),
            normalized_question=str(data.get("normalized_question", "")),
            transcription_backend=str(data.get("transcription_backend", "")),
            transcription_model=str(data.get("transcription_model", "")),
            used_vad=bool(data.get("used_vad", False)),
            imported_into_question=bool(data.get("imported_into_question", False)),
            duration_seconds=float(data.get("duration_seconds", 0.0) or 0.0),
            analyzed_duration_seconds=float(data.get("analyzed_duration_seconds", 0.0) or 0.0),
            voice_activity=VoiceActivityReport.from_dict(data.get("voice_activity", {})),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class AudioSynthesisResult(DictSerializable):
    """Typed result of one bounded local text-to-speech request."""

    target_path: str = ""
    status: str = "idle"
    source_text: str = ""
    clipped_text: str = ""
    synthesis_backend: str = ""
    synthesis_model: str = ""
    duration_seconds: float = 0.0
    sample_rate_hz: int = 0
    warnings: tuple[str, ...] = ()
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.status.strip()), "AudioSynthesisResult.status must not be empty.")
        _require(self.duration_seconds >= 0.0, "AudioSynthesisResult.duration_seconds must be zero or positive.")
        _require(self.sample_rate_hz >= 0, "AudioSynthesisResult.sample_rate_hz must be zero or positive.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AudioSynthesisResult:
        return cls(
            target_path=str(data.get("target_path", "")),
            status=str(data.get("status", "idle")),
            source_text=str(data.get("source_text", "")),
            clipped_text=str(data.get("clipped_text", "")),
            synthesis_backend=str(data.get("synthesis_backend", "")),
            synthesis_model=str(data.get("synthesis_model", "")),
            duration_seconds=float(data.get("duration_seconds", 0.0) or 0.0),
            sample_rate_hz=int(data.get("sample_rate_hz", 0)),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class TextTranslationResult(DictSerializable):
    """Typed result of one bounded local text-translation request."""

    status: str = "idle"
    source_text: str = ""
    translated_text: str = ""
    source_language: str = ""
    target_language: str = ""
    translation_backend: str = ""
    translation_model: str = ""
    source_scope: str = "free_text"
    imported_into_question: bool = False
    warnings: tuple[str, ...] = ()
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.status.strip()), "TextTranslationResult.status must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TextTranslationResult:
        return cls(
            status=str(data.get("status", "idle")),
            source_text=str(data.get("source_text", "")),
            translated_text=str(data.get("translated_text", "")),
            source_language=str(data.get("source_language", "")),
            target_language=str(data.get("target_language", "")),
            translation_backend=str(data.get("translation_backend", "")),
            translation_model=str(data.get("translation_model", "")),
            source_scope=str(data.get("source_scope", "free_text")),
            imported_into_question=bool(data.get("imported_into_question", False)),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CodeSpecialistResult(DictSerializable):
    """Typed result of one bounded optional code-specialist request."""

    status: str = "idle"
    source_scope: str = "snippet"
    source_path: str = ""
    request_text: str = ""
    summary: str = ""
    suggested_actions: tuple[str, ...] = ()
    code_backend: str = ""
    code_model: str = ""
    detected_language: str = ""
    line_count: int = 0
    warnings: tuple[str, ...] = ()
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.status.strip()), "CodeSpecialistResult.status must not be empty.")
        _require(self.line_count >= 0, "CodeSpecialistResult.line_count must be zero or positive.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CodeSpecialistResult:
        return cls(
            status=str(data.get("status", "idle")),
            source_scope=str(data.get("source_scope", "snippet")),
            source_path=str(data.get("source_path", "")),
            request_text=str(data.get("request_text", "")),
            summary=str(data.get("summary", "")),
            suggested_actions=tuple(str(item) for item in data.get("suggested_actions", ())),
            code_backend=str(data.get("code_backend", "")),
            code_model=str(data.get("code_model", "")),
            detected_language=str(data.get("detected_language", "")),
            line_count=int(data.get("line_count", 0)),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CodeQualityReport(DictSerializable):
    """Machine-readable bounded validation summary for generated or reviewed code."""

    tests_passed: bool = False
    lint_passed: bool = False
    complexity_passed: bool = False
    security_passed: bool = False
    maintainability_passed: bool = False
    critique_passed: bool = False
    regression_passed: bool = False
    overall_passed: bool = False
    quality_score: float = 0.0
    findings: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(0.0 <= float(self.quality_score) <= 1.0, "CodeQualityReport.quality_score must be between 0 and 1.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CodeQualityReport:
        raw_metrics = dict(data.get("metrics", {}))
        metrics: dict[str, float] = {}
        for key, value in raw_metrics.items():
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return cls(
            tests_passed=bool(data.get("tests_passed", False)),
            lint_passed=bool(data.get("lint_passed", False)),
            complexity_passed=bool(data.get("complexity_passed", False)),
            security_passed=bool(data.get("security_passed", False)),
            maintainability_passed=bool(data.get("maintainability_passed", False)),
            critique_passed=bool(data.get("critique_passed", False)),
            regression_passed=bool(data.get("regression_passed", False)),
            overall_passed=bool(data.get("overall_passed", False)),
            quality_score=float(data.get("quality_score", 0.0) or 0.0),
            findings=tuple(str(item) for item in data.get("findings", ())),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
            metrics=metrics,
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CodingTaskArtifact(DictSerializable):
    """One bounded artifact emitted by a coding task or practice cycle."""

    artifact_id: str = ""
    artifact_type: str = "code"
    title: str = ""
    language: str = ""
    path: str = ""
    content_preview: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CodingTaskArtifact:
        return cls(
            artifact_id=str(data.get("artifact_id", "")),
            artifact_type=str(data.get("artifact_type", "code")),
            title=str(data.get("title", "")),
            language=str(data.get("language", "")),
            path=str(data.get("path", "")),
            content_preview=str(data.get("content_preview", "")),
            metadata=dict(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CodingPatternValidation(DictSerializable):
    """Validation history entry attached to one learned coding pattern."""

    validation_id: str = ""
    checks_passed: tuple[str, ...] = ()
    checks_failed: tuple[str, ...] = ()
    reviewer_summary: str = ""
    quality_report: CodeQualityReport = field(default_factory=CodeQualityReport)
    validated_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CodingPatternValidation:
        return cls(
            validation_id=str(data.get("validation_id", "")),
            checks_passed=tuple(str(item) for item in data.get("checks_passed", ())),
            checks_failed=tuple(str(item) for item in data.get("checks_failed", ())),
            reviewer_summary=str(data.get("reviewer_summary", "")),
            quality_report=CodeQualityReport.from_dict(data.get("quality_report", {})),
            validated_at=_parse_datetime(data.get("validated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CodingPattern(DictSerializable):
    """Structured local coding-memory pattern stored with gated promotion tiers."""

    pattern_id: str = ""
    title: str = ""
    summary: str = ""
    tier: CodingPatternTier = CodingPatternTier.CANDIDATE
    category: str = "good_practice"
    language: str = ""
    framework: str = ""
    task_type: CodingTaskType = CodingTaskType.CODE_REVIEW
    source: str = ""
    quality_score: float = 0.0
    reuse_count: int = 0
    code_snippet: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    validation_history: tuple[CodingPatternValidation, ...] = ()
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(0.0 <= float(self.quality_score) <= 1.0, "CodingPattern.quality_score must be between 0 and 1.")
        _require(self.reuse_count >= 0, "CodingPattern.reuse_count must be zero or positive.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CodingPattern:
        return cls(
            pattern_id=str(data.get("pattern_id", "")),
            title=str(data.get("title", "")),
            summary=str(data.get("summary", "")),
            tier=_parse_enum(CodingPatternTier, data.get("tier", CodingPatternTier.CANDIDATE)),
            category=str(data.get("category", "good_practice")),
            language=str(data.get("language", "")),
            framework=str(data.get("framework", "")),
            task_type=_parse_enum(CodingTaskType, data.get("task_type", CodingTaskType.CODE_REVIEW)),
            source=str(data.get("source", "")),
            quality_score=float(data.get("quality_score", 0.0) or 0.0),
            reuse_count=int(data.get("reuse_count", 0)),
            code_snippet=str(data.get("code_snippet", "")),
            metadata=dict(data.get("metadata", {})),
            validation_history=tuple(
                CodingPatternValidation.from_dict(item) for item in data.get("validation_history", ())
            ),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CodingTaskRequest(DictSerializable):
    """One bounded Coding Mode request."""

    request_id: str = ""
    task_type: CodingTaskType = CodingTaskType.CODE_REVIEW
    prompt: str = ""
    language: str = "python"
    framework: str = ""
    source_scope: str = "snippet"
    source_path: str = ""
    source_text: str = ""
    tests_text: str = ""
    target_paths: tuple[str, ...] = ()
    idle_practice: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CodingTaskRequest:
        return cls(
            request_id=str(data.get("request_id", "")),
            task_type=_parse_enum(CodingTaskType, data.get("task_type", CodingTaskType.CODE_REVIEW)),
            prompt=str(data.get("prompt", "")),
            language=str(data.get("language", "python")),
            framework=str(data.get("framework", "")),
            source_scope=str(data.get("source_scope", "snippet")),
            source_path=str(data.get("source_path", "")),
            source_text=str(data.get("source_text", "")),
            tests_text=str(data.get("tests_text", "")),
            target_paths=tuple(str(item) for item in data.get("target_paths", ())),
            idle_practice=bool(data.get("idle_practice", False)),
            metadata=dict(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class CodingTaskResult(DictSerializable):
    """Typed result for one Coding Mode request."""

    request_id: str = ""
    task_type: CodingTaskType = CodingTaskType.CODE_REVIEW
    status: str = "idle"
    active_phase: str = "idle"
    prompt: str = ""
    summary: str = ""
    language: str = ""
    framework: str = ""
    source_scope: str = "snippet"
    role_assignments: dict[str, str] = field(default_factory=dict)
    route_summary: tuple[str, ...] = ()
    artifacts: tuple[CodingTaskArtifact, ...] = ()
    quality_report: CodeQualityReport = field(default_factory=CodeQualityReport)
    verified_patterns: tuple[str, ...] = ()
    candidate_patterns: tuple[str, ...] = ()
    rejected_patterns: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    practice_session_id: str = ""
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.status.strip()), "CodingTaskResult.status must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CodingTaskResult:
        return cls(
            request_id=str(data.get("request_id", "")),
            task_type=_parse_enum(CodingTaskType, data.get("task_type", CodingTaskType.CODE_REVIEW)),
            status=str(data.get("status", "idle")),
            active_phase=str(data.get("active_phase", "idle")),
            prompt=str(data.get("prompt", "")),
            summary=str(data.get("summary", "")),
            language=str(data.get("language", "")),
            framework=str(data.get("framework", "")),
            source_scope=str(data.get("source_scope", "snippet")),
            role_assignments={str(key): str(value) for key, value in dict(data.get("role_assignments", {})).items()},
            route_summary=tuple(str(item) for item in data.get("route_summary", ())),
            artifacts=tuple(CodingTaskArtifact.from_dict(item) for item in data.get("artifacts", ())),
            quality_report=CodeQualityReport.from_dict(data.get("quality_report", {})),
            verified_patterns=tuple(str(item) for item in data.get("verified_patterns", ())),
            candidate_patterns=tuple(str(item) for item in data.get("candidate_patterns", ())),
            rejected_patterns=tuple(str(item) for item in data.get("rejected_patterns", ())),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
            practice_session_id=str(data.get("practice_session_id", "")),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class PracticeSessionResult(DictSerializable):
    """Typed bounded result for one idle Coding Dojo practice cycle."""

    session_id: str = ""
    status: str = "idle"
    task_type: CodingTaskType = CodingTaskType.PRACTICE
    prompt: str = ""
    language: str = "python"
    summary: str = ""
    quality_score: float = 0.0
    validated_patterns: tuple[str, ...] = ()
    rejected_patterns: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    task_result: CodingTaskResult = field(default_factory=CodingTaskResult)
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        _require(0.0 <= float(self.quality_score) <= 1.0, "PracticeSessionResult.quality_score must be between 0 and 1.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PracticeSessionResult:
        raw_completed_at = data.get("completed_at")
        return cls(
            session_id=str(data.get("session_id", "")),
            status=str(data.get("status", "idle")),
            task_type=_parse_enum(CodingTaskType, data.get("task_type", CodingTaskType.PRACTICE)),
            prompt=str(data.get("prompt", "")),
            language=str(data.get("language", "python")),
            summary=str(data.get("summary", "")),
            quality_score=float(data.get("quality_score", 0.0) or 0.0),
            validated_patterns=tuple(str(item) for item in data.get("validated_patterns", ())),
            rejected_patterns=tuple(str(item) for item in data.get("rejected_patterns", ())),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
            task_result=CodingTaskResult.from_dict(data.get("task_result", {})),
            started_at=_parse_datetime(data.get("started_at", utc_now())),
            completed_at=None if raw_completed_at in (None, "") else _parse_datetime(raw_completed_at),
        )


@dataclass(slots=True, frozen=True)
class VisionInspectionResult(DictSerializable):
    """Typed result of one bounded optional visual-role inspection request."""

    status: str = "idle"
    source_path: str = ""
    request_text: str = ""
    role: ModelRole = ModelRole.VISION
    inspection_backend: str = ""
    inspection_model: str = ""
    summary: str = ""
    extracted_text: str = ""
    warnings: tuple[str, ...] = ()
    degraded_reason: str = ""
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.status.strip()), "VisionInspectionResult.status must not be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VisionInspectionResult:
        return cls(
            status=str(data.get("status", "idle")),
            source_path=str(data.get("source_path", "")),
            request_text=str(data.get("request_text", "")),
            role=_parse_enum(ModelRole, data.get("role", ModelRole.VISION)),
            inspection_backend=str(data.get("inspection_backend", "")),
            inspection_model=str(data.get("inspection_model", "")),
            summary=str(data.get("summary", "")),
            extracted_text=str(data.get("extracted_text", "")),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
            degraded_reason=str(data.get("degraded_reason", "")),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class DashboardLocalTaskSessionState(DictSerializable):
    """Read-only dashboard projection for the active local task execution session."""

    session_id: str = ""
    label: str = ""
    profile_name: str = ""
    status: str = "inactive"
    control_mode: str = "local_task"
    current_target: str = ""
    last_action_summary: str = ""
    last_request_id: str = ""
    continuous_capture_active: bool = False
    continuous_capture_directory: str = ""
    continuous_capture_frame_count: int = 0
    continuous_capture_retained_frame_count: int = 0
    continuous_capture_last_frame_path: str = ""
    continuous_capture_region: str = "full_screen"
    continuous_capture_fps: float = 0.0
    continuous_capture_max_width: int = 0
    continuous_capture_max_height: int = 0
    continuous_capture_last_diff_ratio: float = 0.0
    continuous_capture_warnings: tuple[str, ...] = ()
    continuous_capture_last_capture_at: datetime | None = None
    requested_observation_tier: str = "screenshot_on_demand"
    effective_observation_tier: str = "screenshot_on_demand"
    observation_degraded_reason: str = ""
    observation_degraded_features: tuple[str, ...] = ()
    last_observation_tier: str = ""
    last_observation_status: str = ""
    last_observation_summary: str = ""
    last_observation_output_ref: str = ""
    last_observation_text_preview: str = ""
    last_observation_backend: str = ""
    last_observation_warnings: tuple[str, ...] = ()
    last_observation_at: datetime | None = None
    pending_approval_summaries: tuple[str, ...] = ()
    pause_requested: bool = False
    stop_requested: bool = False
    kill_switch_engaged: bool = False
    last_control_reason: str = ""
    last_error: str = ""
    created_at: datetime | None = None
    updated_at: datetime = field(default_factory=utc_now)
    ended_at: datetime | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DashboardLocalTaskSessionState:
        raw_continuous_capture_last_capture_at = data.get("continuous_capture_last_capture_at")
        raw_last_observation_at = data.get("last_observation_at")
        raw_created_at = data.get("created_at")
        raw_ended_at = data.get("ended_at")
        return cls(
            session_id=str(data.get("session_id", "")),
            label=str(data.get("label", "")),
            profile_name=str(data.get("profile_name", "")),
            status=str(data.get("status", "inactive")),
            control_mode=str(data.get("control_mode", "local_task")),
            current_target=str(data.get("current_target", "")),
            last_action_summary=str(data.get("last_action_summary", "")),
            last_request_id=str(data.get("last_request_id", "")),
            continuous_capture_active=bool(data.get("continuous_capture_active", False)),
            continuous_capture_directory=str(data.get("continuous_capture_directory", "")),
            continuous_capture_frame_count=int(data.get("continuous_capture_frame_count", 0)),
            continuous_capture_retained_frame_count=int(data.get("continuous_capture_retained_frame_count", 0)),
            continuous_capture_last_frame_path=str(data.get("continuous_capture_last_frame_path", "")),
            continuous_capture_region=str(data.get("continuous_capture_region", "full_screen")),
            continuous_capture_fps=float(data.get("continuous_capture_fps", 0.0)),
            continuous_capture_max_width=int(data.get("continuous_capture_max_width", 0)),
            continuous_capture_max_height=int(data.get("continuous_capture_max_height", 0)),
            continuous_capture_last_diff_ratio=float(data.get("continuous_capture_last_diff_ratio", 0.0)),
            continuous_capture_warnings=tuple(
                str(item) for item in data.get("continuous_capture_warnings", ())
            ),
            continuous_capture_last_capture_at=(
                None
                if raw_continuous_capture_last_capture_at in (None, "")
                else _parse_datetime(raw_continuous_capture_last_capture_at)
            ),
            requested_observation_tier=str(
                data.get("requested_observation_tier", "screenshot_on_demand")
            ),
            effective_observation_tier=str(
                data.get("effective_observation_tier", "screenshot_on_demand")
            ),
            observation_degraded_reason=str(data.get("observation_degraded_reason", "")),
            observation_degraded_features=tuple(
                str(item) for item in data.get("observation_degraded_features", ())
            ),
            last_observation_tier=str(data.get("last_observation_tier", "")),
            last_observation_status=str(data.get("last_observation_status", "")),
            last_observation_summary=str(data.get("last_observation_summary", "")),
            last_observation_output_ref=str(data.get("last_observation_output_ref", "")),
            last_observation_text_preview=str(data.get("last_observation_text_preview", "")),
            last_observation_backend=str(data.get("last_observation_backend", "")),
            last_observation_warnings=tuple(
                str(item) for item in data.get("last_observation_warnings", ())
            ),
            last_observation_at=(
                None if raw_last_observation_at in (None, "") else _parse_datetime(raw_last_observation_at)
            ),
            pending_approval_summaries=tuple(
                str(item) for item in data.get("pending_approval_summaries", ())
            ),
            pause_requested=bool(data.get("pause_requested", False)),
            stop_requested=bool(data.get("stop_requested", False)),
            kill_switch_engaged=bool(data.get("kill_switch_engaged", False)),
            last_control_reason=str(data.get("last_control_reason", "")),
            last_error=str(data.get("last_error", "")),
            created_at=None if raw_created_at in (None, "") else _parse_datetime(raw_created_at),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
            ended_at=None if raw_ended_at in (None, "") else _parse_datetime(raw_ended_at),
        )


@dataclass(slots=True, frozen=True)
class ActivityChip(DictSerializable):
    """Compact shell activity badge derived from dashboard and runtime state."""

    chip_id: str = ""
    label: str = ""
    tone: str = "neutral"
    detail: str = ""
    active: bool = True
    created_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActivityChip:
        return cls(
            chip_id=str(data.get("chip_id", "")),
            label=str(data.get("label", "")),
            tone=str(data.get("tone", "neutral")),
            detail=str(data.get("detail", "")),
            active=bool(data.get("active", True)),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class TimelineEntry(DictSerializable):
    """One bounded timeline entry shown in the shell task tray."""

    entry_id: str = ""
    label: str = ""
    stage: str = ""
    detail: str = ""
    severity: str = "info"
    created_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TimelineEntry:
        return cls(
            entry_id=str(data.get("entry_id", "")),
            label=str(data.get("label", "")),
            stage=str(data.get("stage", "")),
            detail=str(data.get("detail", "")),
            severity=str(data.get("severity", "info")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class ShellNotification(DictSerializable):
    """Operator-facing shell notification kept separate from raw notices."""

    notification_id: str = ""
    message: str = ""
    severity: str = "info"
    created_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShellNotification:
        return cls(
            notification_id=str(data.get("notification_id", "")),
            message=str(data.get("message", "")),
            severity=str(data.get("severity", "info")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class ConversationItem(DictSerializable):
    """Presentation-only conversation/task card shown in the shell center pane."""

    item_id: str = ""
    role: str = "system"
    title: str = ""
    body: str = ""
    status: str = ""
    chips: tuple[str, ...] = ()
    task_id: str = ""
    created_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConversationItem:
        return cls(
            item_id=str(data.get("item_id", "")),
            role=str(data.get("role", "system")),
            title=str(data.get("title", "")),
            body=str(data.get("body", "")),
            status=str(data.get("status", "")),
            chips=tuple(str(item) for item in data.get("chips", ())),
            task_id=str(data.get("task_id", "")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class OrbEffectState(DictSerializable):
    """Bounded transient orb overlays derived from recent runtime events."""

    transient_effects: tuple[str, ...] = ()
    active_overlay: str = ""
    insight_flash_pending: bool = False
    consensus_shimmer_pending: bool = False
    verification_lock_pending: bool = False
    checkpoint_pulse_pending: bool = False
    approval_hold: bool = False
    degraded_undertone: bool = False
    updated_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OrbEffectState:
        return cls(
            transient_effects=tuple(str(item) for item in data.get("transient_effects", ())),
            active_overlay=str(data.get("active_overlay", "")),
            insight_flash_pending=bool(data.get("insight_flash_pending", False)),
            consensus_shimmer_pending=bool(data.get("consensus_shimmer_pending", False)),
            verification_lock_pending=bool(data.get("verification_lock_pending", False)),
            checkpoint_pulse_pending=bool(data.get("checkpoint_pulse_pending", False)),
            approval_hold=bool(data.get("approval_hold", False)),
            degraded_undertone=bool(data.get("degraded_undertone", False)),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class ShellState(DictSerializable):
    """Typed projection between backend state and premium shell visuals."""

    orb_mode: str = "offline"
    orb_palette: str = "slate_blue"
    orb_intensity: float = 0.12
    ring_mode: str = "dormant"
    particle_mode: str = "sparse"
    ambient_mode: str = "dormant"
    status_text: str = "Offline"
    sub_status_text: str = "Local shell is not ready."
    active_agent: str = ""
    secondary_agents: tuple[str, ...] = ()
    active_tools: tuple[str, ...] = ()
    active_roles: tuple[str, ...] = ()
    confidence_band: str = "low"
    verifier_state: str = "idle"
    retrieval_state: str = "idle"
    compression_state: str = "idle"
    optimizer_state: str = "idle"
    coding_state: str = "idle"
    workspace_mode: str = "assistant"
    active_route_summary: tuple[str, ...] = ()
    active_model_roles: tuple[str, ...] = ()
    candidate_count: int = 0
    evidence_count: int = 0
    elapsed_seconds: float = 0.0
    current_file: str = ""
    current_project: str = ""
    sandbox_state: str = "idle"
    quality_gate_state: str = "idle"
    pattern_tier_counts: dict[str, int] = field(default_factory=dict)
    practice_session_state: str = "idle"
    approval_prompt_summary: str = ""
    resource_ribbon_flags: tuple[str, ...] = ()
    panel_visibility_state: dict[str, bool] = field(default_factory=dict)
    hero_metric_strip: tuple[str, ...] = ()
    long_horizon_state: str = ""
    checkpoint_count: int = 0
    degraded_reason: str = ""
    fallback_reason: str = ""
    resource_pressure_level: str = "nominal"
    speaking_state: str = "idle"
    approval_pending: bool = False
    capability_session_state: str = "inactive"
    observation_tier: str = "screenshot_on_demand"
    cloud_helper_state: str = "disabled"
    activity_chips: tuple[ActivityChip, ...] = ()
    timeline_entries: tuple[TimelineEntry, ...] = ()
    shell_notifications: tuple[ShellNotification, ...] = ()
    conversation_items: tuple[ConversationItem, ...] = ()
    orb_effects: OrbEffectState = field(default_factory=OrbEffectState)
    current_task_summary: str = ""
    updated_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShellState:
        return cls(
            orb_mode=str(data.get("orb_mode", "offline")),
            orb_palette=str(data.get("orb_palette", "slate_blue")),
            orb_intensity=float(data.get("orb_intensity", 0.12) or 0.12),
            ring_mode=str(data.get("ring_mode", "dormant")),
            particle_mode=str(data.get("particle_mode", "sparse")),
            ambient_mode=str(data.get("ambient_mode", "dormant")),
            status_text=str(data.get("status_text", "Offline")),
            sub_status_text=str(data.get("sub_status_text", "Local shell is not ready.")),
            active_agent=str(data.get("active_agent", "")),
            secondary_agents=tuple(str(item) for item in data.get("secondary_agents", ())),
            active_tools=tuple(str(item) for item in data.get("active_tools", ())),
            active_roles=tuple(str(item) for item in data.get("active_roles", ())),
            confidence_band=str(data.get("confidence_band", "low")),
            verifier_state=str(data.get("verifier_state", "idle")),
            retrieval_state=str(data.get("retrieval_state", "idle")),
            compression_state=str(data.get("compression_state", "idle")),
            optimizer_state=str(data.get("optimizer_state", "idle")),
            coding_state=str(data.get("coding_state", "idle")),
            workspace_mode=str(data.get("workspace_mode", "assistant")),
            active_route_summary=tuple(str(item) for item in data.get("active_route_summary", ())),
            active_model_roles=tuple(str(item) for item in data.get("active_model_roles", ())),
            candidate_count=int(data.get("candidate_count", 0)),
            evidence_count=int(data.get("evidence_count", 0)),
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0) or 0.0),
            current_file=str(data.get("current_file", "")),
            current_project=str(data.get("current_project", "")),
            sandbox_state=str(data.get("sandbox_state", "idle")),
            quality_gate_state=str(data.get("quality_gate_state", "idle")),
            pattern_tier_counts={
                str(key): int(value)
                for key, value in dict(data.get("pattern_tier_counts", {})).items()
            },
            practice_session_state=str(data.get("practice_session_state", "idle")),
            approval_prompt_summary=str(data.get("approval_prompt_summary", "")),
            resource_ribbon_flags=tuple(str(item) for item in data.get("resource_ribbon_flags", ())),
            panel_visibility_state={
                str(key): bool(value)
                for key, value in dict(data.get("panel_visibility_state", {})).items()
            },
            hero_metric_strip=tuple(str(item) for item in data.get("hero_metric_strip", ())),
            long_horizon_state=str(data.get("long_horizon_state", "")),
            checkpoint_count=int(data.get("checkpoint_count", 0)),
            degraded_reason=str(data.get("degraded_reason", "")),
            fallback_reason=str(data.get("fallback_reason", "")),
            resource_pressure_level=str(data.get("resource_pressure_level", "nominal")),
            speaking_state=str(data.get("speaking_state", "idle")),
            approval_pending=bool(data.get("approval_pending", False)),
            capability_session_state=str(data.get("capability_session_state", "inactive")),
            observation_tier=str(data.get("observation_tier", "screenshot_on_demand")),
            cloud_helper_state=str(data.get("cloud_helper_state", "disabled")),
            activity_chips=tuple(ActivityChip.from_dict(item) for item in data.get("activity_chips", ())),
            timeline_entries=tuple(TimelineEntry.from_dict(item) for item in data.get("timeline_entries", ())),
            shell_notifications=tuple(
                ShellNotification.from_dict(item) for item in data.get("shell_notifications", ())
            ),
            conversation_items=tuple(
                ConversationItem.from_dict(item) for item in data.get("conversation_items", ())
            ),
            orb_effects=OrbEffectState.from_dict(data.get("orb_effects", {})),
            current_task_summary=str(data.get("current_task_summary", "")),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


@dataclass(slots=True, frozen=True)
class DashboardAppState(DictSerializable):
    """Typed read-only snapshot consumed by the dashboard shell."""

    last_stage: str = ""
    event_count: int = 0
    dropped_events: int = 0
    active_task: DashboardTaskState = field(default_factory=DashboardTaskState)
    local_task_session: DashboardLocalTaskSessionState = field(default_factory=DashboardLocalTaskSessionState)
    runtime_health: DashboardRuntimeHealth = field(default_factory=DashboardRuntimeHealth)
    statuses: dict[str, AgentStatus] = field(default_factory=dict)
    recent_conditions: tuple[RuntimeCondition, ...] = ()
    user_settings: UserSettingsProfile = field(default_factory=UserSettingsProfile)
    settings_profiles: tuple[UserSettingsProfile, ...] = ()
    task_history: tuple[DashboardTaskHistoryEntry, ...] = ()
    selected_task: DashboardTaskInspector = field(default_factory=DashboardTaskInspector)
    knowledge_sources: tuple[DashboardKnowledgeSource, ...] = ()
    readiness_report: DashboardReadinessReport = field(default_factory=DashboardReadinessReport)
    capability_registry_view: CapabilityRegistryView = field(default_factory=CapabilityRegistryView)
    model_registry_view: ModelRegistryView = field(default_factory=ModelRegistryView)
    model_role_action: ModelRoleActionReport = field(default_factory=ModelRoleActionReport)
    demo_pack_status: DemoPackStatus = field(default_factory=DemoPackStatus)
    sample_tasks: tuple[SampleTaskDefinition, ...] = ()
    selected_sample_task: SampleTaskDefinition = field(default_factory=SampleTaskDefinition)
    audio_input: AudioTranscriptionResult = field(default_factory=AudioTranscriptionResult)
    audio_output: AudioSynthesisResult = field(default_factory=AudioSynthesisResult)
    translation_output: TextTranslationResult = field(default_factory=TextTranslationResult)
    code_output: CodeSpecialistResult = field(default_factory=CodeSpecialistResult)
    coding_output: CodingTaskResult = field(default_factory=CodingTaskResult)
    coding_practice: PracticeSessionResult = field(default_factory=PracticeSessionResult)
    coding_patterns: tuple[CodingPattern, ...] = ()
    last_notice: str = ""
    last_notice_severity: str = "info"
    updated_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DashboardAppState:
        raw_statuses = data.get("statuses", {})
        return cls(
            last_stage=str(data.get("last_stage", "")),
            event_count=int(data.get("event_count", 0)),
            dropped_events=int(data.get("dropped_events", 0)),
            active_task=DashboardTaskState.from_dict(data.get("active_task", {})),
            local_task_session=DashboardLocalTaskSessionState.from_dict(data.get("local_task_session", {})),
            runtime_health=DashboardRuntimeHealth.from_dict(data.get("runtime_health", {})),
            statuses={
                str(key): AgentStatus.from_dict(value)
                for key, value in dict(raw_statuses).items()
            },
            recent_conditions=tuple(
                RuntimeCondition.from_dict(item) for item in data.get("recent_conditions", ())
            ),
            user_settings=UserSettingsProfile.from_dict(data.get("user_settings", {})),
            settings_profiles=tuple(
                UserSettingsProfile.from_dict(item) for item in data.get("settings_profiles", ())
            ),
            task_history=tuple(
                DashboardTaskHistoryEntry.from_dict(item) for item in data.get("task_history", ())
            ),
            selected_task=DashboardTaskInspector.from_dict(data.get("selected_task", {})),
            knowledge_sources=tuple(
                DashboardKnowledgeSource.from_dict(item)
                for item in data.get("knowledge_sources", ())
            ),
            readiness_report=DashboardReadinessReport.from_dict(
                data.get("readiness_report", {})
            ),
            capability_registry_view=CapabilityRegistryView.from_dict(
                data.get("capability_registry_view", {})
            ),
            model_registry_view=ModelRegistryView.from_dict(data.get("model_registry_view", {})),
            model_role_action=ModelRoleActionReport.from_dict(data.get("model_role_action", {})),
            demo_pack_status=DemoPackStatus.from_dict(data.get("demo_pack_status", {})),
            sample_tasks=tuple(
                SampleTaskDefinition.from_dict(item) for item in data.get("sample_tasks", ())
            ),
            selected_sample_task=SampleTaskDefinition.from_dict(
                data.get("selected_sample_task", {})
            ),
            audio_input=AudioTranscriptionResult.from_dict(data.get("audio_input", {})),
            audio_output=AudioSynthesisResult.from_dict(data.get("audio_output", {})),
            translation_output=TextTranslationResult.from_dict(data.get("translation_output", {})),
            code_output=CodeSpecialistResult.from_dict(data.get("code_output", {})),
            coding_output=CodingTaskResult.from_dict(data.get("coding_output", {})),
            coding_practice=PracticeSessionResult.from_dict(data.get("coding_practice", {})),
            coding_patterns=tuple(CodingPattern.from_dict(item) for item in data.get("coding_patterns", ())),
            last_notice=str(data.get("last_notice", "")),
            last_notice_severity=str(data.get("last_notice_severity", "info")),
            updated_at=_parse_datetime(data.get("updated_at", utc_now())),
        )


def coerce_plan(value: Plan | Mapping[str, Any]) -> Plan:
    """Convert mapping payloads to Plan while preserving existing Plan values."""
    if isinstance(value, Plan):
        return value
    return Plan.from_dict(value)


def coerce_evidence_bundle(value: EvidenceBundle | Mapping[str, Any]) -> EvidenceBundle:
    """Convert mapping payloads to EvidenceBundle while preserving existing values."""
    if isinstance(value, EvidenceBundle):
        return value
    return EvidenceBundle.from_dict(value)


def coerce_opcode_entry(value: OpcodeEntry | Mapping[str, Any]) -> OpcodeEntry:
    """Convert mapping payloads to OpcodeEntry while preserving existing values."""
    if isinstance(value, OpcodeEntry):
        return value
    return OpcodeEntry.from_dict(value)


def coerce_decoder_entry(value: DecoderEntry | Mapping[str, Any]) -> DecoderEntry:
    """Convert mapping payloads to DecoderEntry while preserving existing values."""
    if isinstance(value, DecoderEntry):
        return value
    return DecoderEntry.from_dict(value)


def coerce_symbol_table_snapshot(
    value: SymbolTableSnapshot | Mapping[str, Any],
) -> SymbolTableSnapshot:
    """Convert mapping payloads to SymbolTableSnapshot while preserving existing values."""
    if isinstance(value, SymbolTableSnapshot):
        return value
    return SymbolTableSnapshot.from_dict(value)


def coerce_proof_hash_record(value: ProofHashRecord | Mapping[str, Any]) -> ProofHashRecord:
    """Convert mapping payloads to ProofHashRecord while preserving existing values."""
    if isinstance(value, ProofHashRecord):
        return value
    return ProofHashRecord.from_dict(value)


def coerce_compression_runtime_subset(
    value: CompressionRuntimeSubset | Mapping[str, Any],
) -> CompressionRuntimeSubset:
    """Convert mapping payloads to CompressionRuntimeSubset while preserving existing values."""
    if isinstance(value, CompressionRuntimeSubset):
        return value
    return CompressionRuntimeSubset.from_dict(value)


def coerce_compressed_trace(value: CompressedTrace | Mapping[str, Any]) -> CompressedTrace:
    """Convert mapping payloads to CompressedTrace while preserving existing values."""
    if isinstance(value, CompressedTrace):
        return value
    return CompressedTrace.from_dict(value)


def coerce_research_reasoner_handoff(
    value: ResearchReasonerHandoff | Mapping[str, Any],
) -> ResearchReasonerHandoff:
    """Convert mapping payloads to ResearchReasonerHandoff while preserving existing values."""
    if isinstance(value, ResearchReasonerHandoff):
        return value
    return ResearchReasonerHandoff.from_dict(value)


def coerce_reasoner_critic_handoff(
    value: ReasonerCriticHandoff | Mapping[str, Any],
) -> ReasonerCriticHandoff:
    """Convert mapping payloads to ReasonerCriticHandoff while preserving existing values."""
    if isinstance(value, ReasonerCriticHandoff):
        return value
    return ReasonerCriticHandoff.from_dict(value)


def coerce_web_evidence_record(value: WebEvidenceRecord | Mapping[str, Any]) -> WebEvidenceRecord:
    """Convert mapping payloads to WebEvidenceRecord while preserving existing values."""
    if isinstance(value, WebEvidenceRecord):
        return value
    return WebEvidenceRecord.from_dict(value)


def coerce_critique_report(value: CritiqueReport | Mapping[str, Any]) -> CritiqueReport:
    """Convert mapping payloads to CritiqueReport while preserving existing values."""
    if isinstance(value, CritiqueReport):
        return value
    return CritiqueReport.from_dict(value)


def coerce_task_result(value: TaskResult | Mapping[str, Any]) -> TaskResult:
    """Convert mapping payloads to TaskResult while preserving existing values."""
    if isinstance(value, TaskResult):
        return value
    return TaskResult.from_dict(value)


def coerce_optimizer_replay_sample(
    value: OptimizerReplaySample | Mapping[str, Any],
) -> OptimizerReplaySample:
    """Convert mapping payloads to OptimizerReplaySample while preserving existing values."""
    if isinstance(value, OptimizerReplaySample):
        return value
    return OptimizerReplaySample.from_dict(value)


def coerce_optimizer_replay_evaluation(
    value: OptimizerReplayEvaluation | Mapping[str, Any],
) -> OptimizerReplayEvaluation:
    """Convert mapping payloads to OptimizerReplayEvaluation while preserving existing values."""
    if isinstance(value, OptimizerReplayEvaluation):
        return value
    return OptimizerReplayEvaluation.from_dict(value)


def coerce_optimizer_proposal_record(
    value: OptimizerProposalRecord | Mapping[str, Any],
) -> OptimizerProposalRecord:
    """Convert mapping payloads to OptimizerProposalRecord while preserving existing values."""
    if isinstance(value, OptimizerProposalRecord):
        return value
    return OptimizerProposalRecord.from_dict(value)


def coerce_macro_effectiveness_record(
    value: MacroEffectivenessRecord | Mapping[str, Any],
) -> MacroEffectivenessRecord:
    """Convert mapping payloads to MacroEffectivenessRecord while preserving existing values."""
    if isinstance(value, MacroEffectivenessRecord):
        return value
    return MacroEffectivenessRecord.from_dict(value)


def coerce_optimizer_activation_record(
    value: OptimizerActivationRecord | Mapping[str, Any],
) -> OptimizerActivationRecord:
    """Convert mapping payloads to OptimizerActivationRecord while preserving existing values."""
    if isinstance(value, OptimizerActivationRecord):
        return value
    return OptimizerActivationRecord.from_dict(value)


def coerce_optimizer_rollback_record(
    value: OptimizerRollbackRecord | Mapping[str, Any],
) -> OptimizerRollbackRecord:
    """Convert mapping payloads to OptimizerRollbackRecord while preserving existing values."""
    if isinstance(value, OptimizerRollbackRecord):
        return value
    return OptimizerRollbackRecord.from_dict(value)


def coerce_optimizer_suggestion_record(
    value: OptimizerSuggestionRecord | Mapping[str, Any],
) -> OptimizerSuggestionRecord:
    """Convert mapping payloads to OptimizerSuggestionRecord while preserving existing values."""
    if isinstance(value, OptimizerSuggestionRecord):
        return value
    return OptimizerSuggestionRecord.from_dict(value)


def coerce_optimizer_suggestion_usage_record(
    value: OptimizerSuggestionUsageRecord | Mapping[str, Any],
) -> OptimizerSuggestionUsageRecord:
    """Convert mapping payloads to OptimizerSuggestionUsageRecord while preserving existing values."""
    if isinstance(value, OptimizerSuggestionUsageRecord):
        return value
    return OptimizerSuggestionUsageRecord.from_dict(value)


def coerce_verified_deep_trace_export(
    value: VerifiedDeepTraceExport | Mapping[str, Any],
) -> VerifiedDeepTraceExport:
    """Convert mapping payloads to VerifiedDeepTraceExport while preserving existing values."""
    if isinstance(value, VerifiedDeepTraceExport):
        return value
    return VerifiedDeepTraceExport.from_dict(value)


def coerce_runtime_event(value: RuntimeEvent | Mapping[str, Any]) -> RuntimeEvent:
    """Convert mapping payloads to RuntimeEvent while preserving existing values."""
    if isinstance(value, RuntimeEvent):
        return value
    return RuntimeEvent.from_dict(value)


def coerce_agent_status(value: AgentStatus | Mapping[str, Any]) -> AgentStatus:
    """Convert mapping payloads to AgentStatus while preserving existing values."""
    if isinstance(value, AgentStatus):
        return value
    return AgentStatus.from_dict(value)


def coerce_runtime_condition(value: RuntimeCondition | Mapping[str, Any]) -> RuntimeCondition:
    """Convert mapping payloads to RuntimeCondition while preserving existing values."""
    if isinstance(value, RuntimeCondition):
        return value
    return RuntimeCondition.from_dict(value)


def coerce_long_horizon_candidate_snapshot(
    value: LongHorizonCandidateSnapshot | Mapping[str, Any],
) -> LongHorizonCandidateSnapshot:
    """Convert mapping payloads to LongHorizonCandidateSnapshot while preserving existing values."""
    if isinstance(value, LongHorizonCandidateSnapshot):
        return value
    return LongHorizonCandidateSnapshot.from_dict(value)


def coerce_long_horizon_checkpoint(
    value: LongHorizonCheckpoint | Mapping[str, Any],
) -> LongHorizonCheckpoint:
    """Convert mapping payloads to LongHorizonCheckpoint while preserving existing values."""
    if isinstance(value, LongHorizonCheckpoint):
        return value
    return LongHorizonCheckpoint.from_dict(value)


def coerce_long_horizon_session(
    value: LongHorizonSession | Mapping[str, Any],
) -> LongHorizonSession:
    """Convert mapping payloads to LongHorizonSession while preserving existing values."""
    if isinstance(value, LongHorizonSession):
        return value
    return LongHorizonSession.from_dict(value)


def coerce_file_operation_spec(
    value: FileOperationSpec | Mapping[str, Any],
) -> FileOperationSpec:
    """Convert mapping payloads to FileOperationSpec while preserving existing values."""
    if isinstance(value, FileOperationSpec):
        return value
    return FileOperationSpec.from_dict(value)


def coerce_shell_command_spec(
    value: ShellCommandSpec | Mapping[str, Any],
) -> ShellCommandSpec:
    """Convert mapping payloads to ShellCommandSpec while preserving existing values."""
    if isinstance(value, ShellCommandSpec):
        return value
    return ShellCommandSpec.from_dict(value)


def coerce_browser_action_spec(
    value: BrowserActionSpec | Mapping[str, Any],
) -> BrowserActionSpec:
    """Convert mapping payloads to BrowserActionSpec while preserving existing values."""
    if isinstance(value, BrowserActionSpec):
        return value
    return BrowserActionSpec.from_dict(value)


def coerce_app_focus_spec(
    value: AppFocusSpec | Mapping[str, Any],
) -> AppFocusSpec:
    """Convert mapping payloads to AppFocusSpec while preserving existing values."""
    if isinstance(value, AppFocusSpec):
        return value
    return AppFocusSpec.from_dict(value)


def coerce_clipboard_action_spec(
    value: ClipboardActionSpec | Mapping[str, Any],
) -> ClipboardActionSpec:
    """Convert mapping payloads to ClipboardActionSpec while preserving existing values."""
    if isinstance(value, ClipboardActionSpec):
        return value
    return ClipboardActionSpec.from_dict(value)


def coerce_screenshot_spec(
    value: ScreenshotSpec | Mapping[str, Any],
) -> ScreenshotSpec:
    """Convert mapping payloads to ScreenshotSpec while preserving existing values."""
    if isinstance(value, ScreenshotSpec):
        return value
    return ScreenshotSpec.from_dict(value)


def coerce_ocr_request_spec(
    value: OCRRequestSpec | Mapping[str, Any],
) -> OCRRequestSpec:
    """Convert mapping payloads to OCRRequestSpec while preserving existing values."""
    if isinstance(value, OCRRequestSpec):
        return value
    return OCRRequestSpec.from_dict(value)


def coerce_desktop_input_spec(
    value: DesktopInputSpec | Mapping[str, Any],
) -> DesktopInputSpec:
    """Convert mapping payloads to DesktopInputSpec while preserving existing values."""
    if isinstance(value, DesktopInputSpec):
        return value
    return DesktopInputSpec.from_dict(value)


def coerce_capability_request(
    value: CapabilityRequest | Mapping[str, Any],
) -> CapabilityRequest:
    """Convert mapping payloads to CapabilityRequest while preserving existing values."""
    if isinstance(value, CapabilityRequest):
        return value
    return CapabilityRequest.from_dict(value)


def coerce_capability_policy_decision(
    value: CapabilityPolicyDecision | Mapping[str, Any],
) -> CapabilityPolicyDecision:
    """Convert mapping payloads to CapabilityPolicyDecision while preserving existing values."""
    if isinstance(value, CapabilityPolicyDecision):
        return value
    return CapabilityPolicyDecision.from_dict(value)


def coerce_capability_registration(
    value: CapabilityRegistration | Mapping[str, Any],
) -> CapabilityRegistration:
    """Convert mapping payloads to CapabilityRegistration while preserving existing values."""
    if isinstance(value, CapabilityRegistration):
        return value
    return CapabilityRegistration.from_dict(value)


def coerce_capability_audit_record(
    value: CapabilityAuditRecord | Mapping[str, Any],
) -> CapabilityAuditRecord:
    """Convert mapping payloads to CapabilityAuditRecord while preserving existing values."""
    if isinstance(value, CapabilityAuditRecord):
        return value
    return CapabilityAuditRecord.from_dict(value)


def coerce_cloud_offload_record(
    value: CloudOffloadRecord | Mapping[str, Any],
) -> CloudOffloadRecord:
    """Convert mapping payloads to CloudOffloadRecord while preserving existing values."""
    if isinstance(value, CloudOffloadRecord):
        return value
    return CloudOffloadRecord.from_dict(value)


def coerce_capability_execution_result(
    value: CapabilityExecutionResult | Mapping[str, Any],
) -> CapabilityExecutionResult:
    """Convert mapping payloads to CapabilityExecutionResult while preserving existing values."""
    if isinstance(value, CapabilityExecutionResult):
        return value
    return CapabilityExecutionResult.from_dict(value)


def coerce_capability_registry_view(
    value: CapabilityRegistryView | Mapping[str, Any],
) -> CapabilityRegistryView:
    """Convert mapping payloads to CapabilityRegistryView while preserving existing values."""
    if isinstance(value, CapabilityRegistryView):
        return value
    return CapabilityRegistryView.from_dict(value)


def coerce_model_registration(
    value: ModelRegistration | Mapping[str, Any],
) -> ModelRegistration:
    """Convert mapping payloads to ModelRegistration while preserving existing values."""
    if isinstance(value, ModelRegistration):
        return value
    return ModelRegistration.from_dict(value)


def coerce_model_route_decision(
    value: ModelRouteDecision | Mapping[str, Any],
) -> ModelRouteDecision:
    """Convert mapping payloads to ModelRouteDecision while preserving existing values."""
    if isinstance(value, ModelRouteDecision):
        return value
    return ModelRouteDecision.from_dict(value)


def coerce_bounded_cache_snapshot(
    value: BoundedCacheSnapshot | Mapping[str, Any],
) -> BoundedCacheSnapshot:
    """Convert mapping payloads to BoundedCacheSnapshot while preserving existing values."""
    if isinstance(value, BoundedCacheSnapshot):
        return value
    return BoundedCacheSnapshot.from_dict(value)


def coerce_compression_insight_summary(
    value: CompressionInsightSummary | Mapping[str, Any],
) -> CompressionInsightSummary:
    """Convert mapping payloads to CompressionInsightSummary while preserving existing values."""
    if isinstance(value, CompressionInsightSummary):
        return value
    return CompressionInsightSummary.from_dict(value)


def coerce_model_registry_view(
    value: ModelRegistryView | Mapping[str, Any],
) -> ModelRegistryView:
    """Convert mapping payloads to ModelRegistryView while preserving existing values."""
    if isinstance(value, ModelRegistryView):
        return value
    return ModelRegistryView.from_dict(value)


def coerce_model_role_action_report(
    value: ModelRoleActionReport | Mapping[str, Any],
) -> ModelRoleActionReport:
    """Convert mapping payloads to ModelRoleActionReport while preserving existing values."""
    if isinstance(value, ModelRoleActionReport):
        return value
    return ModelRoleActionReport.from_dict(value)


def coerce_audio_transcription_result(
    value: AudioTranscriptionResult | Mapping[str, Any],
) -> AudioTranscriptionResult:
    """Convert mapping payloads to AudioTranscriptionResult while preserving existing values."""
    if isinstance(value, AudioTranscriptionResult):
        return value
    return AudioTranscriptionResult.from_dict(value)


def coerce_audio_synthesis_result(
    value: AudioSynthesisResult | Mapping[str, Any],
) -> AudioSynthesisResult:
    """Convert mapping payloads to AudioSynthesisResult while preserving existing values."""
    if isinstance(value, AudioSynthesisResult):
        return value
    return AudioSynthesisResult.from_dict(value)


def coerce_text_translation_result(
    value: TextTranslationResult | Mapping[str, Any],
) -> TextTranslationResult:
    """Convert mapping payloads to TextTranslationResult while preserving existing values."""
    if isinstance(value, TextTranslationResult):
        return value
    return TextTranslationResult.from_dict(value)


def coerce_code_specialist_result(
    value: CodeSpecialistResult | Mapping[str, Any],
) -> CodeSpecialistResult:
    """Convert mapping payloads to CodeSpecialistResult while preserving existing values."""
    if isinstance(value, CodeSpecialistResult):
        return value
    return CodeSpecialistResult.from_dict(value)


def coerce_code_quality_report(
    value: CodeQualityReport | Mapping[str, Any],
) -> CodeQualityReport:
    """Convert mapping payloads to CodeQualityReport while preserving existing values."""
    if isinstance(value, CodeQualityReport):
        return value
    return CodeQualityReport.from_dict(value)


def coerce_coding_task_request(
    value: CodingTaskRequest | Mapping[str, Any],
) -> CodingTaskRequest:
    """Convert mapping payloads to CodingTaskRequest while preserving existing values."""
    if isinstance(value, CodingTaskRequest):
        return value
    return CodingTaskRequest.from_dict(value)


def coerce_coding_task_result(
    value: CodingTaskResult | Mapping[str, Any],
) -> CodingTaskResult:
    """Convert mapping payloads to CodingTaskResult while preserving existing values."""
    if isinstance(value, CodingTaskResult):
        return value
    return CodingTaskResult.from_dict(value)


def coerce_practice_session_result(
    value: PracticeSessionResult | Mapping[str, Any],
) -> PracticeSessionResult:
    """Convert mapping payloads to PracticeSessionResult while preserving existing values."""
    if isinstance(value, PracticeSessionResult):
        return value
    return PracticeSessionResult.from_dict(value)


def coerce_coding_pattern(
    value: CodingPattern | Mapping[str, Any],
) -> CodingPattern:
    """Convert mapping payloads to CodingPattern while preserving existing values."""
    if isinstance(value, CodingPattern):
        return value
    return CodingPattern.from_dict(value)


def coerce_long_horizon_export_bundle(
    value: LongHorizonExportBundle | Mapping[str, Any],
) -> LongHorizonExportBundle:
    """Convert mapping payloads to LongHorizonExportBundle while preserving existing values."""
    if isinstance(value, LongHorizonExportBundle):
        return value
    return LongHorizonExportBundle.from_dict(value)


def coerce_dashboard_runtime_health(
    value: DashboardRuntimeHealth | Mapping[str, Any],
) -> DashboardRuntimeHealth:
    """Convert mapping payloads to DashboardRuntimeHealth while preserving existing values."""
    if isinstance(value, DashboardRuntimeHealth):
        return value
    return DashboardRuntimeHealth.from_dict(value)


def coerce_dashboard_task_state(
    value: DashboardTaskState | Mapping[str, Any],
) -> DashboardTaskState:
    """Convert mapping payloads to DashboardTaskState while preserving existing values."""
    if isinstance(value, DashboardTaskState):
        return value
    return DashboardTaskState.from_dict(value)


def coerce_local_task_pending_approval(
    value: LocalTaskPendingApproval | Mapping[str, Any],
) -> LocalTaskPendingApproval:
    """Convert mapping payloads to LocalTaskPendingApproval while preserving existing values."""
    if isinstance(value, LocalTaskPendingApproval):
        return value
    return LocalTaskPendingApproval.from_dict(value)


def coerce_local_task_session(
    value: LocalTaskSession | Mapping[str, Any],
) -> LocalTaskSession:
    """Convert mapping payloads to LocalTaskSession while preserving existing values."""
    if isinstance(value, LocalTaskSession):
        return value
    return LocalTaskSession.from_dict(value)


def coerce_dashboard_local_task_session_state(
    value: DashboardLocalTaskSessionState | Mapping[str, Any],
) -> DashboardLocalTaskSessionState:
    """Convert mapping payloads to DashboardLocalTaskSessionState while preserving existing values."""
    if isinstance(value, DashboardLocalTaskSessionState):
        return value
    return DashboardLocalTaskSessionState.from_dict(value)


def coerce_user_settings_profile(
    value: UserSettingsProfile | Mapping[str, Any],
) -> UserSettingsProfile:
    """Convert mapping payloads to UserSettingsProfile while preserving existing values."""
    if isinstance(value, UserSettingsProfile):
        return value
    return UserSettingsProfile.from_dict(value)


def coerce_demo_document_fixture(
    value: DemoDocumentFixture | Mapping[str, Any],
) -> DemoDocumentFixture:
    """Convert mapping payloads to DemoDocumentFixture while preserving existing values."""
    if isinstance(value, DemoDocumentFixture):
        return value
    return DemoDocumentFixture.from_dict(value)


def coerce_sample_task_definition(
    value: SampleTaskDefinition | Mapping[str, Any],
) -> SampleTaskDefinition:
    """Convert mapping payloads to SampleTaskDefinition while preserving existing values."""
    if isinstance(value, SampleTaskDefinition):
        return value
    return SampleTaskDefinition.from_dict(value)


def coerce_demo_runtime_pack_summary(
    value: DemoRuntimePackSummary | Mapping[str, Any],
) -> DemoRuntimePackSummary:
    """Convert mapping payloads to DemoRuntimePackSummary while preserving existing values."""
    if isinstance(value, DemoRuntimePackSummary):
        return value
    return DemoRuntimePackSummary.from_dict(value)


def coerce_demo_pack_status(value: DemoPackStatus | Mapping[str, Any]) -> DemoPackStatus:
    """Convert mapping payloads to DemoPackStatus while preserving existing values."""
    if isinstance(value, DemoPackStatus):
        return value
    return DemoPackStatus.from_dict(value)


def coerce_dashboard_task_history_entry(
    value: DashboardTaskHistoryEntry | Mapping[str, Any],
) -> DashboardTaskHistoryEntry:
    """Convert mapping payloads to DashboardTaskHistoryEntry while preserving existing values."""
    if isinstance(value, DashboardTaskHistoryEntry):
        return value
    return DashboardTaskHistoryEntry.from_dict(value)


def coerce_dashboard_task_inspector(
    value: DashboardTaskInspector | Mapping[str, Any],
) -> DashboardTaskInspector:
    """Convert mapping payloads to DashboardTaskInspector while preserving existing values."""
    if isinstance(value, DashboardTaskInspector):
        return value
    return DashboardTaskInspector.from_dict(value)


def coerce_dashboard_knowledge_source(
    value: DashboardKnowledgeSource | Mapping[str, Any],
) -> DashboardKnowledgeSource:
    """Convert mapping payloads to DashboardKnowledgeSource while preserving existing values."""
    if isinstance(value, DashboardKnowledgeSource):
        return value
    return DashboardKnowledgeSource.from_dict(value)


def coerce_dashboard_capability_availability(
    value: DashboardCapabilityAvailability | Mapping[str, Any],
) -> DashboardCapabilityAvailability:
    """Convert mapping payloads to DashboardCapabilityAvailability while preserving existing values."""
    if isinstance(value, DashboardCapabilityAvailability):
        return value
    return DashboardCapabilityAvailability.from_dict(value)


def coerce_dashboard_readiness_check(
    value: DashboardReadinessCheck | Mapping[str, Any],
) -> DashboardReadinessCheck:
    """Convert mapping payloads to DashboardReadinessCheck while preserving existing values."""
    if isinstance(value, DashboardReadinessCheck):
        return value
    return DashboardReadinessCheck.from_dict(value)


def coerce_dashboard_readiness_report(
    value: DashboardReadinessReport | Mapping[str, Any],
) -> DashboardReadinessReport:
    """Convert mapping payloads to DashboardReadinessReport while preserving existing values."""
    if isinstance(value, DashboardReadinessReport):
        return value
    return DashboardReadinessReport.from_dict(value)


def coerce_dashboard_app_state(value: DashboardAppState | Mapping[str, Any]) -> DashboardAppState:
    """Convert mapping payloads to DashboardAppState while preserving existing values."""
    if isinstance(value, DashboardAppState):
        return value
    return DashboardAppState.from_dict(value)


def coerce_shell_state(value: ShellState | Mapping[str, Any]) -> ShellState:
    """Convert mapping payloads to ShellState while preserving existing values."""
    if isinstance(value, ShellState):
        return value
    return ShellState.from_dict(value)
