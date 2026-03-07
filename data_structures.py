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


@dataclass(slots=True, frozen=True)
class ResourceBudget(DictSerializable):
    """Budget envelope attached to a single task."""

    retrieval_top_k: int = 4
    max_web_queries: int = 1
    reasoner_passes: int = 1
    critic_passes: int = 1
    macro_depth: int = 2

    def __post_init__(self) -> None:
        _require(self.retrieval_top_k > 0, "retrieval_top_k must be positive.")
        _require(self.max_web_queries >= 0, "max_web_queries must be zero or positive.")
        _require(self.reasoner_passes > 0, "reasoner_passes must be positive.")
        _require(self.critic_passes > 0, "critic_passes must be positive.")
        _require(self.macro_depth > 0, "macro_depth must be positive.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResourceBudget:
        return cls(
            retrieval_top_k=int(data.get("retrieval_top_k", 4)),
            max_web_queries=int(data.get("max_web_queries", 1)),
            reasoner_passes=int(data.get("reasoner_passes", 1)),
            critic_passes=int(data.get("critic_passes", 1)),
            macro_depth=int(data.get("macro_depth", 2)),
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
        payload = super().to_dict()
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
            super().to_dict(),
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
            super().to_dict(),
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
            super().to_dict(),
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
            super().to_dict(),
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
            super().to_dict(),
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
            super().to_dict(),
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
            super().to_dict(),
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
            super().to_dict(),
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
    proof_hash: str = ""
    decode_hints: tuple[DecodeHint, ...] = ()
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "CompressedTrace.task_id must not be empty.")
        _require(len(self.tokens) > 0, "CompressedTrace.tokens must not be empty.")
        _require(0.0 <= self.confidence <= 1.0, "CompressedTrace.confidence must be between 0 and 1.")

    def to_dict(self) -> dict[str, Any]:
        payload = _compact_payload(
            super().to_dict(),
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
