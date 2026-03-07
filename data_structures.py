"""Typed runtime contracts used by agents, orchestrator, and tests."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Mapping, TypeVar


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

    def __post_init__(self) -> None:
        _require(bool(self.macro_name.strip()), "macro_name must not be empty.")
        _require(len(self.expansion) > 0, "expansion must include at least one step.")
        _require(self.version > 0, "version must be positive.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Macro:
        return cls(
            macro_name=str(data["macro_name"]),
            expansion=tuple(str(item) for item in data["expansion"]),
            version=int(data.get("version", 1)),
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
class CompressedTrace(DictSerializable):
    """Compressed reasoning chain used by critic and compressor."""

    task_id: str
    tokens: tuple[str, ...]
    expanded_preview: tuple[str, ...]
    macros_used: tuple[str, ...]
    confidence: float
    reasoner_notes: str = ""
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "CompressedTrace.task_id must not be empty.")
        _require(len(self.tokens) > 0, "CompressedTrace.tokens must not be empty.")
        _require(0.0 <= self.confidence <= 1.0, "CompressedTrace.confidence must be between 0 and 1.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompressedTrace:
        return cls(
            task_id=str(data["task_id"]),
            tokens=tuple(str(item) for item in data.get("tokens", data.get("compressed_chain", []))),
            expanded_preview=tuple(str(item) for item in data.get("expanded_preview", [])),
            macros_used=tuple(str(item) for item in data.get("macros_used", [])),
            confidence=float(data.get("confidence", 0.0)),
            reasoner_notes=str(data.get("reasoner_notes", "")),
            created_at=_parse_datetime(data.get("created_at", utc_now())),
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
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require(bool(self.task_id.strip()), "CritiqueReport.task_id must not be empty.")
        _require(0.0 <= self.evidence_coverage <= 1.0, "evidence_coverage must be between 0 and 1.")

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


def coerce_compressed_trace(value: CompressedTrace | Mapping[str, Any]) -> CompressedTrace:
    """Convert mapping payloads to CompressedTrace while preserving existing values."""
    if isinstance(value, CompressedTrace):
        return value
    return CompressedTrace.from_dict(value)


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

