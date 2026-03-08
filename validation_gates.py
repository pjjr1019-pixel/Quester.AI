"""Explicit Phase 16 validation gates by subsystem and project scope."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ValidationCommand:
    """One required validation command plus the files it depends on."""

    label: str
    command: str
    required_paths: tuple[str, ...]
    purpose: str

    def missing_paths(self, repo_root: Path) -> tuple[str, ...]:
        """Return any referenced repo paths that are currently missing."""
        missing = [path for path in self.required_paths if not (repo_root / path).exists()]
        return tuple(missing)


@dataclass(frozen=True, slots=True)
class SubsystemValidationGate:
    """Required local and compatibility commands before a subsystem is done."""

    subsystem: str
    summary: str
    local_commands: tuple[ValidationCommand, ...]
    compatibility_commands: tuple[ValidationCommand, ...]
    done_criteria: tuple[str, ...]

    def all_commands(self) -> tuple[ValidationCommand, ...]:
        """Return both local and compatibility commands in execution order."""
        return self.local_commands + self.compatibility_commands

    def missing_paths(self, repo_root: Path | None = None) -> tuple[str, ...]:
        """Return any missing files referenced by this gate."""
        root = repo_root or Path(__file__).resolve().parent
        missing: list[str] = []
        for command in self.all_commands():
            missing.extend(command.missing_paths(root))
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True, slots=True)
class ProjectValidationGate:
    """Required validation commands and dependencies for project-wide gates."""

    gate_id: str
    summary: str
    commands: tuple[ValidationCommand, ...]
    required_subsystems: tuple[str, ...] = ()
    required_project_gates: tuple[str, ...] = ()
    done_criteria: tuple[str, ...] = ()

    def all_commands(self) -> tuple[ValidationCommand, ...]:
        """Return the required commands in execution order."""
        return self.commands

    def missing_paths(self, repo_root: Path | None = None) -> tuple[str, ...]:
        """Return any missing files referenced by this gate."""
        root = repo_root or Path(__file__).resolve().parent
        missing: list[str] = []
        for command in self.all_commands():
            missing.extend(command.missing_paths(root))
        return tuple(dict.fromkeys(missing))


DATA_LAYER_GATE = SubsystemValidationGate(
    subsystem="data",
    summary=(
        "Data-layer changes are done only when dataclass round-trip, validation, "
        "and compatibility checks all pass together."
    ),
    local_commands=(
        ValidationCommand(
            label="contracts",
            command="python -m pytest -q tests/test_phase2_contracts.py",
            required_paths=(
                "tests/test_phase2_contracts.py",
                "data_structures.py",
                "agent_schema.py",
                "orchestrator.py",
            ),
            purpose=(
                "Proves typed contracts round-trip, malformed payloads fail early, "
                "and the typed pipeline boundary still returns the expected dataclasses."
            ),
        ),
    ),
    compatibility_commands=(
        ValidationCommand(
            label="compatibility",
            command="python -m pytest -q tests/test_compatibility_contract.py",
            required_paths=("tests/test_compatibility_contract.py", "data_structures.py"),
            purpose=(
                "Keeps serialized payloads, stable field projections, and backward-compatible "
                "deserialization additive during future data-structure growth."
            ),
        ),
    ),
    done_criteria=(
        "Typed contracts round-trip through to_dict and from_dict without loss.",
        "Malformed budgets, plans, and evidence payloads fail fast instead of leaking invalid state.",
        "Boundary schema helpers still rebuild the typed planner, reasoner, and critic contracts.",
        "Compatibility checks keep serialized payloads and stable field projections additive.",
    ),
)


STORAGE_LAYER_GATE = SubsystemValidationGate(
    subsystem="storage",
    summary=(
        "Storage changes are done only when SQLite persistence, JSONL mirrors, "
        "restart safety, and compatibility checks all pass together."
    ),
    local_commands=(
        ValidationCommand(
            label="persistence",
            command="python -m pytest -q tests/test_phase5_persistence.py",
            required_paths=("tests/test_phase5_persistence.py", "storage.py", "data_structures.py"),
            purpose=(
                "Proves SQLite repositories, JSONL mirrors, optimizer lifecycle persistence, "
                "and storage restart behavior stay healthy under the current schema."
            ),
        ),
    ),
    compatibility_commands=(
        ValidationCommand(
            label="compatibility",
            command="python -m pytest -q tests/test_compatibility_contract.py",
            required_paths=("tests/test_compatibility_contract.py", "storage.py"),
            purpose=(
                "Keeps legacy StorageManager logging and key-value compatibility intact while "
                "storage internals continue to evolve."
            ),
        ),
    ),
    done_criteria=(
        "SQLite persistence starts cleanly, reopens an existing database, and reads prior records without schema errors.",
        "Append-only JSONL mirrors are still written for task, status, trace, and web evidence surfaces.",
        "Typed runtime registries, task results, and optimizer lifecycle records remain readable across restarts.",
        "Compatibility checks keep the public storage surface additive.",
    ),
)


MODEL_LAYER_GATE = SubsystemValidationGate(
    subsystem="model",
    summary=(
        "Model-layer changes are done only when runtime, readiness, and "
        "compatibility checks all pass together."
    ),
    local_commands=(
        ValidationCommand(
            label="runtime",
            command="python -m pytest -q tests/test_phase3_runtime.py",
            required_paths=("tests/test_phase3_runtime.py", "model_manager.py", "model_backends.py"),
            purpose=(
                "Proves bounded runtime behavior, including generation and embedding "
                "semaphore limits, fallback activation, and idle unload handling."
            ),
        ),
        ValidationCommand(
            label="readiness",
            command="python -m pytest -q tests/test_phase12_preflight.py",
            required_paths=("tests/test_phase12_preflight.py", "orchestrator.py", "config.py"),
            purpose=(
                "Proves actionable dependency and preflight failures for real-mode "
                "model/runtime setup."
            ),
        ),
    ),
    compatibility_commands=(
        ValidationCommand(
            label="compatibility",
            command="python -m pytest -q tests/test_compatibility_contract.py",
            required_paths=("tests/test_compatibility_contract.py", "model_manager.py", "orchestrator.py"),
            purpose=(
                "Keeps the model-layer public surface additive so new work cannot break "
                "earlier contract expectations."
            ),
        ),
    ),
    done_criteria=(
        "Generation semaphore tests prove only the allowed generation jobs run at once.",
        "Embedding semaphore tests prove only the allowed embedding jobs run at once.",
        "Real-mode readiness failures remain actionable instead of collapsing into generic startup errors.",
        "Compatibility checks keep the public model and runtime surface additive.",
    ),
)


MACRO_LAYER_GATE = SubsystemValidationGate(
    subsystem="macro",
    summary=(
        "Macro-layer changes are done only when deterministic expansion guards, "
        "round-trip verification, and compatibility checks all pass together."
    ),
    local_commands=(
        ValidationCommand(
            label="macro_engine",
            command="python -m pytest -q tests/test_phase6_macro_engine.py",
            required_paths=("tests/test_phase6_macro_engine.py", "macro_engine.py", "data_structures.py"),
            purpose=(
                "Proves nested expansion, recursion blocking, proof-hash stability, "
                "and round-trip validation for deterministic macro traces."
            ),
        ),
    ),
    compatibility_commands=(
        ValidationCommand(
            label="compatibility",
            command="python -m pytest -q tests/test_compatibility_contract.py",
            required_paths=("tests/test_compatibility_contract.py", "macro_engine.py"),
            purpose=(
                "Keeps the public MacroEngine surface additive while later macro or compression work evolves."
            ),
        ),
    ),
    done_criteria=(
        "Nested macro expansion remains deterministic and recursion-guarded.",
        "Semantically equivalent traces keep stable proof hashes across compression and replay.",
        "Round-trip verification rejects drifted or noncanonical macro traces.",
        "Compatibility checks keep the public MacroEngine surface additive.",
    ),
)


AGENT_LAYER_GATE = SubsystemValidationGate(
    subsystem="agent",
    summary=(
        "Agent-layer changes are done only when injected unit coverage, foreground "
        "task coverage, and compatibility checks all pass together."
    ),
    local_commands=(
        ValidationCommand(
            label="unit",
            command="python -m pytest -q tests/test_phase12_agent_units.py",
            required_paths=(
                "tests/test_phase12_agent_units.py",
                "planner.py",
                "researcher.py",
                "reasoner.py",
                "critic.py",
                "compressor.py",
            ),
            purpose=(
                "Proves each agent delegates through its injected service boundary and "
                "builds the expected typed handoffs."
            ),
        ),
        ValidationCommand(
            label="acceptance",
            command="python -m pytest -q tests/test_phase12_end_to_end.py tests/test_phase12_preflight.py",
            required_paths=(
                "tests/test_phase12_end_to_end.py",
                "tests/test_phase12_preflight.py",
                "orchestrator.py",
                "planner_service.py",
                "research_service.py",
                "reasoning_service.py",
                "critique_service.py",
                "compression_service.py",
            ),
            purpose=(
                "Keeps the foreground stub path green and keeps real-mode dependency "
                "failures actionable before agent work begins."
            ),
        ),
    ),
    compatibility_commands=(
        ValidationCommand(
            label="compatibility",
            command="python -m pytest -q tests/test_compatibility_contract.py",
            required_paths=("tests/test_compatibility_contract.py", "orchestrator.py"),
            purpose=(
                "Keeps the public agent and orchestrator entrypoints additive during later refactors."
            ),
        ),
    ),
    done_criteria=(
        "Planner, Researcher, Reasoner, Critic, and Compressor unit tests pass through injected service seams.",
        "Foreground stub-mode runs still exercise the full agent stack end to end.",
        "Real-mode dependency failures stay actionable before agent execution starts.",
        "Compatibility checks keep the public agent-facing surface additive.",
    ),
)


OPTIMIZER_GATE = SubsystemValidationGate(
    subsystem="optimizer",
    summary=(
        "Optimizer changes are done only when replay, validation, activation blocking, "
        "and compatibility checks all pass together."
    ),
    local_commands=(
        ValidationCommand(
            label="optimizer_lifecycle",
            command="python -m pytest -q tests/test_phase12_resource_optimizer.py tests/test_phase5_persistence.py",
            required_paths=(
                "tests/test_phase12_resource_optimizer.py",
                "tests/test_phase5_persistence.py",
                "self_optimizer.py",
                "storage.py",
                "orchestrator.py",
            ),
            purpose=(
                "Proves proposals are replay-scored and validated before activation decisions are "
                "recorded, active macros stay unchanged, and lifecycle artifacts persist append-only."
            ),
        ),
    ),
    compatibility_commands=(
        ValidationCommand(
            label="compatibility",
            command="python -m pytest -q tests/test_compatibility_contract.py",
            required_paths=("tests/test_compatibility_contract.py", "self_optimizer.py", "macro_engine.py"),
            purpose=(
                "Keeps the public optimizer and macro entrypoints additive during future compression "
                "and optimizer refactors."
            ),
        ),
    ),
    done_criteria=(
        "Proposals are never activated unless replay simulation and validation both pass first.",
        "Live activation remains blocked by policy even when a proposal becomes activation-eligible.",
        "Optimizer cycles do not mutate the active macro set during foreground-safe operation.",
        "Compatibility checks keep the public optimizer-facing surface additive.",
    ),
)


ORCHESTRATOR_GATE = SubsystemValidationGate(
    subsystem="orchestrator",
    summary=(
        "Orchestrator changes are done only when pipeline stage order, status events, "
        "and compatibility checks all pass together."
    ),
    local_commands=(
        ValidationCommand(
            label="stage_order",
            command="python -m pytest -q tests/test_phase16_subsystem_gates.py",
            required_paths=("tests/test_phase16_subsystem_gates.py", "orchestrator.py", "dashboard.py"),
            purpose=(
                "Proves the pipeline still emits the required stage sequence and per-component "
                "running or idle status updates."
            ),
        ),
        ValidationCommand(
            label="acceptance",
            command="python -m pytest -q tests/test_phase12_end_to_end.py tests/test_phase13_async_safety.py",
            required_paths=(
                "tests/test_phase12_end_to_end.py",
                "tests/test_phase13_async_safety.py",
                "orchestrator.py",
            ),
            purpose=(
                "Keeps the foreground pipeline, cancellation, and dashboard-submission safety path green."
            ),
        ),
    ),
    compatibility_commands=(
        ValidationCommand(
            label="compatibility",
            command="python -m pytest -q tests/test_compatibility_contract.py",
            required_paths=("tests/test_compatibility_contract.py", "orchestrator.py"),
            purpose=(
                "Keeps public orchestrator entrypoints and runtime compatibility expectations additive."
            ),
        ),
    ),
    done_criteria=(
        "The pipeline emits planner through compressor stages in stable order before completion.",
        "Per-component status updates include running and terminal idle states for a successful task.",
        "Cancellation and dashboard-triggered future handling remain bounded.",
        "Compatibility checks keep public orchestrator APIs additive.",
    ),
)


DASHBOARD_GATE = SubsystemValidationGate(
    subsystem="dashboard",
    summary=(
        "Dashboard changes are done only when typed app-state projections, async safety, "
        "and compatibility checks all pass together."
    ),
    local_commands=(
        ValidationCommand(
            label="typed_state",
            command="python -m pytest -q tests/test_phase16_subsystem_gates.py tests/test_phase12_gui_acceptance.py",
            required_paths=(
                "tests/test_phase16_subsystem_gates.py",
                "tests/test_phase12_gui_acceptance.py",
                "dashboard.py",
                "orchestrator.py",
            ),
            purpose=(
                "Proves the headless dashboard exposes typed status, history, readiness, and degraded-state surfaces."
            ),
        ),
        ValidationCommand(
            label="async_safety",
            command="python -m pytest -q tests/test_phase13_async_safety.py tests/test_phase7_boundaries.py",
            required_paths=(
                "tests/test_phase13_async_safety.py",
                "tests/test_phase7_boundaries.py",
                "dashboard.py",
            ),
            purpose=(
                "Keeps queue draining bounded, overflow visible, and controller interactions responsive."
            ),
        ),
    ),
    compatibility_commands=(
        ValidationCommand(
            label="compatibility",
            command="python -m pytest -q tests/test_compatibility_contract.py",
            required_paths=("tests/test_compatibility_contract.py", "dashboard.py"),
            purpose=(
                "Keeps the public dashboard service surface additive during future UI refactors."
            ),
        ),
    ),
    done_criteria=(
        "Headless dashboard state stays typed for active task, status, history, and readiness surfaces.",
        "Degraded or fallback conditions remain visible without reading raw event dicts directly.",
        "Queue overflow and background-controller paths stay bounded and responsive.",
        "Compatibility checks keep public dashboard APIs additive.",
    ),
)


SUBSYSTEM_VALIDATION_GATES: dict[str, SubsystemValidationGate] = {
    DATA_LAYER_GATE.subsystem: DATA_LAYER_GATE,
    STORAGE_LAYER_GATE.subsystem: STORAGE_LAYER_GATE,
    AGENT_LAYER_GATE.subsystem: AGENT_LAYER_GATE,
    DASHBOARD_GATE.subsystem: DASHBOARD_GATE,
    MACRO_LAYER_GATE.subsystem: MACRO_LAYER_GATE,
    MODEL_LAYER_GATE.subsystem: MODEL_LAYER_GATE,
    OPTIMIZER_GATE.subsystem: OPTIMIZER_GATE,
    ORCHESTRATOR_GATE.subsystem: ORCHESTRATOR_GATE,
}


RESOURCE_GATE = ProjectValidationGate(
    gate_id="resource",
    summary=(
        "Resource validation is green only when explicit thresholds, bounded runtime behavior, "
        "and long-horizon scheduling limits all pass together."
    ),
    commands=(
        ValidationCommand(
            label="thresholds",
            command="python -m pytest -q tests/test_phase12_acceptance_thresholds.py",
            required_paths=("tests/test_phase12_acceptance_thresholds.py", "acceptance_thresholds.py", "config.py"),
            purpose=(
                "Locks the explicit 4 GB / 8 GB development calibration, 6 GB / 8 GB baseline target, "
                "bounded queue policy, and foreground compression limits."
            ),
        ),
        ValidationCommand(
            label="bounded_runtime",
            command=(
                "python -m pytest -q tests/test_phase3_runtime.py "
                "tests/test_phase12_resource_optimizer.py tests/test_phase17_long_horizon.py"
            ),
            required_paths=(
                "tests/test_phase3_runtime.py",
                "tests/test_phase12_resource_optimizer.py",
                "tests/test_phase17_long_horizon.py",
                "model_manager.py",
                "orchestrator.py",
                "reasoning_service.py",
                "critique_service.py",
            ),
            purpose=(
                "Proves semaphore limits, deep-mode caps, optimizer-safe boundedness, and long-horizon "
                "cycle scheduling stay within the locked consumer-hardware assumptions."
            ),
        ),
    ),
    required_subsystems=("model", "optimizer", "orchestrator"),
    done_criteria=(
        "The supported resource envelope remains 4 GB / 8 GB for development and 6 GB / 8 GB for the locked baseline target.",
        "Generation and embedding work stay semaphore-bounded with explicit fallback or degradation behavior.",
        "Deep-mode reasoning, dashboard backpressure, and long-horizon scheduling remain capped instead of scaling unboundedly.",
    ),
)


PRE_RELEASE_SMOKE_GATE = ProjectValidationGate(
    gate_id="pre_release_smoke",
    summary=(
        "Pre-release smoke validation is green only when both the stub end-to-end path and "
        "the real-backend launch-smoke path pass under the locked target assumptions."
    ),
    commands=(
        ValidationCommand(
            label="stub_end_to_end",
            command="python -m pytest -q tests/test_phase12_end_to_end.py",
            required_paths=("tests/test_phase12_end_to_end.py", "orchestrator.py"),
            purpose=(
                "Provides the required end-to-end stub-mode smoke path before a release can be called ready."
            ),
        ),
        ValidationCommand(
            label="real_launch_smoke",
            command="python -m pytest -q tests/test_phase12_packaged_smoke.py tests/test_phase12_preflight.py",
            required_paths=(
                "tests/test_phase12_packaged_smoke.py",
                "tests/test_phase12_preflight.py",
                "orchestrator.py",
                "config.py",
            ),
            purpose=(
                "Proves real-mode launch readiness, actionable dependency failures, and safe packaged fallback "
                "behavior against the locked 6 GB / 8 GB assumptions."
            ),
        ),
    ),
    required_subsystems=("model", "agent", "orchestrator", "dashboard"),
    done_criteria=(
        "At least one end-to-end stub-mode task path passes.",
        "At least one real-backend launch smoke path passes without collapsing into generic startup failure.",
        "Packaged launch can either remain in real mode when ready or fall back safely to stub mode with actionable guidance.",
    ),
)


RELEASE_GATE = ProjectValidationGate(
    gate_id="release",
    summary=(
        "Release validation is green only when acceptance, packaged smoke, and readiness checks all pass "
        "alongside the already-defined subsystem and resource gates."
    ),
    commands=(
        ValidationCommand(
            label="acceptance_bundle",
            command=(
                "python -m pytest -q tests/test_phase12_acceptance_thresholds.py "
                "tests/test_phase12_end_to_end.py tests/test_phase12_translation_exports.py"
            ),
            required_paths=(
                "tests/test_phase12_acceptance_thresholds.py",
                "tests/test_phase12_end_to_end.py",
                "tests/test_phase12_translation_exports.py",
                "acceptance_thresholds.py",
                "translation_service.py",
                "orchestrator.py",
            ),
            purpose=(
                "Keeps the acceptance checklist green for bounded correctness, verified output rendering, "
                "and export eligibility."
            ),
        ),
        ValidationCommand(
            label="packaged_bundle",
            command=(
                "python -m pytest -q tests/test_phase12_packaged_smoke.py "
                "tests/test_phase12_gui_acceptance.py tests/test_phase12_preflight.py"
            ),
            required_paths=(
                "tests/test_phase12_packaged_smoke.py",
                "tests/test_phase12_gui_acceptance.py",
                "tests/test_phase12_preflight.py",
                "dashboard.py",
                "orchestrator.py",
            ),
            purpose=(
                "Keeps packaged stub launch, GUI acceptance, readiness parity, and actionable real-mode failures green."
            ),
        ),
    ),
    required_subsystems=tuple(SUBSYSTEM_VALIDATION_GATES),
    required_project_gates=("resource", "pre_release_smoke"),
    done_criteria=(
        "End-to-end tests stay green for the supported stub-mode path.",
        "The packaged Windows path launches cleanly in stub mode and exports diagnostics.",
        "Real-backend preflight failures remain actionable rather than opaque.",
        "The acceptance checklist stays green together with the resource and subsystem gates.",
    ),
)


PROJECT_COMPLETION_GATE = ProjectValidationGate(
    gate_id="project_completion",
    summary=(
        "Whole-project completion is green only when the acceptance bundle, compatibility contract, "
        "subsystem gates, and release gates are all green at the same time."
    ),
    commands=(
        ValidationCommand(
            label="compatibility",
            command="python -m pytest -q tests/test_compatibility_contract.py",
            required_paths=("tests/test_compatibility_contract.py", "data_structures.py", "orchestrator.py"),
            purpose=(
                "Keeps the compatibility contract green while the project-wide completion gate is evaluated."
            ),
        ),
        ValidationCommand(
            label="acceptance_bundle",
            command=(
                "python -m pytest -q tests/test_phase12_acceptance_thresholds.py tests/test_phase12_end_to_end.py "
                "tests/test_phase12_gui_acceptance.py tests/test_phase12_packaged_smoke.py "
                "tests/test_phase12_preflight.py tests/test_phase12_translation_exports.py"
            ),
            required_paths=(
                "tests/test_phase12_acceptance_thresholds.py",
                "tests/test_phase12_end_to_end.py",
                "tests/test_phase12_gui_acceptance.py",
                "tests/test_phase12_packaged_smoke.py",
                "tests/test_phase12_preflight.py",
                "tests/test_phase12_translation_exports.py",
                "orchestrator.py",
                "dashboard.py",
            ),
            purpose=(
                "Evaluates the acceptance checklist as one project-wide bundle instead of relying on isolated green tests."
            ),
        ),
    ),
    required_subsystems=tuple(SUBSYSTEM_VALIDATION_GATES),
    required_project_gates=("resource", "pre_release_smoke", "release"),
    done_criteria=(
        "The acceptance checklist, subsystem gates, and compatibility contract are all green at the same time.",
        "Project completion cannot be claimed unless the release and resource gates are already green.",
    ),
)


PROJECT_VALIDATION_GATES: dict[str, ProjectValidationGate] = {
    PRE_RELEASE_SMOKE_GATE.gate_id: PRE_RELEASE_SMOKE_GATE,
    PROJECT_COMPLETION_GATE.gate_id: PROJECT_COMPLETION_GATE,
    RELEASE_GATE.gate_id: RELEASE_GATE,
    RESOURCE_GATE.gate_id: RESOURCE_GATE,
}


def get_validation_gate(subsystem: str) -> SubsystemValidationGate:
    """Return the configured validation gate for a subsystem."""
    return SUBSYSTEM_VALIDATION_GATES[subsystem]


def get_project_validation_gate(gate_id: str) -> ProjectValidationGate:
    """Return the configured project-wide validation gate."""
    return PROJECT_VALIDATION_GATES[gate_id]
