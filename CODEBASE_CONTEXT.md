# Quester.AI Codebase Context

Generated: March 9, 2026
Scope: Static codebase inspection of the current working tree in `c:\Users\Pgiov\OneDrive\Documents\Custom programs\Quester.AI`
Method: Source, config, tests, docs, fixture, and SQLite-shape analysis. I did not run the full test suite for this document.

## 1. What This App Is

Quester.AI is a local-first, Windows-oriented, async multi-agent runtime scaffold. The design center is:

- bounded resource use on consumer hardware
- typed contracts instead of loose dict passing
- one shared model manager for all model-facing work
- SQLite plus JSONL local persistence
- a Tkinter app shell that doubles as an operator dashboard
- a policy-gated local desktop capability layer instead of unrestricted OS control

The app is not a thin chat wrapper. It is an orchestration runtime with:

- a foreground reasoning pipeline
- long-horizon task support
- optional specialist roles
- a bounded replay-only self-optimizer
- explicit local task sessions for desktop control and observation work

## 2. Current Snapshot

Implemented according to the repo docs and code:

- Phases 0.5 through 8 are implemented or functionally complete in the intended constrained mode.
- Phase 11 packaged demo content exists.
- Phase 12 acceptance and readiness surfaces exist.
- Phase 16 validation gates are codified in code.
- Phase 17 long-horizon control exists.
- Phase 18 model registry and control plane exist.
- Phase 19 specialist local roles exist.
- Phase 20 capability contracts and policy layer exist.
- Phase 21 local task sessions and live execution tiers exist.
- Phase 22 observation tiers now include screenshot, OCR, continuous capture, routed visual roles, and a typed hardware governor that degrades heavier observation before core reasoning while accepting bounded optimizer advisory inputs without surrendering live control.
- Phase 23 auxiliary cloud offload is now implemented with a provider-agnostic adapter manager, per-capability dispatch gating, persisted cloud-offload audit records, privacy enforcement, and export-path auxiliary integration that preserves local fallback.
- Phase 24 startup groundwork now exists: source and packaged runners are separated, packaged startup plans requested vs effective mode explicitly, first-run packaged startup forces stub mode, and packaged startup reuses the same readiness/capability checks that power the runtime dashboard.

The next major roadmap target in `ProgressSummary.md` is Phase 24.1 through 24.6: packaged Windows entrypoint, onboarding, setup guidance, reopenable preflight reporting, crash-safe recovery, and release validation.

## 3. Repo Shape

This is a flat top-level Python module repo rather than a nested package layout.

High-centrality modules by size:

- `orchestrator.py`: 6633 lines
- `data_structures.py`: 5774 lines
- `storage.py`: 3891 lines
- `dashboard.py`: 3528 lines
- `capability_runtime.py`: 2421 lines
- `model_manager.py`: 2017 lines

Important directories and artifacts:

- `tests/`: 36 Python test files spanning phases 2, 3, 4, 5, 6, 7, 11, 12, 13, 16, 17, 18, 19, 20, 21, and 22
- `examples/phase11/`: hidden fixture pack with demo corpus, sample tasks, starter macros, starter runtime pack, and one verified trace export example
- `models/`: contains the configured GGUF fallback model and Hugging Face cache artifacts
- `logs/`: JSONL mirrors and runtime output location
- `runtime_logs/`: current stdout/stderr captures
- `quester.sqlite3`: live SQLite database checked into the repo

The repo is currently on a dirty worktree, so this document reflects the current in-place code rather than a clean commit snapshot.

## 4. Entry Points And Runtime Control Flow

Primary public entrypoints:

- CLI script: `quester-ai -> orchestrator:main`
- Primary runtime API: `Orchestrator.run_task(question, thinking_minutes)`
- Compatibility wrapper: `Orchestrator.run_pipeline(question)` which delegates to `run_task(..., thinking_minutes=1)`

Foreground pipeline:

1. `PlannerAgent.plan(...)`
2. `ResearcherAgent.research(...)`
3. `ReasonerAgent.reason_from_handoff(...)`
4. `CriticAgent.review_from_handoff(...)`
5. `TranslationService.render_answer(...)`
6. `CompressorAgent.propose(...)`
7. dashboard and persistence updates

Important orchestration behaviors in `orchestrator.py`:

- startup and shutdown of storage, model manager, dashboard, agents, optimizer, and capability systems
- bounded task execution
- long-horizon session creation, checkpointing, resume, pause, cancel, cooldown, and export
- runtime events and agent status fanout to storage and dashboard
- readiness reporting and dashboard action routing
- local task session lifecycle for desktop control work
- observation execution and continuous capture coordination
- support bundle and packaged-demo actions

## 5. Architectural Center Of Gravity

### 5.1 `data_structures.py`

This is the schema backbone of the whole app. It defines 100+ dataclasses and enums. Nearly every subsystem depends on it.

Major type families:

- core pipeline: `ResourceBudget`, `PlanStep`, `Plan`, `TaskResult`, `ReasoningLog`, `PerformanceMetric`
- evidence and research: `EvidenceItem`, `EvidenceBundle`, `WebEvidenceRecord`
- IR and compression: `CompressedTrace`, `CanonicalReasoningGraph`, `OperationStep`, `DecodeHint`, `Macro`, `OpcodeEntry`, `DecoderEntry`, `ProofHashRecord`, `CompressionRuntimeSubset`
- typed handoffs: `ResearchReasonerHandoff`, `ReasonerCriticHandoff`, `CritiqueReport`
- long-horizon runtime: `LongHorizonSession`, `LongHorizonCheckpoint`, `LongHorizonCandidateSnapshot`, `LongHorizonExportBundle`
- model control plane: `ModelRegistration`, `ModelRouteDecision`, `ModelRegistryView`, `ModelRoleActionReport`, `CompressionInsightSummary`
- capability layer: `CapabilityRequest`, `CapabilityPolicyDecision`, `CapabilityRegistration`, `CapabilityAuditRecord`, `CapabilityExecutionResult`
- local task sessions: `LocalTaskPendingApproval`, `LocalTaskSession`, `DashboardLocalTaskSessionState`
- user and dashboard state: `UserSettingsProfile`, `DashboardAppState`, `DashboardTaskHistoryEntry`, `DashboardTaskInspector`, `DashboardReadinessReport`, `DashboardCapabilityAvailability`
- packaged/demo content: `PackagedSupportBundle`, `DemoDocumentFixture`, `SampleTaskDefinition`, `DemoRuntimePackSummary`
- specialist-role outputs: `AudioTranscriptionResult`, `AudioSynthesisResult`, `TextTranslationResult`, `CodeSpecialistResult`, `VisionInspectionResult`

This file also owns the coercion and backward-compatible deserialization helpers that keep the public contract additive.

### 5.2 `orchestrator.py`

`Orchestrator` is the single owner of application lifecycle and cross-subsystem coordination. It has 168 methods and is the main behavioral hub.

Key responsibilities:

- apply runtime config and persisted settings profiles
- manage planner, researcher, reasoner, critic, compressor, translation, optimizer, dashboard, and storage interactions
- decide bounded vs long-horizon execution
- persist checkpoints and final task results
- gate capability execution behind sessions, policy, approval, and recovery logic
- manage observation tiers and continuous capture
- surface readiness, history, library, control-plane, and local-session state to the dashboard

### 5.3 `storage.py`

`StorageManager` is the persistence hub. It opens the SQLite database, creates repositories, verifies schema version, runs integrity checks, bootstraps the default runtime lexicon, and manages the vector index.

Repository families:

- runtime metadata: events, agent statuses, key-value state, task runs
- retrieval: source documents, chunks, vector entries, web evidence
- compression/runtime registry: macros, opcodes, decoders, symbol tables, proof hashes, reasoning history
- optimizer lifecycle: replay samples/evaluations, proposal records, activation records, rollback records, suggestion records, suggestion usage
- capability layer: capability audit records and persisted capability registry view

Storage writes both SQLite records and append-only JSONL mirrors where appropriate.

### 5.4 `model_manager.py`

`ModelManager` is the only component intended to own model runtime lifecycle, concurrency, backend startup, fallback, health, and specialist routing.

Core features:

- generation and embedding semaphores
- idle maintenance loop and idle unload support
- backend fallback on startup failure or low memory
- typed local model registry
- heavy-role scheduling with a two-heavy-role limit
- route history, fallback history, cache snapshots, and compression insights
- specialist helpers for STT, VAD, TTS, translation, code analysis, and vision

Current model roles:

- `generation`
- `embedding`
- `reranker`
- `speech_to_text`
- `text_to_speech`
- `vad`
- `translation`
- `code_specialist`
- `vision`
- `specialist_perception`

### 5.5 `dashboard.py`

`DashboardService` consumes typed events and either runs headless or renders a Tkinter UI. It has 113 methods and maintains a typed `DashboardAppState`.

UI surfaces include:

- question submission and time controls
- final answer, citations, critique, evidence, provenance, web activity
- runtime health and agent status
- model registry and capability registry views
- examples, history, knowledge library, readiness, audio, translation, and code tabs
- local task session controls
- long-horizon controls

The dashboard is not a source of truth. It is a consumer and controller that routes actions back into the orchestrator.

### 5.6 `capability_runtime.py`

This module owns the policy and execution layer for local desktop capabilities.

Main classes:

- `CapabilityPolicyEngine`
- `CapabilityStubExecutor`
- `CapabilityExecutor`

Capability types modeled in contracts:

- `file_operation`
- `shell_command`
- `browser_action`
- `app_window_focus`
- `clipboard_action`
- `screenshot`
- `ocr_request`
- `desktop_input`

Live executor coverage observed in code:

- live: file operations, shell commands, browser read and navigate, app/window focus, screenshot capture, OCR, desktop input
- contract-only or still stub/deferred: clipboard actions, interactive browser actions (`click`, `type`, `download`)

Policy characteristics:

- default-deny dangerous flags such as elevation, persistence, hidden execution, or credential harvesting
- allowlisted roots, shell commands, browser domains, and apps
- session-gated execution
- approval policies: `approve_risky_only`, `manual_only`, `safe_auto`
- degraded behavior under resource pressure for screenshot, OCR, and desktop input
- Windows-specific live input and window-control paths

## 6. Service And Agent Layer

Thin agent wrappers:

- `planner.py`
- `researcher.py`
- `reasoner.py`
- `critic.py`
- `compressor.py`

Shared service modules:

- `planner_service.py`: schema-constrained planning with deterministic fallback
- `research_service.py`: local-first retrieval, reranking, seed corpus, bounded web fallback
- `reasoning_service.py`: typed trace construction, runtime-subset loading, fast/deep reasoning modes
- `critique_service.py`: deterministic verification first, structured critique second, repair metadata
- `compression_service.py`: bounded macro proposal generation
- `translation_service.py`: render final answer from verified reasoning state
- `structured_generation.py`: one shared bounded structured-output helper

Important behavior details:

- `ResearchService` seeds a deterministic local corpus when appropriate. The current built-in seed corpus count in code is 6 documents.
- `ReasoningService` prepares a task-scoped runtime subset rather than loading the full registry.
- `CritiqueService` runs deterministic checks before accepting model-shaped critique output.
- `CompressionService` is proposal-oriented and bounded by explicit acceptance thresholds.

## 7. Retrieval, Web, And Compression

Retrieval stack:

- `retrieval.py`: IDs, chunking, lexical scoring, vector adapter interfaces, simple in-memory vector index, Chroma adapter
- `retrieval_service.py`: bounded local hybrid lexical/vector retrieval policy
- `storage.py`: SQLite FTS5-backed lexical candidates plus vector record storage

Web lookup:

- `web_adapter.py` provides a deterministic stub adapter and a Wikipedia/MediaWiki-backed real adapter
- `research_service.py` uses web fallback only when local evidence is weak or freshness is implied

Compression and IR:

- `CompressedTrace` is the primary reasoning artifact
- macro proposals are validated and proof-aware
- canonical graph and proof hash are central to round-trip stability
- optimizer remains replay-only and proposal-only by default

## 8. Local Specialist Features

Supporting local specialist modules:

- `local_audio.py`: WAV-only bounded VAD, stub/system transcription, stub/system speech synthesis
- `local_translation.py`: stub or Argos-based local translation
- `local_code_specialist.py`: bounded file/snippet analysis
- `local_vision.py`: current stub image inspection helper

These route through `ModelManager` rather than bypassing the main runtime.

## 9. Config And Default Runtime Behavior

`config.py` defines frozen dataclasses and a validated `AppConfig`.

Default runtime profile as loaded from code:

- Python requirement: `>=3.11`
- dashboard UI: enabled
- stub mode: `True`
- web fallback: allowed
- self optimizer: disabled
- generation backend: `ollama`
- generation model: `qwen2.5:3b-instruct-q4_K_M`
- generation fallback backend: `llama_cpp`
- generation fallback model: `qwen2.5-3b-instruct-q4_k_m.gguf`
- embedding backend: `sentence_transformers`
- embedding model: `intfloat/e5-small-v2`
- embedding fallback backend: `ollama_embeddings`
- embedding fallback model: `nomic-embed-text`
- vector store target: `chromadb`
- concurrency: 1 generation slot, 1 embedding slot
- dev calibration: 4 GB VRAM / 8 GB RAM
- acceptance target: 6 GB VRAM / 8 GB RAM

Observation defaults:

- tier: `screenshot_on_demand`
- continuous capture: disabled
- OCR on step: disabled
- vision on step: disabled
- capture FPS: `0.5`
- capture max size: `960x540`
- frame history cap: `4`
- diff threshold: `0.03`

Desktop capability defaults in `UserSettingsProfile`:

- desktop disabled by default
- approval policy: `approve_risky_only`
- allowlisted roots: `.`, `logs`, `examples`, `models`
- allowlisted shell commands: `python`, `git`, `rg`, `pytest`
- allowlisted browser domains: `localhost`, `127.0.0.1`

## 10. Persistence And Live Data Snapshot

Persistence locations:

- SQLite DB: `quester.sqlite3`
- logs dir: `logs/`
- JSONL mirrors: `events.jsonl`, `status.jsonl`, `traces.jsonl`, `web.jsonl`, `capability_audit.jsonl`

Observed SQLite table counts at inspection time:

- `task_runs`: 3
- `runtime_events`: 116
- `agent_status_history`: 81
- `source_documents`: 6
- `document_chunks`: 6
- `vector_entries`: 6
- `web_evidence`: 2
- `reasoning_history`: 8
- `proof_hash_history`: 4
- `performance_history`: 3
- `optimizer_replay_samples`: 3
- `capability_audit_records`: 0

This means the repo is carrying meaningful runtime state, not just schema.

## 11. Packaged Demo Content

`phase11_content.py` loads a hidden repo fixture pack from `examples/phase11/`.

Observed fixture counts:

- demo documents: 4
- sample tasks: 5
- starter macros: 3
- starter opcodes: 2
- starter decoders: 2
- packaged verified trace exports: 1

The loader can:

- ingest the demo pack into storage
- load sample tasks into the dashboard
- load starter macros and runtime pack entries
- export a packaged verified-trace example

## 12. Validation, Acceptance, And Test Surfaces

Test inventory:

- 36 test files
- coverage spans contracts, runtime, budget policy, persistence, retrieval, density, macro runtime, typed boundaries, GUI acceptance, long-horizon control, model registry, specialist roles, capability policy, local task sessions, live capability execution, browser/app control, desktop safety, screenshot/OCR, continuous capture, and on-step observation behavior

Validation gates are first-class code in `validation_gates.py`.

Subsystem gates:

- `data`
- `storage`
- `model`
- `macro`
- `agent`
- `optimizer`
- `orchestrator`
- `dashboard`

Project gates:

- `resource`
- `pre_release_smoke`
- `release`
- `project_completion`

Acceptance thresholds are explicit in `acceptance_thresholds.py`:

- deep mode must show at least one verifier-backed improvement example
- final selection must remain verifier-backed
- structured degraded outcomes are allowed
- foreground compression is capped at 5 proposals and 6 recent reasoning logs
- resource target stays at 1 generation slot, 1 embedding slot, and bounded queues

## 13. Notable Technical Characteristics

Important architectural traits:

- strongly typed, dataclass-heavy design
- additive compatibility discipline
- large monolithic core files rather than many small packages
- heavy use of persisted machine-readable state
- Windows-first live desktop support
- stub mode remains the default and safest supported path
- the dashboard is part of the architecture, not an afterthought

Central dependency pattern observed from imports:

- most imported modules: `data_structures`, `config`, `model_manager`, `retrieval`, `storage`
- largest outbound coordinator: `orchestrator`

## 14. Risks, Drift, And Maintenance Notes

### 14.1 Packaging Drift

`pyproject.toml` does not list several newer top-level modules in `tool.setuptools.py-modules`.

Observed omissions:

- `agent_schema`
- `capability_runtime`
- `compression_service`
- `critique_service`
- `local_vision`
- `planner_service`
- `reasoning_service`
- `research_service`
- `retrieval_service`
- `structured_generation`

That means editable or packaged installs may not reflect the true runtime surface unless this metadata is updated.

### 14.2 Monolith Risk

The main runtime logic is concentrated in a handful of very large files:

- `orchestrator.py`
- `data_structures.py`
- `storage.py`
- `dashboard.py`

This keeps navigation expensive and raises the risk of cross-cutting regressions.

### 14.3 Platform Coupling

Live capability execution and some specialist fallbacks are strongly tied to Windows APIs or Windows tooling:

- window enumeration and focus
- desktop input
- Windows OCR
- optional `System.Speech`

Cross-platform behavior exists mainly through stub paths, not equivalent live paths.

### 14.4 Repo-State Coupling

The repository currently contains:

- live SQLite state
- local runtime logs
- a local GGUF model artifact
- hidden demo fixture assets

That is helpful for development continuity, but it also means cloning the repo is not the same as starting from a clean artifact-only codebase.

## 15. Where To Start For Common Work

If you need to change:

- overall task flow: start with `orchestrator.py`
- schemas or compatibility: start with `data_structures.py` and `agent_schema.py`
- persistence or retrieval indexing: start with `storage.py`, `retrieval.py`, `retrieval_service.py`
- local model routing or specialist roles: start with `model_manager.py`
- planner/reasoner/critic behavior: start with the corresponding `*_service.py` modules
- desktop capability policy or execution: start with `capability_runtime.py`
- app UI and operator workflows: start with `dashboard.py`
- demo content or sample tasks: start with `phase11_content.py` and `examples/phase11/`
- release discipline or done criteria: start with `validation_gates.py`, `acceptance_thresholds.py`, and the relevant tests

## 16. Short Mental Model

The cleanest way to think about Quester.AI is:

- a typed orchestration kernel (`orchestrator.py`)
- backed by a typed local state model (`data_structures.py`)
- persisted through SQLite and JSONL (`storage.py`)
- powered by one bounded model control plane (`model_manager.py`)
- surfaced through a Tkinter operator shell (`dashboard.py`)
- extended by a local-first desktop capability and observation layer (`capability_runtime.py`)
- protected by explicit tests, acceptance thresholds, and validation gates

That is the current app, more than "a chatbot" or "a local LLM wrapper."
