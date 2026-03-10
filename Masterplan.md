# Masterplan

## Phase-Based Implementation Plan for a Lightweight Multi-Agent AI System

## Summary
Build a complete Python-based, local-first, async multi-agent AI system that preserves the full requested workflow and feature set while staying within a hard target of 6GB VRAM and 8GB system RAM. Implement agents as logical services that share one generation backend and one embedding backend, use local-first retrieval, enforce strict resource limits, and keep all heavy work bounded, testable, and modular.
Extend the same architecture into a first-class Coding Mode so the product can route between multiple local coding AIs, run bounded coding tasks and sandbox validation locally, learn only from validated coding patterns, practice safely while idle, and surface coding activity through the same orb-driven UI, storage, optimizer, and control-plane systems instead of bolting on a disconnected code tool.

## Required Interfaces and Types
- [x] Implement `PlannerAgent.plan(question: str, budget: ResourceBudget) -> Plan`
- [x] Implement `ResearcherAgent.research(plan: Plan, budget: ResourceBudget) -> EvidenceBundle`
- [x] Implement `ReasonerAgent.reason(plan: Plan, evidence: EvidenceBundle, budget: ResourceBudget) -> CompressedTrace`
- [x] Implement `CriticAgent.review(plan: Plan, evidence: EvidenceBundle, trace: CompressedTrace, budget: ResourceBudget) -> CritiqueReport`
- [x] Implement `CompressorAgent.propose(trace: CompressedTrace, logs: list[ReasoningLog]) -> list[MacroProposal]`
- [x] Implement `SelfOptimizer.run_cycle() -> list[MacroProposal]`
- [x] Implement `Orchestrator.run_task(question: str, thinking_minutes: int) -> TaskResult`
- [x] Implement `MacroEngine.compress(steps: list[str]) -> CompressedTrace`
- [x] Implement `MacroEngine.expand(tokens: list[str]) -> list[str]`
- [x] Implement `MacroEngine.verify_round_trip(trace: CompressedTrace) -> bool`
- [x] During Phases 3-4, align the implementation to this public API checklist; `run_pipeline()` is temporary and must become `run_task(question, thinking_minutes)`.
- [x] During Phases 3-4, thread `ResourceBudget` through Researcher, Reasoner, and Critic so the public interface checklist is fully true.

## Required coding-mode Interfaces
- [ ] Implement `CodingModeService.run_task(request: CodingTaskRequest) -> CodingTaskResult`
- [ ] Implement `CodingRouter.route(request: CodingTaskRequest, role: CodingRole) -> ModelRouteDecision`
- [ ] Implement `SandboxRunner.execute(job: SandboxExecutionRequest) -> SandboxExecutionResult`
- [ ] Implement `CodingPracticeService.run_idle_cycle() -> PracticeSessionResult`
- [ ] Implement `CodingMemoryIndex.search(query: CodingPatternQuery) -> list[CodingPattern]`
- [ ] Keep Coding Mode additive to `Orchestrator.run_task(...)`: coding workflows may use specialized services internally, but must still surface through the same typed event/status, storage, history, and dashboard contracts.

## Required Internal Types
- [x] Add `Macro`, `KnowledgeVector`, `ReasoningLog`, `PerformanceMetric`
- [x] Add `Plan`, `PlanStep`, `EvidenceItem`, `EvidenceBundle`
- [x] Add `CompressedTrace`, `CritiqueReport`, `MacroProposal`
- [x] Add `AgentStatus`, `ResourceBudget`, `TaskResult`
- [ ] Add `CodingTaskRequest`, `CodingTaskResult`, `CodingTaskArtifact`, `CodingRoleAssignment`
- [ ] Add `SandboxExecutionRequest`, `SandboxExecutionResult`, `CodeQualityReport`, `RegressionCheckReport`
- [ ] Add `CodingPattern`, `CodingPatternValidation`, `PracticeSessionResult`, `CodingKnowledgeStats`

0.5 Phase 0.5 - Preflight Decision Lock (Do This Before Phase 1)
- [x] 0.5.1 Confirm hard runtime targets: 6GB VRAM and 8GB RAM.
- [x] 0.5.2 Select generation runtime backend and pin one default option.
- [x] 0.5.3 Select embedding backend and pin one default option.
- [x] 0.5.4 Select local vector store adapter and pin one default option.
- [x] 0.5.5 Set `stub_mode=true` as the default for first-pass development and tests.
- [x] 0.5.6 Define a minimal "first successful run" goal: one question completes full pipeline in stub mode.
- [x] 0.5.7 Define a minimal "first real-backend run" goal: one question completes with real model calls and bounded resources.
- [x] 0.5.8 Write these locked decisions in a short decision log section so later phases do not re-open core choices.
- [x] 0.5.9 Do not begin Phase 1 until items 0.5.1 to 0.5.8 are checked.
- [ ] 0.5.10 Pin one default local coding-model bundle and one fallback bundle for planning, generation, debugging, review, test-writing, summarization, and refactoring roles.
- [ ] 0.5.11 Lock the default Coding Mode task taxonomy: feature generation, bug fixing, refactoring, test generation, code review, explanation or summarization, project scaffolding, and architecture planning.
- [ ] 0.5.12 Lock sandbox defaults: temp-workspace isolation, timeout and resource caps, allowlisted tools, and default blocked behaviors.
- [ ] 0.5.13 Lock gated coding-learning tiers and promotion criteria: `candidate`, `verified`, and `rejected`, with required tests, lint, complexity, security, maintainability, critique, and regression gates.
- [ ] 0.5.14 Lock idle coding-practice policy: opt-in or default-off behavior, safe idle detection window, workspace isolation, and max background resource budget.

1. Phase 1 - Project Scaffold and Runtime Rules
- [x] 1.1 Create the required modules: `planner.py`, `researcher.py`, `reasoner.py`, `compressor.py`, `critic.py`, `self_optimizer.py`, `dashboard.py`, `macro_engine.py`, `data_structures.py`.
- [x] 1.2 Add support modules: `orchestrator.py`, `model_manager.py`, `storage.py`, `config.py`, `prompts.py`, `utils.py`, `tests/`.
- [x] 1.3 Use Python 3.11+ and prefer standard library tools first: `asyncio`, `sqlite3`, `dataclasses`, `logging`, `queue`, `threading`, `tkinter`.
- [x] 1.4 Put all tunable settings in `config.py`; do not hard-code magic numbers inside agents.
- [x] 1.5 Add a clear startup path and a clean shutdown path for models, background tasks, database connections, and the GUI.
- [x] 1.6 Treat 6GB VRAM / 8GB RAM as a hard resource budget and mention it in config comments and runtime checks.
- [ ] 1.1.1 Add coding-mode support modules such as `coding_mode.py`, `coding_router.py`, `sandbox_runner.py`, `coding_memory.py`, `practice_mode.py`, and coding-specific tests without breaking the current app entrypoints.
- [ ] 1.6.1 Extend the same hard resource budget to Coding Mode so coding models, sandbox runs, linting, static analysis, and idle practice remain bounded on the locked hardware target.

2. Phase 2 - Core Data Structures, Typed Contracts, and Boundary Validation
- [x] 2.1 Treat Phase 2 as a contract migration phase, not a behavior-expansion phase.
- [x] 2.2 Keep current Phase 1 behavior working while replacing raw dict handoffs with typed dataclass objects.
- [x] 2.3 Implement all required dataclasses in `data_structures.py` using type hints, `slots=True`, and small helper methods like `to_dict()` / `from_dict()`.
- [x] 2.4 Use `frozen=True` for immutable value objects where mutation is not needed.
- [x] 2.5 Implement the original required structures: `Macro`, `KnowledgeVector`, `ReasoningLog`, and `PerformanceMetric`.
- [x] 2.6 Implement the internal pipeline contracts: `PlanStep`, `Plan`, `EvidenceItem`, `EvidenceBundle`, `CompressedTrace`, `CritiqueReport`, `MacroProposal`, `ResourceBudget`, and `TaskResult`.
- [x] 2.7 Keep timestamps as `datetime` inside typed objects and convert to ISO strings only at storage, logging, and dashboard boundaries.
- [x] 2.8 Add simple enums or string constants for agent states, source types, task states, severity levels, and critique result categories.
- [x] 2.9 Define `Plan` with at minimum: `task_id`, `question`, `steps`, `required_evidence`, `success_criteria`, and `budget`.
- [x] 2.10 Define `PlanStep` with at minimum: `step_id`, `description`, `depends_on`, `status`, and `notes`.
- [x] 2.11 Define `EvidenceItem` with at minimum: `id`, `content`, `source_type`, `source_ref`, `score`, `metadata`, and optional `vector_preview`.
- [x] 2.12 Define `EvidenceBundle` with at minimum: `task_id`, `local_results`, `web_results`, `used_web_fallback`, and `created_at`.
- [x] 2.13 Define `CompressedTrace` with at minimum: `task_id`, `tokens`, `expanded_preview`, `macros_used`, `confidence`, and `created_at`.
- [x] 2.14 Define `CritiqueReport` with at minimum: `task_id`, `is_valid`, `issues`, `fixed_trace`, `evidence_coverage`, and `created_at`.
- [x] 2.15 Define `MacroProposal` with at minimum: `proposal_id`, `macro`, `reason`, `examples`, `simulation_score`, and `approved`.
- [x] 2.16 Define `TaskResult` as the final typed return object from the orchestrator, containing plan, evidence, reasoning, critique, compression output, and completion timestamp.
- [x] 2.16.1 Define `CodingTaskRequest` with task type, language, framework, workspace scope, safety policy, requested coding roles, and budget hints.
- [ ] 2.16.2 Define `CodingTaskResult` with produced patches/files, sandbox results, test/lint/static-analysis summaries, critique outcome, learned patterns, and completion metadata.
- [ ] 2.16.3 Define `CodingPattern` with pattern tier, language, framework, task type, source, validation history, quality score, and reuse frequency.
- [ ] 2.16.4 Define `SandboxExecutionResult`, `CodeQualityReport`, and `RegressionCheckReport` as typed machine-readable outputs instead of raw terminal text.
- [x] 2.17 Add lightweight validation methods so invalid data fails early and predictably at object construction or explicit `validate()` calls.
- [x] 2.17.1 Add validation rules for coding-task safety constraints, allowed language/framework labels, and pattern-tier transitions.
- [x] 2.18 Add temporary adapter helpers so Phase 1 dict-based code can be migrated incrementally instead of all at once.
- [x] 2.19 Migrate `MacroEngine` to typed inputs and outputs before expanding its real compression logic in later phases.
- [x] 2.20 Migrate agent boundaries in this order: Planner -> Researcher -> Reasoner -> Critic -> Compressor.
- [x] 2.21 Migrate `AppOrchestrator.run_pipeline()` last so it returns a typed `TaskResult` instead of nested dicts.
- [x] 2.22 Do not mix Phase 2 work with storage redesign, web adapter changes, dashboard redesign, or new optimization behavior.
- [x] 2.23 Add serialization round-trip tests, invalid-constructor tests, and one typed end-to-end pipeline test before marking Phase 2 complete.
- [ ] 2.23.1 Add constructor and round-trip tests for coding dataclasses before wiring Coding Mode into orchestration.
Phase 2 exit note
Typed contracts are complete, but stub/placeholder behavior still exists in runtime, macro, retrieval, optimizer, and dashboard layers. Do not treat Phase 2 completion as feature completion.

Compatibility Contract for Future Phases
- [x] Keep public method signatures stable in `planner.py`, `researcher.py`, `reasoner.py`, `critic.py`, `compressor.py`, and `orchestrator.py`; future phases may change internals but must not break the current public API.
- [x] Evolve dataclasses additively first: add new fields with safe defaults before considering removal or rename of existing fields.
- [x] Keep every `from_dict()` backward-compatible with old serialized payloads; `to_dict()` may emit the newer canonical shape.
- [x] Keep current enum values valid; add new values only when needed and never repurpose an existing value to mean something different.
- [x] Keep `ResourceBudget` backward-compatible; any new budget knobs must default safely so `BudgetPolicy.from_minutes()` still returns a valid object for old callers.
- [x] Keep `PlanStep.description` as the human/debug representation even if a future planner control IR or opcode layer is added.
- [x] Keep `Plan.task_id`, `question`, `steps`, `required_evidence`, `success_criteria`, `budget`, and `planner_notes`; future plan-IR fields must be additive.
- [x] Keep `Macro.macro_name`, `expansion`, and `version`; future fields such as signature, invariants, or proof hash must be additive.
- [x] Keep `ReasoningLog.compressed_chain` and `macros_used` as legacy/debug projections even after proof hashes, motif IDs, or IR metadata are added.
- [x] Keep `EvidenceItem.content`, `source_type`, `source_ref`, `score`, `metadata`, and `vector_preview`; compressed evidence atoms may be added, but raw content must remain for audit and decode.
- [x] Keep `EvidenceBundle.local_results`, `web_results`, and `used_web_fallback`; future evidence-graph or citation-index fields must be additive.
- [x] Keep `CompressedTrace.tokens`, `expanded_preview`, `macros_used`, `confidence`, and `reasoner_notes`; graph-backed IR fields such as op stream, symbol-table references, context frames, and proof hash must be additive.
- [x] Keep `CompressedTrace.tokens` populated as a legacy/debug projection even after graph-backed IR becomes the primary reasoning representation.
- [x] Keep `CritiqueReport.is_valid`, `issues`, `fixed_trace`, `evidence_coverage`, `critic_notes`, and `result`; future drift/provenance/proof-hash fields must be additive.
- [x] Keep `MacroProposal.proposal_id`, `macro`, `reason`, `examples`, `simulation_score`, and `approved`; future motif/fingerprint metadata must be additive.
- [x] Keep `TaskResult.plan`, `evidence`, `reasoning`, `critique`, and `compression`; future answer text, metrics, or proof summaries must be additive.
- [x] Keep `RuntimeEvent.stage`, `payload`, and `timestamp`; future event codes may be embedded inside payloads, but old event consumers must still work.
- [x] Keep `AgentStatus.component`, `state`, `task_id`, `severity`, and `message`; machine-readable status codes may be added, but message remains the readable fallback.
- [x] Keep `Orchestrator.run_task(question, thinking_minutes)` as the primary public entrypoint; keep `run_pipeline(question)` as a compatibility shim until a later explicit cleanup phase.
- [x] Keep `StorageManager` usable while adding specialized repositories/tables; do not break existing logging and key-value behaviors during storage migrations.
- [x] Keep `DashboardService.publish_event(...)` compatible with the current event-dict shape even after richer typed events are introduced.
- [x] Treat `MacroEngine`, Reasoner, Critic, storage schema, optimizer logic, and dashboard internals as replaceable implementation details so long as the compatibility rules above remain true.
- [x] Add and keep compatibility tests that prove old serialized payloads still deserialize, `run_task(...)` still returns a valid `TaskResult`, `run_pipeline(...)` still works during migration, old `CompressedTrace` payloads without IR fields remain valid, and new IR-backed traces still emit legacy `tokens` and `expanded_preview`.
- [ ] Keep Coding Mode contracts additive: `TaskResult`, `RuntimeEvent`, `AgentStatus`, settings profiles, and task history may gain coding fields, but existing non-coding callers must remain valid.
- [ ] Keep coding-memory tiers backward-compatible and explicit: `candidate`, `verified`, and `rejected` must stay machine-readable states instead of overloaded booleans or ad hoc tags.
- [ ] Keep sandbox, test, lint, static-analysis, and regression outputs machine-readable and replayable so the optimizer, dashboard, and history views never rely on raw terminal text as the primary contract.

3. Phase 3 - Resource Budgeting and Shared Model Management (Complete)
- [x] 3.0.1 Before coding Phase 3, freeze the generation and embedding adapter interfaces (`start`, `stop`, `generate` / `embed`, `health`) so backend-specific code does not leak into agents.
- [x] 3.0.2 Before coding Phase 3, extend config first with backend runtime settings such as endpoint/path, timeout, idle-unload threshold, fallback policy, and telemetry flags; do not hard-code them in `ModelManager`.
- [x] 3.0.3 Before coding Phase 3, define explicit runtime error types for startup failure, backend unavailable, timeout, and fallback/OOM conditions so failures stay structured.
- [x] 3.0.4 Before coding Phase 3, preserve stub-mode parity with real mode and add tests for semaphores, fallback, and health snapshots before wiring actual backends.
- [x] 3.1 Keep `ModelManager` as the only component allowed to load, unload, or schedule model work.
- [x] 3.2 Treat Phase 3 as a real backend-abstraction phase, not just a semaphore cleanup phase.
- [x] 3.3 Implement a shared generation backend interface for Planner, Reasoner, Critic, and Compressor.
- [x] 3.4 Implement a separate lightweight embedding backend interface for Researcher.
- [x] 3.5 Add `asyncio.Semaphore` limits so only 1 generation task and 1 embedding task can actively use model resources at the same time.
- [x] 3.6 Add runtime memory checks using optional `psutil` and backend-reported VRAM usage when available.
- [x] 3.7 Add CPU fallback or reduced-budget fallback when VRAM is too low to keep the system usable.
- [x] 3.8 Ensure inactive model sessions are unloaded or released after idle timeout.
- [x] 3.9 Keep queue sizes bounded so the dashboard and optimizer cannot create unbounded memory growth.
- [x] 3.10 Add provider protocols or adapter classes so `ollama`, `llama.cpp`, and embedding backends can be swapped without agent changes.
- [x] 3.11 Add a runtime capability/health snapshot that reports backend name, active jobs, fallback mode, and recent usage.
- [x] 3.12 Replace placeholder real-mode behavior with actual backend calls or explicit adapter stubs that fail clearly when not configured.
- [x] 3.13 Add explicit idle-unload and backend-fallback rules so the system can degrade gracefully under 6GB VRAM / 8GB RAM constraints.
Phase 3 exit note
- [x] Phase 3 is complete for the current codebase: backend abstraction, semaphores, fallback rules, health snapshots, and idle maintenance are implemented. Real-backend smoke remains a later acceptance-gate item because local services, dependencies, and model files are environment prerequisites rather than remaining Phase 3 code.

4. Phase 4 - Budget Policy and Thinking-Time Mapping (Complete)
- [x] 4.0.1 Before coding Phase 4, treat existing `BudgetPolicy.from_minutes()` and `Orchestrator.run_task(...)` as the baseline; calibrate and extend them instead of creating duplicate entrypoints or duplicate budget logic.
- [x] 4.0.2 Before coding Phase 4, decide one source of truth for budget derivation in `config.py` and make all agents consume the same `ResourceBudget` object.
- [x] 4.0.3 Before coding Phase 4, add tests that prove different `thinking_minutes` values actually change downstream behavior instead of only changing metadata.
- [x] 4.0.4 Before coding Phase 4, keep all new budget controls memory-safe and bounded so no slider position can force unbounded retrieval, web use, or reasoning passes.
- [x] 4.0.5 Before closing Phase 4, calibrate the bounded presets against both the locked 6GB VRAM / 8GB RAM baseline and the current 4GB VRAM / 8GB RAM development profile so the budget table is honest for real runs.
- [x] 4.1 Implement `ResourceBudget` and a `BudgetPolicy.from_minutes(minutes)` helper in `config.py`.
- [x] 4.2 Use these default bounded presets:
- [x] 4.3 `1-5 min -> retrieval_top_k=4, max_web_queries=1, reasoner_passes=1, critic_passes=1, macro_depth=2`
- [x] 4.4 `6-30 min -> retrieval_top_k=6, max_web_queries=2, reasoner_passes=2, critic_passes=2, macro_depth=3`
- [x] 4.5 `31-120 min -> retrieval_top_k=8, max_web_queries=3, reasoner_passes=3, critic_passes=2, macro_depth=4`
- [x] 4.6 `121-720 min -> retrieval_top_k=10, max_web_queries=5, reasoner_passes=4, critic_passes=3, macro_depth=4`
- [x] 4.7 Keep all presets memory-safe; never scale budgets linearly without bounds.
- [x] 4.8 Make the slider affect actual work depth, not artificial waiting.
- [x] 4.9 Add public `Orchestrator.run_task(question, thinking_minutes)` and compute `ResourceBudget` from `thinking_minutes` there.
- [x] 4.10 Pass the computed `ResourceBudget` through Planner, Researcher, Reasoner, and Critic so retrieval depth, web gating, reasoning passes, and critic depth all change actual runtime work instead of only metadata.
- [ ] 4.10.1 Extend `ResourceBudget` additively for coding tasks so thinking time can bound generator passes, sandbox runs, test retries, lint/static-analysis passes, and review depth without permitting unbounded local execution.
- [ ] 4.10.2 Map fast, deep, and long-horizon coding presets to real additional work such as more candidate patches, broader regression checks, deeper review, and more selective pattern promotion.
- [x] Phase 4 exit note: thinking-time budgets are now bounded in config, calibrated against the baseline and dev profiles, and verified by tests that prove plan depth, evidence depth, web gating, reasoning depth, and critic depth all change with `thinking_minutes`.
- [x] 4.11 Keep the current `run_pipeline(question)` path only as a compatibility helper until `run_task(...)` is the main entrypoint.

Roadmap Adjustment Note (Current Priority Order)
- [x] Finish the remaining Phase 5 persistence work and the pulled-forward event/status contract work before starting the full Phase 6 IR redesign.
- [x] Treat `sqlite-vec` as deferred infrastructure; do not schedule it ahead of persistence, selective-loading, and event/status work.
- [x] Execute Phase 6 in three checkpoints: `6A` additive IR/storage contracts, `6B` deterministic `MacroEngine`/IR runtime, and `6C` selective-context loading plus agent integration.
- Policy: keep MediaWiki as the only real web provider until persisted web evidence and provenance-bearing task state are stable.
- Policy: do not enable Coding Mode idle practice, verified pattern promotion, or broad multi-model coding routing until typed coding storage, sandbox validation, and gated learning contracts are green.

5. Phase 5 - Storage, Chunking, Retrieval, and Web Adapters
- [x] 5.0.1 Before coding Phase 5, design repository boundaries first: event log, settings/KV, vector index, source document store, macro registry, and reasoning/performance history must be separate responsibilities.
- [x] 5.0.2 Before coding Phase 5, keep the current `StorageManager.log_event(...)` and key-value behavior working while specialized storage layers are added behind it.
- [x] 5.0.3 Before coding Phase 5, lock document chunking, deduplication, and source-identity rules so Researcher, Critic, and decoder all refer to evidence consistently; evidence handles must remain stable across re-embedding and storage migrations.
- [x] 5.0.4 Before coding Phase 5, define web-adapter timeout, retry, and degraded-mode behavior up front; web failure must never break local-only runs.
- [x] 5.0.5 Before coding Phase 5, keep `chromadb` as the pinned primary vector store for v1 using a local persistent client; do not re-open the primary choice unless the decision log changes.
- [x] 5.0.6 Before closing Phase 5, keep `sqlite-vec` as an optional secondary adapter only and defer it until after `5.1`, `5.2`, `5.11` to `5.14`, and the pulled-forward event/status work are complete.
- [x] 5.0.7 Before coding Phase 5, lock the default retrieval embedding strategy: prefer a retrieval-tuned sentence-transformers model and separate query/document encoding paths instead of one generic embedding call.
- [x] 5.0.8 Before starting full Phase 6 work, finish the remaining machine-readable persistence so IR, optimizer, and dashboard features do not have to retrofit storage later.
- [x] 5.0.9 Before broadening the web layer, persist fetched web evidence and lookup provenance in machine-readable form; web results must not stay transient-only.
- [x] 5.0.10 Before treating seeded local content as normal runtime data, lock the policy that demo/stub corpora are clearly separable from user-ingested and runtime-generated knowledge.
- [x] 5.0.11 Before broadening Coding Mode, lock repository boundaries for coding memory: pattern store, sandbox artifacts, validation history, practice sessions, coding metrics, and workspace summaries must remain separate responsibilities.
- [x] 5.1 Build a lightweight persistence layer in `storage.py` using SQLite for task runs, metrics, macros, runtime registries, and source metadata.
- [x] 5.1.1 Persist coding tasks, generated artifacts, sandbox outputs, review reports, and regression summaries in machine-readable form.
- [x] 5.2 Store append-only trace, web, and status logs as JSONL when that is simpler than relational tables.
- [x] 5.3 Implement a vector store adapter that supports insert, search, update, and delete operations.
- [x] 5.3.1 Add a coding knowledge index that can search verified patterns, candidate patterns, rejected patterns, bug fixes, refactor strategies, test patterns, and architecture templates by language, framework, and task type.
- [x] 5.4 Use Chroma PersistentClient as the primary local vector index; keep the adapter generic so it can swap to `sqlite-vec` or a simple fallback index without agent changes.
- [x] 5.4.1 Do not re-upsert the full persisted corpus into a persistent vector backend on every startup; reconcile or warm persistent collections only as needed.
- [x] 5.5 Add a chunker that splits documents before embedding; default to small bounded chunks with overlap.
- [ ] 5.5.1 Chunk and normalize code and pattern documents separately from prose documents so retrieval can respect language syntax, file path, symbol context, and framework metadata.
- [x] 5.6 Add local-first retrieval logic: search local store first, support metadata filters, score results, optionally rerank a small bounded candidate set, and only trigger web fetch if evidence quality is insufficient.
- [x] 5.6.1 When the selected embedding model supports it, use asymmetric query/document encoding APIs instead of one generic embed path.
- [x] 5.6.2 Keep reranking optional and bounded; if used, apply it only to a small top-N set so CPU-heavy runs remain practical on low-resource machines.
- [x] 5.6.3 Move long-term retrieval policy into a dedicated retrieval service/boundary instead of leaving ranking semantics inside a generic `StorageManager`.
- [x] 5.6.4 Add SQLite `FTS5` or an equivalent bounded lexical-candidate stage so hybrid retrieval does not scan all chunks in Python.
- [x] 5.6.5 Avoid deserializing every stored vector for each search; load only the chunk and vector payloads needed for merged candidates and reranking.
- [ ] 5.6.6 Add local-first coding retrieval that prefers verified patterns and relevant project-local examples before broader user corpus or web sources.
- [x] 5.7 Add a web adapter with timeout, retry, source logging, deduplication, and safe parsing.
- [x] 5.8 Never let web fetch block the app indefinitely; all network work must be optional and time-bounded.
- Policy 5.8.1: keep MediaWiki as the only real web provider until a later explicit decision confirms the persisted evidence/provenance path is mature enough to justify broader provider support.
- [x] 5.9 Split storage responsibilities into explicit repositories or tables for vectors, source documents, macro history, reasoning logs, and performance metrics instead of keeping everything behind a generic key-value layer.
- [ ] 5.9.1 Split coding repositories into explicit stores/tables for verified patterns, candidate patterns, rejected or anti-pattern records, validation history, practice session runs, and pattern-usage counters.
- [x] 5.10 Do not bury retrieval behavior in one generic storage class; keep retrieval/indexing adapters separate from event logging and settings persistence.
- [x] 5.10.1 Pull the typed `RuntimeEvent` / `AgentStatus` contract forward from Phase 9 so storage, orchestrator, and dashboard share one machine-readable event shape before the IR migration.
- [x] 5.11 Add explicit storage for the compression runtime: opcode lexicon, macro registry, decoder lexicon, local symbol-table snapshots, and proof-hash history.
- [x] 5.12 Keep the runtime source of truth machine-readable (`SQLite` tables or `JSONL`), not a giant plain-text dictionary loaded into every prompt.
- [x] 5.13 If a human-readable reference file is desired, generate `compression_lexicon.md` or `compression_lexicon.txt` from the registry as a debug/export artifact only; do not make it the primary runtime store.
- [x] 5.14 Load only the active subset of macro/opcode/decoder entries for a task; never inject the full translation registry into every model call.
- [x] 5.15 Persist chunk text, source metadata, embedding model/version, and vector IDs together so re-indexing and compatibility migrations are auditable.
- [x] 5.15.1 Persist language, framework, task type, quality score, reuse count, source provenance, and validation history alongside every learned coding pattern.
- [x] 5.16 Keep source-document storage and vector storage independently rebuildable; re-embedding or index rebuilds must never lose the original text or provenance metadata.
- [x] 5.17 Persist task runs, surfaced warnings, and additive task outputs so later orchestrator and dashboard phases do not retrofit `TaskResult` storage.
- [x] 5.18 Persist fetched web evidence alongside lookup logs and source provenance instead of treating web evidence as temporary display-only data.
- [ ] 5.18.1 Persist code review findings, bug-fix histories, refactor outcomes, and test-strategy artifacts so Coding Mode can reuse validated lessons instead of only raw code text.
- [x] 5.19 Extend `TaskResult` additively with optional answer text, warnings, and metrics before richer orchestrator and dashboard work lands.
- [x] 5.20 Emit and persist typed `AgentStatus` updates at stage start, completion, failure, fallback, and degraded-mode branches before claiming the event/status contract is stable.

6. Phase 6 - Canonical Semantic IR, Macro Registry, and Compressed Reasoning DSL
- [x] 6.0.0 Before starting full Phase 6, complete the remaining Phase 5 persistence work plus the pulled-forward event/status contract items so the IR migration has stable storage and status surfaces.
- [x] 6.0.1 Before coding Phase 6, lock the canonical IR shapes in `data_structures.py` first; do not begin `MacroEngine` work until the graph, opcode, symbol-table, context-frame, and proof-hash structures are defined.
- [x] 6.0.2 Before coding Phase 6, treat `CompressedTrace.tokens` and `expanded_preview` as legacy/debug projections that must remain available throughout the migration.
- [x] 6.0.3 Before coding Phase 6, implement deterministic expansion and validation before optimizer-learned compression; read/verify paths must exist before write/learn paths.
- [x] 6.0.4 Before closing Phase 6, benchmark and reduce stub-trace persistence overhead so the richer IR stays within a bounded accepted envelope instead of exploding far beyond the legacy placeholder shape.
  Accepted compact-persistence baseline: about `1.42x` to `1.61x` serialized JSON overhead and about `2.21x` to `2.59x` recursive payload-memory overhead versus the legacy projection, measured from `CompressedTrace.to_storage_dict()` and locked by the Phase 6 density test.
- [x] 6.0.5 Before coding Phase 6, lock provenance primitives first: entity, activity, agent, and bundle shapes must be explicit in the IR instead of implicit metadata blobs.
- [x] 6.0.6 Split Phase 6 into explicit checkpoints: `6A` additive IR/storage contracts, `6B` deterministic `MacroEngine` runtime, and `6C` selective-context loading plus agent integration.
- [x] 6.0.7 Complete `6A` and the compatibility fixtures before replacing placeholder foreground-agent behavior.
- [x] 6.0.8 Complete `6B` proof-hash and round-trip validation before enabling optimizer-driven compression changes.
- [x] 6.0.9 Complete `6C` active-subset loading before claiming selective-context injection is done.
- [x] 6.1 Treat Phase 6 as an architecture redesign of `MacroEngine`, not a light enhancement.
- [x] 6.2 Define a canonical semantic intermediate representation (IR) before designing shorthand tokens; the compression language must preserve proof semantics, provenance, and uncertainty as first-class semantics rather than best-effort annotations.
- [x] 6.3 Represent reasoning internally as a graph/DAG of typed semantic nodes and dependencies, not just a flat list of string tokens.
- [x] 6.3.1 Model evidence items, derived claims, answer fragments, macro definitions, and intermediate bindings as typed entities with stable IDs.
- [x] 6.3.2 Model retrieval, reasoning, critique, and compression passes as typed activities linked to the entities they use and produce.
- [x] 6.3.3 Model component/model ownership as typed agents so traces can record which service and backend produced each artifact.
- [ ] 6.3.4 Allow code-task provenance to link source files, patch candidates, tests, lint results, static-analysis findings, and regression evidence as first-class typed entities or referenced artifacts.
- [x] 6.4 Define a compact typed opcode lexicon with fixed semantics for core operations such as lookup, bind, compare, infer, aggregate, check, emit, cite, and confidence update.
- [x] 6.5 Keep the core opcode lexicon small and stable; store it once and treat it as part of the runtime, not as repeated prompt text.
- [x] 6.6 Add a machine-readable macro registry with versioned, parameterized macros; each macro must declare its signature, canonical expansion, invariants, and proof fingerprint.
- [x] 6.7 Add a per-task local symbol table that maps entities, predicates, attributes, evidence handles, and temporary registers to short IDs for dense encoding.
- [x] 6.8 Add context frames so provenance sets, confidence, scope, and active assumptions can be inherited instead of repeated on every operation.
- [x] 6.9 Compile the canonical graph into a compact operation stream / bytecode representation inside `CompressedTrace` instead of relying on plain string-token sequences as the primary meaning store.
- [x] 6.10 Extend `CompressedTrace` as needed to carry IR version, operation stream, symbol-table references, evidence handles, context frames, proof hash, and decode hints while preserving current public compatibility.
- [x] 6.11 Implement `compress()` around repeated graph motifs and parameterized macros, not exact-token aliases.
- [x] 6.12 Implement `expand()` by resolving macro IDs, replaying the operation stream, and reconstructing the canonical graph deterministically.
- [x] 6.13 Add structural hashing / proof fingerprints for expanded graphs so equivalent reasoning chains can be deduplicated and verified cheaply.
- [x] 6.14 Add recursion detection with a visited-set and a strict max depth such as `8`.
- [x] 6.15 Add canonical normalization rules for graphs, bindings, and opcode order where order is semantically irrelevant; reject noncanonical encodings.
- [x] 6.16 Implement `verify_round_trip()` as `compressed -> expanded canonical graph -> normalized graph -> recompressed` and require proof-hash stability, not just text equality.
- [x] 6.17 Reject macro proposals that fail loop checks, fingerprint checks, provenance preservation, uncertainty preservation, or semantic consistency checks.
- [x] 6.18 Replace the current identity-style placeholder compression with real graph compilation, macro expansion, and validation logic.
- [x] 6.19 Add a decoder lexicon that maps internal symbols and verified proof structures back to natural-language templates, but keep decoding separate from reasoning.
- [x] 6.19.1 Add a compact human-readable debug/export view for the IR and provenance graph so operators can inspect a task without reading raw JSON blobs.
- [ ] 6.19.2 Add decode/debug helpers that can render verified code-task reasoning plus sandbox and quality evidence into operator-readable summaries without exposing raw scratch reasoning.
- [x] 6.20 Use selective context injection only: Reasoner and Critic may load active opcodes, active macro definitions, and the current local symbol table, but must never carry the whole registry in context.
- [x] 6.21 Ensure macro proposals are validated before they can influence Reasoner, Critic, or Self-Optimizer flows.

7. Phase 7 - Implement the Foreground Agents and Verification Layer
- [x] 7.0.1 Before coding Phase 7, keep agents thin and move shared logic into services; Planner, Researcher, Reasoner, Critic, and Compressor should orchestrate behavior, not own storage/runtime internals.
- [x] 7.0.2 Before coding Phase 7, define the exact handoff shape from Researcher -> Reasoner and Reasoner -> Critic so evidence handles, proof structures, and repair data do not drift.
- [x] 7.0.3 Before coding Phase 7, implement deterministic or stubbed versions of the new agent logic first, then layer real model-backed inference behind the same contracts.
- [x] 7.0.4 Before coding Phase 7, keep translation to natural language as a final step after verification; do not let free-form text become the primary reasoning state again.
- [x] 7.0.5 Before coding Phase 7, lock one structured-output path for model-backed agents; Planner, Reasoner, Critic, and Compressor must emit schema-constrained JSON rather than free-form text.
- [x] 7.0.5.1 When schema validation moves from parse-only decoding to enforced boundary validation, install `jsonschema` as the first optional structured-output dependency; do not add a heavier schema stack before the current `agent_schema.py` path shows clear limits.
- [x] 7.0.6 Before coding Phase 7, keep dataclasses as the domain model and use boundary-only schema helpers to generate and validate model I/O contracts.
- [x] 7.0.6.1 Keep `outlines` and `msgspec` off the default path unless the current dataclass plus boundary-schema approach proves insufficient; do not replace the Ollama-oriented JSON path or the dataclass domain model just to adopt a library.
- [x] 7.0.7 Before replacing placeholder foreground-agent behavior, require the Phase 5 persistence layer and the `6A` / `6B` IR checkpoints to be green.
- [x] 7.0.8 Before coding Phase 7, optimize for a system-first lightweight reasoner: keep the base policy model small and put most precision gains into structured search, tools, and verification rather than assuming a larger standalone model.
- [x] 7.0.9 Before coding Phase 7, prioritize verifiable reasoning domains first (math, code, logic, and retrieval-backed factual synthesis); broad conversational polish is secondary.
- [x] 7.0.10 Before coding Phase 7, support at least two bounded reasoning modes: `fast` for low-latency single-path work and `deep` for slower multi-candidate verified reasoning.
- [ ] 7.0.11 Before broadening Coding Mode, freeze one coding-role orchestration contract covering `planner`, `generator`, `debugger`, `reviewer`, `test_writer`, `summarizer`, and `refactorer` so routing and UI stay explainable.
- [x] 7.1 Replace remaining stub or placeholder agent behavior with real task-aware logic while preserving typed contracts.
- [x] 7.2 Implement `PlannerAgent` to return structured `Plan` objects only via schema-constrained output; do not let it return free-form text as its primary output.
- [ ] 7.2.1 Extend `PlannerAgent` or a coding-planner service to decompose coding tasks by file scope, risk, validation plan, and required coding roles while preserving the current public planner contract.
- [x] 7.3 Implement `ResearcherAgent` to accept a `Plan`, retrieve local evidence first using stable evidence handles and retrieval-tuned embeddings, optionally fetch web evidence, embed/store new content, and return `EvidenceBundle`.
- [ ] 7.3.1 Extend `ResearcherAgent` or a coding-memory retrieval service to fetch verified local code patterns, bug-fix precedents, test strategies, and anti-pattern warnings before generation.
- [x] 7.4 Implement `ReasonerAgent` to consume the `Plan` and `EvidenceBundle`, build the canonical reasoning/provenance graph, compile it into the compressed IR, and stay within the `ResourceBudget`.
- [x] 7.4.1 In `deep` mode, let the Reasoner generate multiple bounded candidate traces instead of one single trajectory; candidates must stay in the canonical IR, not free-form scratchpads.
- [x] 7.4.2 Treat the small model as a proposal generator, not the final authority; final selection must be driven by verifier, evidence, and proof-hash signals.
- [ ] 7.4.3 Add a coding generator/refactorer path that produces bounded patches or file artifacts rather than free-form prose and records per-candidate route/model choices.
- [ ] 7.4.4 Support coding task types at minimum: feature generation, bug fixing, refactoring, test generation, code review, explanation or summarization, project scaffolding, and architecture planning.
- [x] 7.5 Make the Reasoner reuse the per-task symbol table, active opcode lexicon, and active macro registry subset instead of emitting ad hoc symbolic strings.
- [x] 7.6 Implement `CriticAgent` as a hybrid strict verifier: schema validation -> deterministic provenance/IR constraint checks -> deterministic macro/IR expansion -> optional model-assisted semantic fallback only if structural checks pass.
- [x] 7.6.1 Add tool-backed verification for checkable tasks: Python execution, code or unit-test execution, and retrieval-grounded checks should be first-class Critic helpers before optional model fallback.
- [ ] 7.6.1.1 When deterministic contradiction or satisfiability checks become part of the Critic, install `z3-solver` as an optional helper for math, logic, and ordering/constraint validation instead of hand-rolling a full constraint engine first.
- [ ] 7.6.1.2 Add a sandbox-backed code execution path with isolated temp workspaces, interpreter/tool allowlists, timeouts, output capture, and explicit blocked-action handling.
- [ ] 7.6.1.3 Add bounded lint, static-analysis, complexity, maintainability, and security-review helpers as first-class Coding Mode verification tools before optional model-only review.
- [ ] 7.6.1.4 Add regression checks that compare new patches against baseline tests, targeted failure reproductions, and bounded project-specific checks before promotion or answer finalization.
- [ ] 7.6.1.5 Add a coding reviewer path that can critique generated code for correctness, clarity, safety, and maintainability using machine-readable structured findings.
- [ ] 7.6.1.6 Add a test-writer path that can propose or repair tests, but only accept them after they pass the same sandbox and quality gates as generated code.
- [x] 7.6.2 Keep tool-backed verification bounded and explicit; helper execution must respect the same timeout, retry, and resource policies as the rest of the runtime.
- [x] 7.7 Make the Critic validate plan coverage, evidence support, opcode legality, ordering/type/impossibility constraints, macro signature matching, contradiction surfacing, uncertainty preservation, provenance coverage, and proof-hash stability.
- [x] 7.7.1 Make the Critic surface whether failures are schema, provenance, proof-hash, or evidence-coverage violations so repair paths stay targeted instead of generic.
- [x] 7.7.2 Make the Critic score candidates by verifier result, evidence support, proof-hash stability, and cross-candidate agreement so `deep` mode can select, repair, or abstain instead of always emitting the most fluent answer.
- [x] 7.7.3 If no candidate reaches the configured verification threshold, return a structured uncertainty or degraded result rather than a polished but weakly supported answer.
- [ ] 7.7.4 Make the Critic enforce Coding Mode promotion gates: tests, lint, static analysis, complexity, security, maintainability, critique pass, and regression checks must all be represented explicitly in the final verdict.
- [x] 7.8 Extend `CritiqueReport` as needed to expose drift score, proof-hash match, provenance coverage, macro violations, and repair actions while preserving current compatibility where possible.
- [ ] 7.8.1 Extend `CritiqueReport` or additive coding review artifacts with coding-specific findings such as bug risk, regression risk, security concerns, maintainability issues, and test sufficiency.
- [x] 7.9 Implement `CompressorAgent` to learn parameterized graph motifs, frequent subproofs, and symbol-table optimization opportunities; exact-token aliases must not be the main compression strategy.
- [x] 7.10 Add a translation layer that expands only the verified output-relevant proof subgraph and renders it to natural language using templates first and LLM surface realization second.
- [x] 7.10.1 Keep natural-language generation downstream of candidate selection and verification; the translation layer must render verified state, not invent new reasoning.
- [ ] 7.10.2 Add a coding summarization layer that can explain what changed, why it was accepted or rejected, and which checks passed without losing the underlying machine-readable evidence.
- [x] 7.11 Keep each agent small and single-purpose; shared concerns like logging, model access, storage, symbol-table management, and proof hashing must stay in shared services.
- [x] 7.12 Make budget a real input to foreground agents so work depth changes with the thinking-time policy.
- [x] 7.12.1 Let larger thinking budgets increase candidate count, verifier depth, and tool-use budget in `deep` mode without violating the locked RAM and VRAM limits.
- [ ] 7.12.2 Let larger coding budgets buy additional candidate patches, deeper code review, more regression coverage, broader test generation, and more selective memory promotion rather than longer single generations.
- [x] 7.13 Do not claim foreground-agent completion while local retrieval, critic verification depth, graph-backed compression, and macro-aware reasoning are still stubbed.
- [x] 7.14 If a model output fails schema validation, allow at most one bounded repair attempt before returning a structured failure.
- [x] 7.14.1 Freeze the Phase 7 acceptance contract: bounded `fast` / `deep`, persisted candidate traces, verifier-backed selection or abstention, repair-aware orchestration, cited translation from verified state, and green tests are the minimum definition of completion.

8. Phase 8 - Implement the Background Self-Optimizer
- [x] 8.0.1 Before coding Phase 8, keep the optimizer read-only with respect to live runtime state until replay/simulation infrastructure is in place.
- [x] 8.0.2 Before coding Phase 8, lock the evaluation metrics for “better” proposals: compression gain, proof-hash stability, critic validity, latency, and memory cost.
- [x] 8.0.3 Before coding Phase 8, define proposal persistence and rollback records first so every activation decision is auditable.
- [x] 8.0.4 Before coding Phase 8, cap history windows and replay scope so the optimizer does not become the largest memory consumer in the system.
- [x] 8.0.5 Until `8.5` to `8.9` exist, keep the optimizer disabled by default or clearly proposal-only in stub/dev mode; do not treat placeholder cycles as live optimization.
- [x] 8.0.6 Before coding Phase 8, treat any future model improvement as distillation from verified system traces first; do not plan training-from-scratch as the default path.
- [ ] 8.0.7 Before enabling Coding Mode background learning, keep idle practice opt-in, workspace-isolated, resource-bounded, and separate from live user projects unless explicitly allowed.
- [x] 8.1 Implement `SelfOptimizer` as a background async service that never blocks the foreground task pipeline.
- [x] 8.2 Read reasoning logs, macro usage, critique results, and performance metrics on a schedule.
- [x] 8.2.1 Persist candidate-trace sets, verifier scores, agreement scores, tool outputs, and final adjudication so replay and analysis can learn from `deep` mode runs.
- [ ] 8.2.2 Read coding task histories, sandbox results, lint/static-analysis reports, regression outcomes, and pattern reuse metrics alongside reasoning logs and performance data.
- [x] 8.3 Generate only proposals, never direct live changes.
- [x] 8.4 Enforce the lifecycle `propose -> simulate -> validate -> activate`.
- [x] 8.5 Build a small simulation harness that reruns sample tasks using candidate macro changes and compares success and efficiency metrics.
- [x] 8.5.1 Export only verified `deep`-mode traces as datasets for future SFT or process-supervision training; never distill unverified free-form chain-of-thought.
- [x] 8.5.2 Build a coding-practice harness that can generate or select bounded exercises during idle windows, run candidate solutions, run tests/linting/static checks, score results, and record outcomes.
- [ ] 8.5.3 Keep practice tasks sourced from local fixtures, user-approved corpora, or generated task templates; do not let idle practice mutate real user projects by default.
- [x] 8.6 Activate a proposal only if it passes validation and improves results without increasing contradiction risk.
- [x] 8.6.1 Gate coding-memory promotion: only promote patterns to `verified` after passing tests, lint, static analysis, complexity, security, maintainability, critique, and regression checks; store failures as `candidate` or `rejected` with reasons.
- [x] 8.7 Keep the currently active macro set immutable during a foreground task run.
- [x] 8.8 Replace the current placeholder optimizer cycle with real proposal generation based on persisted reasoning, critique, and performance data.
- [x] 8.8.1 If small-model training is pursued later, begin with distilling verified `deep`-mode behavior into the existing lightweight policy model before considering architecture research or larger local models.
- [ ] 8.8.2 Add coding-pattern discovery over validated code tasks and idle practice runs, including reusable bug-fix motifs, refactor recipes, test strategies, and architecture templates.
- [x] 8.9 Track activation decisions and failed simulations so regressions are visible and rollback is explicit.
- [ ] 8.9.1 Track coding-practice session outcomes, pattern promotions, rejection reasons, reuse counts, and regression-detected events so later analysis can distinguish helpful from harmful learning.

9. Phase 9 - Orchestrator and Event Flow
- [x] 9.0.0 Phase 9 sequencing note: the typed event/status contract and additive `TaskResult` fields should already exist from Phase 5; Phase 9 finishes orchestration behavior on top of those contracts.
- [x] 9.0.1 Before coding Phase 9, lock the event schema first so storage, dashboard, and tests all observe the same stage/status payload shape.
 ioy- [x] 9.0.2 Before coding Phase 9, define cancellation, retry, and surfaced-warning semantics before adding more runtime branches to the orchestrator.
- [x] 9.0.3 Before coding Phase 9, keep orchestration logic separate from agent logic; `Orchestrator` should compose services, not absorb reasoning or storage rules.
- [x] 9.0.4 Before coding Phase 9, preserve `run_task(...)` as the primary path and keep `run_pipeline(...)` as a thin compatibility wrapper only.
- [x] 9.0.5 Before coding Phase 9, make degraded-mode/resource-pressure/fallback events first-class so UI, logs, and tests can surface why the runtime changed behavior.
- [x] 9.0.6 Before wiring Coding Mode through orchestration, lock typed coding event/status codes for planning, generating, testing, debugging, reviewing, indexing, practicing, and regression-detected states so UI and storage stay aligned.
- [x] 9.1 Build `Orchestrator` as the only component that wires the main workflow together.
- [x] 9.2 Implement the foreground sequence exactly as `Planner -> Researcher -> Reasoner -> Critic -> Compressor -> Dashboard`.
- [ ] 9.2.1 Add a coding-task orchestration path that can route through coding planner, generator, debugger, reviewer, test-writer, summarizer, and sandbox verification stages while preserving the main orchestrator boundary.
- [x] 9.3 Add an internal event stream or bounded status queue so the dashboard can display live progress without tight coupling.
- [x] 9.4 Record `AgentStatus` updates at each stage start, completion, retry, failure, fallback, and degraded-mode branch.
- [x] 9.4.1 Emit machine-readable coding events for active model route, current file or patch scope, sandbox phase, test status, lint status, indexing status, practice session activity, and regression detection.
- [x] 9.5 Add cancellation handling so a user can stop a long task safely.
- [x] 9.6 Add retry logic only for safe transient failures such as timeouts; do not retry invalid outputs forever.
- [x] 9.7 Return a final `TaskResult` containing the plan, evidence summary, verified trace, answer, metrics, and surfaced warnings.
- [ ] 9.7.1 Extend final task outputs additively with coding artifacts, patch summaries, validation reports, and learned-pattern references so history and UI can present Coding Mode without parsing raw logs.
- [x] 9.8 Expose `run_task(question, thinking_minutes)` as the primary public entrypoint and treat `run_pipeline(question)` as legacy until it can be removed.
- [x] 9.9 Carry resource budget, status events, and surfaced warnings through the full orchestration path so later acceptance gates can be tested directly.

10. Phase 10 - Dashboard Implementation
- [x] 10.0.1 Before coding Phase 10, keep headless mode and UI mode behaviorally aligned so tests and non-GUI runs remain valid.
- [x] 10.0.2 Before coding Phase 10, treat the dashboard as a read-only event consumer; it must not become a source of pipeline state truth.
- [x] 10.0.3 Before coding Phase 10, define queue backpressure and event-drop behavior so the UI cannot cause memory growth under heavy activity.
- [x] 10.0.4 Before coding Phase 10, keep UI updates off the main pipeline thread and expose degraded resource/error states explicitly rather than hiding them.
- [x] 10.0.5 Before coding Phase 10, treat the dashboard as the shell of the local app, not only an event console; runtime controls, history, settings, and readiness surfaces should be planned together so the UI does not need a second redesign later.
- [x] 10.0.6 Before coding Phase 10, define a typed app-state or view-model projection for the UI so Tkinter panes consume stable structured state instead of parsing raw event dicts directly.
- [x] 10.0.7 Before coding Phase 10, keep future desktop, observation, and cloud controls visible but capability-gated; when a control is unavailable the UI must explain whether the reason is policy, missing dependencies, or resource limits.
- [ ] 10.0.8 Before broadening the shell, treat Coding Mode as a first-class workspace mode; coding panels, orb states, and activity strips must be driven from typed shell or dashboard state rather than ad hoc widget logic.
- [x] 10.1 Use `Tkinter` by default to keep memory overhead low.
- [x] 10.2 Create panels for question input, thinking-time slider, agent status, runtime health, evidence inspector, macro/provenance inspector, web query log, final answer, and critique summary.
- [ ] 10.2.1 Add Coding Mode surfaces for active workspace, file/task status, patch preview, test results, lint/static-analysis summaries, pattern-learning summaries, practice history, and active coding models.
- [ ] 10.2.2 Add a coding workspace mode switch that can pivot the shell from general assistant flow to coding flow without fragmenting settings, history, or model/control-plane visibility.
- [x] 10.3 Show live status updates by polling a bounded queue or using a thread-safe event bridge.
- [x] 10.3.1 Show live coding status updates for planning, generating, refactoring, testing, debugging, reviewing, indexing, practicing, learning pattern, and regression detected.
- [x] 10.4 Display current resource usage when available, including RAM and VRAM estimates.
- [x] 10.5 Keep the UI responsive by running model and I/O work outside the main UI thread.
- [x] 10.6 Add clear error states and recovery actions so a failed web fetch or low-memory event does not crash the dashboard.
- [x] 10.7 Replace the current text-only event console with the real structured dashboard panels above; the raw event log view may remain only as a debug pane.
- [x] 10.8 Show when events were dropped due to queue backpressure so UI silence cannot hide runtime activity.
- [x] 10.9 Add a persisted settings surface for runtime/backend, retrieval/web, reasoning, long-horizon control, optimizer policy, desktop control, observation tiers, cloud offload, privacy/logging, and UI preferences instead of requiring users to edit `config.py`.
- [x] 10.10 Add named settings profiles with validation, safe defaults, reset-to-defaults, import/export, and clear unsupported-state messaging so advanced controls remain lightweight but understandable.
- [x] 10.11 Add task history and run-inspector views that surface prior answers, citations, candidate counts, critique outcomes, degraded reasons, repair actions, optimizer lifecycle records, and export links.
- [ ] 10.11.1 Extend task history to store coding sessions, patch summaries, validation results, model-route history by coding role, learned-pattern promotions, and practice session outcomes.
- [x] 10.12 Add a local knowledge-library panel that can ingest documents, show source metadata, keep demo corpus separate from user corpus, rebuild embeddings or indexes, and remove or archive sources without requiring direct SQLite work.
- [ ] 10.12.1 Extend the knowledge-library panel with a coding memory view for verified patterns, candidate patterns, rejected patterns, bug-fix histories, refactor recipes, test strategies, and architecture templates.
- [x] 10.13 Add a preflight and readiness panel that reports stub-mode readiness, real-backend dependency status, backend health, model availability, optional capability availability, and actionable setup guidance.
- [ ] 10.13.1 Extend readiness and setup guidance to include coding-model availability, sandbox prerequisites, interpreter/tooling requirements, lint/test/security tool availability, and blocked capability reasons.
- [x] 10.14 Surface why a capability or setting is unavailable, degraded, or blocked by policy so the operator does not have to read logs to understand the current runtime envelope.
- [x] 10.15 Keep the raw event log only as a collapsible debug pane after the structured app shell exists; normal operation should happen through purpose-built panels instead of scrolling dict output.
- [ ] 10.15.1 Keep raw code execution logs and sandbox stdout/stderr as debug surfaces only; normal Coding Mode UX must surface structured summaries first.
- [x] 10.16 Add coding-specific orb mappings for `Code Planning`, `Generating`, `Refactoring`, `Testing`, `Debugging`, `Reviewing`, `Indexing`, `Practicing`, `Learning Pattern`, and `Regression Detected`.
- [x] 10.16.1 For each coding state, define orb palette, pulse style, ring behavior, particle density, status text, background tint, and shell accent response so coding activity is legible without reading raw logs.
- [x] 10.16.2 Add coding activity chips and progress ribbons showing active coding role, current validator, sandbox/test progress, pattern tier transitions, and regression warnings.
- [ ] 10.16.3 Use these default coding visual mappings: `Code Planning -> cyan structured pulse`, `Generating -> amber-gold active weave`, `Refactoring -> blue-violet contraction and rewire`, `Testing -> green-white checkpoint pulses`, `Debugging -> orange-red focused search ring`, `Reviewing -> purple validation sweep`, `Indexing -> teal inward particles`, `Practicing -> indigo-violet dojo cadence`, `Learning Pattern -> white-gold compression flash`, and `Regression Detected -> red alert ring with degraded undertone`.
- [ ] 10.17 Surface coding metrics in the shell: pass/fail rate, lint/static-analysis counts, regression rate, practice score trend, pattern reuse score, memory growth, and per-model performance by coding task type.

11. Phase 11 - Seed Data, Example Tasks, and Sample Macros
- [x] 11.0.1 Before coding Phase 11, keep all fixtures small, deterministic, and versioned so they remain stable across optimizer and compression changes.
- [x] 11.0.2 Before coding Phase 11, separate local-only demo tasks from web-fallback demo tasks so each behavior can be tested independently.
- [x] 11.0.3 Before coding Phase 11, ensure starter macros and opcode entries are conservative and fully validated; sample data must not depend on optimizer-learned definitions.
- [x] 11.0.4 Before coding Phase 11, include at least one contradiction/conflict example so Critic and decoder behavior can be verified under non-happy paths.
- [x] 11.0.5 Before coding Phase 11, include at least one clearly asymmetric retrieval corpus (question -> passage) so query/document encoding paths are tested honestly instead of only symmetric similarity cases.
- [x] 11.0.6 Before coding Phase 11, keep seed/demo corpora opt-in or clearly marked as stub/demo-only; they must not masquerade as user-ingested knowledge.
- [x] 11.0.7 Before coding Phase 11, include tasks with exact or executable checks so `fast` vs `deep` reasoning can be compared on correctness, not only presentation quality.
- [ ] 11.0.8 Before coding broader Coding Mode demos, add small deterministic coding fixtures with known-good answers, bug-fix tasks, refactor tasks, and anti-pattern examples so practice and validation remain reproducible.
- [x] 11.1 Add a small local knowledge sample so the system can demonstrate local-first retrieval without requiring web access; chunk IDs, source hashes, and metadata filters should be part of the fixture.
- [x] 11.2 Add a few starter macros that are simple, safe, and useful for repeated reasoning patterns.
- [x] 11.3 Add 3-5 sample user questions that exercise planning, retrieval, reasoning, critique, and compression.
- [x] 11.3.1 Ensure the starter task set includes math, code, logic or planning, and multi-hop retrieval-backed questions with concrete pass-fail or high-precision checks where possible.
- [ ] 11.3.2 Add starter coding tasks for feature generation, bug fixing, refactoring, test writing, code review, project scaffolding, and architecture planning across at least one lightweight language/framework stack.
- [x] 11.4 Add a fake or stub web adapter for tests so CI does not depend on live internet.
- [x] 11.5 Add one example task that intentionally triggers web fallback and one that stays fully local.
- [ ] 11.5.1 Add at least one coding sample that requires sandbox execution and one that fails promotion because tests, lint, or security checks reject it.
- [x] 11.6 Ensure at least one sample task exercises the `thinking_minutes -> ResourceBudget -> agent behavior` flow end to end.
- [x] 11.6.1 Include at least one pair where `fast` is weaker or fails and `deep` succeeds through extra candidates, tools, or verification.
- [x] 11.7 Add a small starter opcode lexicon, macro registry, and decoder lexicon so the compressed IR can be demonstrated without requiring the optimizer to invent everything first.
- [x] 11.8 Add one starter provenance bundle example that shows evidence -> reasoning activity -> verified claim links end to end.
- [ ] 11.8.1 Add coding provenance examples linking source files, patch candidates, test evidence, critique findings, and promoted or rejected patterns end to end.
- [x] 11.9 Add one small verified-trace export example so future replay, distillation, and process-supervision code can be exercised without real model training.
- [ ] 11.9.1 Add a small validated coding-practice export example so future replay and optimizer analysis can exercise Coding Mode without live model training.

12. Phase 12 - Tests, Validation, and Best Practices
- [x] 12.0.1 Before coding Phase 12, prioritize tests by risk: compatibility, resource limits, proof-hash stability, provenance preservation, and fallback behavior come before cosmetic coverage gains.
- [x] 12.0.2 Before coding Phase 12, keep stub-mode tests as the default CI path and make real-backend tests opt-in or smoke-only so the suite stays portable.
- [x] 12.0.3 Before coding Phase 12, add regression fixtures for old serialized payloads and legacy `CompressedTrace` shapes before the IR migration lands.
- [x] 12.0.4 Before coding Phase 12, define minimum acceptance thresholds for validity, compression improvement, and bounded resource usage so “passing” is concrete.
- [x] 12.0.4.1 For compression acceptance, define explicit minimum thresholds for realized savings, proof-hash stability, critic-validity preservation, capped proposal count, and bounded scan window so compressor quality stays measurable and lightweight.
- [ ] 12.0.4.2 For Coding Mode acceptance, define minimum thresholds for sandbox containment, test/lint/security pass rate, regression detection coverage, bounded pattern promotion, and per-role route visibility.
- [x] 12.0.5 Before coding Phase 12, add a reproducible environment manifest (`pyproject.toml` with optional extras) before claiming any real-backend path is supported.
- [ ] 12.0.5.1 When schema validation, JSONL logging, or structured-generation parsing becomes a measured hot path, install `orjson` as an optional performance dependency; keep stdlib `json` until profiling shows a real bottleneck.
- [ ] 12.0.5.2 When proof-hash, macro round-trip, and compatibility invariants need stronger generative coverage, install `hypothesis` as the first property-testing dependency instead of expanding only example-based tests.
- [x] 12.0.6 Before coding Phase 12, define acceptance criteria for `fast` vs `deep`: `deep` mode must improve correctness on at least one verifiable benchmark family as test-time compute increases.
- [x] 12.1 Write unit tests for every agent with mocked model and storage backends.
- [x] 12.1.1 Add unit tests for candidate generation, verifier scoring, consensus or adjudication, abstain logic, and tool-backed verification helpers.
- [x] 12.1.2 Add unit tests for coding-role routing, patch generation boundaries, reviewer/test-writer outputs, and coding-task summarization.
- [x] 12.2 Write unit tests for `MacroEngine` compression, expansion, nested macros, recursion blocking, and round-trip verification.
- [x] 12.3 Write retrieval tests for local-first behavior, metadata filtering, query/document encoding split, bounded reranking, web fallback behavior, deduplication, and empty-result handling.
- [x] 12.3.1 Add retrieval tests for lexical-candidate generation, merged FTS/vector ranking behavior, startup reconciliation for persistent vector indexes, and selective vector loading.
- [ ] 12.3.2 Add coding-memory retrieval tests for language/framework filtering, verified-vs-candidate ranking, anti-pattern suppression, and reuse-count updates.
- [x] 12.4 Write resource tests for semaphore limits, queue limits, low-memory fallback, Ollama keep-alive/unload behavior, and model unload behavior.
- [x] 12.4.1 Write resource tests proving `deep` mode remains bounded as candidate count, tool use, and verifier depth increase.
- [ ] 12.4.2 Add resource tests proving multiple local coding models still obey the heavy-slot cap and that sandbox/practice jobs do not starve foreground tasks.
- [x] 12.5 Write optimizer tests proving proposals are simulated before activation.
- [ ] 12.5.1 Add optimizer and practice tests proving rejected coding patterns never auto-promote and verified patterns require the full gated path.
- [x] 12.6 Write end-to-end pipeline tests for at least two example questions.
- [x] 12.6.1 Add end-to-end tests comparing `fast` vs `deep` on at least one math, code, or logic sample and assert that extra test-time compute can improve correctness without breaking budgets.
- [x] 12.6.2 Add end-to-end tests for tool-backed verification paths, including Python execution or unit-test-backed checks where applicable.
- [ ] 12.6.3 Add end-to-end coding workflow tests covering feature generation, bug fixing, refactoring, test generation, review, summarization, and project scaffolding.
- [ ] 12.6.4 Add idle-practice tests proving the system can generate or select exercises, solve them in a sandbox, score outcomes, and persist candidate/verified/rejected patterns without touching live projects.
- [x] 12.7 Write failure-path tests for malformed macro definitions, invalid plan output, web timeout, empty evidence, and critic rejection.
- [ ] 12.7.1 Add failure-path tests for sandbox timeout, blocked system calls, failing lint/security checks, regression detection, invalid patch application, and rejected pattern promotion.
- [x] 12.8 Keep prompts, parsing, storage, and UI code separate to avoid circular dependencies and fragile edits.
- [x] 12.9 Use dependency injection for models, storage, and web adapters so components are easy to mock and swap.
- [x] 12.10 Keep comments concise and useful; explain non-obvious logic, resource guards, and validation rules.
- [x] 12.11 Add a short README with setup steps, optional dependency extras, local service/model prerequisites, architecture notes, runtime limits, and how both the locked 6GB baseline and the 4GB dev profile are enforced.
- [x] 12.12 Do a final pass to remove dead code, duplicate logic, hidden globals, and unbounded caches.
- [ ] 12.12.1 Do a final pass on Coding Mode to remove unbounded artifact retention, stale temp workspaces, duplicate pattern entries, and silent sandbox failures.
- [x] 12.13 Add explicit tests for the top-level public API checklist so phase completion and interface compliance cannot drift apart.
- [x] 12.14 Do not mark the project feature complete while placeholder runtime paths or text-only dashboard behavior remain.
- [x] 12.15 Add proof-hash stability tests so semantically equivalent encodings normalize to the same canonical fingerprint.
- [ ] 12.15.2 If `hypothesis` is installed, add property tests for proof-hash stability, macro round-trip invariants, and structured-payload compatibility fuzzing before broadening lower-value coverage.
- [x] 12.15.1 Add candidate-agreement and consensus tests so `deep` mode selection is based on verifier and agreement score rather than text fluency alone.
- [x] 12.16 Add failure-path tests for provenance loss, uncertainty erasure, macro signature mismatch, and noncanonical encodings.
- [x] 12.17 Add selective-context tests proving only active opcode/macro/symbol-table entries are loaded for a task instead of the whole registry.
- [x] 12.18 Add translation-layer tests that verify final natural-language output preserves the verified claim, evidence basis, and uncertainty markers.
- [x] 12.19 Add setup/smoke checks that fail clearly when required real-mode dependencies, local services, or model files are missing.
- [x] 12.20 Add dataset-export tests ensuring only verified `deep` traces are eligible for future SFT or process-supervision corpora.
- [x] 12.21 Add GUI-state tests for the typed app-state projection, settings round-trip behavior, disabled-control reasons, and headless-vs-UI parity so the lightweight Tkinter app stays reliable.
- [x] 12.21.1 Add UI-state tests proving coding orb states, activity chips, practice history, model-route displays, and validator results stay synchronized with typed backend state.
- [x] 12.22 Add knowledge-library and task-history tests covering document ingest, source removal, demo-vs-user corpus separation, run-history browsing, and export actions.
- [ ] 12.22.1 Add coding-memory and practice-history tests covering pattern promotion or demotion, metadata edits, archive/removal behavior, and run-history browsing.
- [x] 12.23 Add preflight tests for missing Ollama, missing embedding/vector extras, missing model files, unsupported optional capability toggles, and degraded hardware-governor paths.
- [x] 12.24 Add packaged-app smoke tests for the Windows release path: launch, first-run stub mode, readable dependency failures, and log/export bundle generation.
- [ ] 12.24.1 Add packaged-app smoke tests for Coding Mode setup guidance, sandbox dependency failures, coding-model readiness, and safe fallback when optional code tooling is unavailable.

Acceptance Checklist
- [x] The system runs locally as an async, modular, multi-agent pipeline.
- [ ] The system preserves every requested feature and workflow stage.
- [x] The system remains designed for 6GB VRAM and 8GB RAM through shared runtimes, bounded workloads, and fallback behavior.
- [x] The macro engine supports nested macros and round-trip verification safely.
- [x] The compression system uses a canonical graph-backed IR with parameterized macros, proof hashing, and per-task symbol tables instead of plain string-token shorthands.
- [x] The Researcher uses local retrieval before web access.
- [x] The runtime supports bounded `fast` and `deep` reasoning modes, and `deep` mode can trade latency for higher correctness on verifiable tasks without breaking local resource limits.
- [x] For checkable tasks, final answer selection is verifier or tool-backed rather than driven only by text fluency.
- [x] Hard unverified cases degrade to structured uncertainty or surfaced warnings instead of confident unsupported answers.
- [x] Runtime events and agent statuses are machine-readable, persisted, and shared across storage, orchestrator, and dashboard surfaces.
- [x] The Self-Optimizer never applies live changes without simulation and validation.
- [x] The dashboard works and shows live progress, logs, macro usage, and results.
- [x] The local app has a polished lightweight Tkinter shell with persisted settings, profiles, task history, knowledge management, and real-time readiness or health surfaces rather than only a raw event console.
- [ ] The app exposes one integrated local-AI control plane that shows which local roles or models handled work, which optimizer suggestions were considered, and why fallbacks or degradations happened.
- [x] The system supports user-controlled `1 min -> 12 h` reasoning budgets and turns extra time into real additional work rather than artificial delay.
- [x] Long-horizon reasoning stays hardware-bounded through checkpointing, throttling, and cooperative scheduling instead of saturating CPU, RAM, or VRAM for the full wall-clock budget.
- [ ] The system can run bounded local tasks on the user's computer through typed, auditable, policy-checked capabilities instead of unrestricted OS control.
- [ ] Optional desktop observation and control tiers remain lightweight by default and only enable continuous capture, OCR-on-step, vision-on-step, or full desktop input when the user explicitly turns them on.
- [ ] The default runtime remains within the 6GB VRAM / 8GB RAM target by keeping computer control local-first, OCR CPU-first, optional heavy specialist models load-on-demand only, CPU-sidecar helpers bounded separately, and no more than two heavy model backends active at once.
- [ ] Optional cloud offload remains auxiliary, provider-swappable, and non-mandatory; cloud failures fall back to local execution instead of breaking core local operation.
- [x] Tests cover unit, integration, resource, failure, and optimization scenarios.
- [x] The public interface checklist at the top of this file is fully matched by the codebase.
- [x] Placeholder runtime paths have been replaced by real or clearly configured backend behavior.
- [x] The runtime uses selective context injection and does not depend on loading a giant translation dictionary into every model prompt.
- [x] Real-mode prerequisites are reproducible from a manifest plus README, and missing dependencies/services/models fail clearly.
- [x] The packaged Windows app path can launch in stub mode, guide the user through real-backend setup, and degrade gracefully when optional dependencies or capabilities are unavailable.
- [ ] The primary desktop shell is orb-centered, clearly state-driven, and integrates general assistant work plus Coding Mode inside one coherent app surface rather than separate disconnected tools.
- [ ] Every major backend subsystem maps to visible typed shell surfaces such as hero status, activity chips, center task cards, drawers, sheets, notifications, or orb effects instead of hiding critical state in raw logs.
- [ ] Adaptive collapsed drawers keep the orb and current task dominant on desktop while still exposing run inspector, readiness, knowledge, capability, and control-plane depth on demand.
- [ ] The PySide6 shell preserves operator parity with history, run inspector, readiness, knowledge management, settings, capability details, and debug/export flows after migration.
- [ ] The visual system remains premium and bounded: reduced-effects and low-resource modes still look intentional, and the orb only reflects real runtime or coding-state signals rather than decorative fake motion.
- [ ] Coding Mode can route between multiple local coding models or shared-role routes for planning, generation, debugging, review, test writing, summarization, and refactoring without exceeding the heavy-model cap.
- [ ] Coding tasks run locally through a safe sandbox path with timeouts, resource limits, test/lint/static-analysis integration, and structured validation reports.
- [ ] Idle Coding Dojo mode can run bounded practice sessions, score outcomes, and promote only validated patterns into local memory while storing failures as candidate or rejected examples.
- [ ] The coding knowledge index stores verified patterns, candidate patterns, rejected patterns, bug fixes, refactor strategies, test strategies, and architecture templates with searchable metadata and validation history.
- [ ] The orb and shell expose coding states, active coding role/model, validation progress, pattern learning, and regression warnings in real time.
- [ ] Coding metrics and history surface pass/fail rate, lint/static-analysis results, bug-fix success rate, regression rate, pattern reuse score, practice outcomes, model performance by task type, and coding-memory growth.

Assumptions and Defaults
- [x] Default thinking-time control: user-facing slider with `1 min -> 12 h` range and bounded presets
- [x] Default long-horizon policy: above `120 min`, shift from one saturated preset to checkpointed, duty-cycled refinement
- [x] Default GUI: `Tkinter` app shell backed by a typed view-model, with the raw event log kept as a debug pane instead of the primary UX
- [x] Default persistence: `SQLite + JSONL`
- [x] Default settings persistence: local validated user profiles stored in machine-readable form, with import/export and reset support, instead of editing source config directly
- [x] Default architecture: local-first runtime with an extensible model-role registry, one orchestrator, one user interface, and a hard cap of two active heavy model backends at once
- [x] Default local vector store: `Chroma PersistentClient`
- [x] Optional secondary vector store: `sqlite-vec`, deferred until after persistence and event/status work
- [x] Default retrieval embedding model: `sentence-transformers/intfloat/e5-small-v2`
- [x] Default reasoning strategy: system-first lightweight policy model plus verifier and tool stack, not training a new foundation model from scratch
- [x] Default reasoning priority: verifiable reasoning and precision before broad conversational polish
- [x] Default reasoning modes: bounded `fast` and `deep`
- [x] Default `deep`-mode selection policy: generate bounded candidate traces, verify them, then select, repair, or abstain
- [x] Default tool-verification policy: use bounded Python or code-test execution for checkable tasks before optional model-only fallback
- [x] Default compression architecture: canonical graph-backed IR compiled into a compact operation stream with parameterized macros and proof hashing
- [x] Default model-output policy: schema-constrained JSON at agent boundaries
- [x] Default dev calibration profile: `4GB VRAM / 8GB RAM` current laptop while keeping `6GB VRAM / 8GB RAM` as the locked baseline acceptance target
- [x] Default macro recursion limit: `8`
- [x] Default queue policy: bounded queues only
- [x] Default web behavior: optional, adapter-based, timeout-protected
- [x] Default event/status policy: typed `RuntimeEvent` plus typed `AgentStatus` persisted as machine-readable records
- [x] Default seed-corpus policy: stub/demo fixtures remain clearly separable from user and runtime data
- [x] Default optimizer activation policy: proposal-only until replay, validation, activation logging, and rollback exist
- [x] Default translation-registry policy: machine-readable runtime registry plus optional human-readable export; never use a giant plain-text file as the primary runtime source of truth
- [ ] Default computer-control architecture: typed capability adapters with policy checks and audit logs, not unrestricted shell or desktop control
- [ ] Default control-surface policy: per-capability toggles, not one global desktop or cloud switch
- [ ] Default desktop approval policy: allow safe actions automatically and require explicit approval for risky shell, file-destructive, or cross-app control actions
- [x] Default observation policy: screenshot-on-demand only; continuous capture, OCR-on-step, vision-on-step, and free-form mouse or keyboard control are off by default
- [x] Default OCR policy: CPU-first OCR before any multimodal model path, with optional specialist perception only if CPU OCR plus the general vision model prove insufficient
- [x] Default model topology: generation + embedding as the default active heavy pair, optional specialist roles routed on demand, and a hard scheduler cap of two active heavy model backends at once
- [x] Default model scheduling policy: unload optional heavy specialists when idle, swap them in temporarily when needed, and keep small CPU sidecars outside the heavy-model cap
- [ ] Default local-AI integration UX: one app surface shows installed local models, active routed roles, fallback reasons, optimizer advice, and quick actions without splitting the experience into separate model-specific tools
- [ ] Default future specialist recommendations: `jinaai/jina-reranker-v1-tiny-en` for reranking, `whisper.cpp` or `openai/whisper-tiny` for speech-to-text, `Silero VAD` for voice activity detection, `Piper` for text-to-speech, `Argos Translate` for offline translation, `Qwen/Qwen2.5-Coder-1.5B-Instruct` for code-specialist work, `HuggingFaceTB/SmolVLM-256M-Instruct` for vision, and `PaddleOCR` only as an opt-in specialist perception upgrade if CPU OCR plus the general vision model are insufficient
- [x] Default cloud offload policy: auxiliary-only, provider-agnostic, per-capability, and always able to fall back to local execution
- [x] Default cloud content policy: cloud-enabled capabilities may offload approved task content only after the user has explicitly enabled that capability
- [x] Default hardware-governor policy: degrade heavy observation and vision features before degrading core local reasoning
- [x] Default first-run experience: packaged Windows app launches in stub mode first, runs a preflight check for real mode, and gives actionable setup guidance before enabling heavier local or cloud-assisted features
- [x] Default knowledge UX: a local document library manages user sources separately from demo corpus and makes ingest, rebuild, archive, and removal actions visible in the app
- [ ] Default implementation rule: optimize memory and concurrency without removing logic or features
- [ ] Default future desktop shell target: a PySide6 orb shell backed by typed shell state, while headless mode, packaged recovery, and legacy settings compatibility remain preserved
- [ ] Default shell composition: orb-centered hero, bounded activity strip, stage-like current-task surface, adaptive collapsed drawers, lower operator sheets, and a cockpit-style bottom dock
- [ ] Default shell mode behavior: assistant work and Coding Mode share one shell, one control plane, one history system, and one settings model rather than separate apps
- [ ] Default visual direction: shared premium shell chrome with blue-cyan assistant atmosphere, amber-orange coding atmosphere, and structured card-based operator surfaces instead of plain text panes
- [ ] Default operator-depth policy: advanced evidence, provenance, optimizer, readiness, capability, and coding-memory surfaces live in drawers or sheets so the main canvas stays focused on current work
- [ ] Default shell effect-priority policy: `error > approval > regression > verification > checkpoint > speaking > ambient`
- [ ] Default low-resource visual policy: resource pressure stays visible and can recommend reduced-effects modes, but it must not silently change the user's chosen shell preset
- [ ] Default Coding Mode task taxonomy: feature generation, bug fixing, refactoring, test generation, code review, explanation or summarization, project scaffolding, and architecture planning
- [ ] Default coding router policy: prefer one warmed general coding model plus one optional specialist or reviewer route at a time; never keep more heavy coding models resident than the active heavy-slot budget allows
- [ ] Default coding sandbox policy: temp-workspace isolation, allowlisted interpreters/tools, bounded CPU/RAM/time, and blocked network by default unless the capability policy explicitly allows it
- [ ] Default coding learning tiers: `candidate`, `verified`, and `rejected`, with only `verified` eligible for high-priority retrieval or optimizer reuse
- [ ] Default coding promotion rule: generated code must pass tests, lint, complexity, security, maintainability, critique, and regression checks before entering verified memory
- [ ] Default idle-practice policy: user-visible, opt-in, low-priority background practice only when the PC is idle and no foreground task or risky capability session is active
- [ ] Default coding memory metadata: language, framework, task type, quality score, source, validation history, last-used timestamp, and reuse frequency
- [ ] Default coding UI behavior: one integrated shell shows coding workspace state, active role/model, validation progress, learned pattern summaries, and practice history without splitting Coding Mode into a separate disconnected app

13. Phase 13 - Lower-Level AI Implementation Guardrails
- [x] 13.0.1 Before using Phase 13 as guidance, treat it as a standing checklist that must be reviewed at the start of every implementation phase, not as a one-time phase.
- [x] 13.0.2 Before relying on lower-level AI execution, keep tasks small enough that each coding pass changes one subsystem and one test surface at a time.
- [x] 13.0.3 Before delegating work, ensure every task names the target files, required contracts, and success tests explicitly so the implementer does not guess.
- [x] 13.1 Do not skip phase order; complete Phase N acceptance checks before starting Phase N+1.
- [x] 13.2 Build in two modes from day one: `stub_mode=true` (no heavy model calls) and `stub_mode=false` (real backends).
- [ ] 13.3 Keep one public class per module where possible; avoid advanced abstractions unless they remove duplication.
- [x] 13.4 For every agent output, enforce typed parsing and validation before passing to the next stage.
- [x] 13.5 If generation output is invalid JSON/schema, run one repair attempt, then return a structured failure.
- [x] 13.6 Every async loop must support cancellation and timeout; never allow infinite `while True` without sleep and stop signal.
- [x] 13.7 Keep queues bounded and define explicit drop/retry behavior to prevent memory growth.
- [x] 13.8 Add docstrings that state input, output, and failure behavior for each public method.
- [ ] 13.9 Keep feature parity: simplify implementation details, never remove required logic or workflow steps.
- [ ] 13.10 Never promote raw generated code into verified memory without the full gated validation path; store partial wins as candidates with explicit failure reasons.
- [ ] 13.11 Keep sandbox runners, idle practice, and coding-memory promotion behind typed services; do not let ad hoc scripts become silent workflow dependencies.

14. Phase 14 - Micro-Step Build Order (Easiest Safe Path)
- [ ] 14.0.1 Before following Phase 14 literally, reconcile it with any future phase-order changes; this section is the execution sequence and must stay synchronized with the rest of the plan.
- [ ] 14.0.2 Before each subsystem handoff, confirm the prior subsystem’s tests and done gate are already green; do not treat build order as a substitute for validation.
- [ ] 14.0.3 Before using lower-level AI for a micro-step, reduce the task to one contract addition, one behavior change, and one test target where possible.
- [x] 14.1 Finish the remaining Phase 5 persistence layer: task runs, runtime registries, proof-hash history, and machine-readable source/provenance storage.
- [x] 14.2 Add append-only trace, web, and status JSONL logs plus persisted web evidence so retrieval and orchestration history are auditable.
- [x] 14.3 Pull forward the typed `RuntimeEvent` / `AgentStatus` contract and additive `TaskResult` fields (`answer_text`, `warnings`, `metrics`) before broader orchestration or dashboard work.
- [x] 14.4 Extract retrieval policy into a dedicated retrieval service, then add bounded lexical-candidate generation (`FTS5` or equivalent) and selective vector loading.
- [x] 14.5 Improve persistent-vector startup so backends reconcile existing state instead of blindly rebuilding the full corpus on every boot.
- [x] 14.6 Lock and persist the additive IR shapes, symbol-table records, opcode/decoder registries, and proof-hash fields (`6A`).
- [x] 14.7 Rebuild `macro_engine.py` on the new IR with deterministic compile/expand/verify behavior plus recursion and proof guards (`6B`).
- [x] 14.8 Add selective loading of only the active opcode, macro, decoder, and symbol-table subsets for a task (`6C`).
- [x] 14.9 Replace Planner with schema-constrained structured output and boundary validation, then replace Reasoner and Critic with deterministic IR/provenance construction, bounded `fast` and `deep` candidate generation, tool-backed verification, and select or repair or abstain behavior.
- [x] 14.10 Replace Compressor placeholder logic with parameterized motif/proof compression and persisted proposal storage, then add verified `deep`-trace export for future replay, distillation, and process supervision.
- [x] 14.10.1 Make deterministic graph- and proof-aware proposal building the primary compressor path; any model involvement stays optional and limited to bounded reranking or naming of top candidates in `deep` or long-horizon modes.
- [x] 14.10.2 Extract one shared `MacroProposalBuilder` used by both `CompressorAgent` and `SelfOptimizer`; the compressor remains the single proposal generator while the optimizer scores, validates, blocks, or later activates proposals.
- [x] 14.10.3 Allow `Reasoner` and `Critic` to emit typed compression hints such as shared subproofs, stable symbol bundles, repeated evidence paths, or verifier-stable motifs so the compressor can reuse upstream structure instead of rediscovering everything from scratch.
- [x] 14.10.4 Keep the foreground compressor lightweight: use rolling bounded history windows, capped proposal counts, deterministic-first ranking, and no always-on background mining loop for per-task proposal generation.
- [x] 14.11 Add orchestrator cancellation, retry, surfaced-warning flow, and typed status/event emission end to end.
- [ ] 14.12 Implement replay/simulation-gated optimizer activation, then expand the dashboard from the typed event/status surfaces.
- [x] 14.12.1 Persist a bounded macro-effectiveness registry tracking at minimum seen count, replay pass rate, proof-hash stability, critic-validity rate, realized compression gain, last-used timestamp, and context tags so proposal ranking can reuse real outcomes instead of rescanning full history.
- [x] 14.12.2 Scope compression proposals by context such as reasoning mode, verifier family, evidence mix, or task pattern so reusable macros stay relevant and cheap rather than competing in one global undifferentiated pool.
- [x] 14.13 Add typed coding task, sandbox, pattern, and practice-session contracts plus machine-readable storage.
- [ ] 14.14 Implement multi-model coding routing, role assignment, and heavy-slot-aware scheduling on top of the existing registry.
- [ ] 14.15 Add the safe sandbox runner plus bounded test, lint, static-analysis, complexity, security, and regression validators.
- [x] 14.16 Add coding memory indexing with verified/candidate/rejected tiers, metadata, retrieval, and promotion rules.
- [ ] 14.17 Add idle Coding Dojo orchestration, scoring, practice-history persistence, and optimizer hooks.
- [ ] 14.18 Expand the dashboard, orb-state system, history, readiness, and control plane to surface Coding Mode as a first-class workflow.
- [ ] 14.19 Add coding-mode end-to-end, sandbox, idle-practice, UI-state, and packaged-app regression coverage before claiming feature completeness.

15. Phase 15 - Common Failure Patterns and Mandatory Fixes
- [ ] 15.0.1 Before Phase 15 issues happen in production, convert each known failure pattern into a test, log signal, or explicit degraded-mode branch during the earlier phases.
- [ ] 15.0.2 Before closing any phase, review this failure list and verify the phase did not introduce a new unchecked failure mode around memory, schema drift, retries, or optimizer safety.
- [ ] 15.0.3 Before accepting a “works on my machine” result, confirm the failure handling also behaves correctly in stub mode, headless mode, and low-resource mode.
- [ ] 15.1 If RAM usage grows over time, inspect queues, cached traces, and web content buffers; cap or prune immediately.
- [ ] 15.2 If VRAM spikes, reduce active context length, lower batch size, unload idle model sessions, and reduce concurrent model tasks.
- [ ] 15.3 If agent outputs drift in format, enforce strict schema adapters at agent boundaries.
- [ ] 15.4 If round-trip macro checks fail, reject the macro proposal and keep previous stable macro set.
- [ ] 15.5 If web fetch fails, continue with local evidence and return degraded-mode warnings, not hard crashes.
- [ ] 15.6 If critic rejects trace repeatedly, cap retries and return a structured failure report with root causes.
- [ ] 15.7 If optimizer causes regressions in simulation, do not activate proposals; log failure and rollback candidate.
- [ ] 15.8 If a sandbox run exceeds policy, terminate it, record the blocked reason, and mark the coding attempt as failed rather than retrying unsafely.
- [ ] 15.9 If generated code passes tests but fails security, maintainability, or regression checks, keep it out of verified memory and surface the exact blocking gates.
- [ ] 15.10 If idle practice begins while a foreground task, capability session, or resource-pressure event is active, pause or skip practice instead of competing for shared hardware.
- [ ] 15.11 If coding-memory growth becomes noisy or low-quality, tighten promotion thresholds, deduplicate patterns, and downrank or archive stale candidates.

16. Phase 16 - Definition of Done Gates by Subsystem
- [x] 16.0.1 Before marking any subsystem done, require both its local tests and its compatibility tests to pass; no subsystem is done if it breaks an earlier phase contract.
- [x] 16.0.2 Before calling the whole project complete, require the acceptance checklist, subsystem gates, and compatibility contract to all be green at the same time.
- [x] 16.0.3 Before release, run at least one end-to-end stub-mode path and one real-backend smoke path against the 6GB VRAM / 8GB RAM target assumptions.
- [x] 16.1 Data layer done gate: all dataclass validation and serialization tests pass.
- [x] 16.2 Storage layer done gate: SQLite and JSONL smoke tests pass with no schema migration errors.
- [x] 16.3 Model layer done gate: semaphore/concurrency tests prove only allowed model jobs run at once.
- [x] 16.4 Macro layer done gate: nested expansion, recursion guard, and round-trip verification tests pass.
- [x] 16.5 Agent layer done gate: each agent unit test passes in stub mode and real mode (where dependencies exist).
- [ ] 16.5.1 Coding subsystem done gate: coding-role routing, sandbox validation, gated memory promotion, idle-practice safety, and coding history surfaces all pass in stub and real mode where dependencies exist.
- [x] 16.6 Orchestrator done gate: full pipeline executes sample tasks with correct stage order and status events.
- [x] 16.7 Optimizer done gate: proposals are never activated without simulation + validation pass.
- [x] 16.8 Dashboard done gate: UI remains responsive during long-running tasks, displays live logs and status, persists validated settings, and surfaces history, readiness, and degraded-state explanations without relying on raw dict inspection.
- [ ] 16.8.1 Coding UI done gate: orb states, coding workspace panels, test/lint/static-analysis summaries, practice history, and model-route displays remain responsive and synchronized with typed shell state.
- [x] 16.9 Resource done gate: target runtime stays within practical limits for 6GB VRAM / 8GB RAM baseline.
- [ ] 16.9.1 Coding resource done gate: foreground coding tasks, sandbox validators, and idle practice stay within the shared heavy-slot and local hardware limits without starving core assistant flows.
- [x] 16.10 Release done gate: end-to-end tests pass, the packaged Windows path launches cleanly in stub mode, real-backend preflight failures are actionable, and the acceptance checklist is fully checked.

17. Phase 17 - Long-Horizon Reasoning Control and Time Slider
- [x] 17.0.1 Before coding Phase 17, preserve `run_task(question, thinking_minutes)` as the public contract; extend long-horizon behavior additively instead of creating a separate “hours mode” API.
- [x] 17.0.2 Before coding Phase 17, treat `12 hours` as a wall-clock budget ceiling, not permission for unbounded full-load compute; long runs must yield, checkpoint, and stay honest to the 6GB VRAM / 8GB RAM target.
- [x] 17.0.3 Before coding Phase 17, keep the time slider tied to actual additional work: more candidate batches, verification rounds, research refreshes, and critique passes, not artificial sleeping.
- [x] 17.0.4 Before coding Phase 17, replace the current saturated `121-720 min` behavior with a checkpointed long-horizon schedule; per-cycle work stays bounded even when total wall time is large.
- [x] 17.1 Extend `ResourceBudget` and `BudgetPolicy` additively with long-horizon fields such as wall-clock budget, cycle budget, checkpoint cadence, duty cycle, cool-down interval, and max resume count while keeping old callers valid.
- [x] 17.2 Implement a cooperative long-horizon scheduler that alternates bounded work bursts, persistence, and idle or yield periods so hours-long runs do not monopolize CPU, RAM, or VRAM.
- [x] 17.3 Persist long-horizon session state, checkpoints, partial candidate sets, evidence refresh state, and critique summaries so runs can pause, resume, recover, and survive restart.
- [x] 17.4 For budgets above `120` minutes, spend extra time on additional candidate batches, verifier rounds, retrieval refreshes, and consensus or adjudication passes instead of only making single prompts larger.
- [ ] 17.4.1 For coding tasks above `120` minutes, spend extra time on additional candidate patches, broader regression suites, deeper review passes, more test generation, and more selective pattern validation instead of only larger prompts.
- [x] 17.5 Add pause, resume, cancel, and safe shutdown behavior for long-horizon sessions.
- [x] 17.6 Add pressure-aware throttling: if RAM, VRAM, thermal headroom, queue pressure, or backend health degrade, reduce duty cycle, shrink batch depth, or pause background work instead of pushing harder.
- [x] 17.6.1 Allow long-horizon cycles to request bounded advisory help from the Self-Optimizer between cycles for macros, retrieval strategy, critique heuristics, or planning-template hints; never allow advisory calls to create an inner unbounded feedback loop.
- [x] 17.7 Add a dashboard time-control surface with a `1 min -> 12 h` slider, explicit preset labels, ETA or elapsed time, current phase, checkpoint counters, and a clear distinction between interactive and long-horizon modes.
- [x] 17.8 Show what extra time bought: additional evidence gathered, candidate count, verification passes, repairs, abstentions avoided, or confidence improvements.
- [x] 17.8.1 Show which optimizer or advisor suggestions were requested, accepted, rejected, or deferred during long-horizon work so extra-time gains stay explainable in the app.
- [ ] 17.8.2 Show what extra time bought in Coding Mode: candidate patches explored, tests added, bugs reproduced, regressions prevented, verified patterns promoted, or unsafe patches rejected.
- [x] 17.9 Add end-to-end and resource tests that prove `12h` mode remains bounded on both the 4GB dev profile and the 6GB baseline target, checkpoints resume correctly, and longer budgets can improve at least one verifiable benchmark family.
- [ ] 17.9.1 Add long-horizon coding tests proving extended budgets improve at least one coding benchmark family through additional validation and repair rather than unbounded execution.
- [x] 17.10 Do not mark Phase 17 complete while `121` minutes and `720` minutes still collapse to the same effective schedule.
- [x] 17.11 Keep long-horizon work compatible with future optimizer replay or export flows; checkpoint artifacts must stay machine-readable and only verified final traces may feed future distillation or process-supervision datasets.
- [x] 17.12 If a long-horizon run finds no measurable improvement after multiple cycles, allow early stop with a structured explanation rather than consuming the full wall-clock budget pointlessly.

18. Phase 18 - Extensible Model Registry, Routing, and Scheduler Foundation
- [x] 18.0.1 Before broadening the local model stack, preserve `Orchestrator.run_task(question, thinking_minutes)` plus the current generation and embedding paths as compatibility wrappers; the new routing layer must extend the current runtime instead of replacing it.
- [x] 18.0.2 Before adding more local models, replace the fixed future-role assumption with a typed model-role registry so specialists can be added over time without reworking the core scheduler.
- [x] 18.0.3 Before adding specialist models, keep the system lightweight by defining the two-active limit around heavy inference backends only; small CPU sidecars stay separately bounded and do not consume heavy slots.
- [x] 18.1 Add typed model-role contracts covering at minimum `generation`, `embedding`, `reranker`, `speech_to_text`, `text_to_speech`, `vad`, `translation`, `code_specialist`, `vision`, and `specialist_perception`.
- [ ] 18.1.1 Extend the role registry with typed coding subroles or routed role labels for `code_planner`, `code_generator`, `code_debugger`, `code_reviewer`, `code_test_writer`, `code_summarizer`, and `code_refactorer` while keeping the existing `code_specialist` compatibility path valid.
- [x] 18.2 Add typed model registrations capturing role, backend, model identifier or path, resource class, enablement state, preferred device, load policy, and supported capabilities.
- [x] 18.3 Add a routing layer that lets the orchestrator request a capability and receive a routed model decision without hard-coding per-model branches in task logic.
- [ ] 18.3.1 Add a coding router that can switch between multiple local coding models by task type, language/framework, file scope, and required coding subrole without leaking backend-specific logic into the orchestrator.
- [x] 18.4 Add a heavy-slot scheduler that keeps the default active pair at `generation + embedding`, enforces a hard cap of two active heavy model backends, and unloads optional heavy specialists after idle periods.
- [ ] 18.4.1 Make the heavy-slot scheduler coding-aware so planner/generator/debugger/reviewer routes reuse warmed models when possible and never keep more large coding models resident than the shared budget allows.
- [x] 18.5 Add a sidecar classification for lightweight CPU helpers such as VAD, CPU OCR, TTS, or translation helpers so they remain governed by CPU or RAM pressure without counting against the heavy-model cap.
- [x] 18.6 Extend settings, readiness, and storage with a persisted model-registry view showing installed models, enabled roles, preferred models per role, active heavy-slot usage, missing dependencies, and fallback reasons.
- [x] 18.6.1 Extend that registry view into the app shell so model-role state, routing decisions, optimizer subscriptions, advisory availability, and current fallback reasons are visible in one integrated local-AI control plane.
- [ ] 18.6.2 Extend the app control plane with coding-route visibility: active coding model, assigned role, route history, reuse or warm state, and blocked fallback reasons per coding task type.
- [x] 18.7 Keep the current `ModelManager.generate()` and `embed*()` APIs as compatibility wrappers while the new registry and routing layer are introduced underneath them.
- [x] 18.8 Add deterministic registry and scheduler tests proving new roles can be registered additively, the heavy-slot cap is never exceeded, and sidecars do not consume heavy slots.
- [x] 18.9 Reuse the typed `RuntimeEvent` and `AgentStatus` surfaces as the shared advisory telemetry bus for agents, model manager, dashboard, and Self-Optimizer instead of introducing a second untyped event system.
- [x] 18.10 Add typed optimizer suggestion contracts covering macro advice, retrieval strategy hints, planning-template hints, critique-heuristic hints, dashboard or UI hints, model-loading hints, and cache or prefetch hints.
- [ ] 18.10.1 Add typed optimizer suggestion contracts for coding heuristics such as test-selection hints, lint-policy tuning, refactor recipes, bug-reproduction templates, and pattern-index warming.
- [x] 18.11 Keep optimizer suggestions advisory-only by default: agents may subscribe to persisted suggestions, but live behavior changes still require replay, validation, policy, and explicit activation gates.
- [x] 18.12 Add bounded cross-agent strategy promotion so successful patterns discovered in one subsystem can be published as typed reusable suggestions for others without creating hidden live coupling.
- [x] 18.13 Add shared bounded caches for hot embeddings, retrieval candidates, runtime subsets, and reusable strategy artifacts with explicit ownership, size caps, eviction policy, and auditability; optimizer may recommend warming them but may not create unbounded background caches.
- [x] 18.13.1 Reuse that bounded-cache layer for hot compression artifacts and macro-effectiveness summaries so the compressor and optimizer can share lightweight prior scoring without introducing a second unbounded proposal store.
- [x] 18.14 Add deterministic tests proving advisory fan-out, strategy promotion, and cache warming stay bounded, avoid hidden feedback loops, and never mutate live agent behavior without the existing validation gates.

19. Phase 19 - Local Specialist Models and Unified Feature Expansion
- [x] 19.0.1 Before adding desktop-control or vision-heavy work, add lightweight specialist models that materially improve features while preserving the one-interface, one-orchestrator user experience.
- [x] 19.0.2 Before adding each specialist role, prove it can be enabled, routed, unloaded, and disabled independently without breaking the default `generation + embedding` runtime.
- [x] 19.1 Add an optional reranker role first to improve retrieval quality with minimal extra resource use.
- [x] 19.2 Add optional `speech_to_text` plus `vad` roles so the local app can support voice input and bounded transcription without cloud dependence.
- [x] 19.3 Add an optional `text_to_speech` role so the assistant can speak locally without making voice output part of the base runtime.
- [x] 19.4 Add an optional offline `translation` role so multilingual input, output, and document translation can be enabled without making translation always-on.
- [x] 19.5 Add an optional `code_specialist` role for tool-heavy coding or file-maintenance tasks that is loaded only on demand and unloaded when idle.
- [ ] 19.5.1 Expand the single `code_specialist` path into a multi-model Coding Mode that can assign planner, generator, debugger, reviewer, test-writer, summarizer, and refactorer roles while staying additive to the current model registry.
- [ ] 19.5.2 Prefer lightweight or quantized coding models, shared tokenizer/runtime families, and warm-reuse policies so Coding Mode remains practical on consumer Windows hardware.
- [ ] 19.5.3 Add per-task-type route preferences so feature generation, bug fixing, refactoring, code review, test writing, summarization, and architecture planning can choose different local coding specialists when available.
- [x] 19.6 Surface specialist-role readiness and enablement in the existing settings and readiness UI without fragmenting the user experience into multiple separate agents.
- [x] 19.6.1 Surface all local AI roles, routed decisions, recent fallback reasons, and optimizer suggestions in one integrated app panel instead of scattered role-specific controls.
- [x] 19.6.2 Add per-role quick actions in the app for install guidance, enable or disable, warm or unload, test ping, and fallback-reason inspection so local models feel like one managed system rather than disconnected backends.
- [x] 19.6.3 Surface compressor insights in the app shell as lightweight structured summaries: top reusable patterns, estimated gain, validation state, blocked reasons, and whether a suggestion came from deterministic analysis, replay evidence, or optional advisor reranking.
- [ ] 19.6.4 Surface Coding Mode readiness and quick actions in the integrated control plane: install guidance, enable/disable, warm/unload, test sample, sandbox tool readiness, and per-role fallback explanation.
- [x] 19.7 Pin lightweight recommended defaults for future specialist roles: `jinaai/jina-reranker-v1-tiny-en`, `whisper.cpp` or `openai/whisper-tiny`, `Silero VAD`, `Piper`, `Argos Translate`, and `Qwen/Qwen2.5-Coder-1.5B-Instruct`.
- [ ] 19.7.1 Pin one documented default Coding Mode bundle for a general coding model plus optional reviewer or test-writing specialist recommendations, and keep onboarding/readiness aligned to those defaults.
- [x] 19.8 Add tests proving specialist roles are optional, route only when enabled or needed, and do not break the baseline local pipeline when missing.
- [x] 19.8.1 Add app-integration tests proving the UI can explain which local AI role handled a task, which advisor suggestions were considered, and why a role was or was not used.
- [x] 19.8.2 Add app-integration tests proving compressor summaries remain typed, bounded, and explainable, and that blocked or deferred macro suggestions are visible without raw event-log inspection.
- [ ] 19.8.3 Add app-integration tests proving the shell can explain which coding role/model handled each coding stage, why a fallback occurred, and which validation gates blocked or promoted memory updates.

20. Phase 20 - Capability and Policy Foundation for Local Computer Tasks
- [x] 20.0.1 Before coding computer-control features, preserve the local-first design; desktop or cloud helpers must extend the current runtime rather than replace core local execution.
- [x] 20.0.2 Before coding computer-control features, define typed capability contracts and policy outcomes first so the agent never jumps straight from text to unrestricted OS actions.
- [x] 20.0.3 Before coding computer-control features, keep the default runtime lightweight by making all heavy observation or desktop-control tiers opt-in instead of default-on.
- [x] 20.1 Add typed capability contracts for file operations, allowlisted shell commands, browser actions, app or window focus, clipboard actions, screenshots, OCR requests, and desktop input actions.
- [ ] 20.1.1 Add typed capabilities for sandbox execution, test running, linting, static analysis, complexity checks, dependency inspection, and regression-suite execution as first-class coding tools.
- [x] 20.2 Add a capability registry that records whether each capability is available, enabled, denied by policy, degraded by resources, or requires approval.
- [x] 20.3 Add a policy engine that can return `allowed`, `requires_approval`, `denied`, or `degraded` for every requested action before any executor runs.
- [x] 20.4 Keep dangerous capabilities blocked by default: admin elevation, startup persistence, hidden background control, unrestricted shell execution, unrestricted filesystem deletion, and credential harvesting.
- [ ] 20.4.1 Keep Coding Mode dangerous actions blocked by default: package installation, unrestricted shell escape, networked dependency fetch, writes outside the approved workspace, and arbitrary background services.
- [x] 20.5 Add explicit allowlists for filesystem roots, apps, shell commands, browser domains, and background services so the first live version stays bounded.
- [ ] 20.5.1 Add explicit allowlists for interpreters, package managers, linters, test runners, static-analysis tools, and workspace roots so coding tasks remain auditable and bounded.
- [x] 20.6 Add audit records for every requested action, approval, denial, executor result, surfaced warning, and policy reason.
- [ ] 20.6.1 Record coding audit events for patch application, sandbox execution, test/lint/static-analysis runs, regression gates, pattern promotions, and rejected learning attempts.
- [x] 20.7 Add stub executors and deterministic tests for every capability type before adding live OS adapters.

21. Phase 21 - Local Task Execution and Desktop Control Sessions
- [x] 21.0.1 Before allowing local task execution, make the agent operate inside explicit user sessions with start, pause, resume, stop, and kill-switch controls.
- [x] 21.0.2 Before allowing desktop input, implement safer non-visual tools first: file tasks, bounded shell tasks, browser tasks, and app or window control.
- [x] 21.1 Add session-scoped local task execution so the agent can run approved tasks on the user's machine without gaining permanent unrestricted background control.
- [x] 21.2 Implement local file and project-task execution first, including bounded read, write, move, copy, archive, and delete operations inside allowed roots.
- [ ] 21.2.1 Add workspace-scoped coding task execution for reading project files, proposing patches, applying approved changes, and rolling back failed attempts inside allowed roots.
- [x] 21.3 Implement allowlisted shell execution for project maintenance, scripts, and developer workflows before broader system command support.
- [ ] 21.3.1 Add a safe local sandbox runner for generated code, tests, linting, and static analysis using temp copies or isolated worktrees rather than direct execution in the live project by default.
- [ ] 21.3.2 Support bounded coding-task flows for feature generation, bug fixing, refactoring, test generation, code review, summarization, project scaffolding, and architecture planning.
- [x] 21.4 Implement browser and app or window control as typed actions with focus checks, title matching, and visible target-state validation.
- [x] 21.5 Implement free-form mouse and keyboard control only after the earlier capability tiers are stable, logged, and approval-gated.
- [x] 21.6 Keep the default approval policy at `approve risky only`: safe actions may run automatically during an enabled session, while destructive, cross-app, or side-effect-heavy actions require confirmation.
- [x] 21.7 Add visible session indicators showing active control mode, current target app or window, pending approvals, and the last executed action.
- [x] 21.8 Add safe cancellation, emergency stop, and recovery behavior if focus is lost, the target app changes unexpectedly, or a task starts looping.
- [ ] 21.8.1 Add cleanup and recovery paths for coding sessions: revert temp workspaces, discard failed patch candidates, persist diagnostics, and pause pattern promotion when validation is incomplete.

22. Phase 22 - Observation, OCR, Vision, and Hardware Governor
- [x] 22.0.1 Before coding heavy observation features, keep screenshot-on-demand, continuous capture, OCR-on-step, vision-on-step, and full desktop-control observation as separate capability tiers instead of one giant mode.
- [x] 22.0.2 Before adding multimodal or specialist perception models, keep OCR CPU-first and only activate heavier visual roles when screenshot understanding, UI interpretation, or document perception actually requires them.
- [x] 22.0.3 Before enabling continuous capture, define strict caps for FPS, resolution, frame history, diff thresholds, and region-of-interest behavior so the feature cannot silently become the largest resource consumer.
- [x] 22.1 Add screenshot-on-demand as the lightest visual capability and make it the first observation tool used by desktop-task sessions.
- [ ] 22.1.1 Add lightweight coding-observation artifacts such as diff snapshots, test-output captures, and workspace summaries so the UI can explain coding progress without relying on raw terminal spam.
- [x] 22.2 Add CPU-first OCR helpers for screenshots and selected screen regions before any multimodal inference path.
- [x] 22.3 Add optional continuous capture using low-FPS, downscaled, diff-based, and region-aware sampling instead of full-rate desktop recording.
- [x] 22.4 Add optional OCR-on-step and vision-on-step modes that run only when the user has explicitly enabled them and the hardware governor allows them.
- [x] 22.5 Add an optional routed `vision` role, loaded only on demand and unloaded after idle periods so the default runtime remains a two-heavy-model system.
- [x] 22.5.1 Reserve an optional routed `specialist_perception` role for cases where CPU OCR and the general vision role are still insufficient; do not require it for baseline desktop-task support.
- [x] 22.5.2 Pin lightweight future visual-role recommendations: `HuggingFaceTB/SmolVLM-256M-Instruct` for the general vision role and `PaddleOCR` only as an opt-in specialist perception upgrade if CPU OCR plus general vision are not enough.
- [x] 22.6 Add a hardware governor and model scheduler that monitor CPU, RAM, VRAM, queue pressure, and backend health, enforce a hard cap of two active heavy model backends at once, and degrade heavy features in order: continuous capture, OCR cadence, per-step vision, optional heavy-model residency, then nonessential background work.
- [x] 22.6.1 Allow optimizer demand forecasts and advisory suggestions to inform the governor only as bounded inputs; the hardware governor and scheduler remain the authoritative owners of live resource decisions.
- [x] 22.7 Add explicit UI or dashboard visibility showing which observation tier is active, which specialist visual role is loaded, how many heavy slots are in use, and why a feature was degraded or disabled.
- [ ] 22.7.1 Surface when coding-mode validation is degraded by disabled linters, missing security tools, or unavailable sandbox dependencies, and show the impact in the same readiness/degraded-state UX.
- [x] 22.8 Add resource and failure tests proving the default mode stays within the 6GB VRAM / 8GB RAM baseline and that heavier observation tiers degrade automatically under pressure.

23. Phase 23 - Optional Auxiliary Cloud Offload
- [x] 23.0.1 Before adding cloud offload, keep it auxiliary-only; core answer generation and the base local agent path must remain usable without any cloud dependency.
- [x] 23.0.2 Before binding to providers, define one provider-agnostic cloud job contract covering payload class, privacy class, size limits, retries, and fallback behavior.
- [x] 23.0.3 Before turning on cloud offload for any capability, keep the control surface per-capability rather than one global cloud switch so users can enable only what their hardware or privacy tolerance needs.
- [x] 23.1 Add provider adapters for auxiliary offload targets such as offline replay jobs, export jobs, browser or web helper work, optional OCR or vision helpers, embedding helper work, and background maintenance.
- [x] 23.2 Keep the first provider layer provider-agnostic, with initial support planned around low-cost or free helper platforms rather than one hard-coded cloud vendor.
- [x] 23.3 Allow cloud offload only for explicitly enabled capabilities; disabled capabilities must never send prompts, evidence, screenshots, or task data off-machine.
- [ ] 23.3.1 If auxiliary cloud helpers are ever used for Coding Mode, keep them opt-in per coding capability and never offload source code, patches, or test artifacts unless the content class explicitly permits it.
- [x] 23.4 Default cloud failure behavior to local fallback: if a cloud helper fails or is unavailable, continue locally within the existing resource limits whenever possible.
- [x] 23.5 Persist every cloud-offload decision, provider used, payload class, bytes sent, latency, fallback reason, and final outcome so cloud usage is auditable and debuggable.
- [x] 23.6 Keep privacy boundaries explicit by classifying cloud jobs as metadata-only, approved-content, or denied-content and enforcing that classification before dispatch.
- [x] 23.7 Add tests proving cloud offload is optional, auxiliary-only, provider-swappable, privacy-classified, and never required for the baseline local task pipeline.

24. Phase 24 - Packaging, Onboarding, and Release Hardening
- [x] 24.0.1 Before packaging the app, preserve the developer-from-source path; packaging should add a first-class local app path, not replace the repo workflow.
- [x] 24.0.2 Before calling the app complete, treat packaged Windows launch, first-run onboarding, and real-backend setup guidance as part of product quality rather than optional polish.
- [x] 24.0.3 Before enabling real-mode controls in the packaged app, run the same preflight and capability checks used by the runtime so the GUI never promises unavailable backends or features.
- [x] 24.1 Add a packaged Windows entrypoint that launches the Tkinter app directly in stub mode when real-mode prerequisites are missing or disabled.
- [x] 24.2 Add a first-run onboarding flow that explains stub mode, real mode, local model requirements, hardware targets, privacy boundaries, and where user data is stored.
- [x] 24.3 Add actionable setup guidance for local backends, optional extras, model files, and capability prerequisites so the app can guide users without sending them to the codebase first.
- [x] 24.3.1 Pin and maintain one documented default local model bundle for generation, generation fallback, embedding, and embedding fallback so onboarding, README, readiness, and packaged setup all point to the same concrete models.
- [x] 24.3.2 Keep exact step-by-step local model download and install instructions in repo docs outside the roadmap itself, and link that guide from onboarding, readiness, and packaged preflight surfaces.
- [ ] 24.3.3 Add Coding Mode onboarding and setup guidance for local coding models, sandbox prerequisites, interpreter/toolchain allowlists, optional linters/security scanners, and practice-mode controls.
- [x] 24.4 Add a packaged-app preflight report that can be reopened later and clearly shows backend readiness, model availability, dependency gaps, disabled capability reasons, active heavy-slot policy, and current hardware budget assumptions.
- [ ] 24.4.1 Extend packaged preflight with coding-model readiness, sandbox capability status, toolchain availability, blocked-policy reasons, and whether idle practice can run safely on the current machine.
- [x] 24.5 Add crash-safe startup and recovery behavior for the packaged app, including log capture, exportable diagnostics, and a safe path back to stub mode when startup fails.
- [x] 24.6 Add packaged release validation covering launch, shutdown, stub-mode task execution, readable failure messaging, and persistence-path setup on a clean machine profile.
- [ ] 24.6.1 Add packaged release validation for Coding Mode launch, sample sandbox execution, readable toolchain failures, coding-model fallback messaging, and disabled idle-practice behavior when prerequisites are missing.

25. Phase 25 - Unified Orb Shell Architecture and ShellState Completion
- [ ] 25.0.1 Treat the premium orb shell as a shell migration and integration phase, not a backend rewrite; preserve `Orchestrator.run_task(question, thinking_minutes)`, headless mode, packaged stub-mode recovery, and the existing typed contracts.
- [ ] 25.0.2 Finish the explicit UI-host interface so the service layer can start and stop the shell, refresh from typed state, submit tasks, request actions, and open sheets or drawers without renderer-specific assumptions.
- [ ] 25.0.3 Keep legacy settings payloads loadable; add new shell settings additively instead of breaking old profiles or requiring a destructive migration.
- [ ] 25.1 Make the PySide6 shell the planned primary desktop renderer after parity while keeping the source-from-repo path, headless mode, and packaged recovery behavior valid throughout the migration.
- [ ] 25.2 Extend the typed shell projection additively with at minimum `workspace_mode`, `active_route_summary`, `active_model_roles`, `candidate_count`, `evidence_count`, `elapsed_seconds`, `current_file`, `current_project`, `sandbox_state`, `quality_gate_state`, `pattern_tier_counts`, `practice_session_state`, `approval_prompt_summary`, `resource_ribbon_flags`, `panel_visibility_state`, and `hero_metric_strip`.
- [ ] 25.3 Lock shell-state priority rules: `error` overrides all primary modes, `approval_pending` overlays without erasing the base mode, deep reasoning overrides normal thinking, degraded mode overlays without erasing the base mode, and transient effects stay bounded and auto-expire.
- [ ] 25.4 Add a feature-to-surface coverage matrix mapping every major subsystem to at least one visible shell surface such as hero region, activity strip, center task card, drawer panel, lower sheet, notification, or orb effect so features cannot silently exist only in logs.
- [ ] 25.5 Add an orb-state coverage matrix for planner, local retrieval, web retrieval, fast reasoning, deep reasoning, critic verification, compression, responding, speaking, long-horizon checkpointing, approval pending, degraded mode, capability sessions, observation tiers, cloud helpers, and coding states.
- [ ] 25.6 Treat Coding Mode as a full workspace mode inside the same shell rather than a separate app; switching between general assistant flow and coding flow must preserve shared history, settings, readiness, and control-plane visibility.
- [ ] 25.7 Make desktop layout default to adaptive collapsed drawers so the orb and current task stay dominant while deeper operator surfaces remain reachable on demand.
- [ ] 25.8 Keep shell widgets consuming only typed shell state and typed actions; do not let PySide surfaces parse raw event dicts or reimplement orchestration logic.

26. Phase 26 - Orb Visual System and Design Language
- [ ] 26.0.1 Treat the reference orb workstation look as a concrete shell target: one unified premium shell with a shared chrome, not a generic dashboard with a decorative orb added later.
- [ ] 26.0.2 Keep one shared shell frame with two atmosphere variants: general assistant work uses a blue-cyan family and Coding Mode uses an amber-orange family while preserving the same geometry, materials, spacing, and interaction model.
- [ ] 26.1 Define a shell design-token layer for background gradients, orb palettes by state, panel fills, glass overlays, accent rails, chip colors, timeline markers, warning/success/error tones, typography, spacing, corner radius, border glow, and shadow behavior.
- [ ] 26.2 Choose Windows-safe local fonts and fallbacks for body text, status text, technical badges, and condensed operator labels; lock readable sizing and spacing rules before broad visual polish.
- [ ] 26.3 Complete the orb renderer with layered core, inner shimmer, shell surface, outer glow, halo ring, reflection bed, segmented tool ring, role constellation markers, confidence arc, speaking waveform halo, critic verification sweep, compressor contraction pulse, checkpoint pulse, approval hold ring, optimizer advisory aura, degraded undertone, and bounded particle modes.
- [ ] 26.4 Add a strict orb effect-priority system so only one dominant effect wins at a time; default priority should be `error > approval > regression > verification > checkpoint > speaking > ambient`.
- [ ] 26.5 Carry the orb-state mappings into the real renderer for both general and coding states, including the defined Coding Mode mappings for planning, generating, refactoring, testing, debugging, reviewing, indexing, practicing, learning pattern, and regression detected.
- [ ] 26.6 Add bounded background layers tied to the orb state: subtle starfield or noise bed, horizon energy line, state-aware haze, slow energy arcs, and a lower reflection band that reinforces the orb color without overpowering the task surface.
- [ ] 26.7 Keep the center of the shell stage-like: one dominant active card under the orb, one verified/final card beneath it, and older messages or history visually secondary.
- [ ] 26.8 Add reduced-effects versions of all important orb and background effects so Minimal and low-resource modes still look intentional rather than broken.
- [ ] 26.9 Replace plain text operator panes with structured visual cards for evidence, control plane, runtime, practice logs, pattern memory, and validation summaries.

27. Phase 27 - Main Shell Surfaces, Drawers, and Operator Parity
- [ ] 27.0.1 Build the main shell around the intended hierarchy: orb hero at the top, status and sub-status beneath it, a bounded activity strip, a central active-work surface, adaptive side drawers, and a cockpit-like bottom control dock.
- [x] 27.1 Complete the hero region so status, sub-status, and hero metrics are fully driven by typed shell state and transition smoothly with runtime changes.
- [x] 27.2 Expand the activity strip to cover local retrieval, web query, verification, deep candidate pass, compression review, optimizer advisory, fallback active, approval needed, capability session, observation tier, cloud helper, checkpoint saved, resource pressure, route change, and coding-specific validator or regression states.
- [x] 27.3 Build a dedicated active task card showing current phase, elapsed time, candidate count, evidence count, verification state, fallback status, confidence band, route summary, and relevant warnings instead of forcing every active run through generic chat cards.
- [x] 27.4 Build a dedicated final answer card showing final answer, evidence summary, citations, warnings, verification result, degraded or uncertainty notes, export shortcuts, and expandable sections for why this answer, how it was verified, and what deep mode changed.
- [x] 27.5 Complete the bottom input dock with text entry, send, mic affordance, thinking-time slider, Fast/Deep/Long Horizon controls, stop and pause/resume controls, and compact toggles for local-only, web, verification-priority, capability session, and cloud-helper behavior.
- [ ] 27.6 Complete the left drawer with agent monitor, session controls, current-task timeline, compact diagnostics, and pause/resume/cancel shortcuts using smooth open/close transitions that do not stall the main shell.
- [x] 27.7 Complete the right drawer with interactive evidence, provenance, compressor, optimizer, runtime-health, and integrated local-AI control-plane widgets instead of read-only summary panes.
- [x] 27.8 Preserve run-inspector parity in the lower sheets so task history, prior answers, citations, candidate counts, critique outcomes, degraded reasons, repair actions, optimizer lifecycle records, and export links remain fully reachable after the shell migration.
- [x] 27.9 Add a dedicated long-horizon tray showing elapsed time, current cycle type, checkpoint count, candidate growth, verification passes, evidence refresh count, confidence-improvement trend, advisory suggestions considered, early-stop reason, duty cycle, and what extra time bought.
- [x] 27.10 Surface compact capability and policy context in the main shell with active session badge, target app/window badge, pending approval pill, observation-tier badge, last-action preview, and policy-state summary.
- [x] 27.11 Build calm shell-consistent approval overlays that explain what action is requested, why approval is needed, what the target is, and what the risk level is; keep approval state visible in the orb and activity strip.
- [x] 27.12 Keep export/import actions, diagnostics, and support-bundle access reachable from the shell without dropping the user into raw debug views for routine workflows.

28. Phase 28 - Coding Workspace, Validation Surfaces, and Local-AI Control Plane
- [x] 28.0.1 Treat the coding workspace as a first-class shell mode that reuses the same orb, chrome, bottom dock, drawers, and lower sheets rather than opening a disconnected code tool.
- [x] 28.1 Add a coding-task timeline in the left drawer covering code planning, generation, debugging/testing, review/refactor, indexing, and practice/regression events.
- [x] 28.2 Add a coding model switcher and route summary surface that shows active coding role, active model, warm state, fallback reason, and heavy-slot impact without hiding the shared local-AI control plane.
- [x] 28.3 Build dedicated coding center cards for active coding work, generated artifacts/patches, sandbox/test results, review/refactor outcomes, and final validated patch or summary output.
- [x] 28.4 Add coding progress details such as current file, workspace/project scope, current task, active validator, pattern match/reuse hint, and blocked-gate explanations without requiring raw terminal text.
- [x] 28.5 Add right-drawer coding surfaces for idle practice log, indexed code patterns, rejected anti-patterns, validation history, coding route decisions, and pattern-learning summaries.
- [ ] 28.6 Surface coding memory tiers explicitly in the shell with verified, candidate, and rejected sections that show language, framework, task type, quality score, source, validation history, reuse count, and last-used timestamps.
- [ ] 28.7 Surface coding metrics in the shell: pass/fail rate, lint/static-analysis counts, bug-fix success rate, regression rate, practice score trend, pattern reuse score, coding-memory growth, and per-model performance by coding task type.
- [x] 28.8 Add coding-specific degraded-state UX showing when validation is limited by missing linters, unavailable security tools, sandbox restrictions, blocked policies, or missing specialist routes and what that means for trust/promotion.
- [x] 28.9 Add bottom-dock coding chips for current file, current project, primary model, sandbox mode, and quick coding-depth controls while preserving the shared send/stop/session controls.
- [ ] 28.10 Keep raw code execution logs, stdout/stderr, and sandbox traces as secondary debug surfaces only; the primary Coding Mode UX must surface structured summaries and explicit validation gates first.

29. Phase 29 - Shell Reactivity, Settings, Accessibility, Performance, and Packaging
- [ ] 29.0.1 Add shell-wide reactivity tied to typed state: background tinting, animated haze, panel accent rails, button hover/focus styling, verification accents, degraded/fallback styling, and bounded notifications.
- [ ] 29.0.2 Keep shell motion meaningful rather than decorative; no visual effect should fire without mapping to a real runtime or coding-state signal.
- [x] 29.1 Add persisted shell settings for orb size, animation intensity, reduced motion, ambient reactivity, particle density, side-drawer defaults, activity-strip visibility, timeline visibility, resource-ribbon visibility, and notification visibility.
- [x] 29.2 Add persisted performance settings for low-resource mode, reduced-effects mode, animation frame cap, and simplified orb mode, plus presets for Minimal, Balanced, and Immersive.
- [x] 29.3 Add accessibility controls for low motion, higher contrast, larger status text, and simpler accent behavior while keeping settings import/export compatible with existing profiles.
- [ ] 29.4 Use a shared animation-clock strategy instead of per-widget free-running loops, cap particle counts by preset, and keep runtime pressure visible without silently changing the user's chosen visual preset.
- [ ] 29.5 Preserve the integrated resource ribbon and extend it with degraded-mode flagging, capability pressure, heavy-slot usage, observation tier, and cloud-helper state so the shell always explains why it looks or behaves differently under pressure.
- [x] 29.6 Make onboarding, readiness/preflight, capability details, settings, knowledge library, task history, run inspector, and debug/event access all reachable from the new shell without losing packaged-app parity.
- [x] 29.7 Update the packaged Windows entrypoint so the PySide6 orb shell becomes the planned default UI after parity while preserving stub-mode startup, readable failures, support-bundle export, and safe recovery to a minimal path when startup fails.
- [x] 29.8 Extend onboarding, preflight, and packaged setup guidance to match the shell language and include Coding Mode readiness, sandbox/toolchain requirements, optional model bundles, and idle-practice safety controls.

30. Phase 30 - Shell Validation, Coverage, and Rollout
- [ ] 30.0.1 Keep the full backend and compatibility suite green throughout the shell migration; UI work is not complete if it regresses orchestration, storage, long-horizon runs, capability policy, or packaged startup.
- [x] 30.1 Add shell-state mapping tests for idle, listening, planning, local retrieval, web retrieval, fast reasoning, deep reasoning, critic verification, compression, responding, speaking, degraded mode, approval pending, capability sessions, observation tiers, offline state, and all coding states.
- [x] 30.2 Add PySide6 widget tests for orb demo states, drawer open/close behavior, timeline population, conversation/task cards, coding workspace cards, resource ribbon, settings persistence, reduced-effects mode, low-resource mode, resize behavior, and shell shutdown without leaked Qt state.
- [x] 30.3 Add mode-switch tests proving the user can move between general assistant flow and Coding Mode without stale chips, cards, orb states, route labels, or hidden panel state leaking across modes.
- [x] 30.4 Add feature-to-surface audit tests proving every major subsystem emits at least one visible shell artifact and that operator-critical features do not disappear into debug-only views.
- [x] 30.5 Add flow tests for stub-mode launch, real-mode launch, fast task, deep task, long-horizon task, degraded/fallback path, web fallback, critic rejection, compressor activity, optimizer advisory visibility, task history access, knowledge-library access, settings/profile round-trip, readiness/preflight flow, local-AI control plane flow, capability-session visibility, approval flow, Coding Mode flow, idle-practice flow, and packaged Windows launch.
- [x] 30.6 Add coding-workspace tests covering planning, generating, testing, debugging, reviewing, indexing, practice, regression-detected state mapping, pattern-tier transitions, and validator-result visibility.
- [x] 30.7 Run Qt tests in offscreen mode in CI, keep event-burst throttling covered, and verify that the shell remains bounded under resize churn, state bursts, and low-resource conditions.
- [ ] 30.8 Follow this rollout order: complete the backend/UI host split, finish the full shell-state contract, finalize the standalone orb widget, build the main shell frame, add hero/activity/task surfaces, migrate drawers and sheets, wire live runtime updates, add advanced orb effects, add timeline and long-horizon trays, upgrade the local-AI control plane, integrate capability/approval overlays, finish shell-wide reactivity and settings, and only then switch packaged builds to the new shell by default.

