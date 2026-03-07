# Decision Log

## Phase 0.5 Locked Decisions (2026-03-06)

These choices are locked before Phase 1 to prevent architecture churn.

### 0.5.1 Hard Runtime Targets
- VRAM budget: 6GB max target
- System RAM budget: 8GB max target
- Constraint type: hard design constraint for defaults and concurrency

### 0.5.2 Generation Backend (Pinned Default)
- Primary backend: `ollama`
- Primary model: `qwen2.5:3b-instruct-q4_K_M`
- Fallback backend: `llama_cpp`
- Fallback model: `qwen2.5-3b-instruct-q4_k_m.gguf`

### 0.5.3 Embedding Backend (Pinned Default)
- Primary backend: `sentence_transformers`
- Primary model: `intfloat/e5-small-v2`
- Fallback backend: `ollama_embeddings`
- Fallback model: `nomic-embed-text`
- Retrieval encoding rule: prefer asymmetric query/document embedding APIs and fall back to `query:` / `passage:` prefixes when the backend does not expose dedicated methods

### 0.5.4 Vector Store Adapter (Pinned Default)
- Primary adapter: `chromadb` local persistent collection
- Fallback adapter: `simple_inmemory` for zero-dependency development

### 0.5.4a Web Fallback Defaults
- Real-mode provider: bounded MediaWiki API lookup
- Stub-mode provider: deterministic stub adapter to keep local tests portable
- Failure policy: web errors degrade to local-only evidence and structured logs rather than hard crashes

## Phase 6A IR Contract Rule (2026-03-07)

- `CompressedTrace.tokens` and `expanded_preview` remain required legacy/debug projections during the Phase 6 migration.
- The additive IR contract is carried alongside them via `ir_version`, `canonical_graph`, `operation_stream`, `symbol_table_refs`, `evidence_handles`, `context_frames`, `proof_hash`, and `decode_hints`.
- Proof fingerprints may be persisted independently in proof-hash history, but old serialized traces must remain readable.

## Phase 6B Macro Runtime Rule (2026-03-07)

- `MacroEngine` proof hashes are derived from normalized IR payloads, not raw token text alone.
- Equivalent macro-expanded and literal-expanded traces should resolve to the same proof hash when their replayed canonical graphs are equivalent.
- Default macro recursion limit: `8`

## Phase 6C Selective Runtime Loading Rule (2026-03-07)

- `ReasonerAgent` and `CriticAgent` must load only the task-scoped active opcode, macro, decoder, and symbol-table subset needed for the current task or trace; they must not pull the full registry into context by default.
- Fresh storage bootstraps the compact built-in runtime lexicon required by the current IR path: opcodes `lookup`, `bind`, `compare`, `infer`, `aggregate`, `check`, `emit`, `cite`, `confidence_update`, plus decoders `verified_answer` and `compressed_trace_summary`.
- Historical macro usage may inform loading only when the caller does not explicitly request a narrower runtime subset.

## Phase 6 Density Rule (2026-03-07)

- Generated traces may persist `canonical_graph` as a derived view instead of a required stored blob when the graph can be deterministically rebuilt from compact IR fields.
- `CompressedTrace` now uses `canonical_graph_builder` to mark generated traces whose graph can be reconstructed on read without losing compatibility or proof-check behavior.
- Manual or externally supplied traces without a known builder must keep persisting the explicit `canonical_graph`.
- Generated traces may also persist a compact storage form via `CompressedTrace.to_storage_dict()` that omits rebuildable debug projections such as expanded preview text, trace notes, repeated evidence handles, and builder-specific per-step metadata.

## Phase 6 Semantic Graph Rule (2026-03-07)

- Builder-generated canonical graphs must model more than just question/evidence-set/answer shells: evidence items, intermediate bindings, answer fragments, macro definitions, and typed activities should appear explicitly when that semantic role exists in the trace.
- Typed agent records must preserve component ownership and backend identity for builder-generated traces.

## Phase 6 Normalization Rule (2026-03-07)

- Proof hashes and round-trip validation must run against canonicalized IR payloads, not raw list ordering.
- For commutative operations such as `bind`, `compare`, `aggregate`, `cite`, and `confidence_update`, argument order must be canonicalized before hashing or verification.
- Graph collections, decode hints, and inherited assumptions should normalize to a stable order before equivalence checks.

## Phase 6 Macro Proposal Gate Rule (2026-03-07)

- Macro proposals must carry validation state and a proof fingerprint before they are persisted or surfaced as candidates for later activation.
- Proposal validation must reject missing loop/fingerprint checks, missing opcode patterns, and missing invariants for deterministic round-trip, provenance preservation, and uncertainty preservation.
- `CompressorAgent` and `SelfOptimizer` may emit candidate proposals, but those proposals remain non-authoritative until they pass the validation gate.

## Phase 7 Boundary Rule (2026-03-07)

- Foreground agents should stay as thin lifecycle/orchestration wrappers; shared logic belongs in dedicated service modules.
- The Researcher -> Reasoner boundary is locked by `ResearchReasonerHandoff`, and the Reasoner -> Critic boundary is locked by `ReasonerCriticHandoff`.
- Natural-language answer rendering stays downstream of critique/verification; typed IR and critique state remain the primary reasoning artifacts.
- Dataclasses remain the domain model; `agent_schema.py` provides boundary-only structured-output and handoff schema helpers for future model-backed JSON I/O.

## Phase 7 Structured Planner Rule (2026-03-07)

- Schema-constrained model I/O should go through one shared bounded helper instead of ad hoc JSON parsing in each agent.
- The first live structured-output path is the planner: one model attempt, at most one repair attempt, then deterministic fallback.
- Parsed planner output must be normalized back onto the real user question and active `ResourceBudget` so model drift cannot rewrite runtime budget policy.

## Phase 7 Structured Reasoner/Critic Rule (2026-03-07)

- `ReasoningService` may accept schema-constrained `CompressedTrace` JSON, but it must normalize the result back onto the real handoff, active runtime subset, symbol-table refs, evidence handles, and proof-hash rules before the trace is trusted.
- `CritiqueService` must run deterministic structural and runtime-subset checks first; schema-constrained model critique output is only advisory after those checks pass.
- Both services use the same bounded policy as the planner: one model attempt, at most one repair attempt, then deterministic fallback.

## Phase 7 Deep Candidate Rule (2026-03-07)

- `deep` mode must keep multiple candidate traces as canonical IR artifacts, not only as summary metadata on one selected answer.
- Candidate selection remains verifier-first: exact tool checks outrank retrieval-grounded fluency, and abstention is preferred over unsupported polish when no candidate clears the threshold.
- Larger budgets may increase candidate count, but the candidate budget stays explicitly bounded.

## Phase 7 Bounded Verifier Rule (2026-03-07)

- Checkable questions should use deterministic helper execution before optional model fallback.
- Python code and unit-test helpers stay on a bounded path: limited AST surface, limited line/node budgets, no imports, and no unrestricted attribute access.
- Critique output must surface machine-readable failure categories and repair actions so orchestration can repair or abstain without guessing.

## Phase 7 Acceptance Rule (2026-03-07)

- Phase 7 is only considered complete when `fast` and `deep` reasoning stay bounded, `deep` mode persists candidate traces, the critic selects or abstains from verifier-backed state, the orchestrator can repair or abstain without free-form guesswork, and final answer rendering comes from verified trace state with citations.
- Future phases may extend the verifier/toolset, but they should not reopen the Phase 7 handoff contracts or reintroduce free-form text as the primary reasoning artifact.

## Phase 8 Replay Gate Rule (2026-03-07)

- `SelfOptimizer` remains proposal-only by default; replay evaluation may score proposals, but it must not activate live macro/runtime changes until explicit activation logging and rollback records exist.
- Replay input should come from persisted task results, reasoning traces, critique outcomes, and performance metrics rather than ad hoc in-memory snapshots.
- Replay scope stays bounded by a configurable history window so optimizer work cannot become the dominant memory consumer.

## Phase 8 Optimizer Metric Rule (2026-03-07)

- The offline optimizer contract is locked to five metrics: compression gain, proof-hash stability, critique validity, latency ratio, and memory ratio.
- Default replay weights are `0.30 / 0.25 / 0.25 / 0.10 / 0.10` in that order, with a default minimum simulation score of `0.55`.
- Default replay gates cap both latency and memory ratios at `1.15`; proposals may pass replay evaluation, but proposal approval remains false until the later activation path exists.

### 0.5.5 Development Mode Default
- `stub_mode=true` by default for first-pass implementation and tests

### 0.5.6 Minimal First Successful Run Goal (Stub Mode)
- One input question runs full stage order:
  - Planner -> Researcher -> Reasoner -> Critic -> Compressor -> Dashboard
- Pipeline returns a structured `TaskResult`-shaped object without crash

### 0.5.7 Minimal First Real-Backend Run Goal
- One input question runs full stage order with real model calls enabled
- Runtime remains bounded with shared backend scheduling and status logging

### 0.5.8 Decision Lock Rule
- These defaults can be changed only by updating this file and `config.py`.
- Later phases must not silently override locked Phase 0.5 defaults.

### 0.5.9 Phase Gate
- Phase 1 starts only after all items 0.5.1-0.5.8 are locked above.
