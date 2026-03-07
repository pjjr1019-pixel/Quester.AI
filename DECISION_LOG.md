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
