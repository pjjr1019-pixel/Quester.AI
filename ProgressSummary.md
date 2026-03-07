# Progress Summary

Last updated: March 7, 2026

## Current status

- Phases 0.5, 1, 2, 3, and 4 are complete.
- The compatibility contract is mostly enforced and covered by tests.
- Phase 5 is functionally complete except for deferred `sqlite-vec` follow-up, Phase 6 is complete, Phase 7 is functionally complete in stub mode, and Phase 8 is functionally complete in proposal-only mode.
- The optimizer now records proposal lifecycle, activation decisions, rollback snapshots, and verified `deep`-trace exports while keeping live runtime activation blocked by policy.

## Completed in the current checkpoint

- Added typed contracts, compatibility tests, and a stable public API surface.
- Implemented shared model management, bounded concurrency, backend fallback rules, and health snapshots.
- Finished Phase 4 budget mapping so `thinking_minutes` changes real planner, researcher, reasoner, and critic behavior.
- Added a retrieval/storage foundation:
  - `retrieval.py` now provides chunking, stable document/chunk IDs, lexical scoring, a simple in-memory vector index, and a Chroma adapter.
  - `storage.py` now separates event logging, KV storage, source documents, chunk storage, vector entries, macro registry, reasoning history, and performance history into explicit repository boundaries behind `StorageManager`.
  - `researcher.py` now performs local-first retrieval from persisted chunks, supports metadata-filtered search, seeds a small local corpus for stub-mode development, and only uses bounded web fallback when local evidence is weak or freshness is requested.
- Changed the default retrieval embedding strategy to `intfloat/e5-small-v2` and added explicit query/document embedding paths through the model manager, storage ingestion, and researcher retrieval flow.
- Added a bounded web adapter layer:
  - `web_adapter.py` now provides a deterministic stub adapter for stub mode and a real MediaWiki-backed adapter with timeout, retry, degraded-mode handling, dedupe, and safe JSON parsing.
  - `researcher.py` now logs bounded web lookups, converts adapter responses into typed `EvidenceItem` web evidence, and degrades cleanly to local-only evidence when web lookup fails.
- Persisted fetched web evidence itself:
  - `data_structures.py` now defines a typed `WebEvidenceRecord`.
  - `storage.py` now persists fetched web evidence plus query/provider/reason/warning provenance in SQLite and mirrors those records into `web.jsonl`.
  - `researcher.py` now records deduplicated fetched web evidence before emitting the final `researcher.web_lookup` event.
- Added optional bounded reranking:
  - `storage.py` now supports a bounded rerank stage over only a small top-N retrieval set.
  - `researcher.py` now enables reranking only for larger retrieval budgets and surfaces rerank metadata on local evidence items.
- Extracted the remaining retrieval search policy out of `StorageManager`:
  - `retrieval_service.py` now owns bounded hybrid lexical/vector search behavior.
  - `storage.py` now uses SQLite `FTS5` for lexical candidate generation and only loads merged candidate chunk/vector payloads instead of scanning all stored vectors per search.
- Improved persistent vector startup behavior:
  - `retrieval.py` now lets vector adapters declare whether startup needs a storage reload.
  - `storage.py` now reloads all vectors on startup only for non-persistent backends or when a persistent backend reports a missing/count-mismatched collection.
- Added machine-readable compression-runtime registries:
  - `data_structures.py` now defines typed opcode, decoder, symbol-table, proof-hash, and active-runtime-subset records.
  - `storage.py` now persists opcode lexicon entries, decoder entries, symbol-table snapshots, proof-hash history, and exposes task-scoped active-subset loading plus `compression_lexicon.md` export generation.
- Locked seeded-corpus separation policy:
  - `config.py` now defines stub-only seed-corpus defaults plus explicit seed/demo metadata tags.
  - `researcher.py` now tags seeded documents as demo corpus and excludes them from non-stub retrieval by default, while keeping stub-mode development deterministic.
- Added pulled-forward task/status persistence:
  - `data_structures.py` now extends `TaskResult` additively with optional `answer_text`, `warnings`, and `metrics`.
  - `storage.py` now persists typed task runs, runtime events, agent statuses, reasoning traces/logs, performance metrics, and macro registry entries, with append-only `events.jsonl`, `traces.jsonl`, `web.jsonl`, and `status.jsonl` mirrors where appropriate.
  - `orchestrator.py` now emits typed stage-start/stage-done/stage-failed runtime events, persists `AgentStatus` updates, records task results and metrics, and surfaces degraded web-fallback warnings in the final `TaskResult`.
- Began Phase `6A` and locked the additive IR/storage contract:
  - `data_structures.py` now defines typed semantic entities, activities, agents, provenance bundles, context frames, operation steps, decode hints, and a canonical reasoning graph.
  - `CompressedTrace` now carries additive IR fields: `ir_version`, `canonical_graph`, `operation_stream`, `symbol_table_refs`, `evidence_handles`, `context_frames`, `proof_hash`, and `decode_hints`, while preserving legacy `tokens` and `expanded_preview`.
  - `reasoner.py` now emits a minimal IR-backed trace in the current stub path, and `storage.py` now exposes typed `list_reasoning_traces(...)` plus automatic proof-hash history capture when a stored trace already has a proof fingerprint.
- Implemented the core of Phase `6B` in `macro_engine.py`:
  - `MacroEngine.compress(...)` now expands nested macros with a strict recursion guard, compiles the expanded sequence into a deterministic operation stream, replays it into a canonical graph, and emits a proof-hashed IR-backed `CompressedTrace`.
  - `MacroEngine.expand_to_graph(...)` now replays macro-engine traces into canonical graph form for validation instead of relying on token text alone.
  - `MacroEngine.verify_round_trip(...)` now checks proof-hash stability and normalized graph stability rather than only expanded-token equality.
- Implemented the next Phase 6 runtime-hardening tranche:
  - `macro_engine.py` now supports registry-backed `Macro` objects, parameterized macro invocations, repeated-motif compression, macro proof fingerprints, canonical normalization for commutative opcode arguments, and explicit proposal validation gates.
  - `storage.py` now bootstraps the full compact core opcode lexicon (`lookup`, `bind`, `compare`, `infer`, `aggregate`, `check`, `emit`, `cite`, `confidence_update`), keeps decoder lexicon entries in the runtime registry, filters inactive macros from task-scoped loads, and can export a compact trace debug view.
  - `compressor.py` and `self_optimizer.py` now emit validated `MacroProposal` payloads with proof fingerprints and explicit validation results before those proposals can influence later runtime work.
- Finished the remaining semantic-graph work that closes Phase 6:
  - `reasoner.py` now models evidence items, an intermediate binding node, and answer emission as typed canonical entities and activities instead of flattening everything into question/evidence-set/answer only.
  - `macro_engine.py` and `data_structures.py` now model macro definitions as typed canonical entities and expose `macro_expand` as an explicit activity in the canonical graph.
  - Typed agent ownership now carries backend identity through the rebuilt canonical graph for builder-generated traces.
- Hardened the Phase 6 migration boundary:
  - `tests/fixtures/compatibility/` now contains frozen legacy/stable payload fixtures for `CompressedTrace` and `TaskResult`.
  - Compatibility tests now prove both old payloads still deserialize and current payloads still preserve the frozen stable projections.
- Added Phase `6B` regression coverage:
  - `tests/test_phase6_macro_engine.py` now covers IR-backed compression, nested expansion, recursion blocking, proof-hash stability across equivalent encodings, and proof-hash drift detection.
- Implemented Phase `6C` selective-context loading and agent integration:
  - `reasoner.py` now loads only the task-scoped active opcode and decoder subset it needs, persists or reuses the active symbol-table snapshot, and exposes the last loaded runtime subset for diagnostics/tests.
  - `critic.py` now loads only the macro, opcode, decoder, and symbol-table subset referenced by the current trace and validates runtime-subset alignment before accepting the trace.
  - `storage.py` now bootstraps the compact built-in runtime lexicon on fresh startup so clean repos still have the core `lookup` / `bind` / `compare` / `infer` / `aggregate` / `check` / `emit` / `cite` / `confidence_update` opcodes plus `verified_answer` and `compressed_trace_summary` decoders available for bounded selective loading.
- Added Phase `6C` regression coverage:
  - `tests/test_phase5_persistence.py` now proves the reasoner excludes unrelated historical macros and unused registry entries, and proves the critic only loads the trace-scoped macro/opcode/decoder subset plus the current task symbol table.
- Added coverage for parameterized macros, canonical normalization, proposal validation, and trace/debug export:
  - `tests/test_phase6_macro_engine.py` now covers parameterized motif compression/expansion, noncanonical commutative-argument rejection, and macro-proposal validation success/failure.
  - `tests/test_phase5_persistence.py` now proves inactive macros stay out of active runtime subsets, the richer lexicon export includes invariants and fingerprints, and `storage.py` can export a compact trace debug view while bootstrapping the full core opcode/decoder set.
- Added Phase `6.0.4` benchmark hardening:
  - `phase6_benchmark.py` now provides a repeatable stub-mode harness that compares legacy trace payload size against the current IR-backed trace payload and reports structural counts, JSON bytes, and recursive payload-memory estimates.
  - `tests/test_phase6_density.py` now keeps that harness deterministic and adds a clean-start orchestrator regression proving the built-in runtime lexicon bootstrap keeps the critic green on a fresh database.
- Applied the first optimization half of `6.0.4`:
  - `reasoner.py` now emits a denser stub canonical graph using compact symbolic IDs, a shared evidence-set entity instead of one graph node per evidence item, and less duplicated activity/operation metadata.
  - `data_structures.py` now emits more compact serialized IR payloads by dropping only default-empty IR fields while preserving round-trip equality.
- Applied the second optimization half of `6.0.4`:
  - Generated traces now persist `canonical_graph` as a derived view when the graph can be deterministically rebuilt from compact IR fields.
  - `CompressedTrace` now carries an additive `canonical_graph_builder` marker so storage can omit generated graphs on write and rebuild them on read without breaking equality or compatibility.
  - `macro_engine.py` and `reasoner.py` now stamp generated traces with deterministic graph-builder IDs and shared graph timestamps so storage round-trips remain exact.
- Applied the latest density pass and tightened the regression cap:
  - `data_structures.py` now omits derived `symbol_table_refs` and `decode_hints` from generated-trace payloads when the builder can deterministically reconstruct them on read.
  - `CompressedTrace.to_storage_dict()` now emits a compact persisted form for generated traces, and `storage.py` now records that compact payload instead of the full public debug projection.
  - `tests/test_phase6_density.py` now locks the improved benchmark ceiling at `< 1.7x` JSON growth and `< 2.7x` recursive payload-memory growth.
- Recorded the current density baseline from `python phase6_benchmark.py`:
  - The compact persisted IR-backed stub trace payload is currently about `1.42x` to `1.61x` larger in serialized JSON bytes and about `2.21x` to `2.59x` larger in recursive payload-memory footprint than the legacy projection, which is the accepted bounded density target for Phase 6 completion.
- Added packaging/project metadata with `pyproject.toml` and a repo `README.md`.
- Completed the Phase 7 boundary pass:
  - `planner.py`, `researcher.py`, `reasoner.py`, `critic.py`, and `compressor.py` are now thin lifecycle wrappers that delegate shared logic to `planner_service.py`, `research_service.py`, `reasoning_service.py`, `critique_service.py`, and `compression_service.py`.
  - `data_structures.py` now defines explicit `ResearchReasonerHandoff` and `ReasonerCriticHandoff` contracts so evidence handles, proof hashes, required runtime subsets, repair counters, and final-text policy stay typed across agent boundaries.
  - `agent_schema.py` now locks boundary-only structured-output and handoff schemas for planner, reasoner, critic, and compressor contracts while keeping dataclasses as the domain model.
  - `orchestrator.py` now wires the typed Researcher -> Reasoner and Reasoner -> Critic handoffs explicitly instead of relying on implicit tuple/dataclass coupling between agent calls.
- Completed `7.2` on top of that boundary layer:
  - `structured_generation.py` now provides one bounded schema-constrained JSON generation path with one repair attempt and deterministic fallback.
  - `planner_service.py` now uses `agent_schema.py` plus the shared structured-generation helper to parse a typed `Plan` from model JSON, normalize it to the real budget/question inputs, and fall back to the deterministic planner path if generation or repair fails.
  - `tests/test_phase7_boundaries.py` now covers valid JSON, one repair attempt, and bounded fallback for the planner output path.
- Completed the next Phase 7 foreground replacements on the same path:
  - `reasoning_service.py` now uses the shared structured-generation helper and `agent_schema.py` to accept schema-constrained `CompressedTrace` JSON, normalize it back onto the real handoff, active runtime subset, symbol-table refs, and proof-hash rules, and fall back to the deterministic IR builder if generation or repair fails.
  - `critique_service.py` now runs deterministic verifier checks first and only then accepts schema-constrained `CritiqueReport` JSON from the model path, with one bounded repair attempt and deterministic fallback if JSON decoding fails.
  - `tests/test_phase7_boundaries.py` now covers valid JSON, bounded repair, and fallback for both reasoner and critic structured-output paths.
- Completed the remaining stub-mode Phase 7 behavior:
  - `reasoning_service.py` now supports bounded `fast` and `deep` modes, materializes bounded candidate traces in canonical IR, and scales candidate count with the active budget.
  - `verification_tools.py` and `critique_service.py` now cover arithmetic, Python expression, Python code execution, unit-test style checks, evidence count, and evidence-grounding verification with bounded deterministic helpers.
  - `data_structures.py` and `agent_schema.py` now carry additive Phase 7 metadata for candidate traces, failure categories, provenance coverage, macro violations, and drift score.
  - `translation_service.py` now renders final answers from verified state with source citations, and `orchestrator.py` now applies bounded repair actions and emits richer reasoning/critique metadata into runtime events for the dashboard path.
  - `compression_service.py` now proposes candidate-subproof, graph-path, and symbol-bundle macros in addition to token/opcode motifs, so graph-backed compression is no longer absent from the foreground path.
- Completed the pre-Phase 8 optimizer gate:
  - `data_structures.py` now defines `OptimizerReplaySample` and `OptimizerReplayEvaluation` so replay inputs and simulation outputs stay typed.
  - `storage.py` now derives replay samples directly from persisted `TaskResult` objects and stores offline replay evaluations in dedicated repositories.
  - `config.py` now locks the replay history cap, proposal limit, weighted metric contract, and latency/memory gates for offline optimizer scoring.
  - `self_optimizer.py` now evaluates validated macro proposals against bounded persisted replay samples and traces, records those evaluations, and keeps every proposal proposal-only.
  - `tests/test_phase5_persistence.py`, `tests/test_phase7_boundaries.py`, and `tests/test_phase2_contracts.py` now cover replay-sample persistence, replay-evaluation persistence, and metric-contract validation.
- Closed the remaining Phase 8 lifecycle and export tranche:
  - `data_structures.py` now defines typed optimizer proposal, activation, rollback, and verified-`deep`-trace export records.
  - `storage.py` now persists append-only proposal lifecycle records, activation decisions, rollback snapshots, and can export only verified `deep`-mode traces as machine-readable dataset rows.
  - `orchestrator.py` now marks foreground task entry/exit explicitly so future optimizer activation work cannot mutate the active macro set during a live user run.
  - `self_optimizer.py` now emits a recorded `propose -> simulate -> validate -> activate` lifecycle, logs blocked or rejected activation decisions, prepares rollback snapshots for eligible proposals, defers cycles while foreground work is active, and keeps live activation policy-blocked by default.
  - `tests/test_phase5_persistence.py` and `tests/test_phase7_boundaries.py` now cover lifecycle persistence, verified-trace export, and foreground-safe optimizer deferral.

## Phase 5 items completed

- `5.0.1` repository boundaries designed first
- `5.0.2` legacy `StorageManager.log_event(...)` and KV behavior preserved
- `5.0.3` chunking, dedupe, and stable source-identity rules implemented
- `5.0.4` web timeout, retry, and degraded-mode behavior defined and implemented
- `5.0.5` Chroma kept as the pinned primary vector backend for v1
- `5.0.7` retrieval-tuned embedding default and query/document encoding strategy locked
- `5.0.9` fetched web evidence and lookup provenance persisted in machine-readable form
- `5.0.10` seeded demo corpus now stays explicitly separable from user/runtime knowledge
- `5.1` SQLite persistence now covers task runs, metrics, macros, runtime registries, and source metadata
- `5.3` vector adapter implemented
- `5.4` Chroma-primary generic vector index path implemented
- `5.4.1` persistent vector startup now reconciles only when needed
- `5.5` bounded chunker implemented
- `5.6` local-first retrieval implemented
- `5.6.1` query/document embedding split implemented when the backend supports it
- `5.6.2` reranking kept optional and bounded
- `5.6.3` retrieval policy moved into a dedicated retrieval service boundary
- `5.6.4` SQLite `FTS5` lexical candidate generation implemented
- `5.6.5` selective chunk/vector loading implemented for merged retrieval candidates
- `5.7` bounded web adapter implemented with source logging, dedupe, and safe parsing
- `5.8` web work is optional and time-bounded
- `5.9` storage split into explicit repositories/tables
- `5.10` retrieval/index logic kept separate from event/KV storage
- `5.10.1` typed `RuntimeEvent` / `AgentStatus` contract pulled forward
- `5.11` compression runtime storage added for opcodes, macros, decoders, symbol tables, and proof hashes
- `5.12` runtime source of truth kept machine-readable
- `5.13` human-readable compression lexicon export generated from registries only as a debug artifact
- `5.14` task-scoped active subset loading added for compression runtime registries
- `5.15` chunk text, metadata, embedding model, and vector IDs persisted together
- `5.16` source-document storage and vector storage kept independently rebuildable
- `5.17` task runs, surfaced warnings, and additive task outputs persisted
- `5.18` fetched web evidence persisted alongside lookup logs and provenance
- `5.19` `TaskResult` additively extended with answer text, warnings, and metrics
- `5.20` typed `AgentStatus` persistence implemented for stage start/completion/failure/degraded paths
- `5.0.8` machine-readable persistence gates needed before full Phase 6 are now in place

## Verified state

- Full test suite passes: `python -m pytest -q --cache-clear`
- Current passing count: `105 passed`

## Main gaps remaining

- `5.0.6`: optional `sqlite-vec` secondary adapter

## Recommended next steps

1. Expand the dashboard from the current event console into a richer operator view over verifier type, candidate score, repair actions, degraded reasons, replay scores, activation decisions, rollback snapshots, and citations.
2. Keep live optimizer activation policy-gated and off by default until there is a concrete operator workflow for approval, rollback, and post-activation monitoring.
3. Keep `5.0.6` deferred infrastructure, not the next coding target.
4. Keep the real web provider policy unchanged: do not broaden beyond MediaWiki until the persisted evidence/provenance path has seen more use.
