# Quester.AI

Quester.AI is a local-first, async, multi-agent AI runtime scaffold designed around a shared model manager, typed pipeline contracts, and strict consumer-hardware limits.

Current repo state:
- Phases 0.5-4 are implemented.
- Phase 5 retrieval/storage foundation, dedicated bounded retrieval service, smarter persistent-vector startup, bounded web fallback, persisted web evidence, typed task/status persistence, machine-readable compression-runtime registries, and explicit seed/demo corpus separation are implemented.
- Phase 6A additive IR/storage contracts are implemented: `CompressedTrace` now carries a typed canonical graph, operation stream, context frames, proof hash, and decode hints while preserving legacy `tokens` and `expanded_preview`.
- Phase 6B core macro runtime is implemented: `macro_engine.py` now performs deterministic nested macro expansion, parameterized motif compression, macro proof fingerprinting, canonical normalization, recursion-guarded replay, and normalized round-trip verification.
- Phase 6C selective-context loading is implemented: `reasoner.py` and `critic.py` now load only the task-scoped active runtime subset they need, and storage bootstraps the compact built-in opcode/decoder lexicon required for fresh runtimes.
- Phase 6 is complete: persisted traces now use a compact builder-backed storage form, canonical graphs now model evidence items, intermediate bindings, macro definitions, typed activities, and agent/backend ownership, and the density benchmark is locked by tests.
- Phase 7 is functionally complete in stub mode: foreground agents stay thin, Planner/Reasoner/Critic/Compressor all use bounded schema-constrained JSON paths with deterministic fallback, and typed handoffs lock the Researcher -> Reasoner -> Critic boundary.
- `reasoning_service.py` now supports bounded `fast` and `deep` modes; `deep` mode materializes multiple candidate traces in canonical IR, scores them with verifier/evidence/agreement/proof-hash signals, and stays budget-bounded.
- `critique_service.py` now performs hybrid strict verification with bounded arithmetic, Python-expression, Python-code, unit-test, evidence-count, and retrieval-grounding helpers, emits machine-readable failure categories and repair actions, and degrades or abstains instead of polishing weak answers.
- `translation_service.py` now renders final answers strictly from verified state with templates and source citations, and `orchestrator.py` now applies a bounded repair loop plus richer dashboard/runtime-event metadata.
- `compression_service.py` now proposes graph-path, candidate-subproof, and symbol-bundle macros in addition to token/opcode motifs, so compression is no longer limited to exact-token aliasing.
- Phase 8 is functionally complete in proposal-only mode: `storage.py` persists replay samples, replay evaluations, proposal lifecycle records, activation decisions, rollback snapshots, and verified `deep`-trace exports, while `self_optimizer.py` enforces a bounded `propose -> simulate -> validate -> activate` lifecycle without applying live macro changes.
- Stub mode is the default and is the supported path for local tests.
- Remaining near-term work is dashboard/operator visibility over the richer reasoning, critique, and optimizer metadata, plus future policy-gated live activation work if it is ever pursued.

## Requirements

- Python `3.11+`
- Windows is the current development environment
- Baseline target: `6 GB VRAM / 8 GB RAM`
- Current dev calibration profile: `4 GB VRAM / 8 GB RAM`

## Quick Start

Create a virtual environment and install the local project with test dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[test]
```

Run the test suite:

```bash
python -m pytest -q -ra
```

Run the Phase 6 trace-density benchmark harness:

```bash
python phase6_benchmark.py
```

Run the stub-mode app entrypoint:

```bash
python -m orchestrator
quester-ai-packaged
```

## Optional Extras

Install only what you need:

```bash
python -m pip install -e .[test]
python -m pip install -e .[telemetry]
python -m pip install -e .[embeddings]
python -m pip install -e .[vector]
python -m pip install -e .[llama-cpp]
python -m pip install -e .[dev]
```

Extras in this repo:
- `test`: pytest
- `telemetry`: psutil
- `embeddings`: sentence-transformers
- `vector`: chromadb
- `llama-cpp`: llama-cpp-python
- `dev`: test + telemetry helpers

## Real-Backend Prerequisites

Python extras are only part of real mode. You also need local runtime prerequisites:

- `ollama` must be installed separately and reachable at `http://localhost:11434`
- If using the `llama_cpp` fallback backend, place the configured GGUF model under `models/` or point config at an absolute path
- If using the sentence-transformers embedding backend, install the `embeddings` extra
- If using the Chroma vector backend, install the `vector` extra

Pinned defaults from the current decision log:
- Generation backend: `ollama` with `qwen2.5:3b-instruct-q4_K_M`
- Generation fallback: `llama_cpp` with `qwen2.5-3b-instruct-q4_k_m.gguf`
- Embedding backend: `sentence-transformers/intfloat/e5-small-v2`
- Retrieval encoding: separate query/document embedding paths when the backend supports them
- Retrieval ranking: dedicated bounded hybrid retrieval service with SQLite `FTS5` lexical candidates, selective chunk/vector loading, and optional bounded reranking on larger budgets
- Web fallback provider: bounded MediaWiki API lookup in real mode, deterministic stub adapter in stub mode
- Vector store target for future retrieval work: local Chroma persistent collection

Step-by-step setup for that exact bundle lives in [`LOCAL_MODEL_SETUP.md`](./LOCAL_MODEL_SETUP.md).

Current optional specialist-role support:
- `reranker`: implemented as an opt-in routed local role; the dashboard Settings and Readiness views can enable it without changing the base `generation + embedding` runtime
- `speech_to_text` and `vad`: implemented as opt-in local voice-input roles with bounded `.wav` processing, deterministic local VAD, a dashboard voice-input tab, stub-mode validation paths, and optional Windows `System.Speech` transcription when available
- `text_to_speech`: implemented as an opt-in routed local role with bounded `.wav` synthesis output, deterministic stub speech generation for validation, optional Windows `System.Speech` synthesis when available, and audio-tab actions for speaking custom text or the current answer
- `translation`: implemented as an opt-in routed local role with bounded free-text, answer, and document-style translation in the dashboard Translation tab, deterministic stub translation for validation, and optional Argos Translate routing when installed
- `code_specialist`: implemented as an opt-in routed local role for bounded file or snippet review in the dashboard Code tab, with deterministic stub maintenance analysis for validation and on-demand unload support
- The Local AI control-plane panel now surfaces all typed local roles, recent routed decisions, recent fallback reasons, bounded cache status, and recent optimizer suggestions in one place
- The same panel now exposes per-role quick actions for install guidance, enable/disable, warm, unload, test ping, and fallback inspection without splitting specialist features into separate agent UIs
- The control plane now also surfaces typed compressor insights with estimated gain, validation state, blocked reasons, and whether a reusable pattern came from deterministic analysis or replay evidence

Pinned lightweight specialist-role defaults:
- `reranker`: `jinaai/jina-reranker-v1-tiny-en`
- `speech_to_text`: `whisper.cpp` or `openai/whisper-tiny`
- `vad`: `Silero VAD`
- `text_to_speech`: `Piper`
- `translation`: `Argos Translate`
- `code_specialist`: `Qwen/Qwen2.5-Coder-1.5B-Instruct`

Phase 20 foundation:
- `capability_guardrails.py` now codifies the first desktop/control guardrail: future desktop or cloud helpers must extend the current local-first runtime, keep `Orchestrator.run_task(question, thinking_minutes)` as the public task entrypoint, preserve the base `generation + embedding` pair plus local storage and audit path, remain opt-in, and never make cloud helpers the primary execution path.
- `data_structures.py` and `capability_runtime.py` now define typed capability requests for file, shell, browser, app-focus, clipboard, screenshot, OCR, and desktop-input actions, plus typed policy outcomes, capability registrations, audit records, and execution results.
- `orchestrator.py`, `storage.py`, and `dashboard.py` now route those requests through a persisted capability registry, explicit allowlists, audit logging, approval-gated policy decisions, and bounded execution paths instead of unrestricted OS control.

Phase 21 session foundation:
- `data_structures.py` and `storage.py` now define and persist explicit local task sessions, including session state, pending approvals, active-session tracking, and dashboard session projections.
- `orchestrator.py` now requires an active local task session before capability execution can proceed, recovers or pauses sessions safely on restart or shutdown, and exposes start, pause, resume, stop, and kill-switch controls before any live OS adapters exist.
- `dashboard.py` now surfaces visible local-session indicators for control mode, current target, pending approvals, kill-switch state, and last action alongside the existing long-horizon run controls.

Phase 21 live executor tranche:
- `capability_runtime.py` now executes bounded live file operations inside allowlisted roots for read, write, copy, move, archive, delete, and directory listing requests.
- `capability_runtime.py` now executes allowlisted shell commands inside allowlisted working directories with bounded timeout plus stdout or stderr capture.
- `capability_runtime.py` now executes bounded browser `read` and `navigate` actions for allowlisted domains, visible app/window focus actions with title matching and foreground validation, and approval-gated Windows desktop input for bounded typing, key chords, mouse moves, and mouse clicks against validated targets.
- `orchestrator.py` and `dashboard.py` now surface the live control tier explicitly, including per-capability executor kind, session safety state, loop-guard pauses, recovery pauses, and desktop-control readiness across file, shell, browser, app/window, and direct-input controls.

Phase 22 observation tranche:
- `capability_runtime.py` now executes screenshot-on-demand capture into allowlisted output paths and CPU-first OCR against allowlisted local images or selected image regions, using Windows OCR first with local `tesseract` fallback when available.
- `config.py`, `data_structures.py`, and `orchestrator.py` now define and enforce strict continuous-capture caps for low FPS, downscaled resolution, bounded frame history, diff-threshold retention, and region-of-interest behavior, with the capture loop tied directly to local task session start, pause, resume, stop, and shutdown.
- `orchestrator.py` now treats `ocr_on_step` and `vision_on_step` as explicit per-step observation modes instead of passive labels: successful UI-facing steps can trigger bounded observation captures, low-headroom states skip those captures, and `vision_on_step` records route-aware CPU-OCR fallback until the routed vision executor lands.
- `model_manager.py` now registers optional routed `vision` and `specialist_perception` roles with pinned lightweight recommendations (`HuggingFaceTB/SmolVLM-256M-Instruct` and `PaddleOCR`), and the heavy-slot scheduler can swap those roles in on demand without letting the runtime exceed two active heavy roles at once.
- `orchestrator.py` now reports `screenshot_on_demand`, `ocr_on_step`, and `continuous_capture` as live observation tiers in readiness, while `vision_on_step` becomes a routed ready tier when a concrete vision role is enabled and otherwise degrades visibly to CPU OCR fallback.
- `tests/test_phase22_observation_execution.py` now locks the live screenshot and OCR paths directly, while the existing readiness tests expect the new partial-live observation state.
- `tests/test_phase22_continuous_capture.py` now locks continuous-capture frame retention, hard-cap enforcement, and readiness projection.
- `tests/test_phase22_on_step_modes.py` now locks explicit per-step OCR execution, route-aware vision fallback, and low-headroom observation gating.

The code already fails clearly when real-mode prerequisites are missing:
- missing Ollama service -> backend unavailable/startup error
- missing `llama-cpp-python` -> startup error
- missing GGUF file -> startup error
- missing `sentence-transformers` -> startup error

## Architecture

Foreground pipeline:

```text
Planner -> Researcher -> Reasoner -> Critic -> Compressor -> Dashboard
```

Key runtime pieces:
- `orchestrator.py`: top-level pipeline owner
- `model_manager.py`: the only component allowed to load/unload/schedule model work
- `data_structures.py`: typed contracts and serialization boundaries
- `storage.py`: SQLite task/event/status/trace persistence plus retrieval storage
- `macro_engine.py`: deterministic macro registry runtime, proof fingerprints, canonical normalization, and round-trip verification
- `retrieval_service.py`: bounded local lexical/vector retrieval policy
- `dashboard.py`: headless/Tkinter event consumer

## Implementation Workflow

Phase 13 is now codified as a standing workflow, not just roadmap text.

- Review the relevant phase checklist before each implementation pass.
- Keep each change focused on one subsystem and one primary test surface where practical.
- Name the target files, required typed contracts, and success tests before coding.
- Do not start later-phase work until the earlier phase acceptance path is green.
- Preserve both `stub_mode=true` and `stub_mode=false` behavior on runtime or readiness changes.
- Keep structured agent outputs typed and validated, with one bounded repair attempt before deterministic fallback.

Use [`IMPLEMENTATION_TASK_TEMPLATE.md`](./IMPLEMENTATION_TASK_TEMPLATE.md) for the repo's standard task brief.

## Runtime Limits

The repo is designed around bounded resource behavior:
- one generation slot
- one embedding slot
- idle backend unload support
- low-memory fallback behavior
- bounded dashboard and optimizer queues

Config lives in [`config.py`](./config.py). Runtime choices are locked in [`DECISION_LOG.md`](./DECISION_LOG.md).

Current Phase 12 acceptance thresholds are explicit:
- validity: at least one verifiable `deep` example must improve with extra bounded test-time compute, final selection must stay verifier-backed, structured degraded outcomes are allowed, and real-mode failures must stay actionable
- compression: foreground compression is capped at `5` validated proposals per task, scans at most `6` recent reasoning logs in the foreground path, preserves proof-hash stability, and must not regress critic validity
- resources: the supported bounded baseline remains `1` generation slot, `1` embedding slot, bounded queues, `4 GB VRAM / 8 GB RAM` dev calibration, and `6 GB VRAM / 8 GB RAM` acceptance target

Those thresholds live in [`acceptance_thresholds.py`](./acceptance_thresholds.py) and are locked by
[`tests/test_phase12_acceptance_thresholds.py`](./tests/test_phase12_acceptance_thresholds.py).

## Testing

Current supported local validation commands:

```bash
python -m pytest -q -ra
python -m unittest discover -v
```

Prefer `python -m pytest` over bare `pytest` so the invoked interpreter matches the repo's Python `3.11+` requirement.

Required Phase 16 subsystem gates currently defined:

- Data:
  `python -m pytest -q tests/test_phase2_contracts.py`
  `python -m pytest -q tests/test_compatibility_contract.py`
- Storage:
  `python -m pytest -q tests/test_phase5_persistence.py`
  `python -m pytest -q tests/test_compatibility_contract.py`
- Model:
  `python -m pytest -q tests/test_phase3_runtime.py`
  `python -m pytest -q tests/test_phase12_preflight.py`
  `python -m pytest -q tests/test_compatibility_contract.py`
- Macro:
  `python -m pytest -q tests/test_phase6_macro_engine.py`
  `python -m pytest -q tests/test_compatibility_contract.py`
- Agent:
  `python -m pytest -q tests/test_phase12_agent_units.py`
  `python -m pytest -q tests/test_phase12_end_to_end.py tests/test_phase12_preflight.py`
  `python -m pytest -q tests/test_compatibility_contract.py`
- Orchestrator:
  `python -m pytest -q tests/test_phase16_subsystem_gates.py`
  `python -m pytest -q tests/test_phase12_end_to_end.py tests/test_phase13_async_safety.py`
  `python -m pytest -q tests/test_compatibility_contract.py`
- Optimizer:
  `python -m pytest -q tests/test_phase12_resource_optimizer.py tests/test_phase5_persistence.py`
  `python -m pytest -q tests/test_compatibility_contract.py`
- Dashboard:
  `python -m pytest -q tests/test_phase16_subsystem_gates.py tests/test_phase12_gui_acceptance.py`
  `python -m pytest -q tests/test_phase13_async_safety.py tests/test_phase7_boundaries.py`
  `python -m pytest -q tests/test_compatibility_contract.py`

Each defined subsystem gate requires both its local commands and its
compatibility command to pass together before the subsystem is marked done.
The typed gate registry lives in [`validation_gates.py`](./validation_gates.py)
and is locked by
[`tests/test_phase16_validation_gates.py`](./tests/test_phase16_validation_gates.py).

Required Phase 16 project-wide gates currently defined:

- Resource:
  `python -m pytest -q tests/test_phase12_acceptance_thresholds.py`
  `python -m pytest -q tests/test_phase3_runtime.py tests/test_phase12_resource_optimizer.py tests/test_phase17_long_horizon.py`
- Pre-release smoke:
  `python -m pytest -q tests/test_phase12_end_to_end.py`
  `python -m pytest -q tests/test_phase12_packaged_smoke.py tests/test_phase12_preflight.py`
- Release:
  `python -m pytest -q tests/test_phase12_acceptance_thresholds.py tests/test_phase12_end_to_end.py tests/test_phase12_translation_exports.py`
  `python -m pytest -q tests/test_phase12_packaged_smoke.py tests/test_phase12_gui_acceptance.py tests/test_phase12_preflight.py`
- Project completion:
  `python -m pytest -q tests/test_compatibility_contract.py`
  `python -m pytest -q tests/test_phase12_acceptance_thresholds.py tests/test_phase12_end_to_end.py tests/test_phase12_gui_acceptance.py tests/test_phase12_packaged_smoke.py tests/test_phase12_preflight.py tests/test_phase12_translation_exports.py`

The project-wide gate registry also lives in [`validation_gates.py`](./validation_gates.py)
and is locked by
[`tests/test_phase16_project_gates.py`](./tests/test_phase16_project_gates.py).

Test coverage currently includes:
- typed contract serialization and validation
- compatibility-contract checks
- orchestrator stub-mode pipeline execution
- task/status persistence, persisted web evidence, degraded web-fallback warnings, and append-only JSONL mirrors
- retrieval-service, `FTS5` lexical candidate generation, persistent-vector startup reconciliation, bounded reranking, selective vector-loading behavior, compression-runtime registry persistence, task-scoped selective runtime loading, and seed/demo corpus separation behavior
- repeatable Phase 6 trace-density benchmarking and fresh-start runtime-lexicon bootstrap regression coverage
- model manager runtime/fallback/semaphore behavior
- explicit Phase 16 data, storage, model, macro, agent, orchestrator, optimizer, and dashboard validation-gate definitions
- explicit Phase 16 resource, pre-release smoke, release, and project-completion gate definitions

## Limitations

This repo is not feature-complete yet.

Known planned areas:
- the current real web fallback is intentionally narrow and uses MediaWiki as the default provider
- the packaged path now includes a separate `quester-ai-packaged` entrypoint, startup planning, exported preflight/onboarding artifacts, and stub-mode recovery diagnostics, but broader clean-machine release validation is still the remaining packaging hardening work
- `jsonschema` is the first optional structured-output helper; heavier structured-output libraries (`outlines`, `msgspec`) and solver helpers (`z3-solver`) remain deliberately off the default path

The authoritative roadmap is in [`Masterplan.txt`](./Masterplan.txt).
