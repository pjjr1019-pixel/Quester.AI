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
- Phase 8 prep is in place: `storage.py` persists optimizer replay samples, `config.py` locks the offline optimizer metric contract, and `self_optimizer.py` evaluates proposals against a bounded replay window while remaining proposal-only.
- Stub mode is the default and is the supported path for local tests.
- Remaining near-term work is Phase 8 activation audit trails and rollback records, plus expanding the dashboard from its current minimal event console.

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

## Runtime Limits

The repo is designed around bounded resource behavior:
- one generation slot
- one embedding slot
- idle backend unload support
- low-memory fallback behavior
- bounded dashboard and optimizer queues

Config lives in [`config.py`](./config.py). Runtime choices are locked in [`DECISION_LOG.md`](./DECISION_LOG.md).

## Testing

Current supported local validation commands:

```bash
python -m pytest -q -ra
python -m unittest discover -v
```

Prefer `python -m pytest` over bare `pytest` so the invoked interpreter matches the repo's Python `3.11+` requirement.

Test coverage currently includes:
- typed contract serialization and validation
- compatibility-contract checks
- orchestrator stub-mode pipeline execution
- task/status persistence, persisted web evidence, degraded web-fallback warnings, and append-only JSONL mirrors
- retrieval-service, `FTS5` lexical candidate generation, persistent-vector startup reconciliation, bounded reranking, selective vector-loading behavior, compression-runtime registry persistence, task-scoped selective runtime loading, and seed/demo corpus separation behavior
- repeatable Phase 6 trace-density benchmarking and fresh-start runtime-lexicon bootstrap regression coverage
- model manager runtime/fallback/semaphore behavior

## Limitations

This repo is not feature-complete yet.

Known planned areas:
- the current real web fallback is intentionally narrow and uses MediaWiki as the default provider
- the dashboard is still a minimal event console
- optional structured-output and solver dependencies (`jsonschema`, `outlines`, `msgspec`, `z3-solver`) remain deliberately off the default path

The authoritative roadmap is in [`Masterplan.txt`](./Masterplan.txt).
