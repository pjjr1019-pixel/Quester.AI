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
- Primary model: `all-MiniLM-L6-v2`
- Fallback backend: `ollama_embeddings`
- Fallback model: `nomic-embed-text`

### 0.5.4 Vector Store Adapter (Pinned Default)
- Primary adapter: `chromadb` local persistent collection
- Fallback adapter: `simple_inmemory` for zero-dependency development

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

