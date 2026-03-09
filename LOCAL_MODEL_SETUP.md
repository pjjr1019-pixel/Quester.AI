# Local Model Setup

This guide is the concrete setup path for Quester.AI's default packaged and source runtime.

## Pinned Default Bundle

- Generation: `ollama:qwen2.5:3b-instruct-q4_K_M`
- Generation fallback: `llama_cpp:qwen2.5-3b-instruct-q4_k_m.gguf`
- Embedding: `sentence_transformers:intfloat/e5-small-v2`
- Embedding fallback: `ollama_embeddings:nomic-embed-text`

## 1. Create The Environment

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[test]
```

Install the real-mode extras you need:

```bash
python -m pip install -e .[embeddings]
python -m pip install -e .[vector]
python -m pip install -e .[llama-cpp]
```

## 2. Install Ollama

Install Ollama separately, then confirm it is reachable:

```bash
ollama --version
ollama list
```

Pull the pinned generation model:

```bash
ollama pull qwen2.5:3b-instruct-q4_K_M
```

If you want the Ollama embedding fallback available too:

```bash
ollama pull nomic-embed-text
```

## 3. Install The GGUF Fallback

If you want `llama_cpp` fallback generation, download:

- `qwen2.5-3b-instruct-q4_k_m.gguf`

Place it under:

- `models/qwen2.5-3b-instruct-q4_k_m.gguf`

Or update the configured path in [`config.py`](./config.py).

## 4. Verify Python Dependencies

The default real-mode checks expect:

- `sentence-transformers` for `intfloat/e5-small-v2`
- `chromadb` for the primary persistent vector store
- `llama-cpp-python` only if you want local GGUF fallback

Quick verification:

```bash
python -m pip show sentence-transformers
python -m pip show chromadb
python -m pip show llama-cpp-python
```

## 5. Validate Readiness

Run the shared readiness regressions:

```bash
python -m pytest -q tests/test_phase12_preflight.py tests/test_phase12_packaged_smoke.py tests/test_phase24_packaged_startup.py
```

Or start the app and use the Readiness tab / exported preflight report:

```bash
python -m orchestrator
quester-ai-packaged
```

## 6. Privacy And Data Paths

- Default SQLite path: `quester.sqlite3`
- Default logs directory: `logs/`
- Default local model directory: `models/`
- Cloud helpers are auxiliary-only and should remain disabled unless you explicitly enable a capability and approve its content policy.

## 7. Recommended Order

1. Start in stub mode.
2. Confirm the UI, storage, and history paths work.
3. Install Ollama plus the pinned generation model.
4. Install `sentence-transformers`.
5. Refresh readiness or export a packaged preflight report.
6. Disable stub mode only after generation and embedding checks are green.
