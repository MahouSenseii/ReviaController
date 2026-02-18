# Revia Controller — LLM / Server Integration Review

## Quick verdict

REVIA is structured **modularly in the right places**: provider logic is abstracted in `AIPluginBase`, OpenAI-compatible transport code is centralized in `OpenAICompatPlugin`, and backend implementations are split by server type (`llama.cpp`, `Ollama`, `LM Studio`, cloud OpenAI-compatible APIs).

That said, there are a few practical integration issues worth addressing.

## Confirmed strengths

- **Plugin architecture is clean**: `PluginManager` discovers modules dynamically and only relies on a shared contract (`Plugin` class deriving from `AIPluginBase`).
- **Transport deduplication is good**: common HTTP logic and chat-completions behavior live in one base plugin.
- **Server-specific logic is isolated**:
  - Ollama overrides model discovery via `/api/tags`.
  - llama.cpp adds a `/health` check.
  - LM Studio just overrides defaults.
- **UI wiring exists**: backend selection, connect/disconnect, model refresh and model switching are all present.

## Issues found (LLM + server connectivity)

### 1) Endpoint footgun (`/v1` duplication)

**Issue:** users often enter `http://host:port/v1`. The plugin then appends `/v1/...` and creates invalid paths (`.../v1/v1/models`).

**Fix implemented:** normalize configured base URL by stripping a trailing `/v1` in `OpenAICompatPlugin.connect()`. Also updated the local address placeholder to show host root (`http://localhost:8080`) instead of `/v1`.

### 2) Provider list implies compatibility that may not exist

Online provider options include names like **Anthropic** and **Google Gemini**, but the implementation is strictly OpenAI-compatible endpoints. This can mislead users into expecting native support.

**Recommendation:**
- Label this clearly as “OpenAI-compatible providers”, or
- Add provider-specific adapters for non-compatible APIs.

### 3) No startup or runtime connection backoff/retry

Connection is attempted once and failure is surfaced immediately. This is okay for MVP but brittle when a local server is still booting.

**Recommendation:** add optional retry/backoff (e.g., 2–3 retries with short delays) for connect and model fetch.

### 4) Minimal error classification in UI

All failures collapse into generic connection/model-switch errors. For users, “bad API key” vs “CORS/proxy issue” vs “model not loaded” should be differentiated.

**Recommendation:** map common HTTP status codes and known payload messages to user-friendly diagnostics.

### 5) Security handling of API keys

API keys are persisted in config/registry workflow, and no at-rest encryption/keychain integration is visible.

**Recommendation:** store secrets in OS keychain where available, keep only references in app config.

### 6) Streaming metrics are approximate

Streaming token count currently increments by chunks, not true tokenization. That is acceptable for rough telemetry but inaccurate for model comparisons.

**Recommendation:** label as estimate or integrate tokenizer-based counts per model family.

## Additional non-LLM gaps that matter operationally

- The app has configuration and control surfaces, but core “conversation loop” orchestration is still thin compared to the UI surface area.
- Some subsystems (filters/system monitor/emotion tick paths) are event-driven but rely on more producers/consumers to become fully operational.

## Suggested priority order

1. Harden LLM connection UX (compatibility labels + retry + precise errors).
2. Add secure credential storage.
3. Add integration tests against mock OpenAI-compatible endpoints for connect/list/chat/stream flows.
4. Expand provider adapters only where true protocol differences require it.
