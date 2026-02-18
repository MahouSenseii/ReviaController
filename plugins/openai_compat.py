"""
Base class for any AI backend that exposes an OpenAI-compatible HTTP API.

Ollama, llama.cpp server, LM Studio, and the real OpenAI API all share
the same ``/v1/chat/completions`` and ``/v1/models`` endpoints, so the
heavy lifting lives here.  Concrete plugins override a handful of
properties (default port, name, model listing) and inherit everything
else.

Uses only ``urllib`` — no ``requests`` / ``httpx`` dependency.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Iterator, Optional

from core.plugin_base import (
    AIPluginBase,
    InferenceMetrics,
    ModelInfo,
    PluginCapability,
)


class OpenAICompatPlugin(AIPluginBase):
    """
    Reusable base for any server that speaks the OpenAI chat-completions
    protocol over HTTP.
    """

    # Subclasses override these
    _default_base_url: str = "http://localhost:8080"
    _plugin_name: str = "OpenAI-Compatible"
    _plugin_version: str = "1.0.0"

    def __init__(self) -> None:
        self._base_url: str = self._default_base_url
        self._api_key: str = ""
        self._connected: bool = False
        self._model: ModelInfo | None = None
        self._models: list[ModelInfo] = []
        self._last_metrics = InferenceMetrics()
        self._timeout: int = 120

    # ── Identity ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._plugin_name

    @property
    def version(self) -> str:
        return self._plugin_version

    @property
    def capabilities(self) -> PluginCapability:
        caps = PluginCapability.TEXT
        if self._model and PluginCapability.VISION in self._model.capabilities:
            caps |= PluginCapability.VISION
        return caps

    # ── Lifecycle ─────────────────────────────────────────────

    def connect(self, config: dict[str, Any]) -> None:
        raw_url = config.get("base_url", "").strip().rstrip("/")
        # Strip trailing /v1 — the code appends /v1/... to the base URL
        if raw_url.endswith("/v1"):
            raw_url = raw_url[:-3]
        self._base_url = raw_url or self._default_base_url
        self._api_key = config.get("api_key", "")
        self._timeout = int(config.get("timeout", 120))

        # Probe the server — verify reachability first
        self._verify_server()

        # Fetch available models
        try:
            self._models = self._fetch_models()
            self._connected = True
            if self._models and self._model is None:
                self._model = self._models[0]
        except Exception:
            self._connected = False
            raise

        if not self._models:
            self._connected = False
            raise ConnectionError(
                f"Connected to {self._base_url} but no models were found. "
                f"Make sure the server has at least one model loaded."
            )

    def disconnect(self) -> None:
        self._connected = False
        self._model = None
        self._models = []

    def is_connected(self) -> bool:
        return self._connected

    # ── Models ────────────────────────────────────────────────

    def list_models(self) -> list[ModelInfo]:
        if not self._models:
            try:
                self._models = self._fetch_models()
            except Exception:
                pass
        return list(self._models)

    def select_model(self, model_id: str) -> None:
        for m in self._models:
            if m.id == model_id:
                self._model = m
                return
        # Allow selecting by name even if not in cached list
        self._model = ModelInfo(id=model_id, name=model_id)

    def active_model(self) -> ModelInfo | None:
        return self._model

    # ── Inference ─────────────────────────────────────────────

    def send_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        if self._model is None:
            raise RuntimeError(
                "No model selected. Open the LLM tab and select a model, "
                "then reconnect."
            )

        url = f"{self._base_url}/v1/chat/completions"
        body: dict[str, Any] = {
            "model": self._model.id,
            "messages": messages,
            "stream": stream,
        }
        # Forward common optional params
        for key in ("temperature", "max_tokens", "top_p", "stop"):
            if key in kwargs:
                body[key] = kwargs[key]

        t0 = time.perf_counter()

        if stream:
            return self._stream_request(url, body, t0)

        data = self._post_json(url, body)
        elapsed = (time.perf_counter() - t0) * 1000

        content = ""
        usage = data.get("usage", {})
        if data.get("choices"):
            content = data["choices"][0].get("message", {}).get("content", "")

        completion_tokens = usage.get("completion_tokens", len(content.split()))
        tps = (completion_tokens / (elapsed / 1000)) if elapsed > 0 else 0

        self._last_metrics = InferenceMetrics(
            tokens_per_sec=round(tps, 1),
            latency_ms=round(elapsed, 1),
            ttft_ms=round(elapsed, 1),
            context_used=usage.get("prompt_tokens", 0),
            context_max=self._model.context_window if self._model else 0,
        )
        return content

    def get_metrics(self) -> InferenceMetrics:
        return self._last_metrics

    # ── Streaming helper ──────────────────────────────────────

    def _stream_request(
        self, url: str, body: dict, t0: float,
    ) -> Iterator[str]:
        req = self._make_request(url, body)
        resp = urllib.request.urlopen(req, timeout=self._timeout)
        first_token = True
        total_tokens = 0
        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        if first_token:
                            ttft = (time.perf_counter() - t0) * 1000
                            self._last_metrics = InferenceMetrics(
                                ttft_ms=round(ttft, 1),
                            )
                            first_token = False
                        total_tokens += 1
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            tps = (total_tokens / (elapsed / 1000)) if elapsed > 0 else 0
            self._last_metrics = InferenceMetrics(
                tokens_per_sec=round(tps, 1),
                latency_ms=round(elapsed, 1),
                ttft_ms=self._last_metrics.ttft_ms,
                context_used=0,
                context_max=self._model.context_window if self._model else 0,
            )
            resp.close()

    # ── HTTP helpers ──────────────────────────────────────────

    def _verify_server(self) -> None:
        """Quick check that the server is reachable."""
        try:
            req = urllib.request.Request(self._base_url, method="GET")
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError:
            pass  # Server is reachable (returned an HTTP response)
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionError(
                f"Cannot reach server at {self._base_url}: {e}"
            ) from e

    def _make_request(
        self, url: str, body: dict | None = None,
    ) -> urllib.request.Request:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        data = json.dumps(body).encode("utf-8") if body else None
        return urllib.request.Request(url, data=data, headers=headers)

    def _post_json(self, url: str, body: dict) -> dict:
        req = self._make_request(url, body)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP {e.code} from {url}: {error_body}"
            ) from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach {url}: {e.reason}"
            ) from e

    def _get_json(self, url: str) -> dict:
        req = self._make_request(url)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return {}

    def _fetch_models(self) -> list[ModelInfo]:
        """Fetch models from the /v1/models endpoint."""
        url = f"{self._base_url}/v1/models"
        try:
            req = self._make_request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise ConnectionError(
                f"Server returned HTTP {e.code} from {url}. "
                f"Check that the address does not already include '/v1'."
            ) from e
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionError(
                f"Cannot reach {url}: {e}"
            ) from e

        models: list[ModelInfo] = []
        for entry in data.get("data", []):
            mid = entry.get("id", "")
            if not mid:
                continue
            models.append(ModelInfo(
                id=mid,
                name=mid,
                capabilities=PluginCapability.TEXT,
            ))
        return models

    # ── Optional config schema ────────────────────────────────

    def get_config_schema(self) -> dict[str, Any]:
        return {
            "base_url": {
                "type": "string",
                "label": "Server URL",
                "default": self._default_base_url,
            },
            "api_key": {
                "type": "string",
                "label": "API Key",
                "secret": True,
                "default": "",
            },
            "temperature": {
                "type": "float",
                "label": "Temperature",
                "min": 0.0,
                "max": 2.0,
                "default": 0.7,
            },
            "max_tokens": {
                "type": "int",
                "label": "Max Tokens",
                "min": 1,
                "max": 131072,
                "default": 2048,
            },
        }
