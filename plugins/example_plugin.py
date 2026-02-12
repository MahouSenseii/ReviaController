"""
Example / stub AI plugin.

Copy this file and replace the placeholder logic with real API calls
to create a plugin for any AI provider (OpenAI, Anthropic, Ollama, etc.).
"""

from __future__ import annotations

from typing import Any, Iterator

from core.plugin_base import (
    AIPluginBase,
    InferenceMetrics,
    ModelInfo,
    PluginCapability,
)


class Plugin(AIPluginBase):
    """Offline stub plugin used for UI development and testing."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "Example (Offline)"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def capabilities(self) -> PluginCapability:
        return PluginCapability.TEXT | PluginCapability.VISION

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._connected = False
        self._model: ModelInfo | None = None
        self._models = [
            ModelInfo(
                id="stub-7b",
                name="Stub 7B",
                capabilities=PluginCapability.TEXT,
                context_window=8192,
            ),
            ModelInfo(
                id="stub-vision",
                name="Stub Vision",
                capabilities=PluginCapability.TEXT | PluginCapability.VISION,
                context_window=4096,
            ),
        ]

    def connect(self, config: dict[str, Any]) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False
        self._model = None

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def list_models(self) -> list[ModelInfo]:
        return list(self._models)

    def select_model(self, model_id: str) -> None:
        for m in self._models:
            if m.id == model_id:
                self._model = m
                return
        raise ValueError(f"Unknown model: {model_id!r}")

    def active_model(self) -> ModelInfo | None:
        return self._model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def send_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        reply = f"[Stub] Echo: {messages[-1].get('content', '')}"
        if stream:
            def _gen() -> Iterator[str]:
                for word in reply.split():
                    yield word + " "
            return _gen()
        return reply

    def get_metrics(self) -> InferenceMetrics:
        return InferenceMetrics(
            tokens_per_sec=42.0,
            latency_ms=120.0,
            ttft_ms=85.0,
            context_used=1024,
            context_max=self._model.context_window if self._model else 8192,
        )

    # ------------------------------------------------------------------
    # Config schema (UI auto-generates matching inputs)
    # ------------------------------------------------------------------
    def get_config_schema(self) -> dict[str, Any]:
        return {
            "api_key": {
                "type": "string",
                "label": "API Key",
                "secret": True,
                "default": "",
            },
            "base_url": {
                "type": "string",
                "label": "Base URL",
                "default": "http://localhost:8080",
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
                "max": 32768,
                "default": 2048,
            },
        }
