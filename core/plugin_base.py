"""
Abstract base class that every AI provider plugin must implement.

A plugin exposes a uniform interface so the controller can swap
between OpenAI, Anthropic, local llama.cpp, Ollama, LM Studio,
or any other backend without touching UI code.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


class PluginCapability(enum.Flag):
    """Capabilities a plugin may advertise."""
    TEXT       = enum.auto()
    VISION     = enum.auto()
    TTS        = enum.auto()
    STT        = enum.auto()
    EMBEDDING  = enum.auto()
    TOOL_USE   = enum.auto()


@dataclass
class ModelInfo:
    """Describes a single model offered by the plugin."""
    id: str
    name: str
    capabilities: PluginCapability = PluginCapability.TEXT
    context_window: int = 4096
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceMetrics:
    """Snapshot of inference performance counters."""
    tokens_per_sec: float = 0.0
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    context_used: int = 0
    context_max: int = 0


class AIPluginBase(ABC):
    """
    Contract every AI-provider plugin must fulfil.

    Lifecycle
    ---------
    1. ``__init__``  – lightweight, no network calls.
    2. ``connect``   – establish connection / validate key.
    3. ``disconnect`` – release resources.

    The controller calls these at the appropriate times.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name shown in the UI (e.g. 'OpenAI')."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Semver string (e.g. '1.0.0')."""

    @property
    @abstractmethod
    def capabilities(self) -> PluginCapability:
        """Union of all capabilities this plugin supports."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abstractmethod
    def connect(self, config: dict[str, Any]) -> None:
        """
        Open the connection / authenticate.

        Parameters
        ----------
        config : dict
            Provider-specific settings (api_key, base_url, …).
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Release resources, close connections."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True when the backend is reachable."""

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """Return available models."""

    @abstractmethod
    def select_model(self, model_id: str) -> None:
        """Set the active model for subsequent calls."""

    @abstractmethod
    def active_model(self) -> ModelInfo | None:
        """Return the currently selected model, or None."""

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @abstractmethod
    def send_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        """
        Send a prompt and return the response.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style message list (role/content dicts).
        stream : bool
            If True, yield partial tokens instead of returning a full string.
        """

    @abstractmethod
    def get_metrics(self) -> InferenceMetrics:
        """Return latest inference performance counters."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------
    def transcribe(self, audio_path: str) -> str:
        """STT – override if the plugin supports STT."""
        raise NotImplementedError

    def synthesize(self, text: str) -> bytes:
        """TTS – override if the plugin supports TTS."""
        raise NotImplementedError

    def describe_image(self, image_path: str, prompt: str = "") -> str:
        """Vision – override if the plugin supports vision."""
        raise NotImplementedError

    def get_config_schema(self) -> dict[str, Any]:
        """
        Return a JSON-schema-style dict describing the settings this
        plugin accepts.  The UI will render matching input widgets.

        Example
        -------
        {
            "api_key":  {"type": "string", "label": "API Key", "secret": True},
            "base_url": {"type": "string", "label": "Base URL"},
            "temperature": {"type": "float", "label": "Temperature",
                            "min": 0.0, "max": 2.0, "default": 0.7},
        }
        """
        return {}
