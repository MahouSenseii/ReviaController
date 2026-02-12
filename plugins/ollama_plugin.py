"""
Ollama plugin — connects to a local Ollama instance.

Default endpoint: ``http://localhost:11434``

Ollama exposes an OpenAI-compatible API at ``/v1/`` as well as its
native ``/api/`` endpoints.  This plugin uses the OpenAI-compatible
layer for inference and the native ``/api/tags`` for model listing
(which returns richer metadata).
"""

from __future__ import annotations

from typing import Any

from core.plugin_base import ModelInfo, PluginCapability

from .openai_compat import OpenAICompatPlugin


class Plugin(OpenAICompatPlugin):
    _default_base_url = "http://localhost:11434"
    _plugin_name = "Ollama (Local)"
    _plugin_version = "1.0.0"

    def _fetch_models(self) -> list[ModelInfo]:
        """Use Ollama's native /api/tags for richer model info."""
        data = self._get_json(f"{self._base_url}/api/tags")
        models: list[ModelInfo] = []
        for entry in data.get("models", []):
            name = entry.get("name", "")
            if not name:
                continue
            details = entry.get("details", {})
            families = details.get("families", [])

            caps = PluginCapability.TEXT
            # Ollama models with "clip" family support vision
            if "clip" in families:
                caps |= PluginCapability.VISION

            ctx = details.get("context_length", 4096)
            if isinstance(ctx, str):
                try:
                    ctx = int(ctx)
                except ValueError:
                    ctx = 4096

            models.append(ModelInfo(
                id=name,
                name=name,
                capabilities=caps,
                context_window=ctx,
                metadata={
                    "parameter_size": details.get("parameter_size", ""),
                    "quantization": details.get("quantization_level", ""),
                    "format": details.get("format", ""),
                    "family": details.get("family", ""),
                    "size": entry.get("size", 0),
                },
            ))
        return models

    def get_config_schema(self) -> dict[str, Any]:
        schema = super().get_config_schema()
        schema["base_url"]["default"] = "http://localhost:11434"
        return schema
