"""
llama.cpp server plugin — connects to a local llama-server instance.

Default endpoint: ``http://localhost:8080``

The llama.cpp server (``llama-server``) exposes a full OpenAI-compatible
API.  This plugin is a thin wrapper that sets the right default URL
and adds a health-check via the ``/health`` endpoint.
"""

from __future__ import annotations

from typing import Any

from .openai_compat import OpenAICompatPlugin


class Plugin(OpenAICompatPlugin):
    _default_base_url = "http://localhost:8080"
    _plugin_name = "llama.cpp (Local)"
    _plugin_version = "1.0.0"

    def connect(self, config: dict[str, Any]) -> None:
        super().connect(config)
        # llama.cpp exposes /health — use it to verify
        health = self._get_json(f"{self._base_url}/health")
        status = health.get("status", "")
        if status and status != "ok":
            self._connected = False
            raise ConnectionError(
                f"llama.cpp server reports status: {status}"
            )

    def get_config_schema(self) -> dict[str, Any]:
        schema = super().get_config_schema()
        schema["base_url"]["default"] = "http://localhost:8080"
        # llama.cpp doesn't need an API key
        schema.pop("api_key", None)
        return schema
