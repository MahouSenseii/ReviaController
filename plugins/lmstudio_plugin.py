"""
LM Studio plugin — connects to a local LM Studio server.

Default endpoint: ``http://localhost:1234``

LM Studio exposes a fully OpenAI-compatible API.  This plugin is
a thin wrapper that sets the right default port.
"""

from __future__ import annotations

from typing import Any

from .openai_compat import OpenAICompatPlugin


class Plugin(OpenAICompatPlugin):
    _default_base_url = "http://localhost:1234"
    _plugin_name = "LM Studio (Local)"
    _plugin_version = "1.0.0"

    def get_config_schema(self) -> dict[str, Any]:
        schema = super().get_config_schema()
        schema["base_url"]["default"] = "http://localhost:1234"
        # LM Studio doesn't require an API key for local use
        schema.pop("api_key", None)
        return schema
