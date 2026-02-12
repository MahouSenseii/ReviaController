"""
Generic OpenAI API plugin — for connecting to the real OpenAI API or
any OpenAI-compatible cloud endpoint (Together AI, Groq, OpenRouter,
DeepSeek, Mistral, etc.).

This is the "online" option.  Users must provide an API key and can
optionally set a custom base URL.
"""

from __future__ import annotations

from typing import Any

from .openai_compat import OpenAICompatPlugin


# Provider → default base URL
_PROVIDER_URLS: dict[str, str] = {
    "OpenAI":       "https://api.openai.com",
    "Anthropic":    "https://api.anthropic.com",
    "Google Gemini": "https://generativelanguage.googleapis.com",
    "Mistral":      "https://api.mistral.ai",
    "Cohere":       "https://api.cohere.com/compatibility",
    "Groq":         "https://api.groq.com/openai",
    "Together AI":  "https://api.together.xyz",
    "OpenRouter":   "https://openrouter.ai/api",
    "Perplexity":   "https://api.perplexity.ai",
    "DeepSeek":     "https://api.deepseek.com",
}


class Plugin(OpenAICompatPlugin):
    _default_base_url = "https://api.openai.com"
    _plugin_name = "OpenAI API (Online)"
    _plugin_version = "1.0.0"

    def connect(self, config: dict[str, Any]) -> None:
        # Resolve provider shortcut to URL if no explicit base_url
        provider = config.get("provider", "")
        if provider and not config.get("base_url"):
            config["base_url"] = _PROVIDER_URLS.get(provider, self._default_base_url)
        super().connect(config)

    def get_config_schema(self) -> dict[str, Any]:
        schema = super().get_config_schema()
        schema["base_url"]["default"] = "https://api.openai.com"
        schema["api_key"]["default"] = ""
        return schema
