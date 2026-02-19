"""
Tests for plugins/openai_compat.py — shared OpenAI-compatible HTTP base.

All network calls are replaced with unittest.mock patches so no real
server is needed.

Covers:
* _normalize_base_url()
* connect() — success, no models, URL stripping
* disconnect()
* select_model() — known and unknown IDs
* send_prompt() non-streaming — content extraction, metrics
* send_prompt() streaming — token iteration, invalid JSON skipped
* _verify_server() — HTTPError swallowed, URLError raises
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import urllib.error

from core.plugin_base import ModelInfo, PluginCapability
from plugins.openai_compat import OpenAICompatPlugin


# ── Test helpers ──────────────────────────────────────────────────────

def make_plugin() -> OpenAICompatPlugin:
    p = OpenAICompatPlugin()
    p._base_url = "http://localhost:8080"
    return p


def models_json(model_ids: list[str]) -> bytes:
    return json.dumps({"data": [{"id": mid} for mid in model_ids]}).encode()


def chat_json(content: str, prompt_tokens: int = 5, completion_tokens: int = 3) -> bytes:
    return json.dumps({
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }).encode()


def make_ctx_response(body: bytes) -> MagicMock:
    """Context-manager-compatible mock response."""
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def sse_lines(tokens: list[str]) -> list[bytes]:
    lines: list[bytes] = []
    for token in tokens:
        chunk = {"choices": [{"delta": {"content": token}}]}
        lines.append(f"data: {json.dumps(chunk)}\n".encode())
    lines.append(b"data: [DONE]\n")
    return lines


# ── _normalize_base_url ───────────────────────────────────────────────

class TestNormalizeBaseUrl:
    def test_strips_v1_suffix(self):
        assert OpenAICompatPlugin._normalize_base_url(
            "http://localhost:8080/v1"
        ) == "http://localhost:8080"

    def test_strips_trailing_slash(self):
        assert OpenAICompatPlugin._normalize_base_url(
            "http://localhost:8080/"
        ) == "http://localhost:8080"

    def test_plain_url_unchanged(self):
        assert OpenAICompatPlugin._normalize_base_url(
            "http://localhost:8080"
        ) == "http://localhost:8080"

    def test_empty_string_returns_empty(self):
        assert OpenAICompatPlugin._normalize_base_url("") == ""

    def test_v1_only_stripped_at_end(self):
        # /v1 in the middle of the path should NOT be stripped
        url = "http://localhost:8080/api/v1"
        result = OpenAICompatPlugin._normalize_base_url(url)
        # Trailing /v1 stripped
        assert result == "http://localhost:8080/api"


# ── connect() ────────────────────────────────────────────────────────

class TestConnect:
    def test_connect_sets_connected_true(self):
        plugin = make_plugin()
        with patch("urllib.request.urlopen",
                   return_value=make_ctx_response(models_json(["gpt-4"]))):
            plugin.connect({"base_url": "http://localhost:8080"})
        assert plugin.is_connected()

    def test_connect_populates_model_list(self):
        plugin = make_plugin()
        with patch("urllib.request.urlopen",
                   return_value=make_ctx_response(models_json(["m1", "m2"]))):
            plugin.connect({"base_url": "http://localhost:8080"})
        models = plugin.list_models()
        assert len(models) == 2
        assert {m.id for m in models} == {"m1", "m2"}

    def test_connect_selects_first_model(self):
        plugin = make_plugin()
        with patch("urllib.request.urlopen",
                   return_value=make_ctx_response(models_json(["first", "second"]))):
            plugin.connect({"base_url": "http://localhost:8080"})
        assert plugin.active_model() is not None
        assert plugin.active_model().id == "first"

    def test_connect_no_models_raises_and_not_connected(self):
        plugin = make_plugin()
        with patch("urllib.request.urlopen",
                   return_value=make_ctx_response(models_json([]))):
            with pytest.raises(ConnectionError):
                plugin.connect({"base_url": "http://localhost:8080"})
        assert not plugin.is_connected()

    def test_connect_strips_v1_from_url(self):
        plugin = make_plugin()
        with patch("urllib.request.urlopen",
                   return_value=make_ctx_response(models_json(["m"]))):
            plugin.connect({"base_url": "http://localhost:8080/v1"})
        assert "/v1/v1" not in plugin._base_url

    def test_connect_uses_api_key(self):
        """api_key must be stored after connect."""
        plugin = make_plugin()
        with patch("urllib.request.urlopen",
                   return_value=make_ctx_response(models_json(["m"]))):
            plugin.connect({"base_url": "http://localhost:8080", "api_key": "sk-test"})
        assert plugin._api_key == "sk-test"


# ── disconnect() ─────────────────────────────────────────────────────

class TestDisconnect:
    def test_disconnect_clears_connected_flag(self):
        plugin = make_plugin()
        with patch("urllib.request.urlopen",
                   return_value=make_ctx_response(models_json(["m"]))):
            plugin.connect({"base_url": "http://localhost:8080"})
        plugin.disconnect()
        assert not plugin.is_connected()

    def test_disconnect_clears_model_list(self):
        plugin = make_plugin()
        with patch("urllib.request.urlopen",
                   return_value=make_ctx_response(models_json(["m"]))):
            plugin.connect({"base_url": "http://localhost:8080"})
        plugin.disconnect()
        assert plugin.list_models() == []

    def test_disconnect_clears_active_model(self):
        plugin = make_plugin()
        with patch("urllib.request.urlopen",
                   return_value=make_ctx_response(models_json(["m"]))):
            plugin.connect({"base_url": "http://localhost:8080"})
        plugin.disconnect()
        assert plugin.active_model() is None


# ── select_model() ────────────────────────────────────────────────────

class TestSelectModel:
    def test_select_known_model(self):
        plugin = make_plugin()
        plugin._models = [
            ModelInfo(id="gpt-4", name="GPT-4"),
            ModelInfo(id="gpt-3.5", name="GPT-3.5"),
        ]
        plugin.select_model("gpt-3.5")
        assert plugin.active_model().id == "gpt-3.5"

    def test_select_unknown_model_creates_fallback(self):
        plugin = make_plugin()
        plugin._models = []
        plugin.select_model("some-custom-model")
        m = plugin.active_model()
        assert m is not None
        assert m.id == "some-custom-model"


# ── send_prompt() — non-streaming ────────────────────────────────────

class TestSendPromptNonStreaming:
    def test_returns_response_content(self):
        plugin = make_plugin()
        plugin._model = ModelInfo(id="m", name="m")
        with patch("urllib.request.urlopen",
                   return_value=make_ctx_response(chat_json("Hello there!"))):
            result = plugin.send_prompt([{"role": "user", "content": "hi"}])
        assert result == "Hello there!"

    def test_empty_response_returns_empty_string(self):
        plugin = make_plugin()
        plugin._model = ModelInfo(id="m", name="m")
        body = json.dumps({"choices": [], "usage": {}}).encode()
        with patch("urllib.request.urlopen",
                   return_value=make_ctx_response(body)):
            result = plugin.send_prompt([{"role": "user", "content": "hi"}])
        assert result == ""

    def test_metrics_latency_is_non_negative(self):
        plugin = make_plugin()
        plugin._model = ModelInfo(id="m", name="m")
        with patch("urllib.request.urlopen",
                   return_value=make_ctx_response(chat_json("Hi"))):
            plugin.send_prompt([{"role": "user", "content": "hi"}])
        assert plugin.get_metrics().latency_ms >= 0.0

    def test_no_model_raises_runtime_error(self):
        plugin = make_plugin()
        plugin._model = None
        with pytest.raises(RuntimeError):
            plugin.send_prompt([{"role": "user", "content": "hi"}])

    def test_kwargs_forwarded_in_body(self):
        """temperature and max_tokens should appear in the POST body."""
        plugin = make_plugin()
        plugin._model = ModelInfo(id="m", name="m")
        captured = []

        def fake_urlopen(req, timeout=None):
            import json as _json
            captured.append(_json.loads(req.data.decode()))
            return make_ctx_response(chat_json("ok"))

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            plugin.send_prompt(
                [{"role": "user", "content": "hi"}],
                temperature=0.9,
                max_tokens=512,
            )
        assert captured[0].get("temperature") == 0.9
        assert captured[0].get("max_tokens") == 512


# ── _verify_server() ─────────────────────────────────────────────────

class TestVerifyServer:
    def test_http_error_is_swallowed(self):
        plugin = make_plugin()
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError(
                       url="http://x", code=404,
                       msg="Not Found", hdrs={}, fp=None,
                   )):
            plugin._verify_server()  # must not raise

    def test_url_error_raises_connection_error(self):
        plugin = make_plugin()
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(ConnectionError):
                plugin._verify_server()

    def test_os_error_raises_connection_error(self):
        plugin = make_plugin()
        with patch("urllib.request.urlopen",
                   side_effect=OSError("socket error")):
            with pytest.raises(ConnectionError):
                plugin._verify_server()


# ── send_prompt() — streaming ─────────────────────────────────────────

class TestSendPromptStreaming:
    def _make_stream_mock(self, lines: list[bytes]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.__iter__ = lambda s: iter(lines)
        mock_resp.close = MagicMock()
        return mock_resp

    def test_yields_tokens_in_order(self):
        plugin = make_plugin()
        plugin._model = ModelInfo(id="m", name="m")
        mock_resp = self._make_stream_mock(sse_lines(["Hello", " ", "world"]))
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = list(plugin.send_prompt(
                [{"role": "user", "content": "hi"}], stream=True
            ))
        assert result == ["Hello", " ", "world"]

    def test_skips_lines_not_starting_with_data(self):
        plugin = make_plugin()
        plugin._model = ModelInfo(id="m", name="m")
        lines = [
            b": keep-alive\n",
            b"\n",
            b"data: " + json.dumps(
                {"choices": [{"delta": {"content": "ok"}}]}
            ).encode() + b"\n",
            b"data: [DONE]\n",
        ]
        mock_resp = self._make_stream_mock(lines)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = list(plugin.send_prompt(
                [{"role": "user", "content": "hi"}], stream=True
            ))
        assert result == ["ok"]

    def test_skips_invalid_json_chunks(self):
        plugin = make_plugin()
        plugin._model = ModelInfo(id="m", name="m")
        lines = [
            b"data: {invalid json}\n",
            b"data: " + json.dumps(
                {"choices": [{"delta": {"content": "good"}}]}
            ).encode() + b"\n",
            b"data: [DONE]\n",
        ]
        mock_resp = self._make_stream_mock(lines)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = list(plugin.send_prompt(
                [{"role": "user", "content": "hi"}], stream=True
            ))
        assert result == ["good"]

    def test_streaming_updates_metrics(self):
        plugin = make_plugin()
        plugin._model = ModelInfo(id="m", name="m")
        mock_resp = self._make_stream_mock(sse_lines(["token1", "token2"]))
        with patch("urllib.request.urlopen", return_value=mock_resp):
            list(plugin.send_prompt(
                [{"role": "user", "content": "hi"}], stream=True
            ))
        metrics = plugin.get_metrics()
        assert metrics.tokens_per_sec >= 0.0
        assert metrics.latency_ms >= 0.0

    def test_empty_stream_yields_nothing(self):
        plugin = make_plugin()
        plugin._model = ModelInfo(id="m", name="m")
        mock_resp = self._make_stream_mock([b"data: [DONE]\n"])
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = list(plugin.send_prompt(
                [{"role": "user", "content": "hi"}], stream=True
            ))
        assert result == []
