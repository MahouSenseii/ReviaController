"""
Text-to-Speech manager — converts AI responses to spoken audio and
plays them through the system audio output.

Supports:
* **OpenAI TTS** — sends text to an OpenAI-compatible
  ``/v1/audio/speech`` endpoint and plays the returned audio.
* **Piper (Local)** — uses the ``piper-tts`` Python package directly
  (requires it to be installed).

Other engines (ElevenLabs, Azure TTS, Google TTS) are listed in the
UI but not yet implemented; they will show a warning pill.

Audio is played through ``sounddevice`` in a background thread so it
does not block the UI.
"""

from __future__ import annotations

import io
import json
import struct
import threading
import urllib.error
import urllib.request
from typing import Any, Optional

from PyQt6.QtCore import QObject

from .config import Config
from .events import EventBus
from .plugin_manager import PluginManager

# Optional dependencies ─────────────────────────────────────────────
try:
    import sounddevice as _sd
    import numpy as _np

    _SD_AVAILABLE = True
except ImportError:
    _sd = None  # type: ignore[assignment]
    _np = None  # type: ignore[assignment]
    _SD_AVAILABLE = False


def _decode_wav_bytes(wav_data: bytes) -> tuple:
    """Decode a WAV byte string into (samples_array, sample_rate).

    Returns float32 numpy array and sample rate.
    """
    buf = io.BytesIO(wav_data)

    # Read RIFF header
    riff = buf.read(4)
    if riff != b"RIFF":
        raise ValueError("Not a WAV file")
    buf.read(4)  # file size
    wave = buf.read(4)
    if wave != b"WAVE":
        raise ValueError("Not a WAV file")

    sample_rate = 24000
    channels = 1
    bits_per_sample = 16
    audio_data = b""

    while True:
        chunk_id = buf.read(4)
        if len(chunk_id) < 4:
            break
        chunk_size = struct.unpack("<I", buf.read(4))[0]

        if chunk_id == b"fmt ":
            fmt_data = buf.read(chunk_size)
            audio_format = struct.unpack("<H", fmt_data[0:2])[0]
            channels = struct.unpack("<H", fmt_data[2:4])[0]
            sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
            bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
        elif chunk_id == b"data":
            audio_data = buf.read(chunk_size)
        else:
            buf.read(chunk_size)

    if not audio_data:
        raise ValueError("No audio data in WAV")

    if bits_per_sample == 16:
        samples = _np.frombuffer(audio_data, dtype=_np.int16).astype(_np.float32) / 32767.0
    elif bits_per_sample == 32:
        samples = _np.frombuffer(audio_data, dtype=_np.int32).astype(_np.float32) / 2147483647.0
    else:
        samples = _np.frombuffer(audio_data, dtype=_np.int16).astype(_np.float32) / 32767.0

    if channels > 1:
        samples = samples.reshape(-1, channels)[:, 0]  # take first channel

    return samples, sample_rate


class TTSManager(QObject):
    """Manages text-to-speech synthesis and audio playback."""

    def __init__(
        self,
        event_bus: EventBus,
        config: Config,
        plugin_manager: PluginManager,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._bus = event_bus
        self._config = config
        self._pm = plugin_manager

        self._enabled: bool = False
        self._playing: bool = False

        self._bus.subscribe("tts_toggled", self._on_tts_toggled)
        self._bus.subscribe("assistant_response", self._on_assistant_response)

    # ── Event handlers ────────────────────────────────────────

    def _on_tts_toggled(self, data: dict) -> None:
        enabled = data.get("enabled", False)
        self._enabled = enabled
        if not enabled:
            self._playing = False
            self._bus.publish("module_status", {
                "module": "tts",
                "status": "off",
                "subtitle": "Disabled",
            })
        else:
            if not _SD_AVAILABLE:
                self._bus.publish("module_status", {
                    "module": "tts",
                    "status": "warn",
                    "subtitle": "Install sounddevice + numpy",
                })
                self._bus.publish("activity_log", {
                    "text": "[Error] TTS requires 'sounddevice' and 'numpy'. "
                            "Install them with: pip install sounddevice numpy",
                })
                return

            engine = self._config.get("voice.tts_engine", "OpenAI TTS")
            if engine not in ("OpenAI TTS", "Piper (Local)"):
                self._bus.publish("module_status", {
                    "module": "tts",
                    "status": "warn",
                    "subtitle": f"{engine} not yet implemented",
                })
                self._bus.publish("activity_log", {
                    "text": f"[Error] TTS engine '{engine}' is not yet implemented. "
                            f"Use 'OpenAI TTS' or 'Piper (Local)'.",
                })
                return

            self._bus.publish("module_status", {
                "module": "tts",
                "status": "on",
                "subtitle": "Ready",
            })

    def _on_assistant_response(self, data: dict) -> None:
        """Speak the AI's response when TTS is enabled."""
        if not self._enabled or not _SD_AVAILABLE:
            return

        text = data.get("text", "").strip()
        if not text or data.get("error"):
            return

        # Don't speak if already speaking
        if self._playing:
            return

        # Run synthesis + playback in background thread
        threading.Thread(
            target=self._speak_threaded,
            args=(text,),
            daemon=True,
        ).start()

    # ── Synthesis + playback ──────────────────────────────────

    def _speak_threaded(self, text: str) -> None:
        """Synthesize and play speech off the main thread."""
        self._playing = True
        self._bus.publish("module_status", {
            "module": "tts",
            "status": "on",
            "subtitle": "Speaking...",
        })

        engine = self._config.get("voice.tts_engine", "OpenAI TTS")

        try:
            if engine == "OpenAI TTS":
                audio_data, sample_rate = self._synthesize_openai(text)
            elif engine == "Piper (Local)":
                audio_data, sample_rate = self._synthesize_piper(text)
            else:
                raise RuntimeError(f"TTS engine '{engine}' not implemented")

            # Play audio
            _sd.play(audio_data, samplerate=sample_rate)
            _sd.wait()

        except Exception as exc:
            self._bus.publish("activity_log", {
                "text": f"[Error] TTS playback failed: {exc}",
            })
            self._bus.publish("module_status", {
                "module": "tts",
                "status": "warn",
                "subtitle": f"Error: {exc}",
            })
        finally:
            self._playing = False
            if self._enabled:
                self._bus.publish("module_status", {
                    "module": "tts",
                    "status": "on",
                    "subtitle": "Ready",
                })

    def _synthesize_openai(self, text: str) -> tuple:
        """Call an OpenAI-compatible /v1/audio/speech endpoint."""
        base_url, api_key = self._get_api_credentials()
        if not base_url:
            raise RuntimeError(
                "No API endpoint configured. Connect an LLM backend first."
            )

        url = f"{base_url}/v1/audio/speech"
        model = self._config.get("models.tts_model", "tts-1")
        voice = self._config.get("voice.tts_voice", "alloy") or "alloy"
        speed = float(self._config.get("voice.tts_speed", 1.0) or 1.0)

        body = {
            "model": model,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": "wav",
        }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                content_type = resp.headers.get("Content-Type", "")
                audio_bytes = resp.read()

                # If response is WAV, decode it
                if audio_bytes[:4] == b"RIFF":
                    return _decode_wav_bytes(audio_bytes)

                # If response is raw PCM or MP3, try to handle
                # For MP3, we'd need additional decoding — fallback to playing raw
                # Assume 24kHz mono float32 PCM as fallback
                samples = _np.frombuffer(audio_bytes, dtype=_np.int16).astype(
                    _np.float32
                ) / 32767.0
                return samples, 24000

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"TTS API HTTP {e.code}: {error_body}") from e

    def _synthesize_piper(self, text: str) -> tuple:
        """Synthesize using the local piper-tts package."""
        try:
            from piper import PiperVoice  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "Piper (Local) requires the 'piper-tts' package. "
                "Install it with: pip install piper-tts"
            )

        voice_path = self._config.get("voice.tts_voice", "")
        if not voice_path:
            raise RuntimeError(
                "No Piper voice model configured. Set the Voice field "
                "in the Voice & Vision tab to a .onnx voice file path."
            )

        voice = PiperVoice.load(voice_path)
        audio_stream = voice.synthesize_stream_raw(text)

        chunks = []
        for chunk in audio_stream:
            chunks.append(chunk)

        raw_audio = b"".join(chunks)
        samples = _np.frombuffer(raw_audio, dtype=_np.int16).astype(
            _np.float32
        ) / 32767.0
        return samples, voice.config.sample_rate

    # ── Helpers ───────────────────────────────────────────────

    def _get_api_credentials(self) -> tuple[str, str]:
        """Get the base URL and API key from the active LLM connection."""
        plugin = self._pm.active_plugin
        if plugin is not None and hasattr(plugin, "_base_url"):
            base_url = getattr(plugin, "_base_url", "")
            api_key = getattr(plugin, "_api_key", "")
            if base_url:
                return base_url, api_key

        # Fallback: read from config
        mode = self._config.get("llm.mode", "online")
        if mode == "online":
            online_models = self._config.get("llm.online_models", [])
            selected = self._config.get("llm.online_selected", "")
            for entry in (online_models or []):
                if entry.get("name") == selected:
                    endpoint = entry.get("endpoint", "")
                    api_key = entry.get("api_key", "")
                    return endpoint, api_key
        else:
            local_models = self._config.get("llm.local_models", [])
            selected = self._config.get("llm.local_selected", "")
            for entry in (local_models or []):
                if entry.get("name") == selected:
                    return entry.get("address", ""), ""

        return "", ""
