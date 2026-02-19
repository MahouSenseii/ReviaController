"""
Speech-to-Text manager — captures audio from the microphone and
transcribes it using the configured STT engine.

Supports:
* **Whisper (API)** — sends audio to an OpenAI-compatible
  ``/v1/audio/transcriptions`` endpoint (works with OpenAI, local
  Whisper servers, and any compatible proxy).
* **Whisper (Local)** — uses the ``openai-whisper`` Python package
  directly (requires it to be installed).

Other engines (Google STT, Azure STT, Deepgram) are listed in the UI
but not yet implemented; they will show a warning pill.

Audio is captured with ``sounddevice`` using a simple amplitude-based
Voice Activity Detection (VAD).  When the user stops speaking, the
recorded audio is transcribed and the text is published as a
``user_message`` event so it flows through the normal chat pipeline.
"""

from __future__ import annotations

import io
import json
import struct
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QObject, QTimer

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


def _write_wav_bytes(samples, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Encode raw float32 samples to a WAV byte string."""
    # Convert float32 [-1, 1] to int16
    int_samples = (samples * 32767).astype("int16")
    raw = int_samples.tobytes()

    buf = io.BytesIO()
    # WAV header
    data_size = len(raw)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM
    buf.write(struct.pack("<H", channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * channels * 2))  # byte rate
    buf.write(struct.pack("<H", channels * 2))  # block align
    buf.write(struct.pack("<H", 16))  # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(raw)
    return buf.getvalue()


class STTManager(QObject):
    """Manages microphone capture and speech-to-text transcription."""

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
        self._recording: bool = False
        self._stream: Any = None  # sounddevice.InputStream
        self._audio_buffer: list = []
        self._silence_count: int = 0
        self._speech_detected: bool = False

        # Check timer — polls whether user stopped speaking
        self._check_timer = QTimer(self)
        self._check_timer.setInterval(100)
        self._check_timer.timeout.connect(self._check_audio_state)

        self._bus.subscribe("stt_toggled", self._on_stt_toggled)

    # ── Public API ────────────────────────────────────────────

    def start(self) -> None:
        """Begin listening on the configured microphone."""
        if not _SD_AVAILABLE:
            self._bus.publish("module_status", {
                "module": "stt",
                "status": "warn",
                "subtitle": "Install sounddevice + numpy",
            })
            self._bus.publish("activity_log", {
                "text": "[Error] STT requires 'sounddevice' and 'numpy'. "
                        "Install them with: pip install sounddevice numpy",
            })
            return

        if self._recording:
            return

        engine = self._config.get("voice.stt_engine", "Whisper (API)")
        if engine not in ("Whisper (API)", "Whisper (Local)"):
            self._bus.publish("module_status", {
                "module": "stt",
                "status": "warn",
                "subtitle": f"{engine} not yet implemented",
            })
            self._bus.publish("activity_log", {
                "text": f"[Error] STT engine '{engine}' is not yet implemented. "
                        f"Use 'Whisper (API)' or 'Whisper (Local)'.",
            })
            return

        # Parse device index from config
        device = None
        device_str = str(self._config.get("voice.stt_device", ""))
        if device_str and not device_str.startswith("("):
            try:
                device = int(device_str.split(":")[0])
            except (ValueError, IndexError):
                pass

        try:
            self._audio_buffer = []
            self._silence_count = 0
            self._speech_detected = False
            self._stream = _sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="float32",
                blocksize=1600,  # 100ms blocks at 16kHz
                device=device,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._recording = True
            self._check_timer.start()
            self._bus.publish("module_status", {
                "module": "stt",
                "status": "on",
                "subtitle": "Listening...",
            })
        except Exception as exc:
            self._bus.publish("module_status", {
                "module": "stt",
                "status": "warn",
                "subtitle": f"Mic error: {exc}",
            })
            self._bus.publish("activity_log", {
                "text": f"[Error] STT microphone error: {exc}",
            })

    def stop(self) -> None:
        """Stop listening."""
        self._check_timer.stop()
        self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._audio_buffer = []
        self._bus.publish("module_status", {
            "module": "stt",
            "status": "off",
            "subtitle": "Disabled",
        })

    # ── Audio callback (runs in audio thread) ─────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if not self._recording:
            return
        self._audio_buffer.append(indata.copy())

    # ── Audio state check (runs on Qt timer) ──────────────────

    def _check_audio_state(self) -> None:
        """Check if user has stopped speaking based on amplitude."""
        if not self._audio_buffer:
            return

        vad_threshold = int(self._config.get("voice.vad_threshold", 50)) / 1000.0

        # Check the most recent audio block
        recent = self._audio_buffer[-1] if self._audio_buffer else None
        if recent is None:
            return

        amplitude = float(_np.abs(recent).mean())

        if amplitude > vad_threshold:
            self._speech_detected = True
            self._silence_count = 0
        elif self._speech_detected:
            self._silence_count += 1

        # 10 consecutive silence blocks (~1 second) after speech → transcribe
        if self._speech_detected and self._silence_count >= 10:
            self._process_audio()
            self._audio_buffer = []
            self._silence_count = 0
            self._speech_detected = False

        # Prevent buffer from growing unbounded when no speech is detected
        if not self._speech_detected and len(self._audio_buffer) > 50:
            self._audio_buffer = self._audio_buffer[-10:]

    # ── Transcription ─────────────────────────────────────────

    def _process_audio(self) -> None:
        """Transcribe the buffered audio and publish the result."""
        if not self._audio_buffer:
            return

        audio_data = _np.concatenate(self._audio_buffer, axis=0)

        # Skip very short recordings (< 0.5 seconds)
        if len(audio_data) < 8000:
            return

        self._bus.publish("module_status", {
            "module": "stt",
            "status": "on",
            "subtitle": "Transcribing...",
        })

        # Run transcription in background thread to avoid blocking UI
        threading.Thread(
            target=self._transcribe_threaded,
            args=(audio_data,),
            daemon=True,
        ).start()

    def _transcribe_threaded(self, audio_data) -> None:
        """Run transcription off the main thread."""
        engine = self._config.get("voice.stt_engine", "Whisper (API)")
        text = ""

        try:
            if engine == "Whisper (API)":
                text = self._transcribe_whisper_api(audio_data)
            elif engine == "Whisper (Local)":
                text = self._transcribe_whisper_local(audio_data)
        except Exception as exc:
            self._bus.publish("activity_log", {
                "text": f"[Error] STT transcription failed: {exc}",
            })
            self._bus.publish("module_status", {
                "module": "stt",
                "status": "warn",
                "subtitle": f"Error: {exc}",
            })
            # Resume listening after error
            QTimer.singleShot(2000, lambda: self._bus.publish("module_status", {
                "module": "stt",
                "status": "on",
                "subtitle": "Listening...",
            }))
            return

        text = text.strip()
        if text:
            # Publish as user message so it enters the normal chat pipeline
            self._bus.publish("user_message", {"text": text, "source": "stt"})
            # Also show it in the activity log
            self._bus.publish("activity_log", {
                "text": f'[STT] Heard: "{text}"',
            })

        # Back to listening
        self._bus.publish("module_status", {
            "module": "stt",
            "status": "on",
            "subtitle": "Listening...",
        })

    def _transcribe_whisper_api(self, audio_data) -> str:
        """Send audio to an OpenAI-compatible /v1/audio/transcriptions endpoint."""
        wav_bytes = _write_wav_bytes(audio_data, sample_rate=16000)

        # Get connection info from the active LLM plugin or config
        base_url, api_key = self._get_api_credentials()
        if not base_url:
            raise RuntimeError(
                "No API endpoint configured. Connect an LLM backend first."
            )

        url = f"{base_url}/v1/audio/transcriptions"
        model = self._config.get("models.stt_model", "whisper-1")

        # Build multipart form data
        boundary = "----ReviaSTTBoundary"
        body = bytearray()

        # File field
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        )
        body.extend(b"Content-Type: audio/wav\r\n\r\n")
        body.extend(wav_bytes)
        body.extend(b"\r\n")

        # Model field
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(b'Content-Disposition: form-data; name="model"\r\n\r\n')
        body.extend(model.encode())
        body.extend(b"\r\n")

        # Language field
        lang = self._config.get("voice.stt_language", "Auto")
        if lang and lang != "Auto":
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(b'Content-Disposition: form-data; name="language"\r\n\r\n')
            body.extend(lang[:2].lower().encode())
            body.extend(b"\r\n")

        body.extend(f"--{boundary}--\r\n".encode())

        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(
            url, data=bytes(body), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("text", "")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"STT API HTTP {e.code}: {error_body}") from e

    def _transcribe_whisper_local(self, audio_data) -> str:
        """Transcribe using the local openai-whisper package."""
        try:
            import whisper  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "Whisper (Local) requires the 'openai-whisper' package. "
                "Install it with: pip install openai-whisper"
            )

        model_name = self._config.get("models.stt_model", "base")
        # Load model (cached after first call by whisper)
        model = whisper.load_model(model_name)

        # Write audio to a temp file
        wav_bytes = _write_wav_bytes(audio_data, sample_rate=16000)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            lang = self._config.get("voice.stt_language", "Auto")
            kwargs: dict[str, Any] = {}
            if lang and lang != "Auto":
                kwargs["language"] = lang[:2].lower()
            result = model.transcribe(tmp_path, **kwargs)
            return result.get("text", "")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── Helpers ───────────────────────────────────────────────

    def _get_api_credentials(self) -> tuple[str, str]:
        """Get the base URL and API key from the active LLM connection."""
        # Try the active plugin first
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

    # ── Event handlers ────────────────────────────────────────

    def _on_stt_toggled(self, data: dict) -> None:
        enabled = data.get("enabled", False)
        self._enabled = enabled
        if enabled:
            self.start()
        else:
            self.stop()
