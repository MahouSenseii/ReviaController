"""
Voice & Vision tab — STT / TTS / Vision engine selection, thresholds.

Reads current state from Config on init, writes changes back, and
publishes events so the sidebar pills and backend stay in sync.
"""

from __future__ import annotations

from .base_tab import BaseTab


class VoiceVisionTab(BaseTab):

    def _build(self) -> None:
        lay = self._layout

        # ── STT ───────────────────────────────────────────────
        lay.addWidget(self._heading("Speech-to-Text (STT)"))

        self._stt_engine = self._make_combo(
            ["Whisper (Local)", "Whisper (API)", "Google STT", "Azure STT", "Deepgram"],
        )
        self._stt_engine.currentTextChanged.connect(
            lambda v: self._save("voice.stt_engine", v)
        )
        lay.addWidget(self._row("Engine", self._stt_engine))

        self._stt_lang = self._make_combo(
            ["Auto", "English", "Spanish", "French", "German", "Japanese", "Chinese"],
        )
        self._stt_lang.currentTextChanged.connect(
            lambda v: self._save("voice.stt_language", v)
        )
        lay.addWidget(self._row("Language", self._stt_lang))

        self._vad_slider = self._make_slider(0, 100, 50)
        self._vad_slider.valueChanged.connect(
            lambda v: self._save("voice.vad_threshold", v)
        )
        lay.addWidget(self._row("VAD Thresh.", self._vad_slider))

        # ── TTS ───────────────────────────────────────────────
        lay.addWidget(self._heading("Text-to-Speech (TTS)"))

        self._tts_engine = self._make_combo(
            ["Piper (Local)", "ElevenLabs", "Azure TTS", "Google TTS", "OpenAI TTS"],
        )
        self._tts_engine.currentTextChanged.connect(
            lambda v: self._save("voice.tts_engine", v)
        )
        lay.addWidget(self._row("Engine", self._tts_engine))

        self._tts_voice = self._make_line_edit(placeholder="Voice ID or name")
        self._tts_voice.textChanged.connect(
            lambda v: self._save("voice.tts_voice", v)
        )
        lay.addWidget(self._row("Voice", self._tts_voice))

        self._tts_speed = self._make_double_spin(0.5, 2.0, 1.0, 0.1)
        self._tts_speed.valueChanged.connect(
            lambda v: self._save("voice.tts_speed", v)
        )
        lay.addWidget(self._row("Speed", self._tts_speed))

        # ── Vision ────────────────────────────────────────────
        lay.addWidget(self._heading("Vision"))

        self._vision_source = self._make_combo(
            ["Screen Capture", "Webcam", "Window Follow", "Clipboard"],
        )
        self._vision_source.currentTextChanged.connect(
            lambda v: self._save("vision.source", v)
        )
        lay.addWidget(self._row("Source", self._vision_source))

        self._vision_interval = self._make_spin(100, 10000, 2000)
        self._vision_interval.setSuffix(" ms")
        self._vision_interval.valueChanged.connect(
            lambda v: self._save("vision.interval_ms", v)
        )
        lay.addWidget(self._row("Interval", self._vision_interval))

        self._load_saved()

    def _load_saved(self) -> None:
        """Restore widget states from persisted config."""
        _restore_combo(self._stt_engine, self.config.get("voice.stt_engine"))
        _restore_combo(self._stt_lang, self.config.get("voice.stt_language"))
        _restore_combo(self._tts_engine, self.config.get("voice.tts_engine"))
        _restore_combo(self._vision_source, self.config.get("vision.source"))

        val = self.config.get("voice.vad_threshold")
        if val is not None:
            self._vad_slider.setValue(int(val))

        voice = self.config.get("voice.tts_voice", "")
        if voice:
            self._tts_voice.setText(str(voice))

        speed = self.config.get("voice.tts_speed")
        if speed is not None:
            self._tts_speed.setValue(float(speed))

        interval = self.config.get("vision.interval_ms")
        if interval is not None:
            self._vision_interval.setValue(int(interval))

    def _save(self, key: str, value) -> None:
        self.config.set(key, value)
        self.bus.publish("voice_vision_changed", {"key": key, "value": value})


def _restore_combo(combo, value) -> None:
    if value is not None:
        idx = combo.findText(str(value))
        if idx >= 0:
            combo.setCurrentIndex(idx)
