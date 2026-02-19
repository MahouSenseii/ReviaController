"""
Voice & Vision tab — STT / TTS / Vision engine selection, thresholds,
voice cloning reference audio, camera device selection, and per-task
model role assignments.

Reads current state from Config on init, writes changes back, and
publishes events so the sidebar pills and backend stay in sync.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

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

        # ── Voice Cloning ─────────────────────────────────────
        lay.addWidget(self._heading("Voice Cloning"))

        self._clone_engine = self._make_combo(
            ["Coqui XTTS", "Fish Speech", "RVC", "So-VITS-SVC", "Tortoise TTS", "Custom"],
        )
        self._clone_engine.currentTextChanged.connect(
            lambda v: self._save("voice.clone_engine", v)
        )
        lay.addWidget(self._row("Engine", self._clone_engine))

        # Reference audio file browser
        ref_widget = QWidget()
        ref_h = QHBoxLayout(ref_widget)
        ref_h.setContentsMargins(0, 0, 0, 0)
        ref_h.setSpacing(8)

        self._clone_ref_edit = self._make_line_edit(
            placeholder="Path to reference audio file…"
        )
        self._clone_ref_edit.textChanged.connect(
            lambda v: self._save("voice.clone_reference", v)
        )
        ref_h.addWidget(self._clone_ref_edit, 1)

        self._clone_ref_btn = QPushButton("Browse…")
        self._clone_ref_btn.setObjectName("ModeButton")
        self._clone_ref_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clone_ref_btn.clicked.connect(self._on_browse_clone_ref)
        ref_h.addWidget(self._clone_ref_btn)

        lay.addWidget(self._row("Reference Audio", ref_widget))

        self._clone_name = self._make_line_edit(
            placeholder="Name for this cloned voice…"
        )
        self._clone_name.textChanged.connect(
            lambda v: self._save("voice.clone_name", v)
        )
        lay.addWidget(self._row("Clone Name", self._clone_name))

        clone_hint = QLabel(
            "Provide 10–30 s of clean, noise-free reference audio for best cloning results."
        )
        clone_hint.setWordWrap(True)
        clone_hint.setStyleSheet("color:#8fa6c3; font-size:11px; padding:2px 0;")
        lay.addWidget(clone_hint)

        # ── Vision ────────────────────────────────────────────
        lay.addWidget(self._heading("Vision"))

        self._vision_source = self._make_combo(
            ["Screen Capture", "Webcam", "Window Follow", "Clipboard"],
        )
        self._vision_source.currentTextChanged.connect(self._on_vision_source_changed)
        lay.addWidget(self._row("Source", self._vision_source))

        # Camera index — shown only when Webcam is selected
        self._camera_index = self._make_spin(0, 9, 0)
        self._camera_index.setToolTip(
            "Device index reported by your OS (0 = first camera, 1 = second, …)"
        )
        self._camera_index.valueChanged.connect(
            lambda v: self._save("vision.camera_index", v)
        )
        self._camera_index_row = self._row("Camera Index", self._camera_index)
        lay.addWidget(self._camera_index_row)

        # Capture resolution — shown only when Webcam is selected
        self._camera_res = self._make_combo(
            ["640×480", "1280×720 (HD)", "1920×1080 (FHD)", "3840×2160 (4K)"],
        )
        self._camera_res.currentTextChanged.connect(
            lambda v: self._save("vision.camera_resolution", v)
        )
        self._camera_res_row = self._row("Resolution", self._camera_res)
        lay.addWidget(self._camera_res_row)

        self._vision_interval = self._make_spin(100, 10000, 2000)
        self._vision_interval.setSuffix(" ms")
        self._vision_interval.valueChanged.connect(
            lambda v: self._save("vision.interval_ms", v)
        )
        lay.addWidget(self._row("Capture Interval", self._vision_interval))

        # ── Model Roles ───────────────────────────────────────
        lay.addWidget(self._heading("Model Roles"))

        role_hint = QLabel(
            "Assign a specific model ID to each task type. "
            "Leave blank to use the currently active LLM for all tasks."
        )
        role_hint.setWordWrap(True)
        role_hint.setStyleSheet("color:#8fa6c3; font-size:11px; padding:2px 0;")
        lay.addWidget(role_hint)

        _role_defs: list[tuple[str, str, str]] = [
            ("Vision",          "models.vision_model",    "e.g. llava:latest, gpt-4o"),
            ("Image-to-Image",  "models.img2img_model",   "e.g. stable-diffusion-xl"),
            ("Text-to-Image",   "models.txt2img_model",   "e.g. flux1-dev, sdxl-turbo"),
            ("STT Model",       "models.stt_model",       "e.g. whisper-large-v3"),
            ("TTS Model",       "models.tts_model",       "e.g. tts-1-hd, piper-en_US"),
            ("Embedding",       "models.embedding_model", "e.g. nomic-embed-text"),
        ]

        self._role_edits: dict[str, object] = {}
        for label, key, placeholder in _role_defs:
            edit = self._make_line_edit(placeholder=placeholder)
            # Use a default-arg capture to avoid the closure-loop variable bug
            edit.textChanged.connect(
                (lambda k: lambda v: self._save(k, v))(key)
            )
            self._role_edits[key] = edit
            lay.addWidget(self._row(label, edit))

        self._load_saved()

    # ── Vision source toggle ───────────────────────────────────

    def _on_vision_source_changed(self, value: str) -> None:
        self._save("vision.source", value)
        is_webcam = value == "Webcam"
        self._camera_index_row.setVisible(is_webcam)
        self._camera_res_row.setVisible(is_webcam)

    # ── Voice cloning reference browse ────────────────────────

    def _on_browse_clone_ref(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a *.aac);;All Files (*)",
        )
        if path:
            self._clone_ref_edit.setText(path)

    # ── Restore saved config ───────────────────────────────────

    def _load_saved(self) -> None:
        """Restore widget states from persisted config."""
        _restore_combo(self._stt_engine,   self.config.get("voice.stt_engine"))
        _restore_combo(self._stt_lang,     self.config.get("voice.stt_language"))
        _restore_combo(self._tts_engine,   self.config.get("voice.tts_engine"))
        _restore_combo(self._clone_engine, self.config.get("voice.clone_engine"))
        _restore_combo(self._vision_source, self.config.get("vision.source"))
        _restore_combo(self._camera_res,   self.config.get("vision.camera_resolution"))

        val = self.config.get("voice.vad_threshold")
        if val is not None:
            self._vad_slider.setValue(int(val))

        voice = self.config.get("voice.tts_voice", "")
        if voice:
            self._tts_voice.setText(str(voice))

        speed = self.config.get("voice.tts_speed")
        if speed is not None:
            self._tts_speed.setValue(float(speed))

        clone_ref = self.config.get("voice.clone_reference", "")
        if clone_ref:
            self._clone_ref_edit.setText(str(clone_ref))

        clone_name = self.config.get("voice.clone_name", "")
        if clone_name:
            self._clone_name.setText(str(clone_name))

        interval = self.config.get("vision.interval_ms")
        if interval is not None:
            self._vision_interval.setValue(int(interval))

        cam_idx = self.config.get("vision.camera_index")
        if cam_idx is not None:
            self._camera_index.setValue(int(cam_idx))

        # Restore model role text fields
        for key, edit in self._role_edits.items():
            saved = self.config.get(key, "")
            if saved:
                edit.setText(str(saved))

        # Apply initial camera-row visibility
        source = self.config.get("vision.source", "Screen Capture")
        is_webcam = source == "Webcam"
        self._camera_index_row.setVisible(is_webcam)
        self._camera_res_row.setVisible(is_webcam)

    def _save(self, key: str, value) -> None:
        self.config.set(key, value)
        self.bus.publish("voice_vision_changed", {"key": key, "value": value})


def _restore_combo(combo, value) -> None:
    if value is not None:
        idx = combo.findText(str(value))
        if idx >= 0:
            combo.setCurrentIndex(idx)
