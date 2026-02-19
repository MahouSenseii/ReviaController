"""
Profile tab — AI character profile editor.

Manages the AI persona definition: character name, persona description,
voice path, voice tone, fallback message, system prompt, and more.
All fields persist to a dedicated ``profile.json`` file and publish
events so other components react in real time.
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QWidget,
)

from .base_tab import BaseTab

_PROFILE_PATH = Path("profile.json")

_DEFAULT_PROFILE = {
    "character_name": "Astra",
    "persona": "A friendly and knowledgeable AI assistant.",
    "voice_path": "",
    "voice_tone": "Neutral",
    "language": "English",
    "fallback_message": "I'm sorry, I didn't understand that. Could you rephrase?",
    "greeting": "Hello! How can I help you today?",
    "personality_traits": "Helpful, Curious, Patient",
    "response_style": "Conversational",
    "verbosity": "Medium",
    "system_prompt": "",
}

_VOICE_TONES = [
    "Neutral",
    "Warm",
    "Professional",
    "Energetic",
    "Calm",
    "Friendly",
    "Serious",
]

_RESPONSE_STYLES = [
    "Conversational",
    "Bullet Points",
    "Technical",
    "ELI5",
]

_VERBOSITY_LEVELS = ["Low", "Medium", "High"]


class BehaviorTab(BaseTab):
    """AI Profile editor — one data-set per profile, persisted in config."""

    def _build(self) -> None:
        lay = self._layout

        # ── Character Identity ────────────────────────────────
        lay.addWidget(self._heading("Character Identity"))

        self._name_edit = self._make_line_edit(placeholder="e.g. Astra")
        self._name_edit.textChanged.connect(
            lambda v: self._update_field("character_name", v)
        )
        lay.addWidget(self._row("Character Name", self._name_edit))

        self._persona_edit = QTextEdit()
        self._persona_edit.setObjectName("SettingsTextEdit")
        self._persona_edit.setPlaceholderText(
            "Describe the AI's personality and role..."
        )
        self._persona_edit.setMaximumHeight(80)
        self._persona_edit.setStyleSheet(
            "background:#101824; border:1px solid #2a3b55; border-radius:8px; "
            "color:#d8e1ee; padding:8px; font-size:12px;"
        )
        self._persona_edit.textChanged.connect(
            lambda: self._update_field("persona", self._persona_edit.toPlainText())
        )
        lay.addWidget(self._row("Persona", self._persona_edit))

        self._traits_edit = self._make_line_edit(
            placeholder="e.g. Helpful, Curious, Patient"
        )
        self._traits_edit.textChanged.connect(
            lambda v: self._update_field("personality_traits", v)
        )
        lay.addWidget(self._row("Traits", self._traits_edit))

        # ── Voice ─────────────────────────────────────────────
        lay.addWidget(self._heading("Voice"))

        # Voice path with browse button
        voice_row = QWidget()
        vh = QHBoxLayout(voice_row)
        vh.setContentsMargins(0, 0, 0, 0)
        vh.setSpacing(6)
        lbl = QLabel("Voice Path")
        lbl.setFixedWidth(120)
        vh.addWidget(lbl)

        self._voice_path_edit = QLineEdit()
        self._voice_path_edit.setPlaceholderText("Path to voice model file...")
        self._voice_path_edit.setObjectName("SettingsLineEdit")
        self._voice_path_edit.textChanged.connect(
            lambda v: self._update_field("voice_path", v)
        )
        vh.addWidget(self._voice_path_edit, 1)

        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("ModeButton")
        browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_btn.clicked.connect(self._browse_voice)
        vh.addWidget(browse_btn)
        lay.addWidget(voice_row)

        self._tone_combo = self._make_combo(_VOICE_TONES, current=0)
        self._tone_combo.currentTextChanged.connect(
            lambda v: self._update_field("voice_tone", v)
        )
        lay.addWidget(self._row("Voice Tone", self._tone_combo))

        self._lang_edit = self._make_line_edit(placeholder="e.g. English")
        self._lang_edit.textChanged.connect(
            lambda v: self._update_field("language", v)
        )
        lay.addWidget(self._row("Language", self._lang_edit))

        # ── Behavior ──────────────────────────────────────────
        lay.addWidget(self._heading("Behavior"))

        self._style_combo = self._make_combo(_RESPONSE_STYLES, current=0)
        self._style_combo.currentTextChanged.connect(
            lambda v: self._update_field("response_style", v)
        )
        lay.addWidget(self._row("Response Style", self._style_combo))

        self._verbosity_combo = self._make_combo(_VERBOSITY_LEVELS, current=1)
        self._verbosity_combo.currentTextChanged.connect(
            lambda v: self._update_field("verbosity", v)
        )
        lay.addWidget(self._row("Verbosity", self._verbosity_combo))

        self._fallback_edit = self._make_line_edit(
            placeholder="Message when AI can't understand..."
        )
        self._fallback_edit.textChanged.connect(
            lambda v: self._update_field("fallback_message", v)
        )
        lay.addWidget(self._row("Fallback Msg", self._fallback_edit))

        self._greeting_edit = self._make_line_edit(
            placeholder="Greeting message on start..."
        )
        self._greeting_edit.textChanged.connect(
            lambda v: self._update_field("greeting", v)
        )
        lay.addWidget(self._row("Greeting", self._greeting_edit))

        # ── System Prompt ─────────────────────────────────────
        lay.addWidget(self._heading("System Prompt"))

        self._sys_prompt = QTextEdit()
        self._sys_prompt.setObjectName("SettingsTextEdit")
        self._sys_prompt.setPlaceholderText("Enter a custom system prompt...")
        self._sys_prompt.setMaximumHeight(100)
        self._sys_prompt.setStyleSheet(
            "background:#101824; border:1px solid #2a3b55; border-radius:8px; "
            "color:#d8e1ee; padding:8px; font-size:12px;"
        )
        self._sys_prompt.textChanged.connect(
            lambda: self._update_field("system_prompt", self._sys_prompt.toPlainText())
        )
        lay.addWidget(self._sys_prompt)

        # ── Save / Load buttons ───────────────────────────────
        btn_row = QWidget()
        bh = QHBoxLayout(btn_row)
        bh.setContentsMargins(0, 8, 0, 0)
        bh.setSpacing(10)

        save_btn = QPushButton("Save Profile")
        save_btn.setObjectName("ModeButton")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.setStyleSheet(
            "QPushButton { color:#43aa8b; border-color:#43aa8b; } "
            "QPushButton:hover { background:#43aa8b22; }"
        )
        save_btn.clicked.connect(self._save_profile)
        bh.addWidget(save_btn)

        load_btn = QPushButton("Load Profile")
        load_btn.setObjectName("ModeButton")
        load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        load_btn.clicked.connect(self._load_profile_from_file)
        bh.addWidget(load_btn)

        export_btn = QPushButton("Export")
        export_btn.setObjectName("ModeButton")
        export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        export_btn.clicked.connect(self._export_profile)
        bh.addWidget(export_btn)

        bh.addStretch(1)
        lay.addWidget(btn_row)

        # ── Load existing profile data ────────────────────────
        self._current_profile_name: str = ""
        self._loading: bool = False  # guard against spurious _update_field calls
        self._profile_data: dict = {}

        # React whenever the user picks a different profile in the sidebar
        self.bus.subscribe("profile_selected", self._on_profile_selected)
        self._init_from_selected_profile()

    # ══════════════════════════════════════════════════════════
    # Profile persistence
    # ══════════════════════════════════════════════════════════

    def _init_from_selected_profile(self) -> None:
        """Load the profile that is currently active in the sidebar."""
        selected = self.config.get("selected_profile", "")
        if selected:
            self._load_for_profile(selected)
        else:
            # Fallback: load from profile.json (legacy / no profile selected yet)
            self._load_legacy()

    def _on_profile_selected(self, data: dict) -> None:
        """Called when the user switches profiles in the sidebar."""
        name = data.get("value")
        if not name:
            return
        if name != self._current_profile_name:
            self._load_for_profile(name)

    def _load_for_profile(self, name: str) -> None:
        """Load the stored data for *name* (or sensible defaults) into the UI."""
        self._current_profile_name = name

        # Retrieve per-profile data stored in config
        all_profiles_data = self.config.get("profiles_data") or {}
        stored: dict = all_profiles_data.get(name, {}) if isinstance(all_profiles_data, dict) else {}

        # Start from defaults, overlay stored values
        profile_data = dict(_DEFAULT_PROFILE)
        profile_data.update(stored)

        # If this is a brand-new profile with no stored data, seed the
        # character name from the sidebar profile name.
        if not stored:
            profile_data["character_name"] = name

        self._profile_data = profile_data
        self._populate_ui()

    def _load_legacy(self) -> None:
        """Load profile.json (or defaults) — used only when no profile is selected."""
        if _PROFILE_PATH.exists():
            try:
                self._profile_data = json.loads(
                    _PROFILE_PATH.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                self._profile_data = {}

        for k, v in _DEFAULT_PROFILE.items():
            self._profile_data.setdefault(k, v)

        self._populate_ui()

    def _populate_ui(self) -> None:
        """Push current profile data into every widget."""
        self._loading = True
        try:
            d = self._profile_data

            self._name_edit.setText(d.get("character_name", ""))
            self._persona_edit.setPlainText(d.get("persona", ""))
            self._traits_edit.setText(d.get("personality_traits", ""))

            self._voice_path_edit.setText(d.get("voice_path", ""))
            self._set_combo(self._tone_combo, d.get("voice_tone", "Neutral"))
            self._lang_edit.setText(d.get("language", "English"))

            self._set_combo(self._style_combo, d.get("response_style", "Conversational"))
            self._set_combo(self._verbosity_combo, d.get("verbosity", "Medium"))
            self._fallback_edit.setText(d.get("fallback_message", ""))
            self._greeting_edit.setText(d.get("greeting", ""))

            self._sys_prompt.setPlainText(d.get("system_prompt", ""))
        finally:
            self._loading = False

    @staticmethod
    def _set_combo(combo: QComboBox, value: str) -> None:
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _update_field(self, key: str, value: str) -> None:
        """Update in-memory profile and notify the bus."""
        if self._loading:
            return

        self._profile_data[key] = value
        self.bus.publish("profile_field_changed", {"key": key, "value": value})

        # Persist the change immediately for the current profile
        if self._current_profile_name:
            all_profiles_data = self.config.get("profiles_data") or {}
            if not isinstance(all_profiles_data, dict):
                all_profiles_data = {}
            profile_entry = dict(all_profiles_data.get(self._current_profile_name, {}))
            profile_entry[key] = value
            all_profiles_data[self._current_profile_name] = profile_entry
            self.config.set("profiles_data", all_profiles_data)

        # Also mirror core behaviour keys into Config for backward compat
        if key in ("persona", "verbosity", "response_style", "system_prompt"):
            self.config.set(f"behavior.{key}", value)
            self.bus.publish("behavior_changed", {"key": f"behavior.{key}", "value": value})

    def _save_profile(self) -> None:
        """Persist the current profile to config and to a named JSON file."""
        # Persist to config under the current profile name
        if self._current_profile_name:
            all_profiles_data = self.config.get("profiles_data") or {}
            if not isinstance(all_profiles_data, dict):
                all_profiles_data = {}
            all_profiles_data[self._current_profile_name] = dict(self._profile_data)
            self.config.set("profiles_data", all_profiles_data)

        # Also write to a file named after the profile for easy export
        safe_name = self._current_profile_name or "profile"
        save_path = Path(f"{safe_name}.json") if self._current_profile_name else _PROFILE_PATH
        try:
            save_path.write_text(
                json.dumps(self._profile_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.bus.publish("profile_saved", dict(self._profile_data))
            QMessageBox.information(
                self,
                "Profile Saved",
                f"Profile saved to {save_path.resolve()}",
            )
        except OSError as exc:
            QMessageBox.warning(self, "Save Error", str(exc))

    def _load_profile_from_file(self) -> None:
        """Open a file dialog and load a JSON profile."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load AI Profile",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Profile must be a JSON object")
            # Merge defaults for missing keys
            for k, v in _DEFAULT_PROFILE.items():
                data.setdefault(k, v)
            self._profile_data = data
            # Persist into the current profile's config slot
            if self._current_profile_name:
                all_profiles_data = self.config.get("profiles_data") or {}
                if not isinstance(all_profiles_data, dict):
                    all_profiles_data = {}
                all_profiles_data[self._current_profile_name] = dict(data)
                self.config.set("profiles_data", all_profiles_data)
            self._populate_ui()
            QMessageBox.information(
                self, "Profile Loaded", f"Loaded profile from:\n{path}"
            )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            QMessageBox.warning(self, "Load Error", str(exc))

    def _export_profile(self) -> None:
        """Export the current profile to a chosen location."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export AI Profile",
            "profile.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(self._profile_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            QMessageBox.information(
                self, "Exported", f"Profile exported to:\n{path}"
            )
        except OSError as exc:
            QMessageBox.warning(self, "Export Error", str(exc))

    def _browse_voice(self) -> None:
        """Open a file dialog to select the voice model file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Voice Model",
            "",
            "Voice Files (*.onnx *.pth *.bin *.wav);;All Files (*)",
        )
        if path:
            self._voice_path_edit.setText(path)
