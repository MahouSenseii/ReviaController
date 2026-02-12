"""
Character tab — edit the active profile's persona, example dialogue,
emotion pool, and neural-network-style emotion weights.

The profile is the data that gets sent to the LLM so it knows *who*
the AI character is.  Everything saved here is stored inside the
profile entry in ``config.json`` and published on ``profile_selected``
so the rest of the system (prompt builder, TTS voice picker, etc.)
can react to it.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config import Config
from core.events import EventBus

from .base_tab import BaseTab


class CharacterTab(BaseTab):
    """Editor for the active profile's character card and emotion weights."""

    def __init__(self, event_bus: EventBus, config: Config):
        self._updating = False  # guard against feedback loops
        super().__init__(event_bus, config)

    # ══════════════════════════════════════════════════════════
    # Build
    # ══════════════════════════════════════════════════════════

    def _build(self) -> None:
        lay = self._layout
        lay.setSpacing(8)

        # ── Scrollable content ───────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border:none; background:transparent; }"
        )

        container = QWidget()
        container.setStyleSheet("background:transparent;")
        self._clayout = QVBoxLayout(container)
        self._clayout.setContentsMargins(0, 0, 0, 0)
        self._clayout.setSpacing(10)

        self._build_identity()
        self._build_persona()
        self._build_emotions()
        self._build_weights()

        self._clayout.addStretch(1)
        scroll.setWidget(container)
        lay.addWidget(scroll, 1)

        # ── Events ───────────────────────────────────────────
        self.bus.subscribe("profile_selected", self._on_profile_switched)

        # Load current profile on startup
        self._load_active_profile()

    # ── Identity section ──────────────────────────────────────

    def _build_identity(self) -> None:
        self._clayout.addWidget(self._heading("Identity"))

        self._name_display = QLabel("(no profile)")
        self._name_display.setStyleSheet(
            "color:#d8e1ee; font-size:13px; font-weight:700; "
            "background:transparent; border:none;"
        )
        self._clayout.addWidget(self._name_display)

        self._creator_edit = QLineEdit()
        self._creator_edit.setObjectName("SettingsLineEdit")
        self._creator_edit.setPlaceholderText("Creator name...")
        self._creator_edit.textChanged.connect(self._on_field_changed)
        self._clayout.addWidget(self._row("Creator", self._creator_edit))

        self._fallback_edit = QLineEdit()
        self._fallback_edit.setObjectName("SettingsLineEdit")
        self._fallback_edit.setPlaceholderText("Fallback message shown when AI is down...")
        self._fallback_edit.textChanged.connect(self._on_field_changed)
        self._clayout.addWidget(self._row("Fallback", self._fallback_edit))

    # ── Persona / dialogue section ────────────────────────────

    def _build_persona(self) -> None:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#263246;")
        self._clayout.addWidget(sep)

        self._clayout.addWidget(self._heading("Persona"))

        self._persona_edit = QTextEdit()
        self._persona_edit.setObjectName("SettingsLineEdit")
        self._persona_edit.setPlaceholderText(
            "Describe the character's personality, quirks, backstory..."
        )
        self._persona_edit.setMinimumHeight(100)
        self._persona_edit.setMaximumHeight(160)
        self._persona_edit.setStyleSheet(
            "QTextEdit { background:#101824; border:1px solid #2a3b55; "
            "border-radius:6px; padding:6px; color:#d8e1ee; font-size:12px; }"
            "QTextEdit:focus { border-color:#4a8cd8; }"
        )
        self._persona_edit.textChanged.connect(self._on_field_changed)
        self._clayout.addWidget(self._persona_edit)

        self._clayout.addWidget(self._heading("Example Dialogue"))

        self._dialogue_edit = QTextEdit()
        self._dialogue_edit.setObjectName("SettingsLineEdit")
        self._dialogue_edit.setPlaceholderText(
            "User: Hello!\nCharacter: Hey hey! What's up?!"
        )
        self._dialogue_edit.setMinimumHeight(100)
        self._dialogue_edit.setMaximumHeight(160)
        self._dialogue_edit.setStyleSheet(
            "QTextEdit { background:#101824; border:1px solid #2a3b55; "
            "border-radius:6px; padding:6px; color:#d8e1ee; font-size:12px; }"
            "QTextEdit:focus { border-color:#4a8cd8; }"
        )
        self._dialogue_edit.textChanged.connect(self._on_field_changed)
        self._clayout.addWidget(self._dialogue_edit)

    # ── Emotions pool ─────────────────────────────────────────

    def _build_emotions(self) -> None:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#263246;")
        self._clayout.addWidget(sep)

        self._clayout.addWidget(self._heading("Available Emotions"))

        hint = QLabel("These are the emotions the AI can express.")
        hint.setStyleSheet(
            "color:#5a7090; font-size:11px; background:transparent; border:none;"
        )
        self._clayout.addWidget(hint)

        # Tag cloud (flow layout approximated with word-wrap label)
        self._emotion_cloud = QLabel()
        self._emotion_cloud.setWordWrap(True)
        self._emotion_cloud.setStyleSheet(
            "color:#8fa6c3; font-size:11px; line-height:1.6; "
            "background:transparent; border:none; padding:4px 0;"
        )
        self._clayout.addWidget(self._emotion_cloud)

        # Add / remove row
        emo_row = QWidget()
        emo_row.setStyleSheet("background:transparent;")
        eh = QHBoxLayout(emo_row)
        eh.setContentsMargins(0, 0, 0, 0)
        eh.setSpacing(6)

        self._emotion_add_edit = QLineEdit()
        self._emotion_add_edit.setObjectName("SettingsLineEdit")
        self._emotion_add_edit.setPlaceholderText("New emotion...")
        eh.addWidget(self._emotion_add_edit, 1)

        add_btn = QPushButton("Add")
        add_btn.setObjectName("ModeButton")
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.clicked.connect(self._on_emotion_add)
        eh.addWidget(add_btn)

        self._emotion_remove_edit = QLineEdit()
        self._emotion_remove_edit.setObjectName("SettingsLineEdit")
        self._emotion_remove_edit.setPlaceholderText("Remove emotion...")
        eh.addWidget(self._emotion_remove_edit, 1)

        rem_btn = QPushButton("Remove")
        rem_btn.setObjectName("ModeButton")
        rem_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        rem_btn.setStyleSheet(
            "QPushButton { color:#ef476f; } QPushButton:hover { border-color:#ef476f; }"
        )
        rem_btn.clicked.connect(self._on_emotion_remove)
        eh.addWidget(rem_btn)

        self._clayout.addWidget(emo_row)

    # ── Weights (neural-network style) ────────────────────────

    def _build_weights(self) -> None:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#263246;")
        self._clayout.addWidget(sep)

        self._clayout.addWidget(self._heading("Emotion Weights"))

        hint = QLabel(
            "Set base weights for the character's mood. "
            "Higher weight = stronger tendency toward that emotion."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(
            "color:#5a7090; font-size:11px; background:transparent; border:none;"
        )
        self._clayout.addWidget(hint)

        # Scrollable weight grid
        self._weight_container = QWidget()
        self._weight_container.setStyleSheet("background:transparent;")
        self._weight_grid = QGridLayout(self._weight_container)
        self._weight_grid.setContentsMargins(0, 0, 0, 0)
        self._weight_grid.setSpacing(4)
        self._weight_rows: list[dict] = []  # [{emotion, spin, remove_btn}]
        self._clayout.addWidget(self._weight_container)

        # Add-weight row
        add_row = QWidget()
        add_row.setStyleSheet("background:transparent;")
        ah = QHBoxLayout(add_row)
        ah.setContentsMargins(0, 4, 0, 0)
        ah.setSpacing(6)

        self._weight_emotion_combo = QComboBox()
        self._weight_emotion_combo.setObjectName("SettingsCombo")
        self._weight_emotion_combo.setMinimumWidth(140)
        ah.addWidget(self._weight_emotion_combo, 1)

        self._weight_value_spin = QSpinBox()
        self._weight_value_spin.setObjectName("SettingsSpin")
        self._weight_value_spin.setRange(1, 10)
        self._weight_value_spin.setValue(3)
        self._weight_value_spin.setFixedWidth(55)
        ah.addWidget(self._weight_value_spin)

        add_w_btn = QPushButton("Add Weight")
        add_w_btn.setObjectName("ModeButton")
        add_w_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_w_btn.clicked.connect(self._on_weight_add)
        ah.addWidget(add_w_btn)

        ah.addStretch(1)
        self._clayout.addWidget(add_row)

    # ══════════════════════════════════════════════════════════
    # Profile data helpers
    # ══════════════════════════════════════════════════════════

    def _active_profile(self) -> dict | None:
        """Return the active profile dict, or None."""
        selected = self.config.get("selected_profile", "")
        if not selected:
            return None
        for p in self.config.get("profiles", []):
            if p.get("name") == selected:
                return p
        return None

    def _save_profiles(self) -> None:
        profiles = self.config.get("profiles", [])
        self.config.set("profiles", profiles)

    # ══════════════════════════════════════════════════════════
    # Load / display
    # ══════════════════════════════════════════════════════════

    def _load_active_profile(self) -> None:
        """Populate all fields from the currently selected profile."""
        self._updating = True
        profile = self._active_profile()

        if not profile:
            self._name_display.setText("(no profile selected)")
            self._creator_edit.clear()
            self._fallback_edit.clear()
            self._persona_edit.clear()
            self._dialogue_edit.clear()
            self._emotion_cloud.setText("")
            self._clear_weight_grid()
            self._updating = False
            return

        self._name_display.setText(
            f"{profile.get('name', '?')}  "
            f"<span style='color:#7fb3ff;'>({profile.get('type', 'Assistant')})</span>"
        )

        self._creator_edit.setText(profile.get("creator", ""))
        self._fallback_edit.setText(profile.get("fallback_message", ""))
        self._persona_edit.setPlainText(profile.get("persona", ""))
        self._dialogue_edit.setPlainText(profile.get("example_dialogue", ""))

        # Emotions cloud
        emotions = profile.get("emotions", [])
        self._refresh_emotion_cloud(emotions)
        self._refresh_weight_combo(emotions, profile.get("weights", {}))

        # Weights grid
        self._rebuild_weight_grid(profile.get("weights", {}))

        self._updating = False

    def _refresh_emotion_cloud(self, emotions: list[str]) -> None:
        if not emotions:
            self._emotion_cloud.setText("<i>No emotions defined.</i>")
            return
        tags = []
        for e in sorted(emotions):
            tags.append(
                f'<span style="background:#1a2e44; border-radius:4px; '
                f'padding:2px 6px; margin:1px; color:#8fc9ff;">{e}</span>'
            )
        self._emotion_cloud.setText("  ".join(tags))

    def _refresh_weight_combo(
        self, emotions: list[str], weights: dict[str, int],
    ) -> None:
        """Populate the add-weight combo with emotions not yet weighted."""
        self._weight_emotion_combo.blockSignals(True)
        self._weight_emotion_combo.clear()
        used = set(weights.keys())
        available = [e for e in sorted(emotions) if e not in used]
        self._weight_emotion_combo.addItems(available)
        self._weight_emotion_combo.blockSignals(False)

    # ── Weight grid management ────────────────────────────────

    def _clear_weight_grid(self) -> None:
        while self._weight_grid.count():
            item = self._weight_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._weight_rows.clear()

    def _rebuild_weight_grid(self, weights: dict[str, int]) -> None:
        self._clear_weight_grid()

        if not weights:
            empty = QLabel("No weights set — add one below.")
            empty.setStyleSheet(
                "color:#5a7090; font-size:11px; background:transparent; border:none;"
            )
            self._weight_grid.addWidget(empty, 0, 0, 1, 3)
            return

        # Header
        for col, text in enumerate(("Emotion", "Weight", "")):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "color:#5a7090; font-size:10px; font-weight:700; "
                "background:transparent; border:none;"
            )
            self._weight_grid.addWidget(lbl, 0, col)

        row_idx = 1
        for emotion in sorted(weights.keys()):
            value = weights[emotion]

            name_lbl = QLabel(emotion)
            name_lbl.setStyleSheet(
                "color:#d8e1ee; font-size:12px; background:transparent; border:none;"
            )
            self._weight_grid.addWidget(name_lbl, row_idx, 0)

            spin = QSpinBox()
            spin.setObjectName("SettingsSpin")
            spin.setRange(0, 10)
            spin.setValue(value)
            spin.setFixedWidth(55)
            spin.valueChanged.connect(
                lambda v, e=emotion: self._on_weight_value_changed(e, v),
            )
            self._weight_grid.addWidget(spin, row_idx, 1)

            remove_btn = QPushButton("x")
            remove_btn.setObjectName("ModeButton")
            remove_btn.setFixedSize(24, 24)
            remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            remove_btn.setStyleSheet(
                "QPushButton { color:#ef476f; font-weight:700; } "
                "QPushButton:hover { border-color:#ef476f; }"
            )
            remove_btn.clicked.connect(
                lambda _, e=emotion: self._on_weight_remove(e),
            )
            self._weight_grid.addWidget(remove_btn, row_idx, 2)

            self._weight_rows.append({
                "emotion": emotion,
                "spin": spin,
                "remove_btn": remove_btn,
            })

            row_idx += 1

    # ══════════════════════════════════════════════════════════
    # Event handlers
    # ══════════════════════════════════════════════════════════

    def _on_profile_switched(self, _data: dict) -> None:
        self._load_active_profile()

    def _on_field_changed(self, *_args) -> None:
        """Save text fields back into the active profile."""
        if self._updating:
            return
        profile = self._active_profile()
        if not profile:
            return

        profile["creator"] = self._creator_edit.text().strip()
        profile["fallback_message"] = self._fallback_edit.text().strip()
        profile["persona"] = self._persona_edit.toPlainText()
        profile["example_dialogue"] = self._dialogue_edit.toPlainText()
        self._save_profiles()

    # ── Emotion handlers ──────────────────────────────────────

    def _on_emotion_add(self) -> None:
        text = self._emotion_add_edit.text().strip().lower()
        if not text:
            return
        profile = self._active_profile()
        if not profile:
            return

        emotions = profile.setdefault("emotions", [])
        if text in emotions:
            QMessageBox.information(
                self, "Already Exists",
                f'"{text}" is already in the emotion pool.',
            )
            return

        emotions.append(text)
        emotions.sort()
        self._save_profiles()
        self._emotion_add_edit.clear()

        self._refresh_emotion_cloud(emotions)
        self._refresh_weight_combo(emotions, profile.get("weights", {}))

    def _on_emotion_remove(self) -> None:
        text = self._emotion_remove_edit.text().strip().lower()
        if not text:
            return
        profile = self._active_profile()
        if not profile:
            return

        emotions = profile.get("emotions", [])
        if text not in emotions:
            QMessageBox.warning(
                self, "Not Found",
                f'"{text}" is not in the emotion pool.',
            )
            return

        emotions.remove(text)
        # Also remove from weights if present
        weights = profile.get("weights", {})
        weights.pop(text, None)
        self._save_profiles()
        self._emotion_remove_edit.clear()

        self._refresh_emotion_cloud(emotions)
        self._refresh_weight_combo(emotions, weights)
        self._rebuild_weight_grid(weights)

    # ── Weight handlers ───────────────────────────────────────

    def _on_weight_add(self) -> None:
        emotion = self._weight_emotion_combo.currentText()
        if not emotion:
            return
        profile = self._active_profile()
        if not profile:
            return

        value = self._weight_value_spin.value()
        weights = profile.setdefault("weights", {})
        weights[emotion] = value
        self._save_profiles()

        self._rebuild_weight_grid(weights)
        self._refresh_weight_combo(
            profile.get("emotions", []), weights,
        )
        self._publish_update()

    def _on_weight_value_changed(self, emotion: str, value: int) -> None:
        if self._updating:
            return
        profile = self._active_profile()
        if not profile:
            return
        weights = profile.setdefault("weights", {})
        if value == 0:
            weights.pop(emotion, None)
        else:
            weights[emotion] = value
        self._save_profiles()
        self._publish_update()

    def _on_weight_remove(self, emotion: str) -> None:
        profile = self._active_profile()
        if not profile:
            return
        weights = profile.get("weights", {})
        weights.pop(emotion, None)
        self._save_profiles()

        self._rebuild_weight_grid(weights)
        self._refresh_weight_combo(
            profile.get("emotions", []), weights,
        )
        self._publish_update()

    def _publish_update(self) -> None:
        """Notify the system that the character data changed."""
        profile = self._active_profile()
        if not profile:
            return
        self.bus.publish("profile_updated", {
            "value": profile.get("name", ""),
            "weights": profile.get("weights", {}),
        })
