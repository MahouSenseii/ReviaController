"""
Left sidebar panel — AI monitoring dashboard.

Shows the active assistant's avatar, name / type, live module-status
indicators (STT, TTS, Vision) with coloured lights, emotion state
display, and a profile dropdown that lets the user switch between or
manage AI assistant profiles.

Publishes events when the user toggles modules or switches profiles.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.config import Config
from core.events import EventBus
from ui.widgets import Pill, SectionLabel, StatusDot

from .base_panel import BasePanel

# Default profile created on first launch
_DEFAULT_PROFILE = {
    "name": "Astra",
    "type": "Assistant",
    "avatar": "assets/avatar.png",
}

# Available role types for the "Add Profile" dialog
_PROFILE_TYPES = [
    "Assistant",
    "Teacher",
    "Debugger",
    "Researcher",
    "Creative",
    "Custom",
]


class _ModeButton(QPushButton):
    """Flat selectable button used for Modes."""

    def __init__(self, text: str, group: str, event_bus: EventBus):
        super().__init__(text)
        self._group = group
        self._bus = event_bus
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("ModeButton")
        self.clicked.connect(self._on_click)

    def _on_click(self) -> None:
        self._bus.publish(f"{self._group}_selected", {"value": self.text()})


class SidebarPanel(BasePanel):
    """Left-hand sidebar: avatar, monitoring indicators, profiles."""

    def __init__(self, event_bus: EventBus, config: Config):
        super().__init__(event_bus, config)
        self.setFixedWidth(260)

    # ══════════════════════════════════════════════════════════
    # Build
    # ══════════════════════════════════════════════════════════

    def _build(self) -> None:
        lay = self._inner_layout

        # ── Avatar ────────────────────────────────────────────
        self._avatar = QLabel()
        self._avatar.setObjectName("AvatarImage")
        self._avatar.setFixedSize(220, 220)
        self._avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._avatar, alignment=Qt.AlignmentFlag.AlignHCenter)

        # ── Name / type row ──────────────────────────────────
        name_row = QWidget()
        nr = QHBoxLayout(name_row)
        nr.setContentsMargins(0, 0, 0, 0)
        nr.setSpacing(8)

        self._status_dot = StatusDot("on", size=10)
        self._status_dot.setObjectName("StatusDot")

        self._name_label = QLabel()
        nr.addStretch(1)
        nr.addWidget(self._status_dot, alignment=Qt.AlignmentFlag.AlignVCenter)
        nr.addWidget(self._name_label, alignment=Qt.AlignmentFlag.AlignVCenter)
        nr.addStretch(1)
        lay.addWidget(name_row)
        lay.addSpacing(6)

        # ── Emotion display ───────────────────────────────────
        self._emotion_widget = self._build_emotion_display()
        lay.addWidget(self._emotion_widget)

        # ── Module status pills ──────────────────────────────
        self._stt_pill = Pill("STT", "Idle", status="off", toggle=True, checked=True)
        self._stt_pill.toggled.connect(lambda on: self.bus.publish("stt_toggled", {"enabled": on}))
        lay.addWidget(self._stt_pill)

        self._tts_pill = Pill("TTS", "Idle", status="off", toggle=True, checked=True)
        self._tts_pill.toggled.connect(lambda on: self.bus.publish("tts_toggled", {"enabled": on}))
        lay.addWidget(self._tts_pill)

        self._vision_pill = Pill("Vision", "Idle", status="off", toggle=True, checked=True)
        self._vision_pill.toggled.connect(lambda on: self.bus.publish("vision_toggled", {"enabled": on}))
        lay.addWidget(self._vision_pill)

        # ── Modes ─────────────────────────────────────────────
        lay.addSpacing(14)
        lay.addWidget(SectionLabel("Modes"))

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_buttons: list[_ModeButton] = []
        for idx, label in enumerate(("Passive (Background)", "Interactive (Voice)", "Teaching / Explain", "Debug")):
            btn = _ModeButton(label, "mode", self.bus)
            self._mode_group.addButton(btn, idx)
            self._mode_buttons.append(btn)
            lay.addWidget(btn)

        # ── Profiles dropdown ─────────────────────────────────
        lay.addSpacing(14)
        lay.addWidget(SectionLabel("Profiles"))

        self._profile_combo = QComboBox()
        self._profile_combo.setObjectName("ProfileCombo")
        self._profile_combo.setMinimumHeight(28)
        self._profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        lay.addWidget(self._profile_combo)

        # Add / Remove buttons
        btn_row = QWidget()
        bh = QHBoxLayout(btn_row)
        bh.setContentsMargins(0, 4, 0, 0)
        bh.setSpacing(8)

        add_btn = QPushButton("Add New")
        add_btn.setObjectName("ModeButton")
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.clicked.connect(self._on_profile_add)
        bh.addWidget(add_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.setObjectName("ModeButton")
        remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_btn.setStyleSheet(
            "QPushButton { color:#ef476f; } QPushButton:hover { border-color:#ef476f; }"
        )
        remove_btn.clicked.connect(self._on_profile_remove)
        bh.addWidget(remove_btn)

        bh.addStretch(1)
        lay.addWidget(btn_row)

        lay.addStretch(1)

        # ── Wire up events ────────────────────────────────────
        self.bus.subscribe("mode_selected", self._on_mode_selected)
        self.bus.subscribe("module_status", self._on_module_status)
        self.bus.subscribe("emotion_state_changed", self._on_emotion_changed)

        # ── Initialise data ───────────────────────────────────
        self._ensure_default_profile()
        self._load_profiles()

    # ══════════════════════════════════════════════════════════
    # Emotion display
    # ══════════════════════════════════════════════════════════

    def _build_emotion_display(self) -> QWidget:
        """Build a compact emotion state widget."""
        w = QWidget()
        w.setObjectName("EmotionDisplay")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(4)

        lay.addWidget(SectionLabel("Emotional State"))

        # Dominant emotion label
        self._emotion_label = QLabel("Neutral")
        self._emotion_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._emotion_label.setStyleSheet(
            "font-weight:700; font-size:14px; color:#f9c74f; padding:2px 0;"
        )
        lay.addWidget(self._emotion_label)

        # Mood label
        self._mood_label = QLabel("Mood: neutral")
        self._mood_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._mood_label.setStyleSheet(
            "font-size:11px; color:#8fa6c3; padding:0;"
        )
        lay.addWidget(self._mood_label)

        # Top emotion bars (up to 3)
        self._emotion_bars: list[tuple[QLabel, QProgressBar]] = []
        for _ in range(3):
            row = QWidget()
            rh = QHBoxLayout(row)
            rh.setContentsMargins(0, 0, 0, 0)
            rh.setSpacing(6)

            name_lbl = QLabel("-")
            name_lbl.setFixedWidth(80)
            name_lbl.setStyleSheet("font-size:11px; color:#c7d3e6;")
            rh.addWidget(name_lbl)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            bar.setStyleSheet(
                "QProgressBar { background:#0a0f18; border:1px solid #2a3b55; border-radius:4px; }"
                "QProgressBar::chunk { background:#f9c74f; border-radius:3px; }"
            )
            rh.addWidget(bar, 1)

            self._emotion_bars.append((name_lbl, bar))
            lay.addWidget(row)

        return w

    def _on_emotion_changed(self, data: dict) -> None:
        """Update the emotion display from emotion_state_changed event."""
        dominant = data.get("dominant", "neutral")
        intensity = data.get("dominant_intensity", 0.0)
        colour = data.get("colour", "#f9c74f")
        mood = data.get("mood", "neutral")
        top = data.get("top_emotions", [])

        # Dominant emotion
        if intensity < 0.05:
            self._emotion_label.setText("Neutral")
            self._emotion_label.setStyleSheet(
                "font-weight:700; font-size:14px; color:#8fa6c3; padding:2px 0;"
            )
        else:
            display = f"{dominant.title()} ({intensity:.0%})"
            self._emotion_label.setText(display)
            self._emotion_label.setStyleSheet(
                f"font-weight:700; font-size:14px; color:{colour}; padding:2px 0;"
            )

        # Mood
        self._mood_label.setText(f"Mood: {mood}")

        # Top emotion bars
        for i, (name_lbl, bar) in enumerate(self._emotion_bars):
            if i < len(top):
                entry = top[i]
                ename = entry.get("name", "-")
                eintensity = entry.get("intensity", 0.0)
                name_lbl.setText(ename)
                bar.setValue(int(eintensity * 100))

                # Colour the bar chunk based on intensity
                bar.setStyleSheet(
                    "QProgressBar { background:#0a0f18; border:1px solid #2a3b55; border-radius:4px; }"
                    f"QProgressBar::chunk {{ background:{colour}; border-radius:3px; }}"
                )
            else:
                name_lbl.setText("-")
                bar.setValue(0)

    # ══════════════════════════════════════════════════════════
    # Profile helpers
    # ══════════════════════════════════════════════════════════

    def _get_profiles(self) -> list[dict]:
        return self.config.get("profiles", [])

    def _save_profiles(self, profiles: list[dict]) -> None:
        self.config.set("profiles", profiles)

    def _ensure_default_profile(self) -> None:
        """Create a default profile if none exist."""
        profiles = self._get_profiles()
        if not profiles:
            self._save_profiles([_DEFAULT_PROFILE.copy()])

    def _load_profiles(self) -> None:
        """Populate the dropdown from config and select the active profile."""
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()

        profiles = self._get_profiles()
        if not profiles:
            self._profile_combo.addItem("(no profiles)")
        else:
            for p in profiles:
                label = f"{p['name']}  ({p['type']})"
                self._profile_combo.addItem(label)

        selected = self.config.get("selected_profile", "")
        if selected:
            for i, p in enumerate(profiles):
                if p["name"] == selected:
                    self._profile_combo.setCurrentIndex(i)
                    break

        self._profile_combo.blockSignals(False)
        self._apply_active_profile()

    def _apply_active_profile(self) -> None:
        """Update the avatar, name label, and status dot for the active profile."""
        profiles = self._get_profiles()
        idx = self._profile_combo.currentIndex()

        if not profiles or idx < 0 or idx >= len(profiles):
            self._name_label.setText(
                '<span style="color:#8fa6c3;">No profile selected</span>'
            )
            self._avatar.clear()
            return

        p = profiles[idx]
        name = p.get("name", "Unknown")
        ptype = p.get("type", "Assistant")
        avatar_path = p.get("avatar", "assets/avatar.png")

        # Name / type
        self._name_label.setText(
            f'<span style="color:#ffffff; font-weight:600;">{name}</span> '
            f'<span style="color:#7fb3ff; font-weight:300;">({ptype})</span>'
        )

        # Avatar
        pix = QPixmap(avatar_path)
        if pix.isNull():
            pix = QPixmap("assets/avatar.png")
        if not pix.isNull():
            self._avatar.setPixmap(
                pix.scaled(
                    self._avatar.size(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        self.config.set("selected_profile", name, save=False)

    # ══════════════════════════════════════════════════════════
    # Profile actions
    # ══════════════════════════════════════════════════════════

    def _on_profile_selected(self, idx: int) -> None:
        profiles = self._get_profiles()
        if not profiles or idx < 0 or idx >= len(profiles):
            return

        p = profiles[idx]
        self.config.set("selected_profile", p["name"])
        self._apply_active_profile()
        self.bus.publish("profile_selected", {
            "value": p["name"],
            "type": p["type"],
        })

    def _on_profile_add(self) -> None:
        """Prompt for a name and type, then create a new profile."""
        name, ok = QInputDialog.getText(
            self, "New Profile", "Profile name:",
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        # Check for duplicate
        profiles = self._get_profiles()
        for p in profiles:
            if p["name"].lower() == name.lower():
                QMessageBox.warning(
                    self, "Duplicate Name",
                    f'A profile named "{p["name"]}" already exists.',
                )
                return

        ptype, ok = QInputDialog.getItem(
            self, "Profile Type",
            "Select a role for this profile:",
            _PROFILE_TYPES, 0, False,
        )
        if not ok:
            return

        profiles.append({
            "name": name,
            "type": ptype,
            "avatar": "assets/avatar.png",
        })
        self._save_profiles(profiles)
        self._load_profiles()

        # Select the newly added profile
        for i, p in enumerate(profiles):
            if p["name"] == name:
                self._profile_combo.setCurrentIndex(i)
                break

        QMessageBox.information(
            self, "Profile Created",
            f'Created profile "{name}" ({ptype}).',
        )

    def _on_profile_remove(self) -> None:
        profiles = self._get_profiles()
        idx = self._profile_combo.currentIndex()

        if not profiles or idx < 0 or idx >= len(profiles):
            QMessageBox.warning(
                self, "No Profile Selected",
                "Please select a profile from the dropdown to remove.",
            )
            return

        name = profiles[idx]["name"]
        reply = QMessageBox.question(
            self, "Remove Profile",
            f'Remove profile "{name}"?\n\nThis cannot be undone.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        profiles.pop(idx)
        self._save_profiles(profiles)

        # If list is now empty, recreate default
        if not profiles:
            self._ensure_default_profile()

        self._load_profiles()
        self.bus.publish("profile_selected", {"value": None, "type": None})

    # ══════════════════════════════════════════════════════════
    # Mode handlers
    # ══════════════════════════════════════════════════════════

    def _on_mode_selected(self, data: dict) -> None:
        value = data.get("value")
        for btn in self._mode_buttons:
            btn.setChecked(btn.text() == value)
        self.config.set("mode", value)

    # ══════════════════════════════════════════════════════════
    # Module status (from backend / plugins)
    # ══════════════════════════════════════════════════════════

    def _on_module_status(self, data: dict) -> None:
        """
        Expected data::

            {"module": "stt"|"tts"|"vision",
             "status": "on"|"warn"|"off",
             "subtitle": "Errors: 2 / 10m"}
        """
        module = data.get("module")
        pill_map = {
            "stt": self._stt_pill,
            "tts": self._tts_pill,
            "vision": self._vision_pill,
        }
        p = pill_map.get(module)
        if p is None:
            return

        if "status" in data:
            p.set_status(data["status"])
            # Update title to reflect state
            status = data["status"]
            label = module.upper()
            if status == "on":
                p.set_title(f"{label} Active")
            elif status == "warn":
                p.set_title(f"{label} Issues")
            else:
                p.set_title(f"{label} Offline")

        if "subtitle" in data:
            p.set_subtitle(data["subtitle"])
