"""
Left sidebar panel — AI monitoring dashboard.

Shows the active assistant's avatar, name / type, live module-status
indicators (STT, TTS, Vision) with coloured lights, and a profile
dropdown that lets the user switch between or manage AI assistant
profiles.

Publishes events when the user toggles modules or switches profiles.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
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
        self.bus.subscribe("module_status", self._on_module_status)

        # ── Initialise data ───────────────────────────────────
        self._ensure_default_profile()
        self._load_profiles()

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
