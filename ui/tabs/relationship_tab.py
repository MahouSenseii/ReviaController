"""
Relationship dashboard tab — visualises rapport/trust trend over
time, stable preferences, boundaries, and communication style.

Subscribes to:
    - ``persona_updated`` / ``persona_drift_detected`` for persona data
    - ``intent_updated`` for inferred preferences/boundaries
    - ``continuity_updated`` for relationship signals
    - ``emotion_state_changed`` for emotional rapport trend
"""

from __future__ import annotations

import time

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .base_tab import BaseTab


class _TrendBar(QWidget):
    """Horizontal bar showing a value on a bipolar or unipolar scale."""

    def __init__(self, label: str, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self._min = min_val
        self._max = max_val

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._label = QLabel(label)
        self._label.setFixedWidth(90)
        self._label.setStyleSheet("color:#d1d5db; font-size:11px;")
        layout.addWidget(self._label, 0, 0)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(50)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(12)
        self._bar.setStyleSheet(
            "QProgressBar { background:#0a0f18; border:1px solid #2a3b55; border-radius:6px; }"
            "QProgressBar::chunk { background:#8fc9ff; border-radius:5px; }"
        )
        layout.addWidget(self._bar, 0, 1)

        self._value_label = QLabel("-")
        self._value_label.setFixedWidth(50)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._value_label.setStyleSheet("color:#9ca3af; font-size:10px;")
        layout.addWidget(self._value_label, 0, 2)

    def set_value(self, value: float) -> None:
        normalised = (value - self._min) / (self._max - self._min + 1e-9)
        clamped = max(0, min(100, int(normalised * 100)))
        self._bar.setValue(clamped)
        self._value_label.setText(f"{value:.2f}")


class _InfoSection(QFrame):
    """A titled section with bullet-point items."""

    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("InfoSection")
        self.setStyleSheet(
            "QFrame#InfoSection { background:#111827; border:1px solid #1e293b; "
            "border-radius:8px; padding:8px; }"
        )

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 6, 8, 6)
        self._layout.setSpacing(3)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("color:#8fc9ff; font-size:12px; font-weight:700;")
        self._layout.addWidget(title_lbl)

        self._items_label = QLabel("No data yet")
        self._items_label.setWordWrap(True)
        self._items_label.setStyleSheet("color:#d1d5db; font-size:11px;")
        self._layout.addWidget(self._items_label)

    def set_items(self, items: list[str]) -> None:
        if not items:
            self._items_label.setText("No data yet")
        else:
            self._items_label.setText("\n".join(f"  \u2022 {item}" for item in items[:10]))


class RelationshipTab(BaseTab):
    """Relationship dashboard with rapport trends and persona profile."""

    def _build(self) -> None:
        lay = self._layout

        lay.addWidget(self._heading("Relationship Dashboard"))

        # Persona style dimensions
        lay.addWidget(QLabel("Persona Style Profile"))
        self._persona_bars: dict[str, _TrendBar] = {}
        dims = [
            "Formality", "Warmth", "Verbosity", "Humor",
            "Assertiveness", "Empathy", "Creativity", "Patience",
        ]
        for dim in dims:
            bar = _TrendBar(dim)
            self._persona_bars[dim.lower()] = bar
            lay.addWidget(bar)

        # Drift indicator
        drift_row = QHBoxLayout()
        drift_row.addWidget(QLabel("Persona Drift:"))
        self._drift_label = QLabel("0.00")
        self._drift_label.setStyleSheet("color:#22c55e; font-size:12px; font-weight:700;")
        drift_row.addWidget(self._drift_label)
        drift_row.addStretch()

        self._drift_status = QLabel("Stable")
        self._drift_status.setStyleSheet(
            "color:#22c55e; background:#052e16; padding:2px 8px; "
            "border-radius:4px; font-size:10px;"
        )
        drift_row.addWidget(self._drift_status)
        lay.addLayout(drift_row)

        # Mood trend
        mood_row = QHBoxLayout()
        mood_row.addWidget(QLabel("Current Mood:"))
        self._mood_label = QLabel("neutral")
        self._mood_label.setStyleSheet("color:#f59e0b; font-size:12px;")
        mood_row.addWidget(self._mood_label)
        mood_row.addStretch()
        lay.addLayout(mood_row)

        # Scroll area for info sections
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        scroll_inner = QWidget()
        scroll_lay = QVBoxLayout(scroll_inner)
        scroll_lay.setContentsMargins(0, 0, 0, 0)
        scroll_lay.setSpacing(8)

        self._preferences_section = _InfoSection("Known Preferences")
        self._boundaries_section = _InfoSection("Boundaries")
        self._commitments_section = _InfoSection("Open Commitments")
        self._threads_section = _InfoSection("Active Threads")

        for section in (
            self._preferences_section, self._boundaries_section,
            self._commitments_section, self._threads_section,
        ):
            scroll_lay.addWidget(section)

        scroll_lay.addStretch()
        scroll.setWidget(scroll_inner)
        lay.addWidget(scroll, 1)

        # Subscribe
        self.bus.subscribe("persona_updated", self._on_persona_updated)
        self.bus.subscribe("persona_drift_detected", self._on_drift)
        self.bus.subscribe("intent_updated", self._on_intent_updated)
        self.bus.subscribe("continuity_updated", self._on_continuity_updated)
        self.bus.subscribe("emotion_state_changed", self._on_emotion)

    def _on_persona_updated(self, data: dict) -> None:
        baseline = data.get("baseline", [])
        dims = [
            "formality", "warmth", "verbosity", "humor",
            "assertiveness", "empathy", "creativity", "patience",
        ]
        for i, dim in enumerate(dims):
            if i < len(baseline):
                bar = self._persona_bars.get(dim)
                if bar:
                    bar.set_value(baseline[i])

        drift = data.get("drift", 0.0)
        self._update_drift(drift)

    def _on_drift(self, data: dict) -> None:
        drift = data.get("drift", 0.0)
        self._update_drift(drift)

    def _update_drift(self, drift: float) -> None:
        self._drift_label.setText(f"{drift:.3f}")
        if drift > 0.35:
            self._drift_label.setStyleSheet("color:#ef4444; font-size:12px; font-weight:700;")
            self._drift_status.setText("DRIFTING")
            self._drift_status.setStyleSheet(
                "color:#ef4444; background:#450a0a; padding:2px 8px; "
                "border-radius:4px; font-size:10px;"
            )
        elif drift > 0.2:
            self._drift_label.setStyleSheet("color:#f59e0b; font-size:12px; font-weight:700;")
            self._drift_status.setText("Warning")
            self._drift_status.setStyleSheet(
                "color:#f59e0b; background:#422006; padding:2px 8px; "
                "border-radius:4px; font-size:10px;"
            )
        else:
            self._drift_label.setStyleSheet("color:#22c55e; font-size:12px; font-weight:700;")
            self._drift_status.setText("Stable")
            self._drift_status.setStyleSheet(
                "color:#22c55e; background:#052e16; padding:2px 8px; "
                "border-radius:4px; font-size:10px;"
            )

    def _on_intent_updated(self, data: dict) -> None:
        by_type = data.get("by_type", {})
        prefs = by_type.get("preference", 0)
        boundaries = by_type.get("boundary", 0)
        goals = by_type.get("goal", 0)

        items = []
        if prefs:
            items.append(f"{prefs} stored preference(s)")
        if goals:
            items.append(f"{goals} active goal(s)")
        self._preferences_section.set_items(items or ["None tracked"])

        if boundaries:
            self._boundaries_section.set_items([f"{boundaries} boundary rule(s) active"])
        else:
            self._boundaries_section.set_items(["None set"])

    def _on_continuity_updated(self, data: dict) -> None:
        threads = data.get("open_threads", 0)
        commits = data.get("pending_commitments", 0)

        self._threads_section.set_items(
            [f"{threads} open thread(s)"] if threads else ["No active threads"]
        )
        self._commitments_section.set_items(
            [f"{commits} pending commitment(s)"] if commits else ["No open commitments"]
        )

    def _on_emotion(self, data: dict) -> None:
        mood = data.get("mood", "neutral")
        self._mood_label.setText(mood)
