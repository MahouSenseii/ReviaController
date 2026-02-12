"""
Memory inspector tab — timeline view of memories with metadata,
source, confidence, importance, TTL, last-used, and one-click
actions (edit, pin, archive, delete).

Subscribes to ``memory_updated`` events and refreshes automatically.
Publishes ``memory_command`` events for actions.
"""

from __future__ import annotations

import time

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .base_tab import BaseTab


class _MemoryCard(QFrame):
    """Single memory entry card in the timeline."""

    def __init__(self, data: dict, bus):
        super().__init__()
        self._bus = bus
        self._entry_id = data.get("id", "")
        self.setObjectName("MemoryCard")
        self.setStyleSheet(
            "QFrame#MemoryCard { background:#111827; border:1px solid #2a3b55; "
            "border-radius:8px; padding:8px; margin:2px 0; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Header row: type badge + source + time
        header = QHBoxLayout()
        entry_type = data.get("entry_type", "event")
        type_colors = {
            "conversation": "#3b82f6", "fact": "#22c55e",
            "event": "#f59e0b", "observation": "#a855f7",
            "summary": "#06b6d4",
        }
        badge_color = type_colors.get(entry_type, "#6b7280")
        type_lbl = QLabel(entry_type.upper())
        type_lbl.setStyleSheet(
            f"background:{badge_color}; color:#fff; padding:1px 6px; "
            f"border-radius:4px; font-size:10px; font-weight:700;"
        )
        type_lbl.setFixedHeight(18)
        header.addWidget(type_lbl)

        source_lbl = QLabel(data.get("source", "unknown"))
        source_lbl.setStyleSheet("color:#9ca3af; font-size:11px;")
        header.addWidget(source_lbl)

        header.addStretch()

        # Importance indicator
        importance = data.get("importance", 0.5)
        imp_color = "#22c55e" if importance > 0.7 else "#f59e0b" if importance > 0.4 else "#6b7280"
        imp_lbl = QLabel(f"imp:{importance:.2f}")
        imp_lbl.setStyleSheet(f"color:{imp_color}; font-size:10px;")
        header.addWidget(imp_lbl)

        # Emotion indicators
        valence = data.get("emotion_valence")
        arousal = data.get("emotion_arousal")
        if valence is not None:
            emo_text = f"v:{valence:+.1f}"
            if arousal is not None:
                emo_text += f" a:{arousal:.1f}"
            emo_lbl = QLabel(emo_text)
            emo_lbl.setStyleSheet("color:#c084fc; font-size:10px;")
            header.addWidget(emo_lbl)

        # Time
        created = data.get("created_at", 0)
        age = _format_age(created)
        time_lbl = QLabel(age)
        time_lbl.setStyleSheet("color:#6b7280; font-size:10px;")
        header.addWidget(time_lbl)

        layout.addLayout(header)

        # Content
        content = data.get("content", "")
        content_lbl = QLabel(content[:200] + ("..." if len(content) > 200 else ""))
        content_lbl.setWordWrap(True)
        content_lbl.setStyleSheet("color:#e5e7eb; font-size:12px;")
        layout.addWidget(content_lbl)

        # Tags row
        tags = data.get("tags", [])
        if tags:
            tags_lbl = QLabel(" ".join(f"#{t}" for t in tags[:5]))
            tags_lbl.setStyleSheet("color:#60a5fa; font-size:10px;")
            layout.addWidget(tags_lbl)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        for label, action in [("Promote", "promote"), ("Delete", "forget")]:
            btn = QPushButton(label)
            btn.setFixedHeight(22)
            btn.setStyleSheet(
                "QPushButton { background:#1e293b; color:#94a3b8; border:1px solid #334155; "
                "border-radius:4px; padding:2px 8px; font-size:10px; }"
                "QPushButton:hover { background:#334155; color:#e2e8f0; }"
            )
            btn.clicked.connect(lambda checked, a=action: self._action(a))
            btn_row.addWidget(btn)

        layout.addLayout(btn_row)

    def _action(self, action: str) -> None:
        if self._bus:
            self._bus.publish("memory_command", {
                "action": action,
                "entry_id": self._entry_id,
            })


class MemoryTab(BaseTab):
    """Memory inspector panel with timeline and filtering."""

    def _build(self) -> None:
        lay = self._layout

        lay.addWidget(self._heading("Memory Inspector"))

        # Filter controls
        filter_row = QWidget()
        flay = QHBoxLayout(filter_row)
        flay.setContentsMargins(0, 0, 0, 0)
        flay.setSpacing(8)

        flay.addWidget(QLabel("Store:"))
        self._store_combo = QComboBox()
        self._store_combo.addItems(["All", "Short-term", "Long-term"])
        self._store_combo.setObjectName("SettingsCombo")
        self._store_combo.setFixedWidth(100)
        flay.addWidget(self._store_combo)

        flay.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox()
        self._type_combo.addItems(["All", "conversation", "fact", "event", "observation", "summary"])
        self._type_combo.setObjectName("SettingsCombo")
        self._type_combo.setFixedWidth(100)
        flay.addWidget(self._type_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("ModeButton")
        refresh_btn.clicked.connect(self._refresh)
        flay.addWidget(refresh_btn)

        flay.addStretch()
        lay.addWidget(filter_row)

        # Stats summary
        self._stats_label = QLabel("No profile loaded")
        self._stats_label.setStyleSheet("color:#9ca3af; font-size:11px;")
        lay.addWidget(self._stats_label)

        # Scroll area for memory cards
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        self._card_container = QWidget()
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setContentsMargins(0, 0, 0, 0)
        self._card_layout.setSpacing(4)
        self._card_layout.addStretch()
        self._scroll.setWidget(self._card_container)
        lay.addWidget(self._scroll, 1)

        # Consolidation button
        action_row = QHBoxLayout()
        consolidate_btn = QPushButton("Run Consolidation")
        consolidate_btn.setObjectName("ModeButton")
        consolidate_btn.clicked.connect(self._consolidate)
        action_row.addWidget(consolidate_btn)

        new_session_btn = QPushButton("New Session")
        new_session_btn.setObjectName("ModeButton")
        new_session_btn.clicked.connect(self._new_session)
        action_row.addWidget(new_session_btn)

        action_row.addStretch()
        lay.addLayout(action_row)

        # Subscribe to updates
        self.bus.subscribe("memory_updated", self._on_memory_updated)

    def _on_memory_updated(self, data: dict) -> None:
        stats = data.get("stats", {})
        profile = data.get("profile", "?")
        st = stats.get("short_term_count", 0)
        lt = stats.get("long_term_count", 0)
        sid = stats.get("session_id", "?")
        self._stats_label.setText(
            f"Profile: {profile}  |  ST: {st}  |  LT: {lt}  |  "
            f"Total: {st + lt}  |  Session: {sid[:8]}"
        )

    def _refresh(self) -> None:
        self.bus.publish("memory_inspector_refresh", {
            "store": self._store_combo.currentText().lower(),
            "entry_type": self._type_combo.currentText(),
        })

    def _consolidate(self) -> None:
        self.bus.publish("memory_command", {"action": "consolidate"})

    def _new_session(self) -> None:
        self.bus.publish("memory_command", {"action": "new_session"})


def _format_age(ts: float) -> str:
    if ts <= 0:
        return "unknown"
    age = time.time() - ts
    if age < 60:
        return f"{int(age)}s ago"
    if age < 3600:
        return f"{int(age / 60)}m ago"
    if age < 86400:
        return f"{int(age / 3600)}h ago"
    return f"{int(age / 86400)}d ago"
