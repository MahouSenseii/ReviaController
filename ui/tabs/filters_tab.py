"""
Filters tab — content filtering level and per-category controls.

Publishes ``filter_changed`` so the backend can adjust in real time.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QButtonGroup, QHBoxLayout, QRadioButton, QWidget

from ui.widgets import Pill

from .base_tab import BaseTab


class FiltersTab(BaseTab):

    def _build(self) -> None:
        lay = self._layout

        lay.addWidget(self._heading("Filter Level"))

        # ── Radio group: None / Low / Strict ──────────────────
        radio_row = QWidget()
        rh = QHBoxLayout(radio_row)
        rh.setContentsMargins(0, 0, 0, 0)
        rh.setSpacing(16)

        self._level_group = QButtonGroup(self)
        for idx, label in enumerate(("None", "Low", "Strict")):
            rb = QRadioButton(label)
            rb.setObjectName("FilterRadio")
            self._level_group.addButton(rb, idx)
            rh.addWidget(rb)

        saved = self.config.get("filters.level", "Low")
        mapping = {"None": 0, "Low": 1, "Strict": 2}
        btn = self._level_group.button(mapping.get(saved, 1))
        if btn:
            btn.setChecked(True)

        self._level_group.idToggled.connect(self._on_level)
        lay.addWidget(radio_row)

        # ── Category toggles ─────────────────────────────────
        lay.addWidget(self._heading("Categories"))

        self._categories: dict[str, Pill] = {}
        for cat in ("Violence", "Sexual Content", "Hate Speech", "Self-Harm", "Profanity"):
            enabled = self.config.get(f"filters.{cat}", True)
            p = Pill(cat, "Enabled" if enabled else "Disabled", toggle=True, checked=bool(enabled))
            p.toggled.connect(lambda on, c=cat: self._on_category(c, on))
            self._categories[cat] = p
            lay.addWidget(p)

    def _on_level(self, id_: int, checked: bool) -> None:
        if not checked:
            return
        levels = {0: "None", 1: "Low", 2: "Strict"}
        level = levels.get(id_, "Low")
        self.config.set("filters.level", level)
        self.bus.publish("filter_changed", {"level": level})

    def _on_category(self, category: str, enabled: bool) -> None:
        self.config.set(f"filters.{category}", enabled)
        pill = self._categories[category]
        pill.set_subtitle("Enabled" if enabled else "Disabled")
        self.bus.publish("filter_changed", {"category": category, "enabled": enabled})
