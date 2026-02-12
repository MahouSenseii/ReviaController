"""
Filters tab — per-category safety toggles and severity thresholds.

Each of the 8 safety categories gets:
* An on/off toggle (``Pill`` widget)
* A severity threshold slider (0–5)

Publishes ``safety_config_changed`` events so the ``SafetyFilterEngine``
can adjust in real time.  Also persists to ``Config`` under the
``safety`` namespace.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.safety.categories import CATEGORIES, CATEGORY_ORDER
from ui.widgets import Pill

from .base_tab import BaseTab


class FiltersTab(BaseTab):

    def _build(self) -> None:
        lay = self._layout

        lay.addWidget(self._heading("Safety Filters"))

        hint = QLabel(
            "Toggle categories on/off and set severity thresholds (0–5). "
            "Messages scoring at or above the threshold are rewritten or blocked."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8fa6c3; font-size:11px; margin-bottom:4px;")
        lay.addWidget(hint)

        # ── Per-category controls ─────────────────────────────
        self._toggles: dict[str, Pill] = {}
        self._sliders: dict[str, QSlider] = {}
        self._level_labels: dict[str, QLabel] = {}

        for slug in CATEGORY_ORDER:
            cat = CATEGORIES[slug]

            # Load saved or default
            enabled = self.config.get(f"safety.{slug}.enabled", cat.enabled)
            threshold = self.config.get(f"safety.{slug}.threshold", cat.default_threshold)

            # Container
            row = QWidget()
            row_lay = QVBoxLayout(row)
            row_lay.setContentsMargins(0, 4, 0, 4)
            row_lay.setSpacing(4)

            # Toggle pill
            subtitle = f"Threshold: {threshold}" if enabled else "Disabled"
            pill = Pill(
                cat.name,
                subtitle,
                toggle=True,
                checked=bool(enabled),
            )
            pill.toggled.connect(lambda on, s=slug: self._on_toggle(s, on))
            self._toggles[slug] = pill
            row_lay.addWidget(pill)

            # Threshold slider row
            slider_row = QWidget()
            sh = QHBoxLayout(slider_row)
            sh.setContentsMargins(12, 0, 4, 0)
            sh.setSpacing(8)

            lbl = QLabel("Threshold:")
            lbl.setFixedWidth(70)
            lbl.setStyleSheet("color:#8fa6c3; font-size:11px;")
            sh.addWidget(lbl)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setObjectName("SettingsSlider")
            slider.setRange(0, 5)
            slider.setValue(int(threshold))
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(1)
            slider.valueChanged.connect(lambda v, s=slug: self._on_threshold(s, v))
            self._sliders[slug] = slider
            sh.addWidget(slider, 1)

            val_lbl = QLabel(str(int(threshold)))
            val_lbl.setFixedWidth(20)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val_lbl.setStyleSheet("color:#8fc9ff; font-weight:700; font-size:12px;")
            self._level_labels[slug] = val_lbl
            sh.addWidget(val_lbl)

            row_lay.addWidget(slider_row)

            # Hide slider if disabled
            slider_row.setVisible(bool(enabled))
            pill._slider_row = slider_row  # stash for toggle handler

            lay.addWidget(row)

    # ── Handlers ─────────────────────────────────────────────

    def _on_toggle(self, slug: str, enabled: bool) -> None:
        self.config.set(f"safety.{slug}.enabled", enabled)
        pill = self._toggles[slug]

        threshold = self._sliders[slug].value()
        pill.set_subtitle(f"Threshold: {threshold}" if enabled else "Disabled")

        # Show/hide slider row
        slider_row = getattr(pill, "_slider_row", None)
        if slider_row is not None:
            slider_row.setVisible(enabled)

        self.bus.publish("safety_config_changed", {
            "slug": slug,
            "enabled": enabled,
        })

    def _on_threshold(self, slug: str, value: int) -> None:
        self.config.set(f"safety.{slug}.threshold", value)
        self._level_labels[slug].setText(str(value))

        pill = self._toggles[slug]
        pill.set_subtitle(f"Threshold: {value}")

        self.bus.publish("safety_config_changed", {
            "slug": slug,
            "threshold": value,
        })
