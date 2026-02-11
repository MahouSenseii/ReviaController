"""
Base class for every settings tab.

Gives each tab access to the EventBus and Config and provides
small helpers for building labelled rows of input widgets.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config import Config
from core.events import EventBus


class BaseTab(QWidget):
    """
    Abstract tab page.

    Subclasses override ``_build()`` to populate ``self._layout``.
    """

    def __init__(self, event_bus: EventBus, config: Config):
        super().__init__()
        self.bus = event_bus
        self.config = config

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 16, 16, 16)
        self._layout.setSpacing(12)

        self._build()
        self._layout.addStretch(1)

    def _build(self) -> None:
        """Override to populate the tab."""

    # ── Helpers for building labelled input rows ──────────────

    @staticmethod
    def _row(label_text: str, widget: QWidget, stretch: int = 1) -> QWidget:
        """Return an HBox with a label on the left and *widget* on the right."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        lbl = QLabel(label_text)
        lbl.setFixedWidth(120)
        h.addWidget(lbl)
        h.addWidget(widget, stretch)
        return row

    @staticmethod
    def _heading(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("TabHeading")
        lbl.setStyleSheet("font-weight:700; font-size:14px; color:#8fc9ff; margin-top:4px;")
        return lbl

    @staticmethod
    def _make_combo(items: list[str], current: int = 0) -> QComboBox:
        cb = QComboBox()
        cb.addItems(items)
        cb.setCurrentIndex(current)
        cb.setObjectName("SettingsCombo")
        return cb

    @staticmethod
    def _make_line_edit(placeholder: str = "", text: str = "", secret: bool = False) -> QLineEdit:
        le = QLineEdit(text)
        le.setPlaceholderText(placeholder)
        if secret:
            le.setEchoMode(QLineEdit.EchoMode.Password)
        le.setObjectName("SettingsLineEdit")
        return le

    @staticmethod
    def _make_spin(min_val: int = 0, max_val: int = 100, value: int = 0) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(min_val, max_val)
        sb.setValue(value)
        sb.setObjectName("SettingsSpin")
        return sb

    @staticmethod
    def _make_double_spin(
        min_val: float = 0.0, max_val: float = 1.0, value: float = 0.5, step: float = 0.05,
    ) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(min_val, max_val)
        sb.setSingleStep(step)
        sb.setValue(value)
        sb.setObjectName("SettingsDoubleSpin")
        return sb

    @staticmethod
    def _make_slider(min_val: int = 0, max_val: int = 100, value: int = 50) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(min_val, max_val)
        s.setValue(value)
        s.setObjectName("SettingsSlider")
        return s
