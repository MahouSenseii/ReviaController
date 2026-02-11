"""
Reusable low-level UI widgets.

Each class has a single responsibility and communicates via Qt signals
or the EventBus — never by reaching into other widgets' internals.
"""

from __future__ import annotations

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QSize,
    Qt,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QAbstractButton,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


# ======================================================================
# Toggle Switch
# ======================================================================

class ToggleSwitch(QAbstractButton):
    """Animated on/off toggle (green ↔ red)."""

    def __init__(self, checked: bool = True, width: int = 44, height: int = 22):
        super().__init__()
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._w = width
        self._h = height
        self.setFixedSize(self._w, self._h)

        self._pos = 1.0 if checked else 0.0

        self._anim = QPropertyAnimation(self, b"pos", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.toggled.connect(self._on_toggled)
        self.setObjectName("ToggleSwitch")

    def sizeHint(self) -> QSize:
        return QSize(self._w, self._h)

    def _on_toggled(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self.isChecked())
            e.accept()
            return
        super().mousePressEvent(e)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        r = self.rect()
        radius = r.height() / 2.0

        if self.isChecked():
            track = QColor("#33d17a")
            border = QColor("#2bd46f")
        else:
            track = QColor("#ef476f")
            border = QColor("#ff5c7a")

        p.setPen(border)
        p.setBrush(track)
        p.drawRoundedRect(r.adjusted(1, 1, -1, -1), radius, radius)

        knob_d = r.height() - 6
        x_min = 3
        x_max = r.width() - knob_d - 3
        x = int(x_min + (x_max - x_min) * self._pos)
        y = 3

        p.setPen(QColor(0, 0, 0, 80))
        p.setBrush(QColor("#0f141c"))
        p.drawEllipse(x, y, knob_d, knob_d)
        p.end()

    # animated property
    def getPos(self) -> float:
        return self._pos

    def setPos(self, v: float) -> None:
        self._pos = float(v)
        self.update()

    pos = pyqtProperty(float, fget=getPos, fset=setPos)


# ======================================================================
# Status Dot
# ======================================================================

class StatusDot(QWidget):
    """Small glowing status indicator: ``on`` / ``warn`` / ``off``."""

    _COLORS = {
        "on":   "#33d17a",
        "warn": "#f9c74f",
        "off":  "#ef476f",
    }

    def __init__(self, status: str = "on", size: int = 10):
        super().__init__()
        self._size = size
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._glow = QGraphicsDropShadowEffect(self)
        self._glow.setBlurRadius(18)
        self._glow.setOffset(0, 0)
        self.setGraphicsEffect(self._glow)

        self.set_status(status)

    def set_status(self, status: str) -> None:
        status = status.lower().strip()
        if status not in self._COLORS:
            status = "off"
        self.setProperty("status", status)
        self._glow.setColor(QColor(self._COLORS[status]))
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self._size, self._size)


# ======================================================================
# Pill (status card with optional toggle)
# ======================================================================

class Pill(QFrame):
    """
    Compact status row: optional dot + title/subtitle + optional toggle.

    Signals
    -------
    toggled(bool)
        Emitted when the toggle changes state.
    """
    toggled = pyqtSignal(bool)

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        status: str | None = None,
        toggle: bool = False,
        checked: bool = True,
    ):
        super().__init__()
        self.setObjectName("Pill")

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)

        self.status_dot: StatusDot | None = None
        if status is not None:
            self.status_dot = StatusDot(status, size=10)
            self.status_dot.setObjectName("StatusDot")
            row.addWidget(self.status_dot, alignment=Qt.AlignmentFlag.AlignVCenter)

        text_col = QWidget()
        col = QVBoxLayout(text_col)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        self._title = QLabel(title)
        self._title.setObjectName("PillTitle")
        self._subtitle = QLabel(subtitle)
        self._subtitle.setObjectName("PillSub")
        col.addWidget(self._title)
        col.addWidget(self._subtitle)
        row.addWidget(text_col, 1)

        self.toggle_switch: ToggleSwitch | None = None
        if toggle:
            self.toggle_switch = ToggleSwitch(checked=checked)
            self.toggle_switch.toggled.connect(self.toggled.emit)
            row.addWidget(self.toggle_switch, alignment=Qt.AlignmentFlag.AlignVCenter)

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def set_subtitle(self, text: str) -> None:
        self._subtitle.setText(text)

    def set_status(self, status: str) -> None:
        if self.status_dot:
            self.status_dot.set_status(status)


# ======================================================================
# Section Label
# ======================================================================

class SectionLabel(QLabel):
    """Styled heading used to separate groups of controls."""

    def __init__(self, text: str):
        super().__init__(text)
        self.setObjectName("SectionLabel")


# ======================================================================
# Ghost Panel (dashed placeholder)
# ======================================================================

class GhostPanel(QFrame):
    """Dashed-border placeholder panel."""

    def __init__(self, text: str, height: int = 120):
        super().__init__()
        self.setObjectName("GhostPanel")
        self.setMinimumHeight(height)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)

        self._label = QLabel(text)
        self._label.setObjectName("GhostText")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._label)

    def set_text(self, text: str) -> None:
        self._label.setText(text)


# ======================================================================
# Panel helpers (thin wrappers kept for convenience)
# ======================================================================

def make_panel(title: str | None = None, title_object: str = "PanelTitle") -> QFrame:
    """Create a standard dark-themed panel frame with optional title."""
    frame = QFrame()
    frame.setObjectName("Panel")
    frame.setFrameShape(QFrame.Shape.NoFrame)

    outer = QVBoxLayout(frame)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(8)

    if title:
        t = QLabel(title)
        t.setObjectName(title_object)
        outer.addWidget(t)

    inner = QFrame()
    inner.setObjectName("PanelInner")
    inner_lay = QVBoxLayout(inner)
    inner_lay.setContentsMargins(0, 0, 0, 0)
    inner_lay.setSpacing(0)

    outer.addWidget(inner)
    return frame


def panel_inner(p: QFrame) -> QFrame:
    """Retrieve the ``PanelInner`` child from a panel created by ``make_panel``."""
    return p.findChild(QFrame, "PanelInner")
