"""
ui/av_indicators.py — AV Status Indicator Widgets

Provides glowing status indicators for STT, TTS, and Vision states.
Each indicator shows the component name, current state, and a colour-coded glow.

Colour scheme:
  - Listening:       blue   (#4a9eff) glow
  - Processing:      yellow (#f9c74f) pulse
  - Speaking:        green  (#33d17a) glow
  - Vision Active:   purple (#b07aff) glow
  - Vision Analyzing:white  (#e0e0e0) glow
  - Error:           red    (#ef476f) glow
  - Idle:            dim    (#3a4a60) no glow
  - Muted:           orange (#ff9f43) dim
"""

from __future__ import annotations

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QSize,
    Qt,
    QTimer,
    pyqtProperty,
)
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from core.events import EventBus


# ── Colour Constants ─────────────────────────────────────────

_COLOURS = {
    "listening":  "#4a9eff",  # Blue
    "processing": "#f9c74f",  # Yellow
    "speaking":   "#33d17a",  # Green
    "active":     "#b07aff",  # Purple
    "analyzing":  "#e0e0e0",  # White
    "error":      "#ef476f",  # Red
    "idle":       "#3a4a60",  # Dim grey
    "muted":      "#ff9f43",  # Orange
}


# ── Glow Dot ─────────────────────────────────────────────────

class GlowDot(QWidget):
    """
    A circular dot with animated glow effect.
    Supports pulsing animation for 'processing' state.
    """

    def __init__(self, size: int = 12, parent: QWidget | None = None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size + 8, size + 8)  # Extra space for glow

        self._colour = QColor(_COLOURS["idle"])
        self._glow_radius = 0.0
        self._max_glow = 20.0

        # Glow effect
        self._glow_effect = QGraphicsDropShadowEffect(self)
        self._glow_effect.setOffset(0, 0)
        self._glow_effect.setBlurRadius(0)
        self._glow_effect.setColor(self._colour)
        self.setGraphicsEffect(self._glow_effect)

        # Pulse animation
        self._pulse_anim = QPropertyAnimation(self, b"glowRadius", self)
        self._pulse_anim.setDuration(800)
        self._pulse_anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._pulse_anim.setStartValue(8.0)
        self._pulse_anim.setEndValue(22.0)
        self._pulse_anim.setLoopCount(-1)  # Infinite

        self._pulsing = False

    def set_colour(self, colour: str) -> None:
        """Set the dot and glow colour."""
        self._colour = QColor(colour)
        self._glow_effect.setColor(self._colour)
        self.update()

    def set_glow(self, on: bool) -> None:
        """Turn steady glow on/off."""
        if on:
            self._glow_effect.setBlurRadius(self._max_glow)
        else:
            self._glow_effect.setBlurRadius(0)
        self.stop_pulse()

    def start_pulse(self) -> None:
        """Start pulsing animation."""
        if not self._pulsing:
            self._pulsing = True
            self._pulse_anim.start()

    def stop_pulse(self) -> None:
        """Stop pulsing animation."""
        if self._pulsing:
            self._pulsing = False
            self._pulse_anim.stop()

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        cx = self.width() // 2
        cy = self.height() // 2
        r = self._size // 2

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self._colour))
        p.drawEllipse(cx - r, cy - r, self._size, self._size)
        p.end()

    # Animated property
    def _get_glow_radius(self) -> float:
        return self._glow_radius

    def _set_glow_radius(self, v: float) -> None:
        self._glow_radius = v
        self._glow_effect.setBlurRadius(v)
        self.update()

    glowRadius = pyqtProperty(float, fget=_get_glow_radius, fset=_set_glow_radius)


# ── AV Status Indicator ─────────────────────────────────────

class AVStatusIndicator(QFrame):
    """
    Single AV component status indicator.

    Shows: [GlowDot] [ComponentName: State]

    Example:
        (o) STT: Listening
        (o) TTS: Speaking
        (o) Vision: Active
    """

    def __init__(
        self,
        component: str,
        initial_state: str = "idle",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("AVIndicator")
        self._component = component

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(8)

        # Glow dot
        self._dot = GlowDot(size=10)
        row.addWidget(self._dot, alignment=Qt.AlignmentFlag.AlignVCenter)

        # Label
        self._label = QLabel(f"{component}: {initial_state.capitalize()}")
        self._label.setObjectName("AVIndicatorLabel")
        self._label.setStyleSheet("color: #c7d3e6; font-size: 12px; font-weight: 600;")
        row.addWidget(self._label, 1)

        self.set_state(initial_state)

    def set_state(self, state: str) -> None:
        """
        Update the indicator to reflect a new state.

        Supported states:
          STT: idle, listening, processing, error
          TTS: idle, speaking, muted, error
          Vision: idle, active, analyzing, error
        """
        state = state.lower()
        self._label.setText(f"{self._component}: {state.capitalize()}")

        colour = _COLOURS.get(state, _COLOURS["idle"])
        self._dot.set_colour(colour)

        # Determine glow behaviour
        if state == "processing" or state == "analyzing":
            self._dot.set_glow(True)
            self._dot.start_pulse()
        elif state in ("listening", "speaking", "active"):
            self._dot.stop_pulse()
            self._dot.set_glow(True)
        elif state == "error":
            self._dot.stop_pulse()
            self._dot.set_glow(True)
        else:
            # idle, muted
            self._dot.stop_pulse()
            self._dot.set_glow(False)


# ── AV Status Bar ────────────────────────────────────────────

class AVStatusBar(QFrame):
    """
    Horizontal bar with STT, TTS, and Vision indicators.
    Subscribes to EventBus for automatic state updates.

    ┌──────────────────────────────────────────────────┐
    │  (o) STT: Listening  (o) TTS: Idle  (o) Vision: Active  │
    └──────────────────────────────────────────────────┘
    """

    def __init__(self, event_bus: EventBus, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("AVStatusBar")
        self.setFixedHeight(40)
        self.bus = event_bus

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 4, 12, 4)
        row.setSpacing(16)

        # Create indicators
        self._stt_indicator = AVStatusIndicator("STT", "idle")
        self._tts_indicator = AVStatusIndicator("TTS", "idle")
        self._vision_indicator = AVStatusIndicator("Vision", "idle")

        row.addWidget(self._stt_indicator)
        row.addWidget(self._tts_indicator)
        row.addWidget(self._vision_indicator)
        row.addStretch(1)

        # Subscribe to state events
        self.bus.subscribe("stt_state_changed", self._on_stt_state)
        self.bus.subscribe("tts_state_changed", self._on_tts_state)
        self.bus.subscribe("vision_state_changed", self._on_vision_state)

    @property
    def stt_indicator(self) -> AVStatusIndicator:
        return self._stt_indicator

    @property
    def tts_indicator(self) -> AVStatusIndicator:
        return self._tts_indicator

    @property
    def vision_indicator(self) -> AVStatusIndicator:
        return self._vision_indicator

    def _on_stt_state(self, data: dict) -> None:
        state = data.get("state", "idle")
        self._stt_indicator.set_state(state)

    def _on_tts_state(self, data: dict) -> None:
        state = data.get("state", "idle")
        self._tts_indicator.set_state(state)

    def _on_vision_state(self, data: dict) -> None:
        state = data.get("state", "idle")
        self._vision_indicator.set_state(state)
