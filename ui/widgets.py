# ui/widgets.py

from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QWidget, QHBoxLayout
from PyQt6.QtCore import Qt, QSize, pyqtProperty, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect, QCheckBox, QAbstractButton


def panel(title: str | None = None, title_object: str = "PanelTitle") -> QFrame:
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
    # panel() always adds PanelInner as the last widget in the outer layout
    return p.findChild(QFrame, "PanelInner")

def ghost_panel(text: str, height: int = 120) -> QFrame:
    box = QFrame()
    box.setObjectName("GhostPanel")
    box.setMinimumHeight(height)

    lay = QVBoxLayout(box)
    lay.setContentsMargins(10, 10, 10, 10)

    lab = QLabel(text)
    lab.setObjectName("GhostText")
    lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(lab)

    return box


def section_label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("SectionLabel")
    return lab

class ToggleSwitch(QAbstractButton):
    def __init__(self, checked: bool = True, width: int = 44, height: int = 22):
        super().__init__()
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._w = width
        self._h = height
        self.setFixedSize(self._w, self._h)

        # knob position: 0.0 (left/off) -> 1.0 (right/on)
        self._pos = 1.0 if checked else 0.0

        self._anim = QPropertyAnimation(self, b"pos", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        # When toggled, animate knob
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

        # Track color: green when ON, red when OFF
        if self.isChecked():
            track = QColor("#33d17a")
            border = QColor("#2bd46f")
        else:
            track = QColor("#ef476f")
            border = QColor("#ff5c7a")

        # Draw track
        p.setPen(border)
        p.setBrush(track)
        p.drawRoundedRect(r.adjusted(1, 1, -1, -1), radius, radius)

        # Knob
        knob_d = r.height() - 6
        x_min = 3
        x_max = r.width() - knob_d - 3
        x = int(x_min + (x_max - x_min) * self._pos)
        y = 3

        p.setPen(QColor(0, 0, 0, 80))
        p.setBrush(QColor("#0f141c"))  # matches your background vibe
        p.drawEllipse(x, y, knob_d, knob_d)

        p.end()

    # Animated property
    def getPos(self) -> float:
        return self._pos

    def setPos(self, v: float) -> None:
        self._pos = float(v)
        self.update()

    pos = pyqtProperty(float, fget=getPos, fset=setPos)


def pill(title: str, sub: str, status: str | None = None, toggle: bool = False, checked: bool = True) -> QFrame:
    p = QFrame()
    p.setObjectName("Pill")

    row = QHBoxLayout(p)
    row.setContentsMargins(10, 8, 10, 8)
    row.setSpacing(10)

    if status is not None:
        dot = StatusDot(status, size=10)
        dot.setObjectName("StatusDot")
        row.addWidget(dot, alignment=Qt.AlignmentFlag.AlignVCenter)

    text_col = QWidget()
    lay = QVBoxLayout(text_col)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)

    t = QLabel(title); t.setObjectName("PillTitle")
    s = QLabel(sub);   s.setObjectName("PillSub")
    lay.addWidget(t); lay.addWidget(s)

    row.addWidget(text_col, 1)

    toggle_widget = None
    if toggle:
        toggle_widget = ToggleSwitch(checked=checked)
        row.addWidget(toggle_widget, alignment=Qt.AlignmentFlag.AlignVCenter)

    # Handy references
    p.toggle = toggle_widget
    p.status_dot = dot if status is not None else None

    return p



def right_tab_placeholder(text: str) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(40, 40, 40, 40)

    lay.addStretch(1)  # push down from top

    ghost = ghost_panel(text, height=260)
    ghost.setMaximumWidth(900)  # prevents it from stretching too wide

    center_layout = QHBoxLayout()
    center_layout.addStretch(1)
    center_layout.addWidget(ghost)
    center_layout.addStretch(1)

    lay.addLayout(center_layout)

    lay.addStretch(1)
    return w

class StatusDot(QWidget):
    """
    Small glowing status indicator.
    status: "on" (green), "warn" (yellow), "off" (red)
    """
    def __init__(self, status: str = "on", size: int = 10):
        super().__init__()
        self._size = size
        self.setFixedSize(size, size)

        # IMPORTANT: ensure stylesheet background is actually painted
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        # Real glow (Qt-supported)
        self._glow = QGraphicsDropShadowEffect(self)
        self._glow.setBlurRadius(18)
        self._glow.setOffset(0, 0)
        self.setGraphicsEffect(self._glow)

        self.set_status(status)

    def set_status(self, status: str) -> None:
        status = status.lower().strip()
        if status not in ("on", "warn", "off"):
            status = "off"

        self.setProperty("status", status)

        # Match glow color to status
        if status == "on":
            self._glow.setColor(QColor("#33d17a"))
        elif status == "warn":
            self._glow.setColor(QColor("#f9c74f"))
        else:
            self._glow.setColor(QColor("#ef476f"))

        # Re-apply style so Qt updates dynamic properties
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self._size, self._size)
