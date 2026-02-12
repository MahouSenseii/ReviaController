"""
System tab — Task-Manager-style live performance graphs.

Polls CPU, RAM, and GPU utilisation with ``psutil`` (and ``nvidia-smi``
for NVIDIA GPUs) and renders scrolling area charts.  A time-range row
lets the user choose how far back the graphs display (10 s – 60 s).

Hardware is auto-detected on startup and shown above the graphs.
"""

from __future__ import annotations

import collections
import platform
import shutil
import subprocess
from typing import Any

import psutil
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .base_tab import BaseTab

# ── Hardware detection helpers ────────────────────────────────────────

_hw_cache: dict[str, str] | None = None


def _detect_hardware() -> dict[str, str]:
    """Return a dict with cpu_name, cpu_cores, ram_total, gpu_name."""
    global _hw_cache
    if _hw_cache is not None:
        return _hw_cache

    info: dict[str, str] = {}

    # CPU
    try:
        brand = platform.processor()
        if not brand or brand == "x86_64":
            # Try /proc/cpuinfo on Linux
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            brand = line.split(":", 1)[1].strip()
                            break
            except OSError:
                brand = brand or "Unknown CPU"
        info["cpu_name"] = brand
    except Exception:
        info["cpu_name"] = "Unknown CPU"

    info["cpu_cores"] = f"{psutil.cpu_count(logical=False) or '?'}C / {psutil.cpu_count(logical=True) or '?'}T"

    # RAM
    try:
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        info["ram_total"] = f"{total_gb:.1f} GB"
    except Exception:
        info["ram_total"] = "Unknown"

    # GPU — try nvidia-smi first
    info["gpu_name"] = _detect_gpu()

    _hw_cache = info
    return info


def _detect_gpu() -> str:
    """Best-effort GPU detection via nvidia-smi, then lspci fallback."""
    # NVIDIA
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                timeout=4,
                text=True,
            ).strip()
            if out:
                return out.splitlines()[0].strip()
        except Exception:
            pass

    # lspci fallback (Linux)
    if shutil.which("lspci"):
        try:
            out = subprocess.check_output(
                ["lspci"], timeout=4, text=True,
            )
            for line in out.splitlines():
                low = line.lower()
                if "vga" in low or "3d" in low or "display" in low:
                    return line.split(":", 2)[-1].strip()
        except Exception:
            pass

    return "Not detected"


def _gpu_utilisation() -> float | None:
    """Return GPU utilisation 0-100 via nvidia-smi, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=3,
            text=True,
        ).strip()
        return float(out.splitlines()[0])
    except Exception:
        return None


# ── Performance graph widget ──────────────────────────────────────────

# Color presets per metric
_GRAPH_COLORS: dict[str, dict[str, str]] = {
    "CPU": {"line": "#33d17a", "fill_top": "#33d17a", "fill_bot": "#0a2618"},
    "RAM": {"line": "#7fb3ff", "fill_top": "#7fb3ff", "fill_bot": "#0a1628"},
    "GPU": {"line": "#f9c74f", "fill_top": "#f9c74f", "fill_bot": "#28200a"},
}

# How often we sample (seconds)
_POLL_INTERVAL_MS = 1000
_MAX_HISTORY = 120  # keep up to 2 min of samples


class PerformanceGraph(QWidget):
    """Single scrolling area-chart, styled like the Windows Task Manager."""

    def __init__(self, label: str, unit: str = "%", parent: QWidget | None = None):
        super().__init__(parent)
        self._label = label
        self._unit = unit
        self._samples: collections.deque[float] = collections.deque(maxlen=_MAX_HISTORY)
        self._window_sec = 60  # default visible window
        self.setMinimumHeight(110)

        colors = _GRAPH_COLORS.get(label, _GRAPH_COLORS["CPU"])
        self._line_color = QColor(colors["line"])
        self._fill_top = QColor(colors["fill_top"])
        self._fill_bot = QColor(colors["fill_bot"])

    # ── Public API ────────────────────────────────────────

    def set_window(self, seconds: int) -> None:
        self._window_sec = seconds
        self.update()

    def add_sample(self, value: float) -> None:
        self._samples.append(max(0.0, min(100.0, value)))
        self.update()

    # ── Painting ──────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        margin_top = 22
        margin_bot = 18
        margin_left = 2
        margin_right = 2
        gx = margin_left
        gy = margin_top
        gw = w - margin_left - margin_right
        gh = h - margin_top - margin_bot
        rect = QRectF(gx, gy, gw, gh)

        # Background
        p.fillRect(self.rect(), QColor("#0a0f18"))

        # Grid
        grid_pen = QPen(QColor("#1a2535"), 1, Qt.PenStyle.SolidLine)
        p.setPen(grid_pen)
        # Horizontal lines (0%, 25%, 50%, 75%, 100%)
        for i in range(5):
            y = gy + gh * i / 4
            p.drawLine(int(gx), int(y), int(gx + gw), int(y))
        # Vertical lines — one per (window/6)
        n_vcols = 6
        for i in range(n_vcols + 1):
            x = gx + gw * i / n_vcols
            p.drawLine(int(x), int(gy), int(x), int(gy + gh))

        # Border
        border_pen = QPen(QColor("#2a3b55"), 1)
        p.setPen(border_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(rect.adjusted(0, 0, -1, -1))

        # Data
        n_points = self._window_sec  # 1 sample/sec
        samples = list(self._samples)
        visible = samples[-n_points:] if len(samples) >= n_points else samples
        if len(visible) < 2:
            self._draw_labels(p, w, h, gy, gh, margin_bot, 0.0)
            p.end()
            return

        # Build path
        count = len(visible)
        step_x = gw / max(n_points - 1, 1)
        x_offset = gw - (count - 1) * step_x  # right-align

        path = QPainterPath()
        first_x = gx + x_offset
        first_y = gy + gh * (1 - visible[0] / 100)
        path.moveTo(first_x, first_y)
        for i in range(1, count):
            px = gx + x_offset + i * step_x
            py = gy + gh * (1 - visible[i] / 100)
            path.lineTo(px, py)

        # Fill under the curve
        fill_path = QPainterPath(path)
        last_x = gx + x_offset + (count - 1) * step_x
        fill_path.lineTo(last_x, gy + gh)
        fill_path.lineTo(first_x, gy + gh)
        fill_path.closeSubpath()

        grad = QLinearGradient(0, gy, 0, gy + gh)
        fill_top = QColor(self._fill_top)
        fill_top.setAlpha(100)
        fill_bot = QColor(self._fill_bot)
        fill_bot.setAlpha(40)
        grad.setColorAt(0, fill_top)
        grad.setColorAt(1, fill_bot)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.setClipRect(rect)
        p.drawPath(fill_path)

        # Line on top
        p.setClipping(True)
        p.setClipRect(rect)
        line_pen = QPen(self._line_color, 2, Qt.PenStyle.SolidLine)
        line_pen.setCosmetic(True)
        p.setPen(line_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.setClipping(False)

        current = visible[-1]
        self._draw_labels(p, w, h, gy, gh, margin_bot, current)
        p.end()

    def _draw_labels(
        self, p: QPainter, w: int, h: int,
        gy: float, gh: float, margin_bot: float, current: float,
    ) -> None:
        # Title (top-left)
        p.setPen(QColor("#d8e1ee"))
        title_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        p.setFont(title_font)
        p.drawText(4, 15, self._label)

        # Current value (top-right)
        val_text = f"{current:.0f}{self._unit}"
        p.setPen(self._line_color)
        val_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        p.setFont(val_font)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(val_text)
        p.drawText(w - tw - 6, 15, val_text)

        # Bottom axis labels
        p.setPen(QColor("#5a7090"))
        axis_font = QFont("Segoe UI", 7)
        p.setFont(axis_font)
        bottom_y = int(gy + gh + margin_bot - 4)
        p.drawText(4, bottom_y, f"{self._window_sec}s ago")
        now_text = "0s"
        fm2 = p.fontMetrics()
        p.drawText(w - fm2.horizontalAdvance(now_text) - 4, bottom_y, now_text)


# ── Time-range button bar ─────────────────────────────────────────────

_TIME_OPTIONS = [10, 20, 30, 40, 50, 60]


class _TimeRangeBar(QWidget):
    """Row of flat buttons: 10 s … 60 s."""

    def __init__(self, on_changed):
        super().__init__()
        self._buttons: list[QPushButton] = []
        self._callback = on_changed

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        lbl = QLabel("Range")
        lbl.setStyleSheet("color:#8fa6c3; font-size:11px; font-weight:700;")
        lbl.setFixedWidth(42)
        lay.addWidget(lbl)

        for sec in _TIME_OPTIONS:
            btn = QPushButton(f"{sec}s")
            btn.setObjectName("ModeButton")
            btn.setCheckable(True)
            btn.setFixedHeight(24)
            btn.setFixedWidth(40)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, s=sec: self._select(s))
            lay.addWidget(btn)
            self._buttons.append(btn)

        lay.addStretch(1)
        # Default 60 s
        self._select(60)

    def _select(self, sec: int) -> None:
        for btn in self._buttons:
            btn.setChecked(btn.text() == f"{sec}s")
        self._callback(sec)


# ── Main tab ──────────────────────────────────────────────────────────


class SystemTab(BaseTab):

    def _build(self) -> None:
        lay = self._layout

        # ── Hardware info ────────────────────────────────────
        lay.addWidget(self._heading("Hardware"))

        hw = _detect_hardware()
        hw_text = (
            f"<b>CPU:</b> {hw['cpu_name']} ({hw['cpu_cores']})<br>"
            f"<b>RAM:</b> {hw['ram_total']}<br>"
            f"<b>GPU:</b> {hw['gpu_name']}"
        )
        self._hw_label = QLabel(hw_text)
        self._hw_label.setWordWrap(True)
        self._hw_label.setStyleSheet(
            "color:#c7d3e6; font-size:11px; padding:4px 0; "
            "background:transparent; border:none;"
        )
        lay.addWidget(self._hw_label)

        # ── Time range selector ──────────────────────────────
        self._time_bar = _TimeRangeBar(self._on_time_changed)
        lay.addWidget(self._time_bar)

        # ── Performance graphs (scrollable) ──────────────────
        lay.addWidget(self._heading("Performance"))

        self._cpu_graph = PerformanceGraph("CPU")
        self._ram_graph = PerformanceGraph("RAM")
        self._gpu_graph = PerformanceGraph("GPU")

        self._graphs = [self._cpu_graph, self._ram_graph, self._gpu_graph]

        graph_container = QWidget()
        gc_lay = QVBoxLayout(graph_container)
        gc_lay.setContentsMargins(0, 0, 0, 0)
        gc_lay.setSpacing(8)
        for g in self._graphs:
            gc_lay.addWidget(g)
        gc_lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(graph_container)
        scroll.setStyleSheet(
            "QScrollArea { border:none; background:transparent; }"
            "QWidget { background:transparent; }"
        )
        lay.addWidget(scroll, 1)

        # ── Polling timer ────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(_POLL_INTERVAL_MS)

        # Also accept runtime_stats events from plugins
        self.bus.subscribe("runtime_stats", self._on_runtime_stats)

    # ── Callbacks ─────────────────────────────────────────

    def _on_time_changed(self, seconds: int) -> None:
        for g in self._graphs:
            g.set_window(seconds)

    def _poll(self) -> None:
        """Sample CPU / RAM from psutil, GPU from nvidia-smi."""
        try:
            cpu = psutil.cpu_percent(interval=0)
            ram = psutil.virtual_memory().percent
        except Exception:
            cpu, ram = 0.0, 0.0

        self._cpu_graph.add_sample(cpu)
        self._ram_graph.add_sample(ram)

        gpu = _gpu_utilisation()
        if gpu is not None:
            self._gpu_graph.add_sample(gpu)
        else:
            self._gpu_graph.add_sample(0.0)

    def _on_runtime_stats(self, data: dict) -> None:
        """Accept stats pushed from plugins / backend."""
        if "CPU" in data:
            self._cpu_graph.add_sample(_pct(data["CPU"]))
        if "RAM" in data:
            self._ram_graph.add_sample(_pct(data["RAM"]))
        if "GPU" in data:
            self._gpu_graph.add_sample(_pct(data["GPU"]))


def _pct(raw: Any) -> float:
    """Try to extract a number from strings like '74%' or '74'."""
    s = str(raw).replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0
