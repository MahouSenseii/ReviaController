"""
System tab — live hardware / resource gauges.

Subscribes to ``runtime_stats`` and ``inference_metrics`` and draws
simple progress-bar style meters.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QLabel,
    QProgressBar,
    QWidget,
)

from .base_tab import BaseTab


class _Gauge(QWidget):
    """Labelled progress bar for a single metric."""

    def __init__(self, label: str, unit: str = "%"):
        super().__init__()
        self._unit = unit

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._label = QLabel(label)
        self._label.setFixedWidth(60)
        layout.addWidget(self._label, 0, 0)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(14)
        self._bar.setStyleSheet(
            "QProgressBar { background:#0a0f18; border:1px solid #2a3b55; border-radius:7px; }"
            "QProgressBar::chunk { background:#33d17a; border-radius:6px; }"
        )
        layout.addWidget(self._bar, 0, 1)

        self._value_label = QLabel("- " + unit)
        self._value_label.setFixedWidth(60)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._value_label, 0, 2)

    def set_value(self, value: float) -> None:
        clamped = max(0, min(100, int(value)))
        self._bar.setValue(clamped)
        self._value_label.setText(f"{value:.0f} {self._unit}")


class SystemTab(BaseTab):

    def _build(self) -> None:
        lay = self._layout

        lay.addWidget(self._heading("Hardware Monitors"))

        self._gpu_gauge = _Gauge("GPU", "%")
        self._cpu_gauge = _Gauge("CPU", "%")
        self._vram_gauge = _Gauge("VRAM", "%")
        self._ram_gauge = _Gauge("RAM", "%")

        for g in (self._gpu_gauge, self._cpu_gauge, self._vram_gauge, self._ram_gauge):
            lay.addWidget(g)

        lay.addWidget(self._heading("Runtime Info"))

        self._info_label = QLabel(
            "Model: -\nHealth: -\nUptime: -"
        )
        self._info_label.setObjectName("MonoInfo")
        self._info_label.setStyleSheet("font-size:13px;")
        lay.addWidget(self._info_label)

        # Subscribe
        self.bus.subscribe("runtime_stats", self._on_runtime_stats)

    def _on_runtime_stats(self, data: dict) -> None:
        if "GPU" in data:
            self._gpu_gauge.set_value(_pct(data["GPU"]))
        if "CPU" in data:
            self._cpu_gauge.set_value(_pct(data["CPU"]))
        if "VRAM" in data:
            self._vram_gauge.set_value(_pct(data["VRAM"]))
        if "RAM" in data:
            self._ram_gauge.set_value(_pct(data["RAM"]))

        info_lines = []
        if "Model" in data:
            info_lines.append(f"Model: {data['Model']}")
        if "Health" in data:
            info_lines.append(f"Health: {data['Health']}")
        if "Uptime" in data:
            info_lines.append(f"Uptime: {data['Uptime']}")
        if info_lines:
            self._info_label.setText("\n".join(info_lines))


def _pct(raw) -> float:
    """Try to extract a number from strings like '74%' or '74'."""
    s = str(raw).replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0
