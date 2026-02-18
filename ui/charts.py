"""
Reusable dark-themed live line chart built on PyQt6-Charts.

Provides ``LiveChart`` — a rolling time-series chart that matches
the Revia dark UI.  Add named series with colours, push data points,
and the chart scrolls automatically.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

from PyQt6.QtCharts import (
    QChart,
    QChartView,
    QLineSeries,
    QValueAxis,
)
from PyQt6.QtCore import Qt, QMargins, QPointF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QVBoxLayout, QWidget


# ── Colours matching the dark theme ──────────────────────────

_BG        = QColor("#0a0f18")
_GRID      = QColor("#1a2535")
_AXIS_LINE = QColor("#263246")
_LABEL     = QColor("#8fa6c3")


class LiveChart(QWidget):
    """
    A compact, dark-themed rolling line chart.

    Parameters
    ----------
    max_points : int
        Number of data points visible on the X axis.
    y_min, y_max : float
        Range for the Y axis.
    y_label : str
        Label shown on the Y axis.
    height : int
        Fixed pixel height of the chart widget.
    show_legend : bool
        Whether to display the series legend.
    """

    def __init__(
        self,
        max_points: int = 30,
        y_min: float = 0.0,
        y_max: float = 100.0,
        y_label: str = "%",
        height: int = 180,
        show_legend: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._max = max_points
        self._tick = 0

        # ── Data stores (series name → deque of values) ──────
        self._data: Dict[str, deque] = {}
        self._series: Dict[str, QLineSeries] = {}

        # ── Chart ─────────────────────────────────────────────
        self._chart = QChart()
        self._chart.setBackgroundBrush(_BG)
        self._chart.setPlotAreaBackgroundBrush(_BG)
        self._chart.setPlotAreaBackgroundVisible(True)
        self._chart.setMargins(QMargins(8, 8, 8, 8))

        # Legend
        if show_legend:
            legend = self._chart.legend()
            legend.setVisible(True)
            legend.setAlignment(Qt.AlignmentFlag.AlignBottom)
            legend.setLabelColor(_LABEL)
            legend.setFont(QFont("Segoe UI", 10))
            legend.setMarkerShape(legend.MarkerShape.MarkerShapeCircle)
        else:
            self._chart.legend().setVisible(False)

        # ── Axes ──────────────────────────────────────────────
        self._x_axis = QValueAxis()
        self._x_axis.setRange(0, max_points)
        self._x_axis.setLabelsVisible(False)
        self._x_axis.setGridLineVisible(True)
        self._x_axis.setGridLineColor(_GRID)
        self._x_axis.setLinePenColor(_AXIS_LINE)
        self._x_axis.setTickCount(2)
        self._chart.addAxis(self._x_axis, Qt.AlignmentFlag.AlignBottom)

        self._y_axis = QValueAxis()
        self._y_axis.setRange(y_min, y_max)
        self._y_axis.setTitleText(y_label)
        self._y_axis.setTitleBrush(_LABEL)
        self._y_axis.setTitleFont(QFont("Segoe UI", 10))
        self._y_axis.setLabelsColor(_LABEL)
        self._y_axis.setLabelsFont(QFont("Segoe UI", 9))
        self._y_axis.setGridLineVisible(True)
        self._y_axis.setGridLineColor(_GRID)
        self._y_axis.setLinePenColor(_AXIS_LINE)
        self._y_axis.setTickCount(5)
        self._chart.addAxis(self._y_axis, Qt.AlignmentFlag.AlignLeft)

        # ── View ──────────────────────────────────────────────
        self._view = QChartView(self._chart)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setStyleSheet("background: transparent; border: none;")
        self._view.setFixedHeight(height)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._view)

    # ── Public API ────────────────────────────────────────────

    def add_series(self, name: str, colour: str, width: float = 2.0) -> None:
        """Register a named data series with a hex colour."""
        if name in self._series:
            return

        series = QLineSeries()
        series.setName(name)
        pen = QPen(QColor(colour))
        pen.setWidthF(width)
        series.setPen(pen)

        self._chart.addSeries(series)
        series.attachAxis(self._x_axis)
        series.attachAxis(self._y_axis)

        self._series[name] = series
        self._data[name] = deque(maxlen=self._max)

    def push(self, values: Dict[str, float]) -> None:
        """
        Push one data point per series.

        ``values`` maps series names to their current values.
        Missing series are silently skipped.
        """
        self._tick += 1

        for name, val in values.items():
            if name not in self._data:
                continue
            self._data[name].append(val)

        # Rebuild all series points
        for name, series in self._series.items():
            buf = self._data[name]
            points = [QPointF(float(i), v) for i, v in enumerate(buf)]
            series.replace(points)

        # Scroll X axis
        n = max(len(d) for d in self._data.values()) if self._data else 0
        if n > self._max:
            self._x_axis.setRange(n - self._max, n)
        else:
            self._x_axis.setRange(0, self._max)

    def clear_all(self) -> None:
        """Clear all data points from every series."""
        self._tick = 0
        for name in self._data:
            self._data[name].clear()
        for series in self._series.values():
            series.clear()
