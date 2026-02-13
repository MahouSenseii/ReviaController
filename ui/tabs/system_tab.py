"""
System tab — live hardware monitoring chart.

Subscribes to ``runtime_stats`` and plots GPU, CPU, VRAM, and RAM
usage over time as a rolling line chart.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel

from ui.charts import LiveChart

from .base_tab import BaseTab


# Series colours matching the dark theme
_SERIES = {
    "GPU":  "#f9c74f",   # gold
    "CPU":  "#43aa8b",   # teal
    "VRAM": "#f4845f",   # coral
    "RAM":  "#577590",   # slate blue
}


class SystemTab(BaseTab):

    def _build(self) -> None:
        lay = self._layout

        lay.addWidget(self._heading("Hardware Monitor"))

        # Rolling line chart
        self._hw_chart = LiveChart(
            max_points=60,
            y_min=0.0,
            y_max=100.0,
            y_label="%",
            height=240,
            show_legend=True,
        )
        for name, colour in _SERIES.items():
            self._hw_chart.add_series(name, colour, width=2.0)
        lay.addWidget(self._hw_chart)

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
        values: dict[str, float] = {}
        for key in ("GPU", "CPU", "VRAM", "RAM"):
            if key in data:
                values[key] = _pct(data[key])

        if values:
            self._hw_chart.push(values)

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
