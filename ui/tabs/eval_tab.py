"""
Evaluation dashboard tab — displays recall precision, hallucination
rate, correction latency, persona-drift score, and other quality
metrics.

Subscribes to ``eval_metrics`` events and renders live gauges.
Metrics can be sliced by profile and time window.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from .base_tab import BaseTab


class _MetricCard(QFrame):
    """A single metric display with label, value, and coloured bar."""

    def __init__(self, title: str, unit: str = "", good_direction: str = "high"):
        super().__init__()
        self._good_dir = good_direction  # "high" or "low"
        self.setObjectName("MetricCard")
        self.setStyleSheet(
            "QFrame#MetricCard { background:#111827; border:1px solid #1e293b; "
            "border-radius:8px; padding:6px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(3)

        self._title = QLabel(title)
        self._title.setStyleSheet("color:#9ca3af; font-size:10px;")
        layout.addWidget(self._title)

        self._value_label = QLabel("-")
        self._value_label.setStyleSheet("color:#e5e7eb; font-size:18px; font-weight:700;")
        layout.addWidget(self._value_label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._bar.setStyleSheet(
            "QProgressBar { background:#0a0f18; border:none; border-radius:3px; }"
            "QProgressBar::chunk { background:#22c55e; border-radius:3px; }"
        )
        layout.addWidget(self._bar)

        self._detail = QLabel(unit)
        self._detail.setStyleSheet("color:#6b7280; font-size:9px;")
        layout.addWidget(self._detail)

    def set_value(self, value: float, detail: str = "") -> None:
        self._value_label.setText(f"{value:.1f}%")
        pct = max(0, min(100, int(value)))
        self._bar.setValue(pct)

        # Colour based on good direction
        if self._good_dir == "high":
            if value >= 80:
                color = "#22c55e"
            elif value >= 50:
                color = "#f59e0b"
            else:
                color = "#ef4444"
        else:  # lower is better
            if value <= 5:
                color = "#22c55e"
            elif value <= 15:
                color = "#f59e0b"
            else:
                color = "#ef4444"

        self._bar.setStyleSheet(
            f"QProgressBar {{ background:#0a0f18; border:none; border-radius:3px; }}"
            f"QProgressBar::chunk {{ background:{color}; border-radius:3px; }}"
        )
        self._value_label.setStyleSheet(f"color:{color}; font-size:18px; font-weight:700;")

        if detail:
            self._detail.setText(detail)


class EvalTab(BaseTab):
    """Evaluation dashboard with quality metrics."""

    def _build(self) -> None:
        lay = self._layout

        lay.addWidget(self._heading("Evaluation Dashboard"))

        # Profile indicator
        self._profile_label = QLabel("Profile: none")
        self._profile_label.setStyleSheet("color:#8fc9ff; font-size:11px;")
        lay.addWidget(self._profile_label)

        # Metric cards in a grid
        grid = QGridLayout()
        grid.setSpacing(8)

        self._recall_precision = _MetricCard("Recall Precision", "relevant / retrieved", "high")
        self._recall_coverage = _MetricCard("Recall Coverage", "retrieved / available", "high")
        self._hallucination = _MetricCard("Hallucination Rate", "fabricated recalls", "low")
        self._correction_rate = _MetricCard("Correction Rate", "corrections / total", "low")
        self._persona_drift = _MetricCard("Persona Drift", "style deviation", "low")
        self._emotion_consistency = _MetricCard("Emotion Consistency", "emotional stability", "high")

        grid.addWidget(self._recall_precision, 0, 0)
        grid.addWidget(self._recall_coverage, 0, 1)
        grid.addWidget(self._hallucination, 1, 0)
        grid.addWidget(self._correction_rate, 1, 1)
        grid.addWidget(self._persona_drift, 2, 0)
        grid.addWidget(self._emotion_consistency, 2, 1)

        lay.addLayout(grid)

        # Detailed stats section
        lay.addWidget(self._heading("Detailed Stats"))

        self._detail_label = QLabel(
            "Memory: -\n"
            "Corrections: -\n"
            "Feedback: -\n"
            "Inferences: -"
        )
        self._detail_label.setStyleSheet("color:#d1d5db; font-size:11px;")
        self._detail_label.setWordWrap(True)
        lay.addWidget(self._detail_label)

        # Subscribe
        self.bus.subscribe("eval_metrics", self._on_eval_metrics)
        self.bus.subscribe("memory_updated", self._on_memory_updated)
        self.bus.subscribe("repair_stats_updated", self._on_repair_stats)
        self.bus.subscribe("persona_drift_detected", self._on_drift)

    def _on_eval_metrics(self, data: dict) -> None:
        if "recall_precision" in data:
            self._recall_precision.set_value(
                data["recall_precision"] * 100,
                f"{data.get('relevant_count', '?')} / {data.get('retrieved_count', '?')}",
            )
        if "recall_coverage" in data:
            self._recall_coverage.set_value(data["recall_coverage"] * 100)
        if "hallucination_rate" in data:
            self._hallucination.set_value(data["hallucination_rate"] * 100)
        if "correction_rate" in data:
            self._correction_rate.set_value(data["correction_rate"] * 100)
        if "persona_drift" in data:
            self._persona_drift.set_value(data["persona_drift"] * 100)
        if "emotion_consistency" in data:
            self._emotion_consistency.set_value(data["emotion_consistency"] * 100)

        if "profile" in data:
            self._profile_label.setText(f"Profile: {data['profile']}")

    def _on_memory_updated(self, data: dict) -> None:
        stats = data.get("stats", {})
        self._profile_label.setText(f"Profile: {data.get('profile', '?')}")

        st = stats.get("short_term_count", 0)
        lt = stats.get("long_term_count", 0)
        lines = [f"Memory: {st} short-term, {lt} long-term ({st + lt} total)"]

        self._detail_label.setText("\n".join(lines))

    def _on_repair_stats(self, data: dict) -> None:
        total_c = data.get("total_corrections", 0)
        acked = data.get("acknowledged", 0)
        feedback = data.get("total_feedback", 0)

        current = self._detail_label.text()
        lines = current.split("\n")
        if len(lines) >= 2:
            lines[1] = f"Corrections: {total_c} ({acked} acknowledged)"
        if len(lines) >= 3:
            lines[2] = f"Feedback: {feedback} entries"
        self._detail_label.setText("\n".join(lines))

    def _on_drift(self, data: dict) -> None:
        drift = data.get("drift", 0)
        self._persona_drift.set_value(drift * 100, f"deviation: {drift:.3f}")
