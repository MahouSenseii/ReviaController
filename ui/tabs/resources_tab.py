"""
Runtime resource tracker tab — comprehensive monitoring of token
usage, model latency, CPU/RAM/GPU usage, storage, cost telemetry,
and reliability counters.

Subscribes to:
    - ``runtime_stats`` — hardware metrics
    - ``inference_metrics`` — LLM performance
    - ``resource_telemetry`` — extended telemetry data
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


class _ResourceGauge(QWidget):
    """Compact gauge for a single resource metric."""

    def __init__(self, label: str, unit: str = "%", bar_color: str = "#33d17a"):
        super().__init__()
        self._unit = unit

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self._label = QLabel(label)
        self._label.setFixedWidth(80)
        self._label.setStyleSheet("color:#d1d5db; font-size:11px;")
        layout.addWidget(self._label, 0, 0)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(10)
        self._bar.setStyleSheet(
            f"QProgressBar {{ background:#0a0f18; border:1px solid #2a3b55; border-radius:5px; }}"
            f"QProgressBar::chunk {{ background:{bar_color}; border-radius:4px; }}"
        )
        layout.addWidget(self._bar, 0, 1)

        self._value_label = QLabel(f"- {unit}")
        self._value_label.setFixedWidth(70)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._value_label.setStyleSheet("color:#9ca3af; font-size:10px;")
        layout.addWidget(self._value_label, 0, 2)

    def set_value(self, value: float, display: str = "") -> None:
        clamped = max(0, min(100, int(value)))
        self._bar.setValue(clamped)
        self._value_label.setText(display or f"{value:.1f} {self._unit}")


class _StatRow(QWidget):
    """Label-value pair for a single statistic."""

    def __init__(self, label: str, color: str = "#d1d5db"):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._label = QLabel(label)
        self._label.setStyleSheet("color:#9ca3af; font-size:10px;")
        self._label.setFixedWidth(110)
        layout.addWidget(self._label)

        self._value = QLabel("-")
        self._value.setStyleSheet(f"color:{color}; font-size:10px; font-weight:600;")
        layout.addWidget(self._value)
        layout.addStretch()

    def set_value(self, text: str) -> None:
        self._value.setText(text)


class ResourcesTab(BaseTab):
    """Runtime resource tracker with comprehensive monitoring."""

    def _build(self) -> None:
        lay = self._layout

        # ── Token Usage ───────────────────────────────────────
        lay.addWidget(self._heading("Token Usage"))

        self._prompt_tokens = _StatRow("Prompt Tokens", "#60a5fa")
        self._completion_tokens = _StatRow("Completion Tokens", "#22c55e")
        self._cached_tokens = _StatRow("Cached Tokens", "#f59e0b")
        self._total_tokens = _StatRow("Total Tokens", "#e5e7eb")
        self._tokens_per_sec = _StatRow("Tokens/sec", "#c084fc")

        for w in (self._prompt_tokens, self._completion_tokens,
                  self._cached_tokens, self._total_tokens, self._tokens_per_sec):
            lay.addWidget(w)

        # ── Model Latency ─────────────────────────────────────
        lay.addWidget(self._heading("Model Latency"))

        self._e2e_latency = _StatRow("End-to-End", "#8fc9ff")
        self._ttft = _StatRow("Time to First", "#60a5fa")
        self._retrieval_lat = _StatRow("Retrieval", "#f59e0b")
        self._embedding_lat = _StatRow("Embedding", "#c084fc")
        self._p50_lat = _StatRow("p50 Latency", "#22c55e")
        self._p95_lat = _StatRow("p95 Latency", "#ef4444")

        for w in (self._e2e_latency, self._ttft, self._retrieval_lat,
                  self._embedding_lat, self._p50_lat, self._p95_lat):
            lay.addWidget(w)

        # ── Hardware ──────────────────────────────────────────
        lay.addWidget(self._heading("Hardware"))

        self._cpu_gauge = _ResourceGauge("CPU", "%", "#60a5fa")
        self._ram_gauge = _ResourceGauge("RAM", "%", "#22c55e")
        self._gpu_gauge = _ResourceGauge("GPU", "%", "#c084fc")
        self._vram_gauge = _ResourceGauge("VRAM", "%", "#f59e0b")

        for w in (self._cpu_gauge, self._ram_gauge, self._gpu_gauge, self._vram_gauge):
            lay.addWidget(w)

        # ── Storage & I/O ─────────────────────────────────────
        lay.addWidget(self._heading("Storage & I/O"))

        self._memory_db_size = _StatRow("Memory DB", "#60a5fa")
        self._vector_index = _StatRow("Vector Index", "#22c55e")
        self._disk_io = _StatRow("Disk I/O", "#f59e0b")

        for w in (self._memory_db_size, self._vector_index, self._disk_io):
            lay.addWidget(w)

        # ── Cost & Reliability ────────────────────────────────
        lay.addWidget(self._heading("Cost & Reliability"))

        self._cost_per_req = _StatRow("Cost/Request", "#f59e0b")
        self._daily_burn = _StatRow("Daily Burn", "#ef4444")
        self._timeout_rate = _StatRow("Timeout Rate", "#ef4444")
        self._cache_hit = _StatRow("Cache Hit Ratio", "#22c55e")
        self._retry_rate = _StatRow("Retry Rate", "#f59e0b")

        for w in (self._cost_per_req, self._daily_burn, self._timeout_rate,
                  self._cache_hit, self._retry_rate):
            lay.addWidget(w)

        # Subscribe
        self.bus.subscribe("runtime_stats", self._on_runtime_stats)
        self.bus.subscribe("inference_metrics", self._on_inference_metrics)
        self.bus.subscribe("resource_telemetry", self._on_telemetry)

    def _on_runtime_stats(self, data: dict) -> None:
        if "CPU" in data:
            self._cpu_gauge.set_value(_pct(data["CPU"]))
        if "RAM" in data:
            self._ram_gauge.set_value(_pct(data["RAM"]))
        if "GPU" in data:
            self._gpu_gauge.set_value(_pct(data["GPU"]))
        if "VRAM" in data:
            self._vram_gauge.set_value(_pct(data["VRAM"]))

    def _on_inference_metrics(self, data: dict) -> None:
        if "prompt_tokens" in data:
            self._prompt_tokens.set_value(f"{data['prompt_tokens']:,}")
        if "completion_tokens" in data:
            self._completion_tokens.set_value(f"{data['completion_tokens']:,}")
        if "cached_tokens" in data:
            self._cached_tokens.set_value(f"{data['cached_tokens']:,}")
        if "total_tokens" in data:
            self._total_tokens.set_value(f"{data['total_tokens']:,}")
        if "tokens_per_sec" in data:
            self._tokens_per_sec.set_value(f"{data['tokens_per_sec']:.1f} tok/s")

        if "latency_ms" in data:
            self._e2e_latency.set_value(f"{data['latency_ms']:.0f} ms")
        if "ttft_ms" in data:
            self._ttft.set_value(f"{data['ttft_ms']:.0f} ms")
        if "retrieval_ms" in data:
            self._retrieval_lat.set_value(f"{data['retrieval_ms']:.0f} ms")
        if "embedding_ms" in data:
            self._embedding_lat.set_value(f"{data['embedding_ms']:.0f} ms")
        if "p50_ms" in data:
            self._p50_lat.set_value(f"{data['p50_ms']:.0f} ms")
        if "p95_ms" in data:
            self._p95_lat.set_value(f"{data['p95_ms']:.0f} ms")

    def _on_telemetry(self, data: dict) -> None:
        if "memory_db_bytes" in data:
            self._memory_db_size.set_value(_format_bytes(data["memory_db_bytes"]))
        if "vector_index_bytes" in data:
            self._vector_index.set_value(_format_bytes(data["vector_index_bytes"]))
        if "disk_io_mbps" in data:
            self._disk_io.set_value(f"{data['disk_io_mbps']:.1f} MB/s")

        if "cost_per_request" in data:
            self._cost_per_req.set_value(f"${data['cost_per_request']:.4f}")
        if "daily_burn_rate" in data:
            self._daily_burn.set_value(f"${data['daily_burn_rate']:.2f}/day")
        if "timeout_rate" in data:
            self._timeout_rate.set_value(f"{data['timeout_rate'] * 100:.1f}%")
        if "cache_hit_ratio" in data:
            self._cache_hit.set_value(f"{data['cache_hit_ratio'] * 100:.1f}%")
        if "retry_rate" in data:
            self._retry_rate.set_value(f"{data['retry_rate'] * 100:.1f}%")


def _pct(raw) -> float:
    s = str(raw).replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def _format_bytes(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    if b < 1024 * 1024 * 1024:
        return f"{b / (1024 * 1024):.1f} MB"
    return f"{b / (1024 * 1024 * 1024):.2f} GB"
