"""
Center panel — runtime bar, assistant status, activity log, inference.

Subscribes to events from the plugin layer and updates labels / widgets
in real time.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from core.config import Config
from core.events import EventBus
from ui.widgets import GhostPanel, make_panel, panel_inner

from .base_panel import BasePanel


class CenterPanel(BasePanel):
    """Main content area: runtime stats, status, activity log, inference."""

    def _build(self) -> None:
        lay = self._inner_layout
        lay.setSpacing(12)

        # ── Top bar (runtime stats) ───────────────────────────
        self._topbar = self._make_topbar()
        lay.addWidget(self._topbar, 0)

        # ── Assistant status ──────────────────────────────────
        status_panel = self._make_status_section()
        lay.addWidget(status_panel, 0)

        # ── Activity log ──────────────────────────────────────
        activity_panel = self._make_activity_section()
        lay.addWidget(activity_panel, 1)

        # ── Inference ─────────────────────────────────────────
        inference_panel = self._make_inference_section()
        lay.addWidget(inference_panel, 1)

        # ── Subscribe to live data ────────────────────────────
        self.bus.subscribe("runtime_stats", self._on_runtime_stats)
        self.bus.subscribe("assistant_status", self._on_assistant_status)
        self.bus.subscribe("activity_log", self._on_activity_log)
        self.bus.subscribe("inference_metrics", self._on_inference_metrics)

    # ── Section builders ──────────────────────────────────────

    def _make_topbar(self) -> QFrame:
        bar = make_panel()
        bar.setFixedHeight(60)
        inner = panel_inner(bar)
        inner.layout().setContentsMargins(12, 8, 12, 8)

        self._topbar_label = QLabel(
            "Runtime: Active   |   Model: -   |   VRAM: -   "
            "|   RAM: -   |   GPU: -   |   CPU: -   |   Health: -"
        )
        self._topbar_label.setObjectName("TopBarText")
        inner.layout().addWidget(self._topbar_label)
        return bar

    def _make_status_section(self) -> QFrame:
        p = make_panel("Assistant Status", title_object="StatusPanelTitle")
        title = p.findChild(QLabel)
        title.setProperty("class", "AccentTitle")
        title.setStyleSheet("color: #8fc9ff;")

        inner = panel_inner(p)
        sl = inner.layout()
        sl.setContentsMargins(12, 12, 12, 12)
        sl.setSpacing(6)

        self._status_label = QLabel(
            "• Listening...\n"
            "• Processing Command...\n"
            "• Vision: Idle\n"
            "• Generating Response..."
        )
        self._status_label.setObjectName("MonoInfo")
        sl.addWidget(self._status_label)
        return p

    def _make_activity_section(self) -> QFrame:
        p = make_panel("Activity", title_object="ActivityPanelTitle")
        title = p.findChild(QLabel)
        title.setProperty("class", "AccentTitle")
        title.setStyleSheet("color: #8fc9ff;")

        inner = panel_inner(p)
        al = inner.layout()
        al.setContentsMargins(12, 12, 12, 12)
        al.setSpacing(6)

        self._activity_label = QLabel(
            'User: "Analyze this screenshot and explain the chart"\n'
            'AI: "Sure! The chart shows..."'
        )
        self._activity_label.setObjectName("MonoInfo")
        self._activity_label.setWordWrap(True)
        al.addWidget(self._activity_label)
        return p

    def _make_inference_section(self) -> QFrame:
        p = make_panel("Inference", title_object="ActivityPanelTitle")
        title = p.findChild(QLabel)
        title.setProperty("class", "AccentTitle")
        title.setStyleSheet("color: #8fc9ff;")

        inner = panel_inner(p)
        old = inner.layout()
        if old is not None:
            while old.count():
                item = old.takeAt(0)
                w = item.widget()
                if w:
                    w.setParent(None)

        il = QHBoxLayout(inner)
        il.setContentsMargins(12, 12, 12, 12)
        il.setSpacing(12)

        self._inference_label = QLabel(
            "LLM: -\nLatency: -\nTokens/sec: -\nTTFT: -\nContext: -"
        )
        self._inference_label.setObjectName("MonoInfo")
        self._inference_label.setFixedWidth(240)
        il.addWidget(self._inference_label)

        self._preview = GhostPanel("Preview / Whiteboard / Vision Frame", height=220)
        il.addWidget(self._preview, 1)
        return p

    # ── Event handlers ────────────────────────────────────────

    def _on_runtime_stats(self, data: dict) -> None:
        parts = [f"{k}: {v}" for k, v in data.items()]
        self._topbar_label.setText("   |   ".join(parts))

    def _on_assistant_status(self, data: dict) -> None:
        lines = data.get("lines", [])
        self._status_label.setText("\n".join(f"• {l}" for l in lines))

    def _on_activity_log(self, data: dict) -> None:
        text = data.get("text", "")
        self._activity_label.setText(text)

    def _on_inference_metrics(self, data: dict) -> None:
        lines = [
            f"LLM: {data.get('llm', '-')}",
            f"Latency: {data.get('latency', '-')}",
            f"Tokens/sec: {data.get('tokens_sec', '-')}",
            f"TTFT: {data.get('ttft', '-')}",
            f"Context: {data.get('context', '-')}",
        ]
        self._inference_label.setText("\n".join(lines))
