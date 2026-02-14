"""
Center panel — runtime bar, assistant status, chat, inference.

The top bar shows live stats (Model, VRAM, RAM, GPU, CPU) with a
glowing Health indicator.  The assistant status section has individual
items with animated glow dots that light up to show what the assistant
is currently doing.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QTextCursor
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config import Config
from core.events import EventBus
from ui.widgets import GhostPanel, StatusDot, make_panel, panel_inner

from .base_panel import BasePanel


# ── Health glow colours ───────────────────────────────────────
_HEALTH = {
    "active":  ("#33d17a", "Active"),
    "warning": ("#f9c74f", "Issues"),
    "error":   ("#ef476f", "Offline"),
}


class _GlowLabel(QLabel):
    """A label with a coloured drop-shadow glow behind the text."""

    def __init__(self, text: str, colour: str = "#33d17a"):
        super().__init__(text)
        self._glow = QGraphicsDropShadowEffect(self)
        self._glow.setBlurRadius(24)
        self._glow.setOffset(0, 0)
        self.setGraphicsEffect(self._glow)
        self.set_colour(colour)

    def set_colour(self, colour: str) -> None:
        self._glow.setColor(QColor(colour))
        self.setStyleSheet(
            f"color:{colour}; font-weight:800; font-size:14px; background:transparent;"
        )


class _StatusItem(QWidget):
    """A single status row: glowing dot + label."""

    def __init__(self, label: str, initial_status: str = "off"):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(8)

        self._dot = StatusDot(initial_status, size=10)
        h.addWidget(self._dot, alignment=Qt.AlignmentFlag.AlignVCenter)

        self._label = QLabel(label)
        self._label.setStyleSheet("color:#c7d3e6; font-size:13px;")
        h.addWidget(self._label, 1)

    def set_active(self, active: bool) -> None:
        self._dot.set_status("on" if active else "off")

    def set_warn(self) -> None:
        self._dot.set_status("warn")


class CenterPanel(BasePanel):
    """Main content area: runtime stats, status, chat, inference."""

    def _build(self) -> None:
        lay = self._inner_layout
        lay.setSpacing(12)

        # ── Top bar (runtime stats) ───────────────────────────
        self._topbar = self._make_topbar()
        lay.addWidget(self._topbar, 0)

        # ── Assistant status ──────────────────────────────────
        status_panel = self._make_status_section()
        lay.addWidget(status_panel, 0)

        # ── Chat ──────────────────────────────────────────────
        chat_panel = self._make_chat_section()
        lay.addWidget(chat_panel, 1)

        # ── Inference ─────────────────────────────────────────
        inference_panel = self._make_inference_section()
        lay.addWidget(inference_panel, 0)

        # ── Subscribe to live data ────────────────────────────
        self.bus.subscribe("runtime_stats", self._on_runtime_stats)
        self.bus.subscribe("assistant_status", self._on_assistant_status)
        self.bus.subscribe("assistant_response", self._on_assistant_response)
        self.bus.subscribe("activity_log", self._on_activity_log)
        self.bus.subscribe("inference_metrics", self._on_inference_metrics)
        self.bus.subscribe("model_changed", self._on_model_changed)
        self.bus.subscribe("plugin_activated", self._on_plugin_activated)
        self.bus.subscribe("plugin_deactivated", self._on_plugin_deactivated)
        self.bus.subscribe("user_message", self._on_user_message_status)

    # ══════════════════════════════════════════════════════════
    # Top bar
    # ══════════════════════════════════════════════════════════

    def _make_topbar(self) -> QFrame:
        bar = make_panel()
        bar.setFixedHeight(56)
        inner = panel_inner(bar)
        il = inner.layout()
        il.setContentsMargins(12, 6, 12, 6)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        # Individual stat labels — each one updates independently
        self._stat_model = self._make_stat("Model", "-")
        self._stat_vram  = self._make_stat("VRAM", "-")
        self._stat_ram   = self._make_stat("RAM", "-")
        self._stat_gpu   = self._make_stat("GPU", "-")
        self._stat_cpu   = self._make_stat("CPU", "-")

        for w in (self._stat_model, self._stat_vram, self._stat_ram,
                  self._stat_gpu, self._stat_cpu):
            h.addWidget(w, 1)
            if w is not self._stat_cpu:
                sep = QLabel("|")
                sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
                sep.setStyleSheet("color:#263246; font-size:16px; padding:0 4px;")
                h.addWidget(sep, 0)

        # Health indicator — glowing word
        sep = QLabel("|")
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep.setStyleSheet("color:#263246; font-size:16px; padding:0 4px;")
        h.addWidget(sep, 0)

        health_w = QWidget()
        hw = QHBoxLayout(health_w)
        hw.setContentsMargins(0, 0, 0, 0)
        hw.setSpacing(6)
        hlbl = QLabel("Health:")
        hlbl.setStyleSheet("color:#8fa6c3; font-size:12px; font-weight:600;")
        hw.addWidget(hlbl, 0)

        self._health_glow = _GlowLabel("Offline", "#ef476f")
        hw.addWidget(self._health_glow, 0)
        hw.addStretch(1)

        h.addWidget(health_w, 1)

        il.addWidget(row)
        return bar

    @staticmethod
    def _make_stat(key: str, value: str) -> QWidget:
        """Create a small key: value stat widget."""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(4, 0, 4, 0)
        h.setSpacing(4)

        k = QLabel(f"{key}:")
        k.setStyleSheet("color:#8fa6c3; font-size:12px; font-weight:600;")
        h.addWidget(k, 0)

        v = QLabel(value)
        v.setObjectName(f"stat_{key}")
        v.setStyleSheet("color:#d8e1ee; font-size:12px;")
        h.addWidget(v, 0)

        h.addStretch(1)
        return w

    def _update_stat(self, key: str, value: str) -> None:
        """Update a single stat value in the top bar."""
        lbl = self._topbar.findChild(QLabel, f"stat_{key}")
        if lbl:
            lbl.setText(value)

    def _set_health(self, level: str) -> None:
        """Set health glow: 'active', 'warning', or 'error'."""
        colour, text = _HEALTH.get(level, _HEALTH["error"])
        self._health_glow.setText(text)
        self._health_glow.set_colour(colour)

    # ══════════════════════════════════════════════════════════
    # Assistant status with glow dots
    # ══════════════════════════════════════════════════════════

    def _make_status_section(self) -> QFrame:
        p = make_panel("Assistant Status", title_object="StatusPanelTitle")
        title = p.findChild(QLabel)
        title.setProperty("class", "AccentTitle")
        title.setStyleSheet("color: #8fc9ff;")

        inner = panel_inner(p)
        sl = inner.layout()
        sl.setContentsMargins(12, 10, 12, 10)
        sl.setSpacing(4)

        self._st_listening   = _StatusItem("Listening...", "off")
        self._st_processing  = _StatusItem("Processing Command...", "off")
        self._st_generating  = _StatusItem("Generating Response...", "off")
        self._st_vision      = _StatusItem("Vision: Idle", "off")

        for item in (self._st_listening, self._st_processing,
                     self._st_generating, self._st_vision):
            sl.addWidget(item)

        # Start in listening state
        self._st_listening.set_active(True)

        return p

    # ══════════════════════════════════════════════════════════
    # Chat section
    # ══════════════════════════════════════════════════════════

    def _make_chat_section(self) -> QFrame:
        p = make_panel("Chat", title_object="ActivityPanelTitle")
        title = p.findChild(QLabel)
        title.setProperty("class", "AccentTitle")
        title.setStyleSheet("color: #8fc9ff;")

        inner = panel_inner(p)
        cl = inner.layout()
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setObjectName("ChatDisplay")
        self._chat_display.setStyleSheet(
            "QTextEdit#ChatDisplay {"
            "  background: #0a0f18;"
            "  border: none;"
            "  border-radius: 0px;"
            "  color: #d8e1ee;"
            "  padding: 12px;"
            "  font-size: 13px;"
            "  font-family: 'Segoe UI', sans-serif;"
            "}"
            "QScrollBar:vertical {"
            "  background: #0a0f18;"
            "  width: 8px;"
            "  border-radius: 4px;"
            "}"
            "QScrollBar::handle:vertical {"
            "  background: #2a3b55;"
            "  border-radius: 4px;"
            "  min-height: 30px;"
            "}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
            "  height: 0px;"
            "}"
        )
        cl.addWidget(self._chat_display)

        input_row = QWidget()
        input_row.setObjectName("ChatInputRow")
        input_row.setStyleSheet(
            "QWidget#ChatInputRow { background: #101824; border-top: 1px solid #263246; }"
        )
        ih = QHBoxLayout(input_row)
        ih.setContentsMargins(8, 8, 8, 8)
        ih.setSpacing(8)

        self._chat_input = QLineEdit()
        self._chat_input.setObjectName("ChatInput")
        self._chat_input.setPlaceholderText("Type a message...")
        self._chat_input.setStyleSheet(
            "QLineEdit#ChatInput {"
            "  background: #0a0f18;"
            "  border: 1px solid #2a3b55;"
            "  border-radius: 8px;"
            "  padding: 8px 12px;"
            "  color: #d8e1ee;"
            "  font-size: 13px;"
            "  min-height: 28px;"
            "}"
            "QLineEdit#ChatInput:focus {"
            "  border-color: #4a8cd8;"
            "}"
        )
        self._chat_input.returnPressed.connect(self._on_send)
        ih.addWidget(self._chat_input, 1)

        self._send_btn = QPushButton("Send")
        self._send_btn.setObjectName("SendButton")
        self._send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_btn.setStyleSheet(
            "QPushButton#SendButton {"
            "  background: #1a3a5c;"
            "  border: 1px solid #4a8cd8;"
            "  border-radius: 8px;"
            "  color: #d8e1ee;"
            "  padding: 8px 20px;"
            "  font-size: 13px;"
            "  font-weight: 600;"
            "  min-height: 28px;"
            "}"
            "QPushButton#SendButton:hover {"
            "  background: #1e4a72;"
            "  border-color: #6aacf8;"
            "}"
            "QPushButton#SendButton:pressed {"
            "  background: #153050;"
            "}"
            "QPushButton#SendButton:disabled {"
            "  background: #101824;"
            "  border-color: #2a3b55;"
            "  color: #5a6a7e;"
            "}"
        )
        self._send_btn.clicked.connect(self._on_send)
        ih.addWidget(self._send_btn)

        cl.addWidget(input_row)
        self._waiting = False
        return p

    # ══════════════════════════════════════════════════════════
    # Inference section
    # ══════════════════════════════════════════════════════════

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

        left = QVBoxLayout()
        left.setSpacing(4)

        self._llm_name_label = QLabel("LLM: -")
        self._llm_name_label.setObjectName("MonoInfo")
        self._llm_name_label.setStyleSheet("font-weight:700; font-size:13px;")
        left.addWidget(self._llm_name_label)

        self._llm_detail_label = QLabel(
            "Size: -   |   Compute: -   |   Quant: -"
        )
        self._llm_detail_label.setObjectName("MonoInfo")
        self._llm_detail_label.setStyleSheet("color:#8fa6c3; font-size:11px;")
        left.addWidget(self._llm_detail_label)

        self._inference_label = QLabel(
            "Latency: -\nTokens/sec: -\nTTFT: -\nContext: -"
        )
        self._inference_label.setObjectName("MonoInfo")
        left.addWidget(self._inference_label)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(280)
        il.addWidget(left_w)

        self._preview = GhostPanel("Preview / Whiteboard / Vision Frame", height=220)
        il.addWidget(self._preview, 1)
        return p

    # ══════════════════════════════════════════════════════════
    # Chat logic
    # ══════════════════════════════════════════════════════════

    def _on_send(self) -> None:
        text = self._chat_input.text().strip()
        if not text or self._waiting:
            return

        self._append_message("You", text, "#7fb3ff")
        self._chat_input.clear()
        self._waiting = True
        self._send_btn.setEnabled(False)
        self._chat_input.setEnabled(False)

        self.bus.publish("user_message", {"text": text})
        self._append_system("Thinking...")

    def _on_assistant_response(self, data: dict) -> None:
        text = data.get("text", "")
        if not text:
            return

        self._remove_last_system()
        model = data.get("model", "AI")
        self._append_message(model, text, "#33d17a")

        self._waiting = False
        self._send_btn.setEnabled(True)
        self._chat_input.setEnabled(True)
        self._chat_input.setFocus()

        # Back to listening
        self._set_status_state("listening")

    def _on_activity_log(self, data: dict) -> None:
        text = data.get("text", "")
        if not text:
            return

        if text.startswith("[System]") or text.startswith("[Error]"):
            self._remove_last_system()
            self._append_system(text)
            if text.startswith("[Error]") or "No AI backend" in text:
                self._waiting = False
                self._send_btn.setEnabled(True)
                self._chat_input.setEnabled(True)
                self._set_status_state("listening")

    def _append_message(self, sender: str, text: str, colour: str) -> None:
        escaped = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        html = (
            f'<div style="margin-bottom:12px;">'
            f'<span style="color:{colour}; font-weight:700; font-size:12px;">'
            f'{sender}</span><br>'
            f'<span style="color:#d8e1ee; font-size:13px; line-height:1.5;">'
            f'{escaped}</span>'
            f'</div>'
        )
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._chat_display.setTextCursor(cursor)
        self._chat_display.insertHtml(html)
        self._scroll_to_bottom()

    def _append_system(self, text: str) -> None:
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = (
            f'<div id="sys_msg" style="margin-bottom:8px;">'
            f'<span style="color:#5a6a7e; font-style:italic; font-size:12px;">'
            f'{escaped}</span>'
            f'</div>'
        )
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._chat_display.setTextCursor(cursor)
        self._chat_display.insertHtml(html)
        self._scroll_to_bottom()

    def _remove_last_system(self) -> None:
        content = self._chat_display.toHtml()
        marker = '<div id="sys_msg"'
        idx = content.rfind(marker)
        if idx == -1:
            return
        end_tag = "</div>"
        end_idx = content.find(end_tag, idx)
        if end_idx == -1:
            return
        new_content = content[:idx] + content[end_idx + len(end_tag):]
        self._chat_display.setHtml(new_content)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        sb = self._chat_display.verticalScrollBar()
        QTimer.singleShot(10, lambda: sb.setValue(sb.maximum()))

    # ══════════════════════════════════════════════════════════
    # Status glow logic
    # ══════════════════════════════════════════════════════════

    def _set_status_state(self, state: str) -> None:
        """Light up the correct status dots.

        States: 'listening', 'processing', 'generating', 'error'
        """
        self._st_listening.set_active(state == "listening")
        self._st_processing.set_active(state == "processing")
        self._st_generating.set_active(state == "generating")

    # ══════════════════════════════════════════════════════════
    # Event handlers
    # ══════════════════════════════════════════════════════════

    def _on_user_message_status(self, data: dict) -> None:
        self._set_status_state("processing")

    def _on_assistant_status(self, data: dict) -> None:
        lines = data.get("lines", [])
        joined = " ".join(lines).lower()
        if "generating" in joined:
            self._set_status_state("generating")
        elif "processing" in joined:
            self._set_status_state("processing")
        elif "listening" in joined:
            self._set_status_state("listening")
        elif "error" in joined:
            self._set_status_state("error")

    def _on_runtime_stats(self, data: dict) -> None:
        for key in ("Model", "VRAM", "RAM", "GPU", "CPU"):
            if key in data:
                self._update_stat(key, str(data[key]))
        if "Health" in data:
            h = str(data["Health"]).lower()
            if h in ("active", "ok", "healthy", "good"):
                self._set_health("active")
            elif h in ("warning", "warn", "degraded"):
                self._set_health("warning")
            else:
                self._set_health("error")

    def _on_plugin_activated(self, data: dict) -> None:
        name = data.get("name", "?")
        self._set_health("active")
        self._update_stat("Model", name)

    def _on_plugin_deactivated(self, data: dict) -> None:
        self._set_health("error")
        self._update_stat("Model", "-")

    def _on_model_changed(self, data: dict) -> None:
        name = data.get("model") or "-"
        reg = data.get("registry", {})
        mode = data.get("mode", "")

        compute = reg.get("compute", "")
        display = f"{name} ({compute})" if compute else name
        self._llm_name_label.setText(f"LLM: {display}")
        self._update_stat("Model", name)

        size = reg.get("size_label", "-")
        quant = reg.get("quant", "") or "-"
        if mode == "online":
            provider = reg.get("provider", data.get("provider", "-"))
            model_id = reg.get("model_id", data.get("model_id", "-"))
            detail = f"Provider: {provider}   |   Model: {model_id}"
        else:
            parts = [f"Size: {size}"]
            params = reg.get("parameters", "")
            if params:
                parts.append(f"Params: {params}")
            parts.append(f"Quant: {quant}")
            detail = "   |   ".join(parts)
        self._llm_detail_label.setText(detail)

    def _on_inference_metrics(self, data: dict) -> None:
        lines = [
            f"Latency: {data.get('latency', '-')}",
            f"Tokens/sec: {data.get('tokens_sec', '-')}",
            f"TTFT: {data.get('ttft', '-')}",
            f"Context: {data.get('context', '-')}",
        ]
        self._inference_label.setText("\n".join(lines))
