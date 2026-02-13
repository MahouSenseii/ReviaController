"""
Center panel — runtime bar, assistant status, chat, inference.

The Activity section is a live chat widget: scrollable message history
with user/assistant messages, a text input, and a send button.
Messages flow through the EventBus so the ConversationManager handles
inference and the chat displays results.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QFrame,
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
from ui.widgets import GhostPanel, make_panel, panel_inner

from .base_panel import BasePanel


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

        # ── Chat (replaces old Activity log) ──────────────────
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

    def _make_chat_section(self) -> QFrame:
        """Build the chat panel with message history and input."""
        p = make_panel("Chat", title_object="ActivityPanelTitle")
        title = p.findChild(QLabel)
        title.setProperty("class", "AccentTitle")
        title.setStyleSheet("color: #8fc9ff;")

        inner = panel_inner(p)
        cl = inner.layout()
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        # Message history
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

        # Input row
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

        # Track if we're waiting for a response
        self._waiting = False

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

        # Left column — model identity + live metrics
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
        """Send the user's message via the EventBus."""
        text = self._chat_input.text().strip()
        if not text or self._waiting:
            return

        # Show user message in chat
        self._append_message("You", text, "#7fb3ff")

        # Clear input
        self._chat_input.clear()

        # Disable input while waiting
        self._waiting = True
        self._send_btn.setEnabled(False)
        self._chat_input.setEnabled(False)

        # Publish for ConversationManager to pick up
        self.bus.publish("user_message", {"text": text})

        # Show typing indicator
        self._append_system("Thinking...")

    def _on_assistant_response(self, data: dict) -> None:
        """Display the assistant's response in the chat."""
        text = data.get("text", "")
        if not text:
            return

        # Remove the "Thinking..." indicator
        self._remove_last_system()

        model = data.get("model", "AI")
        self._append_message(model, text, "#33d17a")

        # Re-enable input
        self._waiting = False
        self._send_btn.setEnabled(True)
        self._chat_input.setEnabled(True)
        self._chat_input.setFocus()

    def _on_activity_log(self, data: dict) -> None:
        """Show system/error messages in the chat."""
        text = data.get("text", "")
        if not text:
            return

        # Show system and error messages
        if text.startswith("[System]") or text.startswith("[Error]"):
            self._remove_last_system()
            self._append_system(text)

            # Re-enable input on error
            if text.startswith("[Error]") or "No AI backend" in text:
                self._waiting = False
                self._send_btn.setEnabled(True)
                self._chat_input.setEnabled(True)

    def _append_message(self, sender: str, text: str, colour: str) -> None:
        """Append a styled chat message to the display."""
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
        """Append a system/status message in muted style."""
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
        """Remove the last system message (e.g. 'Thinking...')."""
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
        """Scroll chat display to the bottom."""
        sb = self._chat_display.verticalScrollBar()
        QTimer.singleShot(10, lambda: sb.setValue(sb.maximum()))

    # ══════════════════════════════════════════════════════════
    # Other event handlers
    # ══════════════════════════════════════════════════════════

    def _on_runtime_stats(self, data: dict) -> None:
        parts = [f"{k}: {v}" for k, v in data.items()]
        self._topbar_label.setText("   |   ".join(parts))

    def _on_assistant_status(self, data: dict) -> None:
        lines = data.get("lines", [])
        self._status_label.setText("\n".join(f"• {l}" for l in lines))

    def _on_model_changed(self, data: dict) -> None:
        name = data.get("model") or "-"
        reg = data.get("registry", {})
        mode = data.get("mode", "")

        compute = reg.get("compute", "")
        params = reg.get("parameters", "")
        if compute:
            display = f"{name} ({compute})"
        else:
            display = name
        self._llm_name_label.setText(f"LLM: {display}")

        size = reg.get("size_label", "-")
        quant = reg.get("quant", "") or "-"
        if mode == "online":
            provider = reg.get("provider", data.get("provider", "-"))
            model_id = reg.get("model_id", data.get("model_id", "-"))
            detail = f"Provider: {provider}   |   Model: {model_id}"
        else:
            parts = [f"Size: {size}"]
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
