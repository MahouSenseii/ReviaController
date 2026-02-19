"""
Center panel — runtime bar, assistant status, chat, inference.

The top bar shows live stats (Model, VRAM, RAM, GPU, CPU) with a
glowing Health indicator.  The assistant status section has individual
items with animated glow dots that light up to show what the assistant
is currently doing.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPixmap, QTextCursor
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
from ui.widgets import StatusDot, make_panel, panel_inner

from .base_panel import BasePanel

# Optional: OpenCV for webcam capture
try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False


# ── Health glow colours ───────────────────────────────────────
_HEALTH = {
    "active":  ("#33d17a", "Active"),
    "warning": ("#f9c74f", "Issues"),
    "error":   ("#ef476f", "Offline"),
}

# Vision status colours
_VISION_STATUS = {
    "on":   ("#33d17a", "Vision Connected"),
    "warn": ("#f9c74f", "Vision Warning"),
    "off":  ("#5a6a7e", "Vision Inactive"),
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


class _WebcamPreview(QWidget):
    """
    Live webcam preview panel for the Inference section.

    Shows real camera frames when the vision source is set to 'Webcam'
    and OpenCV is available.  Falls back to a styled placeholder for all
    other sources or when cv2 is not installed.

    Also displays a small status bar that reflects whether Vision is
    connected correctly to the AI backend.
    """

    _PLACEHOLDER_STYLE = (
        "background:#080d16;"
        "border:1px dashed #2a3b55;"
        "border-radius:6px;"
        "color:#3a4a5e;"
        "font-size:12px;"
    )

    def __init__(self):
        super().__init__()
        self._cap = None          # cv2.VideoCapture or None
        self._camera_index = 0
        self._active = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # ── Frame display ─────────────────────────────────────
        self._frame_label = QLabel()
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumHeight(180)
        self._frame_label.setStyleSheet(self._PLACEHOLDER_STYLE)
        self._show_placeholder("Vision Preview")
        outer.addWidget(self._frame_label, 1)

        # ── Status bar ────────────────────────────────────────
        status_row = QWidget()
        status_row.setStyleSheet("background:transparent;")
        sh = QHBoxLayout(status_row)
        sh.setContentsMargins(4, 0, 4, 0)
        sh.setSpacing(6)

        self._vision_dot = StatusDot("off", size=8)
        sh.addWidget(self._vision_dot, alignment=Qt.AlignmentFlag.AlignVCenter)

        self._vision_status_lbl = QLabel("Vision: Inactive")
        self._vision_status_lbl.setStyleSheet(
            "color:#5a6a7e; font-size:11px; font-style:italic;"
        )
        sh.addWidget(self._vision_status_lbl, 1)

        self._source_lbl = QLabel("")
        self._source_lbl.setStyleSheet("color:#3a4a5e; font-size:11px;")
        sh.addWidget(self._source_lbl, 0)

        outer.addWidget(status_row)

        # ── Capture timer ─────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._capture_frame)

    # ── Public API ────────────────────────────────────────────

    def set_vision_module_status(self, status: str, subtitle: str) -> None:
        """Update the vision connection indicator (on / warn / off)."""
        colour, text = _VISION_STATUS.get(status, _VISION_STATUS["off"])
        self._vision_dot.set_status(status)
        self._vision_status_lbl.setText(subtitle or text)
        self._vision_status_lbl.setStyleSheet(f"color:{colour}; font-size:11px;")

    def start_webcam(self, camera_index: int = 0, interval_ms: int = 66) -> None:
        """Open the webcam at *camera_index* and start the frame timer."""
        if not _CV2_AVAILABLE:
            self._show_placeholder(
                "Install 'opencv-python' to enable webcam preview\n"
                "(pip install opencv-python)"
            )
            self._source_lbl.setText("cv2 not installed")
            return

        self._stop_capture()
        self._camera_index = camera_index
        try:
            self._cap = _cv2.VideoCapture(camera_index)
            if not self._cap.isOpened():
                self._show_placeholder(
                    f"Cannot open camera {camera_index}\n"
                    "Check the Camera Index in Voice & Vision settings."
                )
                self._source_lbl.setText(f"Camera {camera_index} unavailable")
                self._cap = None
                return
        except Exception as exc:
            self._show_placeholder(f"Camera error: {exc}")
            self._source_lbl.setText("Camera error")
            self._cap = None
            return

        self._active = True
        self._source_lbl.setText(f"Camera {camera_index}")
        self._timer.start(interval_ms)

    def stop_webcam(self) -> None:
        """Stop webcam capture and return to placeholder."""
        self._stop_capture()
        self._show_placeholder("Vision Preview")
        self._source_lbl.setText("")

    def set_source_label(self, source: str) -> None:
        """Update the non-webcam source label (Screen Capture, etc.)."""
        if source == "Webcam":
            return  # handled by start_webcam
        self._source_lbl.setText(source)
        self._show_placeholder(f"Source: {source}\n(preview not available here)")

    # ── Internal ──────────────────────────────────────────────

    def _stop_capture(self) -> None:
        self._timer.stop()
        self._active = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _show_placeholder(self, text: str) -> None:
        self._frame_label.setText(text)
        self._frame_label.setPixmap(QPixmap())  # clear any previous frame
        self._frame_label.setStyleSheet(self._PLACEHOLDER_STYLE)

    def _capture_frame(self) -> None:
        if self._cap is None or not _CV2_AVAILABLE:
            return
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._show_placeholder(
                f"Camera {self._camera_index} lost signal.\n"
                "Check connection or try a different Camera Index."
            )
            self._stop_capture()
            return

        # Convert BGR → RGB and show as QPixmap
        rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        # Scale to fit the label while keeping aspect ratio
        label_size = self._frame_label.size()
        scaled = pix.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._frame_label.setPixmap(scaled)
        self._frame_label.setStyleSheet(
            "background:#000; border:1px solid #2a3b55; border-radius:6px;"
        )

    def closeEvent(self, event) -> None:
        self._stop_capture()
        super().closeEvent(event)


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

        # ── Pipeline timing ─────────────────────────────────────
        timing_panel = self._make_timing_section()
        lay.addWidget(timing_panel, 0)

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
        self.bus.subscribe("pipeline_timing", self._on_pipeline_timing)
        self.bus.subscribe("decision_made", self._on_decision_made)
        self.bus.subscribe("metacognition_update", self._on_metacognition)
        self.bus.subscribe("self_dev_update", self._on_self_dev)
        self.bus.subscribe("module_status", self._on_module_status)
        self.bus.subscribe("vision_source_changed", self._on_vision_source_changed)
        self.bus.subscribe("voice_vision_changed", self._on_voice_vision_changed)

        # Apply saved vision source on startup
        self._apply_saved_vision_source()

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

        self._st_listening  = _StatusItem("Listening...", "off")
        self._st_analyzing  = _StatusItem("Analyzing Input...", "off")
        self._st_emotion    = _StatusItem("Processing Emotions...", "off")
        self._st_decision   = _StatusItem("Making Decision...", "off")
        self._st_memory     = _StatusItem("Querying Memory...", "off")
        self._st_generating = _StatusItem("Generating Response...", "off")
        self._st_learning   = _StatusItem("Learning...", "off")
        self._st_vision     = _StatusItem("Vision: Idle", "off")

        for item in (
            self._st_listening, self._st_analyzing, self._st_emotion,
            self._st_decision, self._st_memory, self._st_generating,
            self._st_learning, self._st_vision,
        ):
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
        container = QWidget()
        il = QHBoxLayout(container)
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

        # Emotion injection indicator
        self._emotion_inject_lbl = QLabel("Emotion → LLM: Active")
        self._emotion_inject_lbl.setStyleSheet(
            "color:#33d17a; font-size:11px; margin-top:6px;"
        )
        self._emotion_inject_lbl.setToolTip(
            "The AI's current emotional state (valence, arousal, dominant emotion, "
            "mood trajectory) is injected into every system prompt so it influences "
            "tone and word choice."
        )
        left.addWidget(self._emotion_inject_lbl)

        left.addStretch(1)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(280)
        il.addWidget(left_w)

        # Live webcam / vision preview
        self._webcam_preview = _WebcamPreview()
        il.addWidget(self._webcam_preview, 1)

        inner.layout().addWidget(container)
        return p

    # ══════════════════════════════════════════════════════════
    # Vision helpers
    # ══════════════════════════════════════════════════════════

    def _apply_saved_vision_source(self) -> None:
        """Apply the vision source stored in config on startup."""
        source = self.config.get("vision.source", "Screen Capture")
        cam_idx = int(self.config.get("vision.camera_index", 0))
        interval = int(self.config.get("vision.interval_ms", 66))
        if source == "Webcam":
            self._webcam_preview.start_webcam(cam_idx, interval)
        else:
            self._webcam_preview.set_source_label(source)

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
        is_error = data.get("error", False)
        colour = "#ef476f" if is_error else "#33d17a"
        self._append_message(model, text, colour)

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
        def _esc(s: str) -> str:
            return (
                s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
            )

        # Split into paragraphs on double newline; single newline → <br>
        paragraphs = text.split("\n\n")
        body_parts: list[str] = []
        for para in paragraphs:
            para_html = _esc(para).replace("\n", "<br>")
            body_parts.append(
                f'<p style="margin:0 0 4px 0; padding:0; '
                f'color:#d8e1ee; font-size:13px; line-height:1.6;">'
                f'{para_html}</p>'
            )

        html = (
            f'<div style="margin-bottom:14px;">'
            f'<p style="margin:0 0 3px 0; padding:0; '
            f'color:{colour}; font-weight:700; font-size:12px;">'
            f'{_esc(sender)}</p>'
            + "".join(body_parts)
            + f'</div>'
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
        """Light up the correct status dot.

        States: 'listening', 'analyzing', 'emotion', 'decision',
                'memory', 'generating', 'learning', 'error'
        """
        self._st_listening.set_active(state == "listening")
        self._st_analyzing.set_active(state == "analyzing")
        self._st_emotion.set_active(state == "emotion")
        self._st_decision.set_active(state == "decision")
        self._st_memory.set_active(state == "memory")
        self._st_generating.set_active(state == "generating")
        self._st_learning.set_active(state == "learning")
        if state == "error":
            self._st_generating.set_warn()

    # ══════════════════════════════════════════════════════════
    # Event handlers
    # ══════════════════════════════════════════════════════════

    def _on_user_message_status(self, data: dict) -> None:
        self._set_status_state("analyzing")

    def _on_assistant_status(self, data: dict) -> None:
        # Prefer the explicit stage key (new-style events)
        stage = data.get("stage", "")
        if stage:
            self._set_status_state(stage)
            return

        # Fallback: parse legacy text lines
        lines = data.get("lines", [])
        joined = " ".join(lines).lower()
        if "generating" in joined:
            self._set_status_state("generating")
        elif "processing" in joined:
            self._set_status_state("analyzing")
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

    def _on_module_status(self, data: dict) -> None:
        """Update the webcam preview's vision status indicator."""
        if data.get("module") == "vision":
            status = data.get("status", "off")
            subtitle = data.get("subtitle", "")
            self._webcam_preview.set_vision_module_status(status, subtitle)
            # Mirror to the status dots section
            if status == "on":
                self._st_vision.set_active(True)
                self._st_vision._label.setText("Vision: Active")
            elif status == "warn":
                self._st_vision.set_warn()
                self._st_vision._label.setText(f"Vision: {subtitle}")
            else:
                self._st_vision.set_active(False)
                self._st_vision._label.setText("Vision: Idle")

    def _on_vision_source_changed(self, data: dict) -> None:
        """Handle vision source changes from VoiceVisionTab."""
        source = data.get("source", "Screen Capture")
        cam_idx = int(self.config.get("vision.camera_index", 0))
        interval = int(self.config.get("vision.interval_ms", 66))
        if source == "Webcam":
            self._webcam_preview.start_webcam(cam_idx, interval)
        else:
            self._webcam_preview.stop_webcam()
            self._webcam_preview.set_source_label(source)

    def _on_voice_vision_changed(self, data: dict) -> None:
        """Handle individual vision setting changes (camera index, interval)."""
        key = data.get("key", "")
        value = data.get("value")
        source = self.config.get("vision.source", "Screen Capture")

        if key == "vision.camera_index" and source == "Webcam":
            interval = int(self.config.get("vision.interval_ms", 66))
            self._webcam_preview.start_webcam(int(value), interval)
        elif key == "vision.interval_ms" and source == "Webcam":
            cam_idx = int(self.config.get("vision.camera_index", 0))
            self._webcam_preview.start_webcam(cam_idx, int(value))

    # ══════════════════════════════════════════════════════════
    # Pipeline Timing section
    # ══════════════════════════════════════════════════════════

    def _make_timing_section(self) -> QFrame:
        p = make_panel("Pipeline Timing", title_object="TimingPanelTitle")
        title = p.findChild(QLabel)
        title.setProperty("class", "AccentTitle")
        title.setStyleSheet("color: #8fc9ff;")

        inner = panel_inner(p)
        container = QWidget()
        il = QHBoxLayout(container)
        il.setContentsMargins(12, 8, 12, 8)
        il.setSpacing(16)

        # ── Timing column ──────────────────────────────────────
        timing_col = QVBoxLayout()
        timing_col.setSpacing(2)

        timing_title = QLabel("Stage Latency")
        timing_title.setStyleSheet(
            "color:#8fa6c3; font-size:11px; font-weight:700;"
        )
        timing_col.addWidget(timing_title)

        self._timing_label = QLabel(
            "Stimulus:  -\n"
            "Emotion:   -\n"
            "Decision:  -\n"
            "Inference: -\n"
            "Total:     -"
        )
        self._timing_label.setObjectName("MonoInfo")
        self._timing_label.setStyleSheet(
            "color:#d8e1ee; font-size:12px; font-family:'Consolas','Courier New',monospace;"
        )
        timing_col.addWidget(self._timing_label)

        timing_w = QWidget()
        timing_w.setLayout(timing_col)
        timing_w.setFixedWidth(180)
        il.addWidget(timing_w)

        # ── Decision column ────────────────────────────────────
        decision_col = QVBoxLayout()
        decision_col.setSpacing(2)

        decision_title = QLabel("Decision Strategy")
        decision_title.setStyleSheet(
            "color:#8fa6c3; font-size:11px; font-weight:700;"
        )
        decision_col.addWidget(decision_title)

        self._decision_label = QLabel(
            "Empathy: -  Warmth: -\n"
            "Verbose: -  Assert: -\n"
            "Curious: -  Caution: -"
        )
        self._decision_label.setObjectName("MonoInfo")
        self._decision_label.setStyleSheet(
            "color:#d8e1ee; font-size:12px; font-family:'Consolas','Courier New',monospace;"
        )
        decision_col.addWidget(self._decision_label)

        decision_w = QWidget()
        decision_w.setLayout(decision_col)
        decision_w.setFixedWidth(220)
        il.addWidget(decision_w)

        # ── Metacognition column ───────────────────────────────
        meta_col = QVBoxLayout()
        meta_col.setSpacing(2)

        meta_title = QLabel("Self-Awareness")
        meta_title.setStyleSheet(
            "color:#8fa6c3; font-size:11px; font-weight:700;"
        )
        meta_col.addWidget(meta_title)

        self._meta_label = QLabel(
            "Confidence: -\n"
            "Accuracy:   -\n"
            "Learning:   -"
        )
        self._meta_label.setObjectName("MonoInfo")
        self._meta_label.setStyleSheet(
            "color:#d8e1ee; font-size:12px; font-family:'Consolas','Courier New',monospace;"
        )
        meta_col.addWidget(self._meta_label)

        meta_w = QWidget()
        meta_w.setLayout(meta_col)
        il.addWidget(meta_w, 1)

        inner.layout().addWidget(container)
        return p

    # ── Pipeline timing handler ────────────────────────────────

    def _on_pipeline_timing(self, data: dict) -> None:
        def _fmt(cur: str, avg: str) -> str:
            if avg and avg != "-":
                return f"{cur}  (avg {avg})"
            return cur

        runs = data.get("runs", "")
        runs_str = f"  [{runs} runs]" if runs else ""
        lines = [
            f"Stimulus:  {data.get('stimulus', '-')}",
            f"Emotion:   {data.get('emotion', '-')}",
            f"Decision:  {_fmt(data.get('decision', '-'), data.get('avg_decision', ''))}",
            f"Inference: {_fmt(data.get('inference', '-'), data.get('avg_inference', ''))}",
            f"Total:     {_fmt(data.get('total', '-'), data.get('avg_total', ''))}{runs_str}",
        ]
        self._timing_label.setText("\n".join(lines))

    def _on_decision_made(self, data: dict) -> None:
        def _bar(val: float) -> str:
            """Render a value as a small text bar."""
            filled = int(val * 5)
            return "|" * filled + "." * (5 - filled)

        lines = [
            f"Empathy: {_bar(data.get('empathy', 0))}  "
            f"Warmth: {_bar(data.get('warmth', 0))}",
            f"Verbose: {_bar(data.get('verbosity', 0))}  "
            f"Assert: {_bar(data.get('assertiveness', 0))}",
            f"Curious: {_bar(data.get('curiosity', 0))}  "
            f"Caution: {_bar(data.get('caution', 0))}",
        ]

        flags = data.get("flags", {})
        active_flags = [k for k, v in flags.items() if v]
        if active_flags:
            lines.append(f"Flags: {', '.join(active_flags)}")

        self._decision_label.setText("\n".join(lines))

    def _on_metacognition(self, data: dict) -> None:
        conf = data.get("confidence", 0)
        acc = data.get("last_accuracy", 0)
        lines = [
            f"Confidence: {conf:.0%}",
            f"Accuracy:   {acc:.0%}",
            f"Predictions: {data.get('prediction_count', 0)}",
        ]
        self._meta_label.setText("\n".join(lines))

    def _on_self_dev(self, data: dict) -> None:
        # Update the metacognition label with growth info
        lr = data.get("learning_rate", 0)
        bypass = data.get("bypass_strength", 0)
        interactions = data.get("total_interactions", 0)
        pos_ratio = data.get("positive_ratio", 0.5)
        lines = [
            f"Confidence: {data.get('avg_accuracy', 0):.0%}",
            f"LR: {lr:.4f}  Bypass: {bypass:.2f}",
            f"Interactions: {interactions}  +/-: {pos_ratio:.0%}",
        ]
        self._meta_label.setText("\n".join(lines))
