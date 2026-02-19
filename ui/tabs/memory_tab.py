"""
Memory tab — Docker-based RAG / vector-store connection.

Lets the user configure and connect to a vector database running in
Docker (ChromaDB, Qdrant, Weaviate, etc.).  Connection status is shown
with the same status-dot pattern used by the LLM tab, and a
``rag_status`` event is published so other parts of the UI can react.

Supported health-check endpoints (tried in order):
  GET /api/v1/heartbeat   (ChromaDB)
  GET /health             (Qdrant, Weaviate, generic)
  GET /                   (last-resort fallback)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
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
from ui.widgets import StatusDot

from .base_tab import BaseTab


# ── Health-check worker ───────────────────────────────────────────────


class _TestWorker(QThread):
    """Probes the RAG service in a background thread."""

    succeeded = pyqtSignal(str, list)   # base_url, collection_names
    failed    = pyqtSignal(str)         # error message

    _HEALTH_PATHS = [
        "/api/v1/heartbeat",   # ChromaDB
        "/health",             # Qdrant / Weaviate / generic
        "/",                   # last resort
    ]
    _COLLECTIONS_PATHS = [
        "/api/v1/collections",   # ChromaDB
        "/collections",          # Qdrant
    ]

    def __init__(self, base_url: str, timeout: int = 6):
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._timeout  = timeout

    def run(self) -> None:
        # 1 — health probe
        reachable = False
        for path in self._HEALTH_PATHS:
            try:
                url = self._base_url + path
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=self._timeout) as r:
                    if r.status < 400:
                        reachable = True
                        break
            except Exception:
                continue

        if not reachable:
            self.failed.emit(
                f"Cannot reach {self._base_url} — "
                "make sure the Docker container is running."
            )
            return

        # 2 — try to list collections (best-effort; ignored on error)
        collections: list[str] = []
        for path in self._COLLECTIONS_PATHS:
            try:
                url = self._base_url + path
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=self._timeout) as r:
                    raw = json.loads(r.read().decode())
                    # ChromaDB: list of dicts with "name" key
                    # Qdrant:   {"result": {"collections": [{"name": ...}]}}
                    if isinstance(raw, list):
                        collections = [
                            c.get("name", str(c)) if isinstance(c, dict) else str(c)
                            for c in raw
                        ]
                    elif isinstance(raw, dict):
                        inner = (
                            raw.get("result", {}).get("collections", [])
                            or raw.get("collections", [])
                        )
                        collections = [
                            c.get("name", str(c)) if isinstance(c, dict) else str(c)
                            for c in inner
                        ]
                    if collections:
                        break
            except Exception:
                continue

        self.succeeded.emit(self._base_url, collections)


# ── Tab ──────────────────────────────────────────────────────────────


class MemoryTab(BaseTab):
    """Settings tab for the Docker-hosted RAG / vector store."""

    def __init__(self, event_bus: EventBus, config: Config):
        self._worker: Optional[_TestWorker] = None
        super().__init__(event_bus, config)

    # ══════════════════════════════════════════════════════════
    # Build
    # ══════════════════════════════════════════════════════════

    def _build(self) -> None:
        lay = self._layout

        # ── Connection ────────────────────────────────────────
        lay.addWidget(self._heading("Vector Store Connection"))

        # Backend type
        self._backend_combo = QComboBox()
        self._backend_combo.setObjectName("SettingsCombo")
        self._backend_combo.setMinimumHeight(28)
        self._backend_combo.addItems([
            "ChromaDB",
            "Qdrant",
            "Weaviate",
            "Milvus",
            "Custom",
        ])
        saved_backend = self.config.get("rag.backend", "ChromaDB")
        idx = self._backend_combo.findText(saved_backend)
        if idx >= 0:
            self._backend_combo.setCurrentIndex(idx)
        self._backend_combo.currentTextChanged.connect(self._on_backend_changed)
        lay.addWidget(self._row("Backend", self._backend_combo))

        # URL
        self._url_edit = self._make_line_edit(
            placeholder="http://localhost:8000",
            text=self.config.get("rag.url", "http://localhost:8000"),
        )
        self._url_edit.textChanged.connect(self._on_url_changed)
        lay.addWidget(self._row("URL", self._url_edit))

        # Collection / index name
        self._collection_edit = self._make_line_edit(
            placeholder="e.g. revia_memory",
            text=self.config.get("rag.collection", ""),
        )
        self._collection_edit.textChanged.connect(self._on_collection_changed)
        lay.addWidget(self._row("Collection", self._collection_edit))

        # Connect row
        conn_row = QWidget()
        ch = QHBoxLayout(conn_row)
        ch.setContentsMargins(0, 0, 0, 0)
        ch.setSpacing(8)

        self._connect_btn = QPushButton("Test Connection")
        self._connect_btn.setObjectName("ModeButton")
        self._connect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._connect_btn.clicked.connect(self._on_connect)
        ch.addWidget(self._connect_btn)

        self._disconnect_btn = QPushButton("Disconnect")
        self._disconnect_btn.setObjectName("ModeButton")
        self._disconnect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._disconnect_btn.setStyleSheet(
            "QPushButton { color:#ef476f; } QPushButton:hover { border-color:#ef476f; }"
        )
        self._disconnect_btn.setEnabled(False)
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        ch.addWidget(self._disconnect_btn)

        self._conn_dot = StatusDot("off", size=10)
        ch.addWidget(self._conn_dot, alignment=Qt.AlignmentFlag.AlignVCenter)
        ch.addStretch(1)
        lay.addWidget(conn_row)

        self._conn_status_label = QLabel("Not connected")
        self._conn_status_label.setStyleSheet("color:#8fa6c3; font-size:11px; padding:2px 0;")
        lay.addWidget(self._conn_status_label)

        # ── Discovered collections ─────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#263246;")
        lay.addWidget(sep)

        lay.addWidget(self._heading("Discovered Collections"))

        self._collections_label = QLabel("(connect to discover)")
        self._collections_label.setStyleSheet("color:#8fa6c3; font-size:11px; padding:2px 0;")
        self._collections_label.setWordWrap(True)
        lay.addWidget(self._collections_label)

        # ── Docker hint ───────────────────────────────────────
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#263246;")
        lay.addWidget(sep2)

        lay.addWidget(self._heading("Docker Quick-Start"))

        hint = QTextEdit()
        hint.setReadOnly(True)
        hint.setObjectName("MonoInfo")
        hint.setMaximumHeight(140)
        hint.setStyleSheet(
            "QTextEdit {"
            "  background:#0a0f18;"
            "  border:1px solid #263246;"
            "  border-radius:6px;"
            "  color:#8fa6c3;"
            "  font-family:'Consolas','Courier New',monospace;"
            "  font-size:11px;"
            "  padding:6px;"
            "}"
        )
        hint.setPlainText(
            "# ChromaDB\n"
            "docker run -d -p 8000:8000 chromadb/chroma\n\n"
            "# Qdrant\n"
            "docker run -d -p 6333:6333 qdrant/qdrant\n\n"
            "# Weaviate\n"
            "docker run -d -p 8080:8080 \\\n"
            "  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \\\n"
            "  semitechnologies/weaviate:latest"
        )
        lay.addWidget(hint)

    # ══════════════════════════════════════════════════════════
    # Event handlers
    # ══════════════════════════════════════════════════════════

    def _on_backend_changed(self, text: str) -> None:
        self.config.set("rag.backend", text)
        # Update default URL hint
        defaults = {
            "ChromaDB":  "http://localhost:8000",
            "Qdrant":    "http://localhost:6333",
            "Weaviate":  "http://localhost:8080",
            "Milvus":    "http://localhost:19530",
        }
        if text in defaults:
            current = self._url_edit.text().strip()
            # Only auto-fill if the field looks like a default URL
            if not current or any(
                current == v for v in defaults.values()
            ):
                self._url_edit.blockSignals(True)
                self._url_edit.setText(defaults[text])
                self._url_edit.blockSignals(False)
                self.config.set("rag.url", defaults[text])

    def _on_url_changed(self, text: str) -> None:
        self.config.set("rag.url", text.strip())

    def _on_collection_changed(self, text: str) -> None:
        self.config.set("rag.collection", text.strip())

    def _on_connect(self) -> None:
        url = self._url_edit.text().strip()
        if not url:
            self._conn_status_label.setText("Enter a URL first.")
            self._conn_status_label.setStyleSheet("color:#ef476f; font-size:11px; padding:2px 0;")
            return

        self._connect_btn.setEnabled(False)
        self._conn_dot.set_status("off")
        self._conn_status_label.setText("Connecting to Docker container...")
        self._conn_status_label.setStyleSheet("color:#f9c74f; font-size:11px; padding:2px 0;")
        self._collections_label.setText("(discovering...)")

        self._worker = _TestWorker(url)
        self._worker.succeeded.connect(self._on_connect_success)
        self._worker.failed.connect(self._on_connect_error)
        self._worker.start()

    def _on_connect_success(self, base_url: str, collections: list) -> None:
        backend = self._backend_combo.currentText()
        self._conn_dot.set_status("on")
        self._connect_btn.setEnabled(False)
        self._disconnect_btn.setEnabled(True)
        self._conn_status_label.setText(
            f"Connected: {backend}  |  {base_url}"
        )
        self._conn_status_label.setStyleSheet("color:#33d17a; font-size:11px; padding:2px 0;")

        if collections:
            self._collections_label.setText(", ".join(collections))
            # Auto-fill collection field if it's empty and we found exactly one
            if not self._collection_edit.text().strip() and len(collections) == 1:
                self._collection_edit.setText(collections[0])
                self.config.set("rag.collection", collections[0])
        else:
            self._collections_label.setText("(none found — enter name manually)")

        self.config.set("rag.connected", True)
        self.bus.publish("rag_status", {
            "connected": True,
            "url": base_url,
            "backend": backend,
            "collections": collections,
        })
        col_str = f" ({len(collections)} collections)" if collections else ""
        self.bus.publish("activity_log", {
            "text": f"[System] Memory connected: {backend} at {base_url}{col_str}",
        })

    def _on_connect_error(self, error_msg: str) -> None:
        self._conn_dot.set_status("off")
        self._connect_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(False)
        self._conn_status_label.setText(f"Failed: {error_msg}")
        self._conn_status_label.setStyleSheet("color:#ef476f; font-size:11px; padding:2px 0;")
        self._collections_label.setText("(connect to discover)")

        self.config.set("rag.connected", False)
        self.bus.publish("rag_status", {"connected": False, "url": "", "backend": "", "collections": []})

    def _on_disconnect(self) -> None:
        self._conn_dot.set_status("off")
        self._connect_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(False)
        self._conn_status_label.setText("Disconnected")
        self._conn_status_label.setStyleSheet("color:#8fa6c3; font-size:11px; padding:2px 0;")
        self._collections_label.setText("(connect to discover)")

        self.config.set("rag.connected", False)
        self.bus.publish("rag_status", {"connected": False, "url": "", "backend": "", "collections": []})
