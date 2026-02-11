"""
LLM tab — Local / Online toggle with saved model libraries.

**Local mode**  – browse for a model file on disk, enter a local
server address (llama.cpp, Ollama, LM Studio, etc.).  The file path
and address are saved in a dropdown so the user can switch quickly.

**Online mode** – pick a major API provider (OpenAI, Anthropic, etc.),
enter an API key, and optionally a custom endpoint.  Each saved
configuration appears in a dropdown.

Both sides persist to ``config.json`` and publish ``model_changed``
events so the rest of the UI stays in sync.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.config import Config
from core.events import EventBus
from core.plugin_manager import PluginManager
from ui.widgets import ToggleSwitch

from .base_tab import BaseTab

# Major API providers shown in the Online provider dropdown
_ONLINE_PROVIDERS = [
    "OpenAI",
    "Anthropic",
    "Google Gemini",
    "Mistral",
    "Cohere",
    "Groq",
    "Together AI",
    "OpenRouter",
    "Perplexity",
    "DeepSeek",
    "Custom",
]


class LLMTab(BaseTab):

    def __init__(self, event_bus: EventBus, config: Config, plugin_manager: PluginManager):
        self._pm = plugin_manager
        super().__init__(event_bus, config)

    # ══════════════════════════════════════════════════════════
    # Build
    # ══════════════════════════════════════════════════════════

    def _build(self) -> None:
        lay = self._layout

        # ── Mode toggle (Local ↔ Online) ─────────────────────
        mode_row = QWidget()
        mh = QHBoxLayout(mode_row)
        mh.setContentsMargins(0, 0, 0, 0)
        mh.setSpacing(10)

        self._local_label = QLabel("Local")
        self._local_label.setStyleSheet("font-weight:700; font-size:13px;")
        mh.addWidget(self._local_label)

        is_online = self.config.get("llm.mode", "local") == "online"
        self._mode_toggle = ToggleSwitch(checked=is_online, width=50, height=24)
        self._mode_toggle.toggled.connect(self._on_mode_toggled)
        mh.addWidget(self._mode_toggle, alignment=Qt.AlignmentFlag.AlignVCenter)

        self._online_label = QLabel("Online")
        self._online_label.setStyleSheet("font-weight:700; font-size:13px;")
        mh.addWidget(self._online_label)

        mh.addStretch(1)
        lay.addWidget(mode_row)

        # ── Stacked pages ────────────────────────────────────
        self._stack = QStackedWidget()
        self._local_page = self._build_local_page()
        self._online_page = self._build_online_page()
        self._stack.addWidget(self._local_page)
        self._stack.addWidget(self._online_page)
        lay.addWidget(self._stack, 1)

        # Set initial page
        self._stack.setCurrentIndex(1 if is_online else 0)
        self._update_mode_labels(is_online)

        # Load saved entries
        self._load_local_models()
        self._load_online_models()

    # ── Mode switch ───────────────────────────────────────────

    def _on_mode_toggled(self, online: bool) -> None:
        self._stack.setCurrentIndex(1 if online else 0)
        self.config.set("llm.mode", "online" if online else "local")
        self._update_mode_labels(online)
        self.bus.publish("llm_mode_changed", {"mode": "online" if online else "local"})

    def _update_mode_labels(self, online: bool) -> None:
        self._local_label.setStyleSheet(
            f"font-weight:700; font-size:13px; color:{'#8fa6c3' if online else '#33d17a'};"
        )
        self._online_label.setStyleSheet(
            f"font-weight:700; font-size:13px; color:{'#33d17a' if online else '#8fa6c3'};"
        )

    # ══════════════════════════════════════════════════════════
    # LOCAL page
    # ══════════════════════════════════════════════════════════

    def _build_local_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(10)

        lay.addWidget(self._heading("Local Model"))

        # Saved models dropdown
        self._local_combo = QComboBox()
        self._local_combo.setObjectName("SettingsCombo")
        self._local_combo.setMinimumHeight(28)
        self._local_combo.currentIndexChanged.connect(self._on_local_selected)
        lay.addWidget(self._row("Model", self._local_combo))

        # Browse + Remove
        btn_row = QWidget()
        bh = QHBoxLayout(btn_row)
        bh.setContentsMargins(0, 0, 0, 0)
        bh.setSpacing(8)

        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.setObjectName("ModeButton")
        self._browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._browse_btn.clicked.connect(self._on_browse)
        bh.addWidget(self._browse_btn)

        self._local_remove_btn = QPushButton("Remove")
        self._local_remove_btn.setObjectName("ModeButton")
        self._local_remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._local_remove_btn.setStyleSheet(
            "QPushButton { color:#ef476f; } QPushButton:hover { border-color:#ef476f; }"
        )
        self._local_remove_btn.clicked.connect(self._on_local_remove)
        bh.addWidget(self._local_remove_btn)

        bh.addStretch(1)
        lay.addWidget(btn_row)

        # Rename
        rename_row = QWidget()
        rh = QHBoxLayout(rename_row)
        rh.setContentsMargins(0, 0, 0, 0)
        rh.setSpacing(8)

        self._local_rename_edit = QLineEdit()
        self._local_rename_edit.setObjectName("SettingsLineEdit")
        self._local_rename_edit.setPlaceholderText("Give this model a friendly name...")
        rh.addWidget(self._local_rename_edit, 1)

        self._local_rename_btn = QPushButton("Rename")
        self._local_rename_btn.setObjectName("ModeButton")
        self._local_rename_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._local_rename_btn.clicked.connect(self._on_local_rename)
        rh.addWidget(self._local_rename_btn)

        lay.addWidget(self._row("Name", rename_row))

        # Local server address
        self._local_address = self._make_line_edit(
            placeholder="http://localhost:8080/v1",
        )
        self._local_address.textChanged.connect(self._on_local_address_changed)
        lay.addWidget(self._row("Address", self._local_address))

        # Path display
        self._local_path_label = QLabel("No model selected")
        self._local_path_label.setObjectName("PillSub")
        self._local_path_label.setWordWrap(True)
        self._local_path_label.setStyleSheet("color:#8fa6c3; font-size:11px; padding:4px 0;")
        lay.addWidget(self._local_path_label)

        lay.addStretch(1)
        return page

    # ── Local data helpers ────────────────────────────────────

    def _get_local_models(self) -> list[dict]:
        return self.config.get("llm.local_models", [])

    def _save_local_models(self, models: list[dict]) -> None:
        self.config.set("llm.local_models", models)

    def _load_local_models(self) -> None:
        self._local_combo.blockSignals(True)
        self._local_combo.clear()

        models = self._get_local_models()
        if not models:
            self._local_combo.addItem("(no models added)")
        else:
            for m in models:
                self._local_combo.addItem(m["name"], m["path"])

        selected = self.config.get("llm.local_selected", "")
        if selected:
            idx = self._local_combo.findText(selected)
            if idx >= 0:
                self._local_combo.setCurrentIndex(idx)

        self._local_combo.blockSignals(False)
        self._update_local_details()

    # ── Local event handlers ──────────────────────────────────

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select LLM Model File",
            "",
            "Model Files (*.gguf *.bin *.onnx *.pt *.pth *.safetensors *.ggml);;"
            "All Files (*)",
        )
        if not path:
            return

        models = self._get_local_models()
        for m in models:
            if m["path"] == path:
                QMessageBox.information(
                    self, "Already Added",
                    f"This model is already in your library as \"{m['name']}\".",
                )
                idx = self._local_combo.findData(path)
                if idx >= 0:
                    self._local_combo.setCurrentIndex(idx)
                return

        file_name = Path(path).stem
        models.append({"name": file_name, "path": path, "address": ""})
        self._save_local_models(models)
        self._load_local_models()

        idx = self._local_combo.findData(path)
        if idx >= 0:
            self._local_combo.setCurrentIndex(idx)

        QMessageBox.information(
            self,
            "Model Added",
            f"Added \"{file_name}\" to your library.\n\n"
            "Tip: Use the Name field to give this model a friendly "
            "name so you can identify it easily later.",
        )

    def _on_local_remove(self) -> None:
        idx = self._local_combo.currentIndex()
        models = self._get_local_models()
        if not models or idx < 0 or idx >= len(models):
            return

        name = models[idx]["name"]
        reply = QMessageBox.question(
            self, "Remove Model",
            f"Remove \"{name}\" from your library?\n\n"
            "The file itself is not deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        models.pop(idx)
        self._save_local_models(models)
        self._load_local_models()
        self.bus.publish("model_changed", {"model": None, "mode": "local"})

    def _on_local_rename(self) -> None:
        new_name = self._local_rename_edit.text().strip()
        if not new_name:
            return
        idx = self._local_combo.currentIndex()
        models = self._get_local_models()
        if not models or idx < 0 or idx >= len(models):
            return

        models[idx]["name"] = new_name
        self._save_local_models(models)
        self.config.set("llm.local_selected", new_name, save=False)
        self._load_local_models()
        self._local_rename_edit.clear()

    def _on_local_address_changed(self, text: str) -> None:
        idx = self._local_combo.currentIndex()
        models = self._get_local_models()
        if not models or idx < 0 or idx >= len(models):
            return
        models[idx]["address"] = text
        self._save_local_models(models)

    def _on_local_selected(self, idx: int) -> None:
        models = self._get_local_models()
        if not models or idx < 0 or idx >= len(models):
            self._update_local_details()
            return

        entry = models[idx]
        path = entry["path"]

        if not Path(path).is_file():
            QMessageBox.warning(
                self, "Model File Not Found",
                f"The file for \"{entry['name']}\" no longer exists:\n\n"
                f"{path}\n\n"
                "It may have been moved or deleted. "
                "This entry will be removed from your library.",
            )
            models.pop(idx)
            self._save_local_models(models)
            self._load_local_models()
            self.bus.publish("model_changed", {"model": None, "mode": "local"})
            return

        self.config.set("llm.local_selected", entry["name"])
        self._update_local_details()
        self.bus.publish("model_changed", {
            "model": entry["name"],
            "path": path,
            "address": entry.get("address", ""),
            "mode": "local",
        })

    def _update_local_details(self) -> None:
        idx = self._local_combo.currentIndex()
        models = self._get_local_models()
        has = models and 0 <= idx < len(models)

        self._local_remove_btn.setEnabled(has)
        self._local_rename_btn.setEnabled(has)
        self._local_rename_edit.setEnabled(has)
        self._local_address.setEnabled(has)

        if has:
            entry = models[idx]
            self._local_path_label.setText(f"Path: {entry['path']}")
            self._local_rename_edit.setPlaceholderText(f"Currently: {entry['name']}")
            self._local_address.blockSignals(True)
            self._local_address.setText(entry.get("address", ""))
            self._local_address.blockSignals(False)
        else:
            self._local_path_label.setText("No model selected")
            self._local_rename_edit.setPlaceholderText("Add a model first...")
            self._local_address.blockSignals(True)
            self._local_address.clear()
            self._local_address.blockSignals(False)

    # ══════════════════════════════════════════════════════════
    # ONLINE page
    # ══════════════════════════════════════════════════════════

    def _build_online_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(10)

        lay.addWidget(self._heading("Online LLM"))

        # Saved configurations dropdown
        self._online_combo = QComboBox()
        self._online_combo.setObjectName("SettingsCombo")
        self._online_combo.setMinimumHeight(28)
        self._online_combo.currentIndexChanged.connect(self._on_online_selected)
        lay.addWidget(self._row("Saved", self._online_combo))

        # Add + Remove
        btn_row = QWidget()
        bh = QHBoxLayout(btn_row)
        bh.setContentsMargins(0, 0, 0, 0)
        bh.setSpacing(8)

        self._online_add_btn = QPushButton("Add New...")
        self._online_add_btn.setObjectName("ModeButton")
        self._online_add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._online_add_btn.clicked.connect(self._on_online_add)
        bh.addWidget(self._online_add_btn)

        self._online_remove_btn = QPushButton("Remove")
        self._online_remove_btn.setObjectName("ModeButton")
        self._online_remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._online_remove_btn.setStyleSheet(
            "QPushButton { color:#ef476f; } QPushButton:hover { border-color:#ef476f; }"
        )
        self._online_remove_btn.clicked.connect(self._on_online_remove)
        bh.addWidget(self._online_remove_btn)

        bh.addStretch(1)
        lay.addWidget(btn_row)

        # Rename
        rename_row = QWidget()
        rh = QHBoxLayout(rename_row)
        rh.setContentsMargins(0, 0, 0, 0)
        rh.setSpacing(8)

        self._online_rename_edit = QLineEdit()
        self._online_rename_edit.setObjectName("SettingsLineEdit")
        self._online_rename_edit.setPlaceholderText("Give this config a friendly name...")
        rh.addWidget(self._online_rename_edit, 1)

        self._online_rename_btn = QPushButton("Rename")
        self._online_rename_btn.setObjectName("ModeButton")
        self._online_rename_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._online_rename_btn.clicked.connect(self._on_online_rename)
        rh.addWidget(self._online_rename_btn)

        lay.addWidget(self._row("Name", rename_row))

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#263246;")
        lay.addWidget(sep)

        lay.addWidget(self._heading("Connection"))

        # Provider dropdown
        self._provider_combo = QComboBox()
        self._provider_combo.setObjectName("SettingsCombo")
        self._provider_combo.addItems(_ONLINE_PROVIDERS)
        self._provider_combo.currentTextChanged.connect(self._on_online_field_changed)
        lay.addWidget(self._row("Provider", self._provider_combo))

        # API key
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setObjectName("SettingsLineEdit")
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("sk-...")
        self._api_key_edit.textChanged.connect(self._on_online_field_changed)
        lay.addWidget(self._row("API Key", self._api_key_edit))

        # Endpoint / base URL
        self._endpoint_edit = QLineEdit()
        self._endpoint_edit.setObjectName("SettingsLineEdit")
        self._endpoint_edit.setPlaceholderText("(uses default for provider)")
        self._endpoint_edit.textChanged.connect(self._on_online_field_changed)
        lay.addWidget(self._row("Endpoint", self._endpoint_edit))

        # Model ID
        self._model_id_edit = QLineEdit()
        self._model_id_edit.setObjectName("SettingsLineEdit")
        self._model_id_edit.setPlaceholderText("e.g. gpt-4o, claude-sonnet-4-5-20250929...")
        self._model_id_edit.textChanged.connect(self._on_online_field_changed)
        lay.addWidget(self._row("Model ID", self._model_id_edit))

        # Status label
        self._online_status = QLabel("")
        self._online_status.setObjectName("PillSub")
        self._online_status.setWordWrap(True)
        self._online_status.setStyleSheet("color:#8fa6c3; font-size:11px; padding:4px 0;")
        lay.addWidget(self._online_status)

        lay.addStretch(1)
        return page

    # ── Online data helpers ───────────────────────────────────

    def _get_online_models(self) -> list[dict]:
        return self.config.get("llm.online_models", [])

    def _save_online_models(self, models: list[dict]) -> None:
        self.config.set("llm.online_models", models)

    def _load_online_models(self) -> None:
        self._online_combo.blockSignals(True)
        self._online_combo.clear()

        models = self._get_online_models()
        if not models:
            self._online_combo.addItem("(no configs saved)")
        else:
            for m in models:
                self._online_combo.addItem(m["name"])

        selected = self.config.get("llm.online_selected", "")
        if selected:
            idx = self._online_combo.findText(selected)
            if idx >= 0:
                self._online_combo.setCurrentIndex(idx)

        self._online_combo.blockSignals(False)
        self._update_online_details()

    # ── Online event handlers ─────────────────────────────────

    def _on_online_add(self) -> None:
        provider = self._provider_combo.currentText()
        models = self._get_online_models()

        count = sum(1 for m in models if m.get("provider") == provider) + 1
        name = provider if count == 1 else f"{provider} ({count})"

        models.append({
            "name": name,
            "provider": provider,
            "api_key": "",
            "endpoint": "",
            "model_id": "",
        })
        self._save_online_models(models)
        self._load_online_models()

        idx = self._online_combo.findText(name)
        if idx >= 0:
            self._online_combo.setCurrentIndex(idx)

        QMessageBox.information(
            self,
            "Config Added",
            f"Added \"{name}\" to your saved configs.\n\n"
            "Fill in your API key and model ID, then use the Name "
            "field to give it a name you'll recognise.",
        )

    def _on_online_remove(self) -> None:
        idx = self._online_combo.currentIndex()
        models = self._get_online_models()
        if not models or idx < 0 or idx >= len(models):
            return

        name = models[idx]["name"]
        reply = QMessageBox.question(
            self, "Remove Config",
            f"Remove \"{name}\" and its saved API key?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        models.pop(idx)
        self._save_online_models(models)
        self._load_online_models()
        self.bus.publish("model_changed", {"model": None, "mode": "online"})

    def _on_online_rename(self) -> None:
        new_name = self._online_rename_edit.text().strip()
        if not new_name:
            return
        idx = self._online_combo.currentIndex()
        models = self._get_online_models()
        if not models or idx < 0 or idx >= len(models):
            return

        models[idx]["name"] = new_name
        self._save_online_models(models)
        self.config.set("llm.online_selected", new_name, save=False)
        self._load_online_models()
        self._online_rename_edit.clear()

    def _on_online_selected(self, idx: int) -> None:
        models = self._get_online_models()
        if not models or idx < 0 or idx >= len(models):
            self._update_online_details()
            return

        entry = models[idx]
        self.config.set("llm.online_selected", entry["name"])
        self._update_online_details()
        self.bus.publish("model_changed", {
            "model": entry["name"],
            "provider": entry.get("provider", ""),
            "model_id": entry.get("model_id", ""),
            "mode": "online",
        })

    def _on_online_field_changed(self, *_args) -> None:
        """Save current field values back into the active config entry."""
        idx = self._online_combo.currentIndex()
        models = self._get_online_models()
        if not models or idx < 0 or idx >= len(models):
            return

        models[idx]["provider"] = self._provider_combo.currentText()
        models[idx]["api_key"] = self._api_key_edit.text()
        models[idx]["endpoint"] = self._endpoint_edit.text()
        models[idx]["model_id"] = self._model_id_edit.text()
        self._save_online_models(models)

        # Update status hint
        prov = models[idx]["provider"]
        has_key = bool(models[idx]["api_key"])
        has_model = bool(models[idx]["model_id"])
        if has_key and has_model:
            self._online_status.setText(f"Ready — {prov} / {models[idx]['model_id']}")
            self._online_status.setStyleSheet("color:#33d17a; font-size:11px; padding:4px 0;")
        elif has_key:
            self._online_status.setText("API key set — enter a Model ID to finish setup")
            self._online_status.setStyleSheet("color:#f9c74f; font-size:11px; padding:4px 0;")
        else:
            self._online_status.setText("Enter an API key to connect")
            self._online_status.setStyleSheet("color:#8fa6c3; font-size:11px; padding:4px 0;")

    def _update_online_details(self) -> None:
        idx = self._online_combo.currentIndex()
        models = self._get_online_models()
        has = models and 0 <= idx < len(models)

        self._online_remove_btn.setEnabled(has)
        self._online_rename_btn.setEnabled(has)
        self._online_rename_edit.setEnabled(has)
        self._provider_combo.setEnabled(has)
        self._api_key_edit.setEnabled(has)
        self._endpoint_edit.setEnabled(has)
        self._model_id_edit.setEnabled(has)

        if has:
            entry = models[idx]
            self._online_rename_edit.setPlaceholderText(f"Currently: {entry['name']}")

            self._provider_combo.blockSignals(True)
            pidx = self._provider_combo.findText(entry.get("provider", "OpenAI"))
            if pidx >= 0:
                self._provider_combo.setCurrentIndex(pidx)
            self._provider_combo.blockSignals(False)

            self._api_key_edit.blockSignals(True)
            self._api_key_edit.setText(entry.get("api_key", ""))
            self._api_key_edit.blockSignals(False)

            self._endpoint_edit.blockSignals(True)
            self._endpoint_edit.setText(entry.get("endpoint", ""))
            self._endpoint_edit.blockSignals(False)

            self._model_id_edit.blockSignals(True)
            self._model_id_edit.setText(entry.get("model_id", ""))
            self._model_id_edit.blockSignals(False)

            # Trigger status update
            self._on_online_field_changed()
        else:
            self._online_rename_edit.setPlaceholderText("Add a config first...")
            self._api_key_edit.blockSignals(True)
            self._api_key_edit.clear()
            self._api_key_edit.blockSignals(False)
            self._endpoint_edit.blockSignals(True)
            self._endpoint_edit.clear()
            self._endpoint_edit.blockSignals(False)
            self._model_id_edit.blockSignals(True)
            self._model_id_edit.clear()
            self._model_id_edit.blockSignals(False)
            self._online_status.setText("")
