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
from core.llm_registry import LLMRegistry
from core.plugin_manager import PluginManager
from ui.widgets import StatusDot, ToggleSwitch

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
        self._registry = LLMRegistry()
        super().__init__(event_bus, config)

    # ══════════════════════════════════════════════════════════
    # Build
    # ══════════════════════════════════════════════════════════

    def _build(self) -> None:
        lay = self._layout

        # ── Backend selection (plugin) ────────────────────────
        lay.addWidget(self._heading("Backend"))

        self._backend_combo = QComboBox()
        self._backend_combo.setObjectName("SettingsCombo")
        self._backend_combo.setMinimumHeight(28)
        self._populate_backends()
        lay.addWidget(self._row("Server", self._backend_combo))

        # Connect / Disconnect row
        conn_row = QWidget()
        ch = QHBoxLayout(conn_row)
        ch.setContentsMargins(0, 0, 0, 0)
        ch.setSpacing(8)

        self._connect_btn = QPushButton("Connect")
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

        # ── Active model selector (populated after connect) ────
        model_sel_row = QWidget()
        msr = QHBoxLayout(model_sel_row)
        msr.setContentsMargins(0, 0, 0, 0)
        msr.setSpacing(8)

        self._model_select_combo = QComboBox()
        self._model_select_combo.setObjectName("SettingsCombo")
        self._model_select_combo.setMinimumHeight(28)
        self._model_select_combo.setEnabled(False)
        self._model_select_combo.addItem("(connect to see models)")
        self._model_select_combo.currentIndexChanged.connect(self._on_model_selected)
        msr.addWidget(self._model_select_combo, 1)

        self._refresh_models_btn = QPushButton("Refresh")
        self._refresh_models_btn.setObjectName("ModeButton")
        self._refresh_models_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_models_btn.setEnabled(False)
        self._refresh_models_btn.clicked.connect(self._on_refresh_models)
        msr.addWidget(self._refresh_models_btn)

        lay.addWidget(self._row("Active Model", model_sel_row))

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#263246;")
        lay.addWidget(sep)

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
        self._registry.upsert_local(name=file_name, path=path)
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
            QMessageBox.warning(
                self, "No Model Selected",
                "Please select a model from the dropdown before removing.",
            )
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

        reg_entry = self._registry.find_by_name(name)
        if reg_entry:
            self._registry.remove(reg_entry["id"])
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

        old_name = models[idx]["name"]
        reg_entry = self._registry.find_by_name(old_name)
        if reg_entry:
            self._registry.rename(reg_entry["id"], new_name)

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
        entry = models[idx]
        self._registry.upsert_local(
            name=entry["name"], path=entry["path"], address=text,
        )

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
        reg = self._registry.find_by_name(entry["name"]) or {}
        self.bus.publish("model_changed", {
            "model": entry["name"],
            "path": path,
            "address": entry.get("address", ""),
            "mode": "local",
            "registry": reg,
        })

    def _update_local_details(self) -> None:
        idx = self._local_combo.currentIndex()
        models = self._get_local_models()
        has = bool(models and 0 <= idx < len(models))

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
        self._registry.upsert_online(name=name, provider=provider)
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
            QMessageBox.warning(
                self, "No LLM Selected",
                "Please select an LLM config from the dropdown before removing.",
            )
            return

        name = models[idx]["name"]
        reply = QMessageBox.question(
            self, "Remove Config",
            f"Remove \"{name}\" and its saved API key?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        reg_entry = self._registry.find_by_name(name)
        if reg_entry:
            self._registry.remove(reg_entry["id"])
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

        old_name = models[idx]["name"]
        reg_entry = self._registry.find_by_name(old_name)
        if reg_entry:
            self._registry.rename(reg_entry["id"], new_name)

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
        reg = self._registry.find_by_name(entry["name"]) or {}
        self.bus.publish("model_changed", {
            "model": entry["name"],
            "provider": entry.get("provider", ""),
            "model_id": entry.get("model_id", ""),
            "mode": "online",
            "registry": reg,
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
        self._registry.upsert_online(
            name=models[idx]["name"],
            provider=models[idx]["provider"],
            api_key=models[idx]["api_key"],
            endpoint=models[idx]["endpoint"],
            model_id=models[idx]["model_id"],
        )

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
        has = bool(models and 0 <= idx < len(models))

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

    # ══════════════════════════════════════════════════════════
    # Backend (plugin) management
    # ══════════════════════════════════════════════════════════

    # Plugin name → user-friendly display label
    _BACKEND_LABELS: dict[str, str] = {
        "ollama_plugin":   "Ollama (Local)",
        "llamacpp_plugin": "llama.cpp (Local)",
        "lmstudio_plugin": "LM Studio (Local)",
        "openai_plugin":   "OpenAI API (Online)",
        "example_plugin":  "Example (Offline Stub)",
    }

    def _populate_backends(self) -> None:
        """Fill the backend dropdown with discovered plugins."""
        self._backend_combo.clear()
        available = self._pm.available
        for name in available:
            label = self._BACKEND_LABELS.get(name, name)
            self._backend_combo.addItem(label, name)

        # Restore last used backend
        last = self.config.get("llm.backend", "")
        if last:
            idx = self._backend_combo.findData(last)
            if idx >= 0:
                self._backend_combo.setCurrentIndex(idx)

    def _on_connect(self) -> None:
        """Activate the selected plugin and connect it."""
        idx = self._backend_combo.currentIndex()
        if idx < 0:
            return

        plugin_name = self._backend_combo.currentData()
        if not plugin_name:
            return

        # Build config from current UI state
        cfg: dict = {}
        is_online = self.config.get("llm.mode", "local") == "online"

        if is_online:
            models = self._get_online_models()
            oidx = self._online_combo.currentIndex()
            if models and 0 <= oidx < len(models):
                entry = models[oidx]
                cfg["api_key"] = entry.get("api_key", "")
                cfg["base_url"] = entry.get("endpoint", "")
                cfg["provider"] = entry.get("provider", "")
                cfg["model_id"] = entry.get("model_id", "")
        else:
            models = self._get_local_models()
            lidx = self._local_combo.currentIndex()
            if models and 0 <= lidx < len(models):
                entry = models[lidx]
                cfg["base_url"] = entry.get("address", "")
                cfg["model_name"] = entry.get("name", "")
                cfg["model_path"] = entry.get("path", "")

        try:
            plugin = self._pm.activate(plugin_name, cfg)
            self.config.set("llm.backend", plugin_name)
            self._conn_dot.set_status("on")
            self._connect_btn.setEnabled(False)
            self._disconnect_btn.setEnabled(True)

            # Populate the model selector from the server's loaded models
            self._populate_model_selector(plugin)

            # Select the correct model on the now-connected plugin
            desired_model_id = None
            if is_online and cfg.get("model_id"):
                desired_model_id = cfg["model_id"]
            elif not is_online and cfg.get("model_name"):
                # For local: try to match by name against server models
                desired_model_id = self._match_local_model(
                    plugin, cfg["model_name"], cfg.get("model_path", ""),
                )

            if desired_model_id:
                try:
                    plugin.select_model(desired_model_id)
                    # Sync the combo to match
                    idx = self._model_select_combo.findData(desired_model_id)
                    if idx >= 0:
                        self._model_select_combo.blockSignals(True)
                        self._model_select_combo.setCurrentIndex(idx)
                        self._model_select_combo.blockSignals(False)
                except Exception:
                    pass

            # Report the active model
            active = plugin.active_model()
            model_name = active.name if active else "?"
            self._conn_status_label.setText(
                f"Connected: {plugin.name}  |  Model: {model_name}"
            )
            self._conn_status_label.setStyleSheet(
                "color:#33d17a; font-size:11px; padding:2px 0;"
            )

            # Publish model info so the top bar and inference panel update
            if active:
                self.bus.publish("model_changed", {
                    "model": active.name,
                    "mode": "online" if is_online else "local",
                    "registry": active.metadata,
                })

            self.bus.publish("log_entry", {
                "category": "Allowed",
                "text": f"Connected to {plugin.name} — model: {model_name}",
            })
        except Exception as e:
            self._conn_dot.set_status("off")
            self._conn_status_label.setText(f"Failed: {e}")
            self._conn_status_label.setStyleSheet(
                "color:#ef476f; font-size:11px; padding:2px 0;"
            )
            self._model_select_combo.setEnabled(False)
            self._refresh_models_btn.setEnabled(False)
            self.bus.publish("log_entry", {
                "category": "Filtered",
                "text": f"Connection failed: {e}",
            })

    def _on_disconnect(self) -> None:
        """Deactivate the current plugin."""
        self._pm.deactivate()
        self._conn_dot.set_status("off")
        self._conn_status_label.setText("Disconnected")
        self._conn_status_label.setStyleSheet(
            "color:#8fa6c3; font-size:11px; padding:2px 0;"
        )
        self._connect_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(False)

        # Clear model selector
        self._model_select_combo.blockSignals(True)
        self._model_select_combo.clear()
        self._model_select_combo.addItem("(connect to see models)")
        self._model_select_combo.setEnabled(False)
        self._model_select_combo.blockSignals(False)
        self._refresh_models_btn.setEnabled(False)

        self.bus.publish("log_entry", {
            "category": "Allowed",
            "text": "Disconnected from AI backend",
        })

    # ══════════════════════════════════════════════════════════
    # Model selector (server-side model discovery)
    # ══════════════════════════════════════════════════════════

    def _populate_model_selector(self, plugin) -> None:
        """Populate the Active Model dropdown from the plugin's loaded models."""
        self._model_select_combo.blockSignals(True)
        self._model_select_combo.clear()

        models = plugin.list_models()
        if not models:
            self._model_select_combo.addItem("(no models found on server)")
            self._model_select_combo.setEnabled(False)
        else:
            for m in models:
                # Show name in the label, store model id as data
                label = m.name
                if m.context_window and m.context_window > 0:
                    label += f"  ({m.context_window // 1024}k ctx)"
                meta = m.metadata or {}
                if meta.get("parameter_size"):
                    label += f"  [{meta['parameter_size']}]"
                if meta.get("quantization"):
                    label += f"  {meta['quantization']}"
                self._model_select_combo.addItem(label, m.id)
            self._model_select_combo.setEnabled(True)

            # Select the currently active model
            active = plugin.active_model()
            if active:
                idx = self._model_select_combo.findData(active.id)
                if idx >= 0:
                    self._model_select_combo.setCurrentIndex(idx)

        self._model_select_combo.blockSignals(False)
        self._refresh_models_btn.setEnabled(True)

    def _on_model_selected(self, idx: int) -> None:
        """User picked a different model from the dropdown."""
        if idx < 0:
            return
        model_id = self._model_select_combo.currentData()
        if not model_id:
            return

        plugin = self._pm.active_plugin
        if plugin is None or not plugin.is_connected():
            return

        try:
            plugin.select_model(model_id)
            active = plugin.active_model()
            model_name = active.name if active else model_id
            self._conn_status_label.setText(
                f"Connected: {plugin.name}  |  Model: {model_name}"
            )
            self._conn_status_label.setStyleSheet(
                "color:#33d17a; font-size:11px; padding:2px 0;"
            )

            is_online = self.config.get("llm.mode", "local") == "online"
            self.bus.publish("model_changed", {
                "model": model_name,
                "mode": "online" if is_online else "local",
                "registry": active.metadata if active else {},
            })

            self.bus.publish("log_entry", {
                "category": "Allowed",
                "text": f"Switched model to: {model_name}",
            })
        except Exception as e:
            self._conn_status_label.setText(f"Model switch failed: {e}")
            self._conn_status_label.setStyleSheet(
                "color:#ef476f; font-size:11px; padding:2px 0;"
            )

    def _on_refresh_models(self) -> None:
        """Re-query the server for available models."""
        plugin = self._pm.active_plugin
        if plugin is None or not plugin.is_connected():
            return

        # Force a fresh fetch by clearing the cache
        try:
            plugin._models = []
            plugin._models = plugin._fetch_models()
        except Exception:
            pass

        self._populate_model_selector(plugin)

    @staticmethod
    def _match_local_model(plugin, model_name: str, model_path: str) -> str | None:
        """
        Try to match a user's local library entry against the server's
        loaded models.  Returns the best-matching model ID, or None.

        Matching strategy (in priority order):
        1. Exact model ID match on name
        2. Model ID contains the library name (e.g. Ollama "llama3:latest"
           matches library name "llama3")
        3. Model ID contains the filename stem from the model path
           (e.g. server model "my-model.gguf" matches path
           "/models/my-model.gguf")
        """
        from pathlib import Path as _Path

        server_models = plugin.list_models()
        if not server_models:
            return None

        name_lower = model_name.lower()
        stem_lower = _Path(model_path).stem.lower() if model_path else ""

        # Pass 1: exact ID match
        for m in server_models:
            if m.id.lower() == name_lower or m.name.lower() == name_lower:
                return m.id

        # Pass 2: server model contains the library name
        if name_lower:
            for m in server_models:
                if name_lower in m.id.lower() or name_lower in m.name.lower():
                    return m.id

        # Pass 3: match on filename stem
        if stem_lower:
            for m in server_models:
                if stem_lower in m.id.lower() or stem_lower in m.name.lower():
                    return m.id

        return None
