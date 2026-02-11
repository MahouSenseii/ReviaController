"""
LLM tab — local model file browser + saved model library.

Users browse their PC to select a local LLM file (GGUF, bin, etc.).
The file path is saved and the filename becomes the display name in
a dropdown.  A rename field lets users give models friendly names.
On every selection the path is validated — if the file was moved or
deleted a popup warns the user and the entry is removed.

Also keeps the plugin-based provider / dynamic-settings support from
before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config import Config
from core.events import EventBus
from core.plugin_manager import PluginManager

from .base_tab import BaseTab


class LLMTab(BaseTab):

    def __init__(self, event_bus: EventBus, config: Config, plugin_manager: PluginManager):
        self._pm = plugin_manager
        self._dynamic_widgets: list[QWidget] = []
        super().__init__(event_bus, config)

    # ── Build ─────────────────────────────────────────────────

    def _build(self) -> None:
        lay = self._layout

        # ── Local Model Library ───────────────────────────────
        lay.addWidget(self._heading("Local Model"))

        # Model selector dropdown
        self._model_combo = QComboBox()
        self._model_combo.setObjectName("SettingsCombo")
        self._model_combo.setMinimumHeight(28)
        self._model_combo.currentIndexChanged.connect(self._on_model_selected)
        lay.addWidget(self._row("Model", self._model_combo))

        # Browse + Remove buttons
        btn_row = QWidget()
        bh = QHBoxLayout(btn_row)
        bh.setContentsMargins(0, 0, 0, 0)
        bh.setSpacing(8)

        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.setObjectName("ModeButton")
        self._browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._browse_btn.clicked.connect(self._on_browse)
        bh.addWidget(self._browse_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setObjectName("ModeButton")
        self._remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._remove_btn.setStyleSheet(
            "QPushButton { color:#ef476f; } QPushButton:hover { border-color:#ef476f; }"
        )
        self._remove_btn.clicked.connect(self._on_remove)
        bh.addWidget(self._remove_btn)

        bh.addStretch(1)
        lay.addWidget(btn_row)

        # Rename field
        rename_row = QWidget()
        rh = QHBoxLayout(rename_row)
        rh.setContentsMargins(0, 0, 0, 0)
        rh.setSpacing(8)

        self._rename_edit = QLineEdit()
        self._rename_edit.setObjectName("SettingsLineEdit")
        self._rename_edit.setPlaceholderText("Enter a friendly name for this model...")
        rh.addWidget(self._rename_edit, 1)

        self._rename_btn = QPushButton("Rename")
        self._rename_btn.setObjectName("ModeButton")
        self._rename_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rename_btn.clicked.connect(self._on_rename)
        rh.addWidget(self._rename_btn)

        lay.addWidget(self._row("Name", rename_row))

        # Path display (read-only)
        self._path_label = QLabel("No model selected")
        self._path_label.setObjectName("PillSub")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet("color:#8fa6c3; font-size:11px; padding:4px 0;")
        lay.addWidget(self._path_label)

        # ── Separator ────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#263246;")
        lay.addWidget(sep)

        # ── Plugin Provider ──────────────────────────────────
        lay.addWidget(self._heading("API Provider (optional)"))

        self._provider_combo = QComboBox()
        self._provider_combo.setObjectName("SettingsCombo")
        self._refresh_providers()
        self._provider_combo.currentTextChanged.connect(self._on_provider)
        lay.addWidget(self._row("Provider", self._provider_combo))

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#263246;")
        lay.addWidget(sep2)

        lay.addWidget(self._heading("Provider Settings"))

        # Container for auto-generated plugin config fields
        self._dynamic_container = QWidget()
        self._dynamic_layout = QVBoxLayout(self._dynamic_container)
        self._dynamic_layout.setContentsMargins(0, 0, 0, 0)
        self._dynamic_layout.setSpacing(10)
        lay.addWidget(self._dynamic_container)

        # React to external discovery
        self.bus.subscribe("plugins_discovered", lambda _: self._refresh_providers())
        self.bus.subscribe("plugin_activated", self._on_plugin_activated)

        # Load saved models into the dropdown
        self._load_saved_models()

    # ══════════════════════════════════════════════════════════
    # Local model library
    # ══════════════════════════════════════════════════════════

    def _get_models(self) -> list[dict]:
        """Return the persisted list of local models."""
        return self.config.get("llm.local_models", [])

    def _save_models(self, models: list[dict]) -> None:
        self.config.set("llm.local_models", models)

    def _load_saved_models(self) -> None:
        """Populate the dropdown from persisted model list."""
        self._model_combo.blockSignals(True)
        self._model_combo.clear()

        models = self._get_models()
        if not models:
            self._model_combo.addItem("(no models added)")
        else:
            for m in models:
                self._model_combo.addItem(m["name"], m["path"])

        # Restore last selection
        selected = self.config.get("llm.selected_model", "")
        if selected:
            idx = self._model_combo.findText(selected)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)

        self._model_combo.blockSignals(False)
        self._update_details()

    def _on_browse(self) -> None:
        """Open a file dialog so the user can pick a local model file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select LLM Model File",
            "",
            "Model Files (*.gguf *.bin *.onnx *.pt *.pth *.safetensors *.ggml);;"
            "All Files (*)",
        )
        if not path:
            return

        # Check for duplicates
        models = self._get_models()
        for m in models:
            if m["path"] == path:
                QMessageBox.information(
                    self, "Already Added",
                    f"This model is already in your library as \"{m['name']}\".",
                )
                # Select it
                idx = self._model_combo.findData(path)
                if idx >= 0:
                    self._model_combo.setCurrentIndex(idx)
                return

        # Use filename (without extension) as default name
        file_name = Path(path).stem
        models.append({"name": file_name, "path": path})
        self._save_models(models)
        self._load_saved_models()

        # Select the newly added model
        idx = self._model_combo.findData(path)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)

        # Prompt to rename
        QMessageBox.information(
            self,
            "Model Added",
            f"Added \"{file_name}\" to your library.\n\n"
            "Tip: Use the Name field to give this model a friendly "
            "name so you can identify it easily later.",
        )

    def _on_remove(self) -> None:
        """Remove the currently selected model from the library."""
        idx = self._model_combo.currentIndex()
        models = self._get_models()
        if not models or idx < 0 or idx >= len(models):
            return

        name = models[idx]["name"]
        reply = QMessageBox.question(
            self,
            "Remove Model",
            f"Remove \"{name}\" from your library?\n\n"
            "This only removes it from the list — the file itself is not deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        models.pop(idx)
        self._save_models(models)
        self._load_saved_models()
        self.bus.publish("model_changed", {"model": None, "path": None})

    def _on_rename(self) -> None:
        """Rename the currently selected model."""
        new_name = self._rename_edit.text().strip()
        if not new_name:
            return

        idx = self._model_combo.currentIndex()
        models = self._get_models()
        if not models or idx < 0 or idx >= len(models):
            return

        models[idx]["name"] = new_name
        self._save_models(models)

        # Preserve selection after reload
        self.config.set("llm.selected_model", new_name, save=False)
        self._load_saved_models()

        self._rename_edit.clear()

    def _on_model_selected(self, idx: int) -> None:
        """Called when the user picks a model from the dropdown."""
        models = self._get_models()
        if not models or idx < 0 or idx >= len(models):
            self._update_details()
            return

        entry = models[idx]
        path = entry["path"]

        # Validate that the file still exists
        if not Path(path).is_file():
            QMessageBox.warning(
                self,
                "Model File Not Found",
                f"The file for \"{entry['name']}\" no longer exists:\n\n"
                f"{path}\n\n"
                "It may have been moved or deleted. "
                "This entry will be removed from your library.",
            )
            models.pop(idx)
            self._save_models(models)
            self._load_saved_models()
            self.bus.publish("model_changed", {"model": None, "path": None})
            return

        # Valid selection
        self.config.set("llm.selected_model", entry["name"])
        self._update_details()
        self.bus.publish("model_changed", {
            "model": entry["name"],
            "path": path,
        })

    def _update_details(self) -> None:
        """Update the path label and rename placeholder for current selection."""
        idx = self._model_combo.currentIndex()
        models = self._get_models()

        if not models or idx < 0 or idx >= len(models):
            self._path_label.setText("No model selected")
            self._rename_edit.setPlaceholderText("Add a model first...")
            self._remove_btn.setEnabled(False)
            self._rename_btn.setEnabled(False)
            self._rename_edit.setEnabled(False)
            return

        entry = models[idx]
        self._path_label.setText(f"Path: {entry['path']}")
        self._rename_edit.setPlaceholderText(f"Currently: {entry['name']}")
        self._remove_btn.setEnabled(True)
        self._rename_btn.setEnabled(True)
        self._rename_edit.setEnabled(True)

    # ══════════════════════════════════════════════════════════
    # Plugin provider (API-based LLMs)
    # ══════════════════════════════════════════════════════════

    def _refresh_providers(self) -> None:
        self._provider_combo.blockSignals(True)
        self._provider_combo.clear()
        names = ["(none)"] + list(self._pm.available.keys())
        self._provider_combo.addItems(names)
        saved = self.config.get("llm.provider", "")
        idx = self._provider_combo.findText(saved)
        if idx >= 0:
            self._provider_combo.setCurrentIndex(idx)
        self._provider_combo.blockSignals(False)

    def _on_provider(self, name: str) -> None:
        if name == "(none)":
            self._clear_dynamic()
            return
        self.config.set("llm.provider", name)
        try:
            plugin = self._pm.activate(name, self.config.section(f"llm.{name}"))
        except KeyError:
            return
        self._populate_dynamic(plugin)

    def _on_plugin_activated(self, data: dict) -> None:
        plugin = self._pm.active_plugin
        if plugin:
            self._populate_dynamic(plugin)

    # ── Dynamic config form ───────────────────────────────────

    def _clear_dynamic(self) -> None:
        for w in self._dynamic_widgets:
            w.setParent(None)
        self._dynamic_widgets.clear()

    def _populate_dynamic(self, plugin) -> None:
        self._clear_dynamic()

        schema = plugin.get_config_schema()
        provider = plugin.name

        for key, spec in schema.items():
            widget = self._widget_for_spec(spec, provider, key)
            if widget is None:
                continue
            label = spec.get("label", key)
            row = self._row(label, widget)
            self._dynamic_layout.addWidget(row)
            self._dynamic_widgets.append(row)

    def _widget_for_spec(self, spec: dict[str, Any], provider: str, key: str) -> QWidget | None:
        t = spec.get("type", "string")
        default = spec.get("default", "")
        saved = self.config.get(f"llm.{provider}.{key}", default)

        if t == "string":
            le = self._make_line_edit(
                placeholder=str(default),
                text=str(saved),
                secret=spec.get("secret", False),
            )
            le.textChanged.connect(lambda v, k=key, p=provider: self.config.set(f"llm.{p}.{k}", v))
            return le

        if t == "float":
            sb = self._make_double_spin(
                min_val=spec.get("min", 0.0),
                max_val=spec.get("max", 1.0),
                value=float(saved),
                step=spec.get("step", 0.05),
            )
            sb.valueChanged.connect(lambda v, k=key, p=provider: self.config.set(f"llm.{p}.{k}", v))
            return sb

        if t == "int":
            sb = self._make_spin(
                min_val=spec.get("min", 0),
                max_val=spec.get("max", 100000),
                value=int(saved),
            )
            sb.valueChanged.connect(lambda v, k=key, p=provider: self.config.set(f"llm.{p}.{k}", v))
            return sb

        if t == "choice":
            cb = self._make_combo(spec.get("options", []))
            idx = cb.findText(str(saved))
            if idx >= 0:
                cb.setCurrentIndex(idx)
            cb.currentTextChanged.connect(lambda v, k=key, p=provider: self.config.set(f"llm.{p}.{k}", v))
            return cb

        return None
