"""
LLM tab — provider / model selection and provider-specific settings.

The tab reads the active plugin's ``get_config_schema()`` and
auto-generates matching input widgets (text fields, sliders, combos).
When the user picks a different plugin the form rebuilds itself.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QLabel,
    QLineEdit,
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

    def _build(self) -> None:
        lay = self._layout

        lay.addWidget(self._heading("Provider"))

        self._provider_combo = QComboBox()
        self._provider_combo.setObjectName("SettingsCombo")
        self._refresh_providers()
        self._provider_combo.currentTextChanged.connect(self._on_provider)
        lay.addWidget(self._row("Provider", self._provider_combo))

        lay.addWidget(self._heading("Model"))

        self._model_combo = QComboBox()
        self._model_combo.setObjectName("SettingsCombo")
        self._model_combo.currentTextChanged.connect(self._on_model)
        lay.addWidget(self._row("Model", self._model_combo))

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#263246;")
        lay.addWidget(sep)

        lay.addWidget(self._heading("Provider Settings"))

        # Container for auto-generated fields
        self._dynamic_container = QWidget()
        self._dynamic_layout = QVBoxLayout(self._dynamic_container)
        self._dynamic_layout.setContentsMargins(0, 0, 0, 0)
        self._dynamic_layout.setSpacing(10)
        lay.addWidget(self._dynamic_container)

        # React to external discovery
        self.bus.subscribe("plugins_discovered", lambda _: self._refresh_providers())
        self.bus.subscribe("plugin_activated", self._on_plugin_activated)

    # ── Provider / model helpers ──────────────────────────────

    def _refresh_providers(self) -> None:
        self._provider_combo.blockSignals(True)
        self._provider_combo.clear()
        names = list(self._pm.available.keys())
        if not names:
            names = ["(none)"]
        self._provider_combo.addItems(names)
        saved = self.config.get("llm.provider", "")
        idx = self._provider_combo.findText(saved)
        if idx >= 0:
            self._provider_combo.setCurrentIndex(idx)
        self._provider_combo.blockSignals(False)

    def _on_provider(self, name: str) -> None:
        if name == "(none)":
            return
        self.config.set("llm.provider", name)
        try:
            plugin = self._pm.activate(name, self.config.section(f"llm.{name}"))
        except KeyError:
            return
        self._populate_models(plugin)
        self._populate_dynamic(plugin)

    def _on_plugin_activated(self, data: dict) -> None:
        plugin = self._pm.active_plugin
        if plugin:
            self._populate_models(plugin)
            self._populate_dynamic(plugin)

    def _populate_models(self, plugin) -> None:
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in plugin.list_models():
            self._model_combo.addItem(m.name, m.id)
        saved = self.config.get("llm.model", "")
        idx = self._model_combo.findText(saved)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        self._model_combo.blockSignals(False)

    def _on_model(self, name: str) -> None:
        idx = self._model_combo.currentIndex()
        model_id = self._model_combo.itemData(idx)
        if model_id and self._pm.active_plugin:
            self._pm.active_plugin.select_model(model_id)
            self.config.set("llm.model", name)
            self.bus.publish("model_changed", {"model": name, "model_id": model_id})

    # ── Dynamic config form ───────────────────────────────────

    def _populate_dynamic(self, plugin) -> None:
        # Clear old
        for w in self._dynamic_widgets:
            w.setParent(None)
        self._dynamic_widgets.clear()

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
