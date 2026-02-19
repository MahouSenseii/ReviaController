"""
Module status tracker — monitors STT, TTS, and Vision modules.

Listens for toggle events from the sidebar and plugin lifecycle events
to determine whether each module is actually functional.  Publishes
``module_status`` events so the sidebar pills show the real state
instead of staying on "Idle" forever.

When a module is toggled ON but something is wrong, the pill shows a
warning with a specific message about what is failing, and an
``activity_log`` event is published so the error also appears in chat.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject

from .events import EventBus
from .plugin_base import PluginCapability
from .plugin_manager import PluginManager


# Maps module name → required PluginCapability flag
_MODULE_CAPS = {
    "stt":    PluginCapability.STT,
    "tts":    PluginCapability.TTS,
    "vision": PluginCapability.VISION,
}

# Human-readable module names for error messages
_MODULE_NAMES = {
    "stt":    "Speech-to-Text (STT)",
    "tts":    "Text-to-Speech (TTS)",
    "vision": "Vision",
}


class ModuleStatusTracker(QObject):
    """
    Tracks user-facing module toggles and publishes ``module_status``
    events with accurate state information.

    Status values:
    - ``"on"``   — module enabled and the backend supports it
    - ``"warn"`` — module enabled but the backend cannot provide it
    - ``"off"``  — module disabled by the user
    """

    def __init__(
        self,
        event_bus: EventBus,
        plugin_manager: PluginManager,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._bus = event_bus
        self._pm = plugin_manager

        # Track whether the user has each module toggled on
        self._enabled = {"stt": True, "tts": True, "vision": True}

        # Subscribe to toggle events from sidebar pills
        self._bus.subscribe("stt_toggled", self._on_stt_toggled)
        self._bus.subscribe("tts_toggled", self._on_tts_toggled)
        self._bus.subscribe("vision_toggled", self._on_vision_toggled)

        # Re-evaluate when plugin state changes
        self._bus.subscribe("plugin_activated", self._on_plugin_changed)
        self._bus.subscribe("plugin_deactivated", self._on_plugin_changed)

    # ── toggle handlers ─────────────────────────────────────────

    def _on_stt_toggled(self, data: dict) -> None:
        self._enabled["stt"] = data.get("enabled", False)
        self._publish("stt")

    def _on_tts_toggled(self, data: dict) -> None:
        self._enabled["tts"] = data.get("enabled", False)
        self._publish("tts")

    def _on_plugin_changed(self, _data: dict) -> None:
        for module in ("stt", "tts", "vision"):
            self._publish(module)

    def _on_vision_toggled(self, data: dict) -> None:
        self._enabled["vision"] = data.get("enabled", False)
        self._publish("vision")

    # ── core logic ──────────────────────────────────────────────

    def _publish(self, module: str) -> None:
        """Evaluate and publish the real status of *module*."""
        enabled = self._enabled.get(module, False)
        module_name = _MODULE_NAMES.get(module, module.upper())

        if not enabled:
            self._bus.publish("module_status", {
                "module": module,
                "status": "off",
                "subtitle": "Disabled",
            })
            return

        plugin = self._pm.active_plugin
        if plugin is None:
            subtitle = "No AI backend connected — go to LLM tab"
            self._bus.publish("module_status", {
                "module": module,
                "status": "warn",
                "subtitle": subtitle,
            })
            self._bus.publish("activity_log", {
                "text": f"[Error] {module_name} is unavailable: no AI backend is connected. "
                        f"Connect a backend in the LLM tab.",
            })
            return

        if not plugin.is_connected():
            subtitle = "Backend disconnected — check LLM tab"
            self._bus.publish("module_status", {
                "module": module,
                "status": "warn",
                "subtitle": subtitle,
            })
            self._bus.publish("activity_log", {
                "text": f"[Error] {module_name} is unavailable: the AI backend is not "
                        f"connected. Use the LLM tab to reconnect.",
            })
            return

        required_cap = _MODULE_CAPS.get(module)
        if required_cap and not (plugin.capabilities & required_cap):
            plugin_name = type(plugin).__name__
            subtitle = f"Backend ({plugin_name}) does not support {module.upper()}"
            self._bus.publish("module_status", {
                "module": module,
                "status": "warn",
                "subtitle": subtitle,
            })
            self._bus.publish("activity_log", {
                "text": f"[Error] {module_name} is unavailable: the active backend "
                        f"({plugin_name}) does not advertise the {module.upper()} "
                        f"capability. Switch to a backend that supports it.",
            })
            return

        # Everything looks good
        self._bus.publish("module_status", {
            "module": module,
            "status": "on",
            "subtitle": "Running",
        })

    def refresh_all(self) -> None:
        """Force re-evaluate and publish status for every module."""
        for module in ("stt", "tts", "vision"):
            self._publish(module)
