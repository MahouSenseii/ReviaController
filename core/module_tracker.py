"""
Module status tracker — monitors STT, TTS, and Vision modules.

Listens for toggle events from the sidebar and plugin lifecycle events
to determine whether each module is actually functional.  Publishes
``module_status`` events so the sidebar pills show the real state
instead of staying on "Idle" forever.

When a module is toggled ON but no AI backend supports it (or no
backend is connected at all), the pill shows a warning with the
message "Something is wrong with my AI".
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

        if not enabled:
            self._bus.publish("module_status", {
                "module": module,
                "status": "off",
                "subtitle": "Disabled",
            })
            return

        plugin = self._pm.active_plugin
        if plugin is None or not plugin.is_connected():
            self._bus.publish("module_status", {
                "module": module,
                "status": "warn",
                "subtitle": "Something is wrong with my AI",
            })
            return

        required_cap = _MODULE_CAPS.get(module)
        if required_cap and not (plugin.capabilities & required_cap):
            self._bus.publish("module_status", {
                "module": module,
                "status": "warn",
                "subtitle": "Something is wrong with my AI",
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
