"""
Discovers, loads, and manages AI provider plugins.

Plugins are Python modules inside the ``plugins/`` package that expose
a top-level ``Plugin`` class inheriting from ``AIPluginBase``.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

from .plugin_base import AIPluginBase, PluginCapability
from .events import EventBus


class PluginManager:
    """
    Registry + lifecycle manager for AI plugins.

    Parameters
    ----------
    event_bus : EventBus
        Shared bus – the manager publishes plugin-related events.
    plugin_package : str
        Dotted import path of the package that holds plugins.
    """

    def __init__(self, event_bus: EventBus, plugin_package: str = "plugins"):
        self._bus = event_bus
        self._package = plugin_package
        self._registry: dict[str, type[AIPluginBase]] = {}
        self._active: AIPluginBase | None = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover(self) -> list[str]:
        """
        Scan the plugin package and register every module that exposes
        a ``Plugin`` class deriving from ``AIPluginBase``.

        Returns the list of discovered plugin names.
        """
        try:
            pkg = importlib.import_module(self._package)
        except ModuleNotFoundError:
            return []

        pkg_path = Path(pkg.__file__).parent

        for finder, module_name, _ in pkgutil.iter_modules([str(pkg_path)]):
            full = f"{self._package}.{module_name}"
            try:
                mod = importlib.import_module(full)
            except Exception:
                continue

            cls = getattr(mod, "Plugin", None)
            if cls is not None and isinstance(cls, type) and issubclass(cls, AIPluginBase):
                self._registry[module_name] = cls

        self._bus.publish("plugins_discovered", {
            "names": list(self._registry.keys()),
        })
        return list(self._registry.keys())

    def register(self, name: str, cls: type[AIPluginBase]) -> None:
        """Manually register a plugin class under *name*."""
        self._registry[name] = cls

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def available(self) -> dict[str, type[AIPluginBase]]:
        return dict(self._registry)

    @property
    def active_plugin(self) -> AIPluginBase | None:
        return self._active

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self, name: str, config: dict[str, Any] | None = None) -> AIPluginBase:
        """
        Instantiate and connect the named plugin.

        If another plugin is already active it is deactivated first.
        """
        if name not in self._registry:
            raise KeyError(f"Unknown plugin: {name!r}")

        if self._active is not None:
            self.deactivate()

        plugin = self._registry[name]()
        plugin.connect(config or {})
        self._active = plugin

        self._bus.publish("plugin_activated", {
            "name": name,
            "capabilities": plugin.capabilities,
        })
        return plugin

    def deactivate(self) -> None:
        """Disconnect and drop the current plugin."""
        if self._active is None:
            return
        name = self._active.name
        try:
            self._active.disconnect()
        finally:
            self._active = None
        self._bus.publish("plugin_deactivated", {"name": name})
