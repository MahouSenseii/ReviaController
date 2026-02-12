"""
Centralized event bus for decoupled component communication.

Components publish/subscribe to named events instead of holding
direct references to each other.  Built on top of QObject so it
lives naturally inside the Qt event loop.
"""

from __future__ import annotations

from typing import Any, Callable
from PyQt6.QtCore import QObject, pyqtSignal


class _Signal(QObject):
    """Wrapper around a single pyqtSignal that can carry arbitrary data."""
    fired = pyqtSignal(dict)


class EventBus(QObject):
    """
    Application-wide event bus.

    Usage
    -----
    bus = EventBus()
    bus.subscribe("stt_toggled", lambda d: print(d))
    bus.publish("stt_toggled", {"enabled": True})
    """

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._channels: dict[str, _Signal] = {}

    def _ensure(self, event: str) -> _Signal:
        if event not in self._channels:
            self._channels[event] = _Signal(self)
        return self._channels[event]

    def subscribe(self, event: str, callback: Callable[[dict[str, Any]], None]) -> None:
        self._ensure(event).fired.connect(callback)

    def unsubscribe(self, event: str, callback: Callable[[dict[str, Any]], None]) -> None:
        if event in self._channels:
            try:
                self._channels[event].fired.disconnect(callback)
            except TypeError:
                pass

    def publish(self, event: str, data: dict[str, Any] | None = None) -> None:
        if data is None:
            data = {}
        self._ensure(event).fired.emit(data)
