"""
Simple configuration store backed by a JSON file.

Provides typed get/set, section namespacing, and automatic
persistence so that panels, tabs, and plugins can read/write
settings without coupling to each other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .events import EventBus

_DEFAULT_PATH = Path("config.json")


class Config:
    """
    Hierarchical configuration backed by a JSON file.

    Keys use dot notation: ``"llm.provider"``, ``"voice.stt_enabled"``.
    """

    def __init__(self, event_bus: EventBus, path: Path | str = _DEFAULT_PATH):
        self._bus = event_bus
        self._path = Path(path)
        self._data: dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        node = self._data
        for p in parts[:-1]:
            node = node.get(p, {})
            if not isinstance(node, dict):
                return default
        return node.get(parts[-1], default)

    def set(self, key: str, value: Any, *, save: bool = True) -> None:
        parts = key.split(".")
        node = self._data
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value

        if save:
            self._save()

        self._bus.publish("config_changed", {"key": key, "value": value})

    def section(self, prefix: str) -> dict[str, Any]:
        """Return a shallow copy of everything under *prefix*."""
        parts = prefix.split(".")
        node = self._data
        for p in parts:
            node = node.get(p, {})
            if not isinstance(node, dict):
                return {}
        return dict(node)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
