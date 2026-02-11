"""
Base class for all major panels in the controller.

Provides a uniform frame structure and access to the shared
EventBus and Config so child panels can publish/subscribe
without importing singletons.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QVBoxLayout

from core.events import EventBus
from core.config import Config


class BasePanel(QFrame):
    """
    Abstract panel that every sidebar / center / settings panel inherits.

    Subclasses override ``_build()`` to populate ``self._layout``.
    """

    def __init__(self, event_bus: EventBus, config: Config):
        super().__init__()
        self.bus = event_bus
        self.config = config
        self.setObjectName("Panel")
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Inner content frame (matches dark-panel style)
        self._inner = QFrame()
        self._inner.setObjectName("PanelInner")
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(12, 12, 12, 12)
        self._inner_layout.setSpacing(10)

        self._layout.addWidget(self._inner)

        self._build()

    def _build(self) -> None:
        """Override in subclasses to populate the panel."""
