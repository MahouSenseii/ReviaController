"""
Right-hand settings panel — hosts a QTabWidget populated with
individually-classed tab pages.

Each tab is a self-contained class (see ``ui.tabs``).
"""

from __future__ import annotations

from PyQt6.QtWidgets import QTabWidget

from core.config import Config
from core.events import EventBus
from core.plugin_manager import PluginManager
from ui.tabs import (
    BehaviorTab,
    ControlTab,
    EvalTab,
    FiltersTab,
    LLMTab,
    LogsTab,
    MemoryTab,
    RelationshipTab,
    ResourcesTab,
    RetrievalTab,
    SystemTab,
    VoiceVisionTab,
)

from .base_panel import BasePanel


class SettingsPanel(BasePanel):
    """Tabbed settings panel on the right side of the window."""

    def __init__(
        self,
        event_bus: EventBus,
        config: Config,
        plugin_manager: PluginManager,
    ):
        self._plugin_manager = plugin_manager
        super().__init__(event_bus, config)
        self.setFixedWidth(490)

    def _build(self) -> None:
        lay = self._inner_layout
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(15)

        self._tabs = QTabWidget()
        self._tabs.setObjectName("RightTabs")

        self._behavior_tab = BehaviorTab(self.bus, self.config)
        self._llm_tab = LLMTab(self.bus, self.config, self._plugin_manager)
        self._voice_vision_tab = VoiceVisionTab(self.bus, self.config)
        self._filters_tab = FiltersTab(self.bus, self.config)
        self._logs_tab = LogsTab(self.bus, self.config)
        self._system_tab = SystemTab(self.bus, self.config)
        self._memory_tab = MemoryTab(self.bus, self.config)
        self._retrieval_tab = RetrievalTab(self.bus, self.config)
        self._relationship_tab = RelationshipTab(self.bus, self.config)
        self._eval_tab = EvalTab(self.bus, self.config)
        self._resources_tab = ResourcesTab(self.bus, self.config)
        self._control_tab = ControlTab(self.bus, self.config)

        self._tabs.addTab(self._behavior_tab, "Behavior")
        self._tabs.addTab(self._llm_tab, "LLM")
        self._tabs.addTab(self._voice_vision_tab, "Voice && Vision")
        self._tabs.addTab(self._filters_tab, "Filters")
        self._tabs.addTab(self._memory_tab, "Memory")
        self._tabs.addTab(self._retrieval_tab, "Retrieval")
        self._tabs.addTab(self._relationship_tab, "Relations")
        self._tabs.addTab(self._eval_tab, "Eval")
        self._tabs.addTab(self._resources_tab, "Resources")
        self._tabs.addTab(self._control_tab, "Control")
        self._tabs.addTab(self._logs_tab, "Logs")
        self._tabs.addTab(self._system_tab, "System")

        lay.addWidget(self._tabs, 1)
