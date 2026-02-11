"""
Main application window.

Composes the three major panels (sidebar, center, settings) and wires
them to the shared EventBus, Config, and PluginManager.  Contains no
business logic of its own.
"""

from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from core.config import Config
from core.events import EventBus
from core.plugin_manager import PluginManager
from ui.panels import CenterPanel, SettingsPanel, SidebarPanel
from ui.style import DARK_STYLE


class MainWindow(QMainWindow):

    def __init__(
        self,
        event_bus: EventBus,
        config: Config,
        plugin_manager: PluginManager,
    ):
        super().__init__()
        self.setWindowTitle("Revia Controller")
        self.resize(1600, 950)

        self.bus = event_bus
        self.config = config
        self.pm = plugin_manager

        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.sidebar = SidebarPanel(event_bus, config)
        self.center = CenterPanel(event_bus, config)
        self.settings = SettingsPanel(event_bus, config, plugin_manager)

        layout.addWidget(self.sidebar, 0)
        layout.addWidget(self.center, 1)
        layout.addWidget(self.settings, 0)

        self.setStyleSheet(DARK_STYLE)
