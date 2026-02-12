"""
Revia Controller — Entry Point

Creates the shared infrastructure (EventBus, Config, PluginManager),
discovers plugins, and launches the UI.
"""

import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from core import Config, EventBus, PluginManager
from main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    # Shared infrastructure
    bus = EventBus()
    config = Config(bus)
    pm = PluginManager(bus, plugin_package="plugins")
    pm.discover()

    # Main window
    window = MainWindow(bus, config, pm)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
