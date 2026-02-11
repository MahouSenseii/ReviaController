"""
AI Runtime UI - Entry Point
This file only starts the application
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from main_window import MainWindow


def main():
    """Application entry point"""
    app = QApplication(sys.argv)

    # Set application-wide font
    app.setFont(QFont("Segoe UI", 10))

    # Create and show main window
    window = MainWindow()
    window.show()

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()