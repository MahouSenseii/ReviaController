"""
Revia Controller — Entry Point

Creates the shared infrastructure (EventBus, Config, PluginManager),
discovers plugins, and launches the UI.
"""

import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from core import (
    Config, EventBus, PluginManager,
    EmotionEngine, RAGEngine, SafetyFilterEngine,
    AVPipeline,
)
from main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    # Shared infrastructure
    bus = EventBus()
    config = Config(bus)
    pm = PluginManager(bus, plugin_package="plugins")
    pm.discover()

    # Emotion engine — gives the AI a living emotional state
    emotion_engine = EmotionEngine(bus)

    # RAG memory — profile-isolated short-term + long-term memory
    rag_engine = RAGEngine(bus)

    # Safety filter — input/output content filtering
    safety_filter = SafetyFilterEngine(bus, config)

    # AV pipeline — STT, TTS, Vision orchestration
    av_pipeline = AVPipeline(
        bus,
        emotion_engine=emotion_engine,
        safety_filter=safety_filter,
    )

    # Main window
    window = MainWindow(
        bus, config, pm,
        emotion_engine, rag_engine, safety_filter,
        av_pipeline,
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
