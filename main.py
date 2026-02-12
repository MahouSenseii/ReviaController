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
    ContinuityTracker, PersonaController, IntentMemory,
    RepairMemory, RecallPolicy,
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

    # Conversation continuity — tracks threads, commitments, tasks
    continuity_tracker = ContinuityTracker(bus)

    # Persona consistency — monitors style drift
    persona_controller = PersonaController(bus)

    # Intent memory — inferred user goals and needs
    intent_memory = IntentMemory(bus)

    # Repair memory — tracks corrections and feedback
    repair_memory = RepairMemory(bus)

    # Recall policy — controls when/how to inject memories
    recall_policy = RecallPolicy(bus)

    # Main window
    window = MainWindow(
        bus, config, pm,
        emotion_engine, rag_engine, safety_filter,
        av_pipeline,
    )

    # Store references so they don't get garbage-collected
    window._continuity_tracker = continuity_tracker
    window._persona_controller = persona_controller
    window._intent_memory = intent_memory
    window._repair_memory = repair_memory
    window._recall_policy = recall_policy
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
