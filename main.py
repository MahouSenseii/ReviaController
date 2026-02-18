"""
Revia Controller — Entry Point

Creates the shared infrastructure (EventBus, Config, PluginManager),
discovers plugins, initialises the emotion system with decision-making,
metacognition, self-development, and pipeline timing, then launches
the UI.
"""

import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from core import (
    Config,
    EventBus,
    PluginManager,
    EmotionEngine,
    ConversationManager,
    DecisionEngine,
    MetacognitionEngine,
    SelfDevelopmentEngine,
    PipelineTimer,
)
from core.stimulus import StimulusAnalyser
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

    # Stimulus analyser — converts chat messages into emotion stimuli
    stimulus_analyser = StimulusAnalyser(bus)

    # Decision engine — emotion-driven response strategy selection
    decision_engine = DecisionEngine(bus, emotion_engine)

    # Metacognition — self-monitoring and reflection
    metacognition = MetacognitionEngine(bus, emotion_engine)

    # Self-development — learning from interactions over time
    self_dev = SelfDevelopmentEngine(bus, emotion_engine)

    # Pipeline timer — measures latency of every stage
    timer = PipelineTimer(bus)

    # Conversation manager — orchestrates chat turns with full pipeline
    conversation = ConversationManager(
        bus, config, pm, emotion_engine,
        decision_engine=decision_engine,
        metacognition=metacognition,
        self_dev=self_dev,
        timer=timer,
    )

    # Emotion tick timer — natural decay every 2 seconds
    tick_timer = QTimer()
    tick_timer.setInterval(emotion_engine.cfg.tick_interval_ms)
    tick_timer.timeout.connect(emotion_engine.tick)
    tick_timer.start()

    # Main window
    window = MainWindow(bus, config, pm, emotion_engine)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
