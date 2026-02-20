from .events import EventBus
from .llm_registry import LLMRegistry
from .plugin_base import AIPluginBase, PluginCapability
from .plugin_manager import PluginManager
from .config import Config
from .emotions import EmotionProfile, EMOTION_PROFILES, EMOTION_NAMES, NUM_EMOTIONS
from .emotion_nn import EmotionNeuralNetwork
from .emotion_engine import EmotionEngine, EmotionEngineConfig, Stimulus
from .conversation import ConversationManager
from .stimulus import StimulusAnalyser
from .decision import DecisionEngine, ResponseStrategy
from .metacognition import MetacognitionEngine
from .self_dev import SelfDevelopmentEngine
from .timing import PipelineTimer
from .system_monitor import SystemMonitor
from .module_tracker import ModuleStatusTracker

# Optional audio modules depend on system libraries (e.g., PortAudio).
# Import them defensively so non-audio workflows (tests, headless use)
# can still import ``core`` without hard failures.
try:
    from .stt_manager import STTManager
except Exception:  # pragma: no cover - depends on host audio stack
    STTManager = None  # type: ignore[assignment]

try:
    from .tts_manager import TTSManager
except Exception:  # pragma: no cover - depends on host audio stack
    TTSManager = None  # type: ignore[assignment]
