"""
Conversation manager — maintains message history, assembles prompts
with emotion context, and drives inference through the active plugin.

Responsibilities
----------------
* Maintain an ordered list of ``{"role": ..., "content": ...}`` messages.
* Inject the system prompt (from Behavior settings) and the emotion
  engine's context block.
* Run the **decision engine** to select a response strategy based on
  emotional state and conversation context.
* Inject **metacognition** self-reflection and **learned preferences**
  into the prompt so the AI can course-correct.
* Time every pipeline stage via ``PipelineTimer`` and publish results
  to the UI.
* Send the assembled prompt to the active plugin and publish the
  response through the EventBus.
* Publish ``activity_log`` events so the center panel stays updated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .config import Config
from .emotion_engine import EmotionEngine
from .events import EventBus
from .plugin_manager import PluginManager

if TYPE_CHECKING:
    from .decision import DecisionEngine
    from .metacognition import MetacognitionEngine
    from .self_dev import SelfDevelopmentEngine
    from .timing import PipelineTimer


class ConversationManager:
    """
    Orchestrates chat turns between the user and the AI assistant.

    Usage::

        mgr = ConversationManager(bus, config, pm, emotion_engine)
        mgr.send("Hello!")
        # response arrives via "assistant_response" event
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: Config,
        plugin_manager: PluginManager,
        emotion_engine: Optional[EmotionEngine] = None,
        decision_engine: Optional["DecisionEngine"] = None,
        metacognition: Optional["MetacognitionEngine"] = None,
        self_dev: Optional["SelfDevelopmentEngine"] = None,
        timer: Optional["PipelineTimer"] = None,
    ):
        self.bus = event_bus
        self.config = config
        self.pm = plugin_manager
        self.emotion = emotion_engine
        self.decision = decision_engine
        self.metacognition = metacognition
        self.self_dev = self_dev
        self.timer = timer

        self._history: List[Dict[str, str]] = []
        self._max_history: int = 50  # keep last N turns

        # Listen for user messages from UI
        self.bus.subscribe("user_message", self._on_user_message)

    # ── Public API ────────────────────────────────────────────

    def send(self, user_text: str) -> str | None:
        """
        Send a user message and return the assistant's reply.

        Also publishes events for UI updates.  Returns None if no
        plugin is active.
        """
        # Start pipeline timer
        if self.timer:
            self.timer.begin()

        plugin = self.pm.active_plugin
        if plugin is None or not plugin.is_connected():
            self.bus.publish("activity_log", {
                "text": "[System] No AI backend connected. "
                        "Go to LLM tab and connect a provider.",
            })
            return None

        # Record user turn
        self._history.append({"role": "user", "content": user_text})
        self._trim_history()

        # Publish activity
        self.bus.publish("activity_log", {
            "text": f'User: "{user_text}"',
        })
        self.bus.publish("assistant_status", {
            "lines": ["Processing...", "Generating Response..."],
        })

        # ── Decision engine ───────────────────────────────────
        strategy = None
        if self.decision:
            if self.timer:
                self.timer.start("decision")
            strategy = self.decision.decide(user_text)
            if self.timer:
                self.timer.stop("decision")

        # Build messages (includes emotion + decision + metacognition)
        messages = self._build_messages(strategy)

        # ── Inference ─────────────────────────────────────────
        if self.timer:
            self.timer.start("inference")

        try:
            reply = plugin.send_prompt(messages, stream=False)
            if not isinstance(reply, str):
                # Consume iterator if streaming was forced
                reply = "".join(reply)
        except Exception as e:
            if self.timer:
                self.timer.stop("inference")
                self.timer.finish()
            error_msg = f"[Error] {e}"
            self.bus.publish("activity_log", {"text": error_msg})
            self.bus.publish("assistant_status", {
                "lines": ["Error during inference"],
            })
            return None

        if self.timer:
            self.timer.stop("inference")

        # Record assistant turn
        self._history.append({"role": "assistant", "content": reply})

        # Publish response
        self.bus.publish("assistant_response", {
            "text": reply,
            "model": plugin.active_model().name if plugin.active_model() else "?",
        })
        self.bus.publish("activity_log", {
            "text": f'AI: "{reply[:200]}{"..." if len(reply) > 200 else ""}"',
        })
        self.bus.publish("assistant_status", {
            "lines": ["Listening...", "Vision: Idle"],
        })

        # Publish LLM metrics
        try:
            m = plugin.get_metrics()
            self.bus.publish("inference_metrics", {
                "latency": f"{m.latency_ms:.0f} ms",
                "tokens_sec": f"{m.tokens_per_sec:.1f}",
                "ttft": f"{m.ttft_ms:.0f} ms",
                "context": f"{m.context_used} / {m.context_max}",
            })
        except Exception:
            pass

        # ── Finish pipeline timing ────────────────────────────
        if self.timer:
            self.timer.finish()

        return reply

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self._history)

    # ── Event handler ─────────────────────────────────────────

    def _on_user_message(self, data: Dict[str, Any]) -> None:
        text = data.get("text", "").strip()
        if text:
            self.send(text)

    # ── Internal ──────────────────────────────────────────────

    def _build_messages(self, strategy=None) -> List[Dict[str, str]]:
        """Assemble the full message list for the LLM."""
        messages: List[Dict[str, str]] = []

        # System prompt
        system_parts: list[str] = []

        # Base system prompt from behavior settings
        base_prompt = self.config.get("behavior.system_prompt", "")
        if base_prompt:
            system_parts.append(base_prompt)

        # Personality / style hints
        persona = self.config.get("behavior.persona", "Friendly")
        verbosity = self.config.get("behavior.verbosity", "Medium")
        style = self.config.get("behavior.response_style", "Conversational")
        system_parts.append(
            f"Personality: {persona}. "
            f"Verbosity: {verbosity}. "
            f"Response style: {style}."
        )

        # Emotion context
        if self.emotion:
            ctx = self.emotion.get_llm_context()
            injection = ctx.get("prompt_injection", "")
            if injection:
                system_parts.append(injection)

        # Decision engine strategy — shapes *how* the AI responds
        if strategy:
            strategy_block = strategy.to_prompt_block()
            if strategy_block:
                system_parts.append(strategy_block)

        # Metacognition self-reflection — lets the AI course-correct
        if self.metacognition:
            reflection = self.metacognition.get_reflection_block()
            if reflection:
                system_parts.append(reflection)

        # Learned user preferences — adapt to this specific user
        if self.self_dev:
            prefs = self.self_dev.get_preference_hints()
            if prefs:
                system_parts.append(prefs)

        if system_parts:
            messages.append({
                "role": "system",
                "content": "\n\n".join(system_parts),
            })

        # Conversation history
        messages.extend(self._history)
        return messages

    def _trim_history(self) -> None:
        """Keep only the last N messages to avoid unbounded growth."""
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
