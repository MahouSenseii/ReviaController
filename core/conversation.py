"""
Conversation manager — maintains message history, assembles prompts
with emotion context, and drives inference through the active plugin.

Responsibilities
----------------
* Maintain an ordered list of ``{"role": ..., "content": ...}`` messages.
* Inject the system prompt (from Behavior settings) and the emotion
  engine's context block.
* Send the assembled prompt to the active plugin and publish the
  response through the EventBus.
* Publish ``activity_log`` events so the center panel stays updated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import Config
from .emotion_engine import EmotionEngine
from .events import EventBus
from .plugin_manager import PluginManager


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
    ):
        self.bus = event_bus
        self.config = config
        self.pm = plugin_manager
        self.emotion = emotion_engine

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

        # Build messages
        messages = self._build_messages()

        # Inference
        try:
            reply = plugin.send_prompt(messages, stream=False)
            if not isinstance(reply, str):
                # Consume iterator if streaming was forced
                reply = "".join(reply)
        except Exception as e:
            error_msg = f"[Error] {e}"
            self.bus.publish("activity_log", {"text": error_msg})
            self.bus.publish("assistant_status", {
                "lines": ["Error during inference"],
            })
            return None

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

        # Publish metrics
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

    def _build_messages(self) -> List[Dict[str, str]]:
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
