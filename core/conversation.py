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

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .config import Config
from .emotion_engine import EmotionEngine
from .events import EventBus
from .plugin_manager import PluginManager

_PROFILE_PATH = Path("profile.json")

# Tokens that some models leak into their output but should never be shown.
_DEFAULT_STOP_TOKENS: list[str] = [
    "<|im_end|>",
    "<|im_start|>",
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
    "<|end|>",
    "<|endoftext|>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    "<s>",
    "</s>",
    "Thinking...",
]

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
        self._has_greeted_user: bool = False

        # Cache the active model's registry entry (for stop tokens etc.)
        self._active_model_registry: Dict[str, Any] = {}

        # Cached profile data — loaded from config on profile selection
        self._cached_profile: Dict[str, Any] = {}
        self._load_active_profile()

        # Listen for user messages from UI
        self.bus.subscribe("user_message", self._on_user_message)
        # Cache registry metadata whenever the active model changes
        self.bus.subscribe("model_changed", self._on_model_changed_registry)
        # Reload profile when user selects a different one in the sidebar
        self.bus.subscribe("profile_selected", self._on_profile_selected)
        # Reload profile when fields are edited in the Behavior tab
        self.bus.subscribe("profile_field_changed", self._on_profile_field_changed)
        # Reload profile when it's saved from the Behavior tab
        self.bus.subscribe("profile_saved", self._on_profile_saved)

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
        if plugin is None:
            if self.timer:
                self.timer.finish()
            self._publish_error(
                "I cannot respond right now — no AI backend is connected. "
                "Go to the LLM tab, select a backend, and press Connect."
            )
            return None
        if not plugin.is_connected():
            if self.timer:
                self.timer.finish()
            self._publish_error(
                f"I cannot respond right now — the backend ({type(plugin).__name__}) "
                "is not connected. Go to the LLM tab and press Connect."
            )
            return None

        # Record user turn
        self._history.append({"role": "user", "content": user_text})
        self._trim_history()

        # Publish activity
        self.bus.publish("activity_log", {
            "text": f'User: "{user_text}"',
        })
        self.bus.publish("assistant_status", {"stage": "analyzing"})

        # ── Decision engine ───────────────────────────────────
        strategy = None
        if self.decision:
            self.bus.publish("assistant_status", {"stage": "decision"})
            if self.timer:
                self.timer.start("decision")
            strategy = self.decision.decide(user_text)
            if self.timer:
                self.timer.stop("decision")

        # Build messages (includes emotion + decision + metacognition)
        messages = self._build_messages(strategy)

        # ── Inference ─────────────────────────────────────────
        self.bus.publish("assistant_status", {"stage": "generating"})
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
            self._publish_error(
                f"I encountered an error while generating a response. "
                f"Failure: {e}"
            )
            return None

        if self.timer:
            self.timer.stop("inference")

        # Strip leaked model artifact tokens from every response.
        # Some registries disable custom formatting, but we still want
        # universal cleanup (e.g., leaked role tags / "Thinking...").
        reg = self._active_model_registry
        extra_tokens: List[str] = []
        if reg.get("needs_formatting", True):
            configured_tokens = reg.get("stop_tokens", [])
            if isinstance(configured_tokens, list):
                extra_tokens = configured_tokens
        reply = self._clean_response(reply, extra_tokens)

        # Record assistant turn
        self._history.append({"role": "assistant", "content": reply})

        # Use the AI's character name (from profile.json) as the chat label,
        # falling back to the plugin's model name if no profile is configured.
        profile = self._load_profile()
        display_name = (
            profile.get("character_name")
            or (plugin.active_model().name if plugin.active_model() else "AI")
        )

        # Publish response
        self.bus.publish("assistant_response", {
            "text": reply,
            "model": display_name,
        })
        self.bus.publish("activity_log", {
            "text": f'AI: "{reply[:200]}{"..." if len(reply) > 200 else ""}"',
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

        # Brief "learning" stage before going back to listening
        self.bus.publish("assistant_status", {"stage": "learning"})

        # ── Finish pipeline timing ────────────────────────────
        if self.timer:
            self.timer.finish()

        self.bus.publish("assistant_status", {"stage": "listening"})
        return reply

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()
        self._has_greeted_user = False

    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self._history)

    # ── Event handler ─────────────────────────────────────────

    def _publish_error(self, message: str) -> None:
        """Show an error as a visible chat response and log it."""
        self.bus.publish("assistant_response", {
            "text": message,
            "model": "System",
            "error": True,
        })
        self.bus.publish("activity_log", {
            "text": f"[Error] {message}",
        })
        self.bus.publish("assistant_status", {
            "lines": ["Error during inference"],
        })

    def _on_user_message(self, data: Dict[str, Any]) -> None:
        text = data.get("text", "").strip()
        if text:
            self.send(text)

    def _on_model_changed_registry(self, data: Dict[str, Any]) -> None:
        """Cache the active model's registry metadata for use in send()."""
        self._active_model_registry = data.get("registry", {})

    def _on_profile_selected(self, data: Dict[str, Any]) -> None:
        """Reload profile data when the user switches profiles in the sidebar."""
        name = data.get("value")
        if name:
            self._load_active_profile(name)
            # Reset greeting flag so the new profile can greet
            self._has_greeted_user = False

    def _on_profile_field_changed(self, data: Dict[str, Any]) -> None:
        """Update cached profile when a field is edited in the Behavior tab."""
        key = data.get("key", "")
        value = data.get("value", "")
        if key and self._cached_profile:
            self._cached_profile[key] = value

    def _on_profile_saved(self, data: Dict[str, Any]) -> None:
        """Reload profile when it's saved from the Behavior tab."""
        if isinstance(data, dict) and data:
            self._cached_profile = dict(data)
            self._sync_profile_to_disk()

    # ── Internal ──────────────────────────────────────────────

    def _build_messages(self, strategy=None) -> List[Dict[str, str]]:
        """Assemble the full message list for the LLM."""
        messages: List[Dict[str, str]] = []

        # System prompt
        system_parts: list[str] = []

        # Load AI profile for identity context
        profile = self._load_profile()

        # Base system prompt from profile (preferred) or config fallback
        base_prompt = (
            profile.get("system_prompt")
            or self.config.get("behavior.system_prompt", "")
        )
        if base_prompt:
            system_parts.append(base_prompt)

        # Character identity from profile
        char_name = profile.get("character_name", "")
        persona_desc = profile.get("persona", "")
        traits = profile.get("personality_traits", "")
        greeting = profile.get("greeting", "")
        fallback = profile.get("fallback_message", "")
        voice_tone = profile.get("voice_tone", "")

        if char_name or persona_desc:
            identity_parts = []
            if char_name:
                identity_parts.append(f"Your name is {char_name}.")
            if persona_desc:
                identity_parts.append(f"Persona: {persona_desc}")
            if traits:
                identity_parts.append(f"Personality traits: {traits}.")
            if voice_tone:
                identity_parts.append(f"Voice tone: {voice_tone}.")
            should_greet = greeting and not self._has_greeted_user
            if should_greet:
                identity_parts.append(
                    f"When greeting users for the first time, say: {greeting}"
                )
            if fallback:
                identity_parts.append(
                    f"When you don't understand something, respond with: {fallback}"
                )
            system_parts.append("[Character Identity]\n" + " ".join(identity_parts))

        # Personality / style hints
        persona = profile.get("persona") or self.config.get("behavior.persona", "Friendly")
        verbosity = profile.get("verbosity") or self.config.get("behavior.verbosity", "Medium")
        style = profile.get("response_style") or self.config.get("behavior.response_style", "Conversational")
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

        # If we instructed a first-time greeting this turn, don't repeat it.
        if greeting and not self._has_greeted_user:
            self._has_greeted_user = True
        return messages

    @staticmethod
    def _clean_response(text: str, extra_tokens: List[str] | None = None) -> str:
        """Strip artifact/stop tokens from a raw model reply."""
        tokens = list(_DEFAULT_STOP_TOKENS)
        if extra_tokens:
            tokens.extend(extra_tokens)
        for tok in tokens:
            text = text.replace(tok, "")
        return text.strip()

    def _load_profile(self) -> Dict[str, Any]:
        """Return the currently cached profile data.

        The cache is populated from ``config["profiles_data"]`` whenever a
        profile is selected, and falls back to the legacy ``profile.json``
        file when no profile has been selected yet.
        """
        if self._cached_profile:
            return self._cached_profile
        # Fallback: read legacy file
        if _PROFILE_PATH.exists():
            try:
                data = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _load_active_profile(self, name: str | None = None) -> None:
        """Load profile data from config for *name* (or the selected profile).

        Also syncs the data to ``profile.json`` so other components
        (StimulusAnalyser, etc.) can read it.
        """
        if name is None:
            name = self.config.get("selected_profile", "")

        if not name:
            # No profile selected — try legacy file
            self._cached_profile = {}
            return

        all_data = self.config.get("profiles_data") or {}
        if not isinstance(all_data, dict):
            all_data = {}

        stored = all_data.get(name, {})
        if stored:
            self._cached_profile = dict(stored)
        else:
            # Brand-new profile with no behavior data yet — seed name
            self._cached_profile = {"character_name": name}

        # Sync to profile.json so StimulusAnalyser and other consumers see it
        self._sync_profile_to_disk()

    def _sync_profile_to_disk(self) -> None:
        """Write the cached profile to profile.json for other components."""
        if not self._cached_profile:
            return
        try:
            _PROFILE_PATH.write_text(
                json.dumps(self._cached_profile, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _trim_history(self) -> None:
        """Keep only the last N messages to avoid unbounded growth."""
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
