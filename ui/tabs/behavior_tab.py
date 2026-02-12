"""
Behavior tab — personality, verbosity, response style controls.

All inputs write to Config and publish events so other components
can react in real time.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QTextEdit

from .base_tab import BaseTab


class BehaviorTab(BaseTab):

    def _build(self) -> None:
        lay = self._layout

        lay.addWidget(self._heading("Personality"))

        self._persona_combo = self._make_combo(
            ["Friendly", "Professional", "Concise", "Custom"],
            current=0,
        )
        self._persona_combo.currentTextChanged.connect(self._on_persona)
        lay.addWidget(self._row("Persona", self._persona_combo))

        self._verbosity = self._make_combo(["Low", "Medium", "High"], current=1)
        self._verbosity.currentTextChanged.connect(
            lambda v: self._save("behavior.verbosity", v)
        )
        lay.addWidget(self._row("Verbosity", self._verbosity))

        self._response_style = self._make_combo(
            ["Conversational", "Bullet Points", "Technical", "ELI5"],
            current=0,
        )
        self._response_style.currentTextChanged.connect(
            lambda v: self._save("behavior.response_style", v)
        )
        lay.addWidget(self._row("Style", self._response_style))

        lay.addWidget(self._heading("System Prompt"))

        self._sys_prompt = QTextEdit()
        self._sys_prompt.setObjectName("SettingsTextEdit")
        self._sys_prompt.setPlaceholderText("Enter a custom system prompt...")
        self._sys_prompt.setMaximumHeight(160)
        self._sys_prompt.setStyleSheet(
            "background:#101824; border:1px solid #2a3b55; border-radius:8px; "
            "color:#d8e1ee; padding:8px; font-size:12px;"
        )
        self._sys_prompt.textChanged.connect(self._on_sys_prompt)
        lay.addWidget(self._sys_prompt)

        # load persisted values
        self._load_saved()

    def _load_saved(self) -> None:
        persona = self.config.get("behavior.persona", "Friendly")
        idx = self._persona_combo.findText(persona)
        if idx >= 0:
            self._persona_combo.setCurrentIndex(idx)

        verbosity = self.config.get("behavior.verbosity", "Medium")
        idx = self._verbosity.findText(verbosity)
        if idx >= 0:
            self._verbosity.setCurrentIndex(idx)

        style = self.config.get("behavior.response_style", "Conversational")
        idx = self._response_style.findText(style)
        if idx >= 0:
            self._response_style.setCurrentIndex(idx)

        prompt = self.config.get("behavior.system_prompt", "")
        if prompt:
            self._sys_prompt.setPlainText(prompt)

    def _save(self, key: str, value: str) -> None:
        self.config.set(key, value)
        self.bus.publish("behavior_changed", {"key": key, "value": value})

    def _on_persona(self, text: str) -> None:
        self._save("behavior.persona", text)

    def _on_sys_prompt(self) -> None:
        self._save("behavior.system_prompt", self._sys_prompt.toPlainText())
