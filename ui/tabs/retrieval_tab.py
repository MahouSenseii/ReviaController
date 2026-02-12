"""
Retrieval trace tab — shows why each memory was retrieved,
exposing similarity scores, recency boost, emotion bias, and
importance weighting for debugging and trust.

Subscribes to ``retrieval_trace`` events published by the RAG
engine during recall operations.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .base_tab import BaseTab


class _TraceCard(QFrame):
    """Single retrieval trace entry."""

    def __init__(self, data: dict):
        super().__init__()
        self.setObjectName("TraceCard")
        self.setStyleSheet(
            "QFrame#TraceCard { background:#111827; border:1px solid #2a3b55; "
            "border-radius:8px; padding:8px; margin:2px 0; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(3)

        # Rank and content
        rank = data.get("rank", 0)
        content = data.get("content", "")[:150]
        rank_lbl = QLabel(f"#{rank}  {content}")
        rank_lbl.setWordWrap(True)
        rank_lbl.setStyleSheet("color:#e5e7eb; font-size:12px;")
        layout.addWidget(rank_lbl)

        # Score breakdown
        scores = QHBoxLayout()
        score_items = [
            ("Score", data.get("final_score", 0), "#8fc9ff"),
            ("Sim", data.get("similarity", 0), "#60a5fa"),
            ("Imp", data.get("importance", 0), "#22c55e"),
            ("Recency", data.get("recency_boost", 0), "#f59e0b"),
            ("Emotion", data.get("emotion_bonus", 0), "#c084fc"),
        ]
        for label, value, color in score_items:
            item = QLabel(f"{label}: {value:.3f}")
            item.setStyleSheet(f"color:{color}; font-size:10px;")
            scores.addWidget(item)
        scores.addStretch()

        layout.addLayout(scores)

        # Memory metadata
        meta_items = []
        if data.get("memory_type"):
            meta_items.append(f"store:{data['memory_type']}")
        if data.get("entry_type"):
            meta_items.append(f"type:{data['entry_type']}")
        if data.get("source"):
            meta_items.append(f"src:{data['source']}")
        if data.get("session_id"):
            meta_items.append(f"sid:{data['session_id'][:8]}")

        if meta_items:
            meta_lbl = QLabel("  |  ".join(meta_items))
            meta_lbl.setStyleSheet("color:#6b7280; font-size:10px;")
            layout.addWidget(meta_lbl)


class RetrievalTab(BaseTab):
    """Retrieval trace panel showing memory ranking explanations."""

    def _build(self) -> None:
        lay = self._layout

        lay.addWidget(self._heading("Retrieval Trace"))

        # Query display
        self._query_label = QLabel("Query: (none)")
        self._query_label.setStyleSheet("color:#8fc9ff; font-size:12px;")
        self._query_label.setWordWrap(True)
        lay.addWidget(self._query_label)

        # Summary
        self._summary_label = QLabel("No retrieval traces yet")
        self._summary_label.setStyleSheet("color:#9ca3af; font-size:11px;")
        lay.addWidget(self._summary_label)

        # Scroll area for trace cards
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        self._card_container = QWidget()
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setContentsMargins(0, 0, 0, 0)
        self._card_layout.setSpacing(4)
        self._card_layout.addStretch()
        self._scroll.setWidget(self._card_container)
        lay.addWidget(self._scroll, 1)

        self.bus.subscribe("retrieval_trace", self._on_trace)

    def _on_trace(self, data: dict) -> None:
        query = data.get("query", "")
        results = data.get("results", [])
        profile = data.get("profile", "?")

        self._query_label.setText(f"Query: \"{query[:100]}\"")
        self._summary_label.setText(
            f"Profile: {profile}  |  Results: {len(results)}  |  "
            f"Top score: {results[0].get('final_score', 0):.3f}" if results else
            f"Profile: {profile}  |  No results"
        )

        # Clear old cards
        while self._card_layout.count() > 1:
            item = self._card_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # Add new trace cards
        for i, result in enumerate(results):
            result["rank"] = i + 1
            card = _TraceCard(result)
            self._card_layout.insertWidget(self._card_layout.count() - 1, card)
