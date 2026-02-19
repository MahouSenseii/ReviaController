"""
Timing tracker — measures latency of every pipeline stage and
publishes results to the UI.

Tracks
------
* **Stimulus analysis** — how long keyword/pattern matching takes
* **Emotion processing** — neural network forward pass + blending
* **Decision making** — strategy selection time
* **LLM inference** — total round-trip, TTFT, tokens/sec
* **Metacognition** — prediction evaluation time
* **Total pipeline** — end-to-end from user message to response

All times are in milliseconds.  Results are published via the
``pipeline_timing`` event so the center panel can display them.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .events import EventBus

_HISTORY_FILE = Path("timing_history.json")
_MAX_SAVED = 500  # cap on-disk records


# ------------------------------------------------------------------
# Timing record
# ------------------------------------------------------------------

@dataclass
class TimingRecord:
    """Timing for a single pipeline run."""
    stimulus_ms: float = 0.0
    emotion_ms: float = 0.0
    decision_ms: float = 0.0
    metacognition_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, str]:
        """Format for UI display."""
        return {
            "stimulus": f"{self.stimulus_ms:.1f} ms",
            "emotion": f"{self.emotion_ms:.1f} ms",
            "decision": f"{self.decision_ms:.1f} ms",
            "metacognition": f"{self.metacognition_ms:.1f} ms",
            "inference": f"{self.inference_ms:.0f} ms",
            "total": f"{self.total_ms:.0f} ms",
        }


# ------------------------------------------------------------------
# Timer context manager
# ------------------------------------------------------------------

class _StopWatch:
    """Simple stopwatch that records elapsed time in ms."""

    __slots__ = ("_start", "elapsed_ms")

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


# ------------------------------------------------------------------
# Pipeline Timer
# ------------------------------------------------------------------

class PipelineTimer:
    """
    Measures latency of each stage in the response pipeline.

    Usage by ConversationManager::

        timer = PipelineTimer(bus)
        timer.begin()                          # start total clock

        timer.start("stimulus")
        # ... do stimulus analysis ...
        timer.stop("stimulus")

        timer.start("inference")
        # ... LLM call ...
        timer.stop("inference")

        timer.finish()                         # stop total, publish
    """

    def __init__(self, event_bus: EventBus):
        self.bus = event_bus

        # Current run
        self._watches: Dict[str, _StopWatch] = {}
        self._total_watch = _StopWatch()
        self._running: bool = False

        # History
        self._history: Deque[TimingRecord] = deque(maxlen=100)

    # ── Pipeline lifecycle ────────────────────────────────────

    def begin(self) -> None:
        """Start timing a new pipeline run."""
        self._watches.clear()
        self._total_watch = _StopWatch()
        self._total_watch.start()
        self._running = True

    def start(self, stage: str) -> None:
        """Start timing a specific stage."""
        if not self._running:
            return
        sw = _StopWatch()
        sw.start()
        self._watches[stage] = sw

    def stop(self, stage: str) -> None:
        """Stop timing a specific stage."""
        sw = self._watches.get(stage)
        if sw:
            sw.stop()

    def finish(self) -> TimingRecord:
        """
        Stop the total timer, build a TimingRecord, publish it,
        persist it to disk, and return it.
        """
        self._total_watch.stop()
        self._running = False

        record = TimingRecord(
            stimulus_ms=self._get_ms("stimulus"),
            emotion_ms=self._get_ms("emotion"),
            decision_ms=self._get_ms("decision"),
            metacognition_ms=self._get_ms("metacognition"),
            inference_ms=self._get_ms("inference"),
            total_ms=self._total_watch.elapsed_ms,
            timestamp=time.time(),
        )

        self._history.append(record)
        self._persist(record)

        # Build event data: current values + rolling averages
        data = record.to_dict()
        n = len(self._history)
        avg_n = min(n, 10)
        data["avg_inference"] = f"{self.average_ms('inference', avg_n):.0f} ms"
        data["avg_total"] = f"{self.average_ms('total', avg_n):.0f} ms"
        data["avg_decision"] = f"{self.average_ms('decision', avg_n):.1f} ms"
        data["runs"] = n

        # Publish for UI
        self.bus.publish("pipeline_timing", data)

        return record

    # ── Persistence ──────────────────────────────────────────

    def _persist(self, record: TimingRecord) -> None:
        """Append a timing record to the on-disk history file."""
        try:
            if _HISTORY_FILE.exists():
                existing: List[dict] = json.loads(_HISTORY_FILE.read_text("utf-8"))
                if not isinstance(existing, list):
                    existing = []
            else:
                existing = []

            existing.append({
                "ts": record.timestamp,
                "stimulus_ms": record.stimulus_ms,
                "emotion_ms": record.emotion_ms,
                "decision_ms": record.decision_ms,
                "metacognition_ms": record.metacognition_ms,
                "inference_ms": record.inference_ms,
                "total_ms": record.total_ms,
            })

            # Trim to max saved records
            if len(existing) > _MAX_SAVED:
                existing = existing[-_MAX_SAVED:]

            _HISTORY_FILE.write_text(json.dumps(existing, indent=2), "utf-8")
        except Exception:
            pass  # Never crash the pipeline over a log write

    def load_history(self) -> List[dict]:
        """Return the on-disk timing history (raw dicts)."""
        try:
            if _HISTORY_FILE.exists():
                data = json.loads(_HISTORY_FILE.read_text("utf-8"))
                return data if isinstance(data, list) else []
        except Exception:
            pass
        return []

    # ── Query ─────────────────────────────────────────────────

    def last_record(self) -> Optional[TimingRecord]:
        """Return the most recent timing record."""
        return self._history[-1] if self._history else None

    def average_ms(self, stage: str, n: int = 10) -> float:
        """Return average ms for a stage over the last N runs."""
        recent = list(self._history)[-n:]
        if not recent:
            return 0.0
        vals = [getattr(r, f"{stage}_ms", 0.0) for r in recent]
        return sum(vals) / len(vals)

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of timing stats."""
        if not self._history:
            return {"message": "No timing data yet"}

        last = self._history[-1]
        return {
            "last": last.to_dict(),
            "avg_total_ms": round(self.average_ms("total"), 1),
            "avg_inference_ms": round(self.average_ms("inference"), 1),
            "avg_emotion_ms": round(self.average_ms("emotion"), 2),
            "avg_decision_ms": round(self.average_ms("decision"), 2),
            "runs": len(self._history),
        }

    # ── Internal ──────────────────────────────────────────────

    def _get_ms(self, stage: str) -> float:
        sw = self._watches.get(stage)
        return sw.elapsed_ms if sw else 0.0
