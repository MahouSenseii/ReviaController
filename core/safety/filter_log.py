"""
Structured safety-filter logging.

Every message that passes through the filter — whether allowed,
rewritten, or blocked — gets a log entry written to:

1. A JSON-Lines file (``safety_logs/filter_log.jsonl``) for
   post-hoc auditing.
2. The ``EventBus`` as a ``log_entry`` event so the Logs tab can
   display it in real time.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOG_DIR = Path("safety_logs")
_LOG_FILE = _LOG_DIR / "filter_log.jsonl"


# ------------------------------------------------------------------
# Log entry builder
# ------------------------------------------------------------------

def build_log_entry(
    stage: str,                      # "input" | "output"
    action: str,                     # "allow" | "rewrite" | "block"
    original_text: str,
    scores: Dict[str, int],          # {category_slug: score}
    thresholds: Dict[str, int],      # {category_slug: threshold}
    enabled: Dict[str, bool],        # {category_slug: enabled}
    reason: str = "",
    rewritten_text: Optional[str] = None,
    matched_details: Optional[Dict[str, Any]] = None,
    store_original: bool = True,
) -> Dict[str, Any]:
    """Build a structured log entry dict."""
    entry: Dict[str, Any] = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "stage": stage,
        "action": action,
        "scores": scores,
        "thresholds": thresholds,
        "enabled_categories": enabled,
        "reason": reason,
    }

    if store_original:
        entry["original_text"] = original_text
    else:
        entry["original_text"] = f"[redacted, length={len(original_text)}]"

    if rewritten_text is not None:
        entry["rewritten_text"] = rewritten_text

    if matched_details:
        entry["matched"] = matched_details

    return entry


# ------------------------------------------------------------------
# Log writer
# ------------------------------------------------------------------

class FilterLogger:
    """
    Append-only JSON-Lines logger for safety filter events.

    Also publishes each entry to the ``EventBus`` so the UI Logs tab
    can display it live.
    """

    def __init__(self, event_bus: Optional[Any] = None, log_dir: Optional[Path] = None):
        self._bus = event_bus
        self._dir = log_dir or _LOG_DIR
        self._file = self._dir / "filter_log.jsonl"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._store_original = True   # can be toggled off for privacy

    @property
    def store_original_text(self) -> bool:
        return self._store_original

    @store_original_text.setter
    def store_original_text(self, val: bool) -> None:
        self._store_original = val

    def log(self, entry: Dict[str, Any]) -> None:
        """Write one log entry to disk and publish to EventBus."""
        # Append to JSONL
        try:
            with self._file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

        # Publish to EventBus for UI
        if self._bus is not None:
            action = entry.get("action", "allow")
            stage = entry.get("stage", "?")
            reason = entry.get("reason", "")

            # Map to the log categories the LogsTab already understands
            if action == "allow":
                ui_category = "Allowed"
            elif action == "rewrite":
                ui_category = "Rewritten"
            else:
                ui_category = "Filtered"

            summary = f"[{stage.upper()}] {action.upper()}"
            if reason:
                summary += f" — {reason}"

            self._bus.publish("log_entry", {
                "category": ui_category,
                "text": summary,
                "detail": entry,
            })

    def read_recent(self, n: int = 50) -> List[Dict[str, Any]]:
        """Read the last *n* log entries from disk."""
        if not self._file.exists():
            return []
        try:
            lines = self._file.read_text(encoding="utf-8").strip().split("\n")
            entries = []
            for line in lines[-n:]:
                if line.strip():
                    entries.append(json.loads(line))
            return entries
        except (json.JSONDecodeError, OSError):
            return []

    def clear(self) -> None:
        """Clear the log file."""
        try:
            self._file.write_text("", encoding="utf-8")
        except OSError:
            pass
