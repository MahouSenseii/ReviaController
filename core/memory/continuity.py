"""
Conversation continuity state — tracks unresolved threads, open
commitments, pending tasks, and "last discussed" context.

Gives the AI the ability to pick up where it left off, remember
what it promised, and carry forward pending work.

Integration
-----------
The ``ContinuityTracker`` subscribes to ``chat_message`` events and
publishes ``continuity_updated`` when threads change.  Call
``get_context()`` to get an LLM prompt injection block.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# Thread types and states
# ------------------------------------------------------------------

class ThreadStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    STALE = "stale"


class CommitmentStatus(str, Enum):
    PENDING = "pending"
    FULFILLED = "fulfilled"
    BROKEN = "broken"
    EXPIRED = "expired"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class ConversationThread:
    """An open topic or line of discussion."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    topic: str = ""
    status: ThreadStatus = ThreadStatus.OPEN
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    summary: str = ""
    messages: int = 0              # number of messages in this thread
    session_id: str = ""
    user_id: str = ""


@dataclass
class Commitment:
    """Something the AI promised or the user asked for."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    description: str = ""
    status: CommitmentStatus = CommitmentStatus.PENDING
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None   # optional unix timestamp
    fulfilled_at: Optional[float] = None
    session_id: str = ""
    user_id: str = ""


@dataclass
class PendingTask:
    """A task inferred from conversation that needs follow-up."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    description: str = ""
    priority: float = 0.5          # 0..1
    created_at: float = field(default_factory=time.time)
    completed: bool = False
    completed_at: Optional[float] = None
    session_id: str = ""
    user_id: str = ""


@dataclass
class ContinuityConfig:
    thread_stale_hours: float = 24.0      # mark threads stale after this
    max_open_threads: int = 20            # cap on tracked open threads
    max_pending_tasks: int = 50           # cap on pending tasks
    max_commitments: int = 30             # cap on tracked commitments
    commitment_expire_hours: float = 72.0 # auto-expire unfulfilled commitments


# ------------------------------------------------------------------
# Continuity tracker
# ------------------------------------------------------------------

class ContinuityTracker:
    """
    Tracks conversation threads, commitments, and pending tasks.

    Subscribes to:
        - ``chat_message`` — to track conversation flow
        - ``continuity_command`` — explicit thread/task management

    Publishes:
        - ``continuity_updated`` — when state changes
    """

    def __init__(
        self,
        event_bus: Optional[Any] = None,
        config: Optional[ContinuityConfig] = None,
    ):
        self.bus = event_bus
        self.cfg = config or ContinuityConfig()

        self._threads: Dict[str, ConversationThread] = {}
        self._commitments: Dict[str, Commitment] = {}
        self._tasks: Dict[str, PendingTask] = {}
        self._last_discussed: str = ""
        self._last_discussed_at: float = 0.0

        if self.bus:
            self.bus.subscribe("chat_message", self._on_chat_message)
            self.bus.subscribe("continuity_command", self._on_command)

    # ── Thread management ─────────────────────────────────────

    def open_thread(
        self,
        topic: str,
        session_id: str = "",
        user_id: str = "",
    ) -> ConversationThread:
        """Open a new conversation thread."""
        thread = ConversationThread(
            topic=topic,
            session_id=session_id,
            user_id=user_id,
        )
        self._threads[thread.id] = thread
        self._enforce_thread_cap()
        self._publish()
        return thread

    def resolve_thread(self, thread_id: str) -> bool:
        """Mark a thread as resolved."""
        thread = self._threads.get(thread_id)
        if thread and thread.status == ThreadStatus.OPEN:
            thread.status = ThreadStatus.RESOLVED
            self._publish()
            return True
        return False

    def touch_thread(self, thread_id: str, summary: str = "") -> None:
        """Update a thread's last-active time and optionally its summary."""
        thread = self._threads.get(thread_id)
        if thread:
            thread.last_active = time.time()
            thread.messages += 1
            if summary:
                thread.summary = summary
            self._publish()

    def get_open_threads(self) -> List[ConversationThread]:
        """Return all open (non-resolved, non-stale) threads."""
        self._age_threads()
        return [t for t in self._threads.values() if t.status == ThreadStatus.OPEN]

    # ── Commitment management ─────────────────────────────────

    def add_commitment(
        self,
        description: str,
        deadline: Optional[float] = None,
        session_id: str = "",
        user_id: str = "",
    ) -> Commitment:
        """Record a new commitment."""
        c = Commitment(
            description=description,
            deadline=deadline,
            session_id=session_id,
            user_id=user_id,
        )
        self._commitments[c.id] = c
        self._enforce_commitment_cap()
        self._publish()
        return c

    def fulfill_commitment(self, commitment_id: str) -> bool:
        c = self._commitments.get(commitment_id)
        if c and c.status == CommitmentStatus.PENDING:
            c.status = CommitmentStatus.FULFILLED
            c.fulfilled_at = time.time()
            self._publish()
            return True
        return False

    def get_pending_commitments(self) -> List[Commitment]:
        self._age_commitments()
        return [c for c in self._commitments.values() if c.status == CommitmentStatus.PENDING]

    # ── Task management ───────────────────────────────────────

    def add_task(
        self,
        description: str,
        priority: float = 0.5,
        session_id: str = "",
        user_id: str = "",
    ) -> PendingTask:
        t = PendingTask(
            description=description,
            priority=priority,
            session_id=session_id,
            user_id=user_id,
        )
        self._tasks[t.id] = t
        self._enforce_task_cap()
        self._publish()
        return t

    def complete_task(self, task_id: str) -> bool:
        t = self._tasks.get(task_id)
        if t and not t.completed:
            t.completed = True
            t.completed_at = time.time()
            self._publish()
            return True
        return False

    def get_pending_tasks(self) -> List[PendingTask]:
        return sorted(
            [t for t in self._tasks.values() if not t.completed],
            key=lambda t: t.priority,
            reverse=True,
        )

    # ── Context for LLM ──────────────────────────────────────

    @property
    def last_discussed(self) -> str:
        return self._last_discussed

    def get_context(self) -> Dict[str, Any]:
        """Build LLM prompt context for conversation continuity."""
        open_threads = self.get_open_threads()
        pending_commits = self.get_pending_commitments()
        pending_tasks = self.get_pending_tasks()

        lines: List[str] = []

        if self._last_discussed:
            lines.append(f"Last discussed topic: {self._last_discussed}")

        if open_threads:
            lines.append("Open conversation threads:")
            for t in open_threads[:5]:
                desc = t.summary or t.topic
                lines.append(f"  - {desc}")

        if pending_commits:
            lines.append("Unfulfilled commitments:")
            for c in pending_commits[:5]:
                lines.append(f"  - {c.description}")

        if pending_tasks:
            lines.append("Pending tasks:")
            for t in pending_tasks[:5]:
                lines.append(f"  - {t.description} (priority: {t.priority:.1f})")

        if not lines:
            prompt_injection = (
                "[Continuity Context]\n"
                "No open threads, commitments, or pending tasks."
            )
        else:
            block = "\n".join(lines)
            prompt_injection = (
                "[Continuity Context]\n"
                f"{block}\n"
                "Use this context to maintain conversation flow and "
                "follow through on commitments."
            )

        return {
            "open_threads": len(open_threads),
            "pending_commitments": len(pending_commits),
            "pending_tasks": len(pending_tasks),
            "last_discussed": self._last_discussed,
            "prompt_injection": prompt_injection,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "open_threads": len(self.get_open_threads()),
            "resolved_threads": sum(
                1 for t in self._threads.values()
                if t.status == ThreadStatus.RESOLVED
            ),
            "pending_commitments": len(self.get_pending_commitments()),
            "fulfilled_commitments": sum(
                1 for c in self._commitments.values()
                if c.status == CommitmentStatus.FULFILLED
            ),
            "pending_tasks": len(self.get_pending_tasks()),
            "completed_tasks": sum(1 for t in self._tasks.values() if t.completed),
            "last_discussed": self._last_discussed,
        }

    # ── Event handlers ────────────────────────────────────────

    def _on_chat_message(self, data: Dict[str, Any]) -> None:
        content = data.get("content", "")
        if content:
            self._last_discussed = content[:200]
            self._last_discussed_at = time.time()

    def _on_command(self, data: Dict[str, Any]) -> None:
        action = data.get("action", "")
        if action == "open_thread":
            self.open_thread(
                topic=data.get("topic", ""),
                session_id=data.get("session_id", ""),
                user_id=data.get("user_id", ""),
            )
        elif action == "resolve_thread":
            self.resolve_thread(data.get("thread_id", ""))
        elif action == "add_commitment":
            self.add_commitment(
                description=data.get("description", ""),
                session_id=data.get("session_id", ""),
                user_id=data.get("user_id", ""),
            )
        elif action == "fulfill_commitment":
            self.fulfill_commitment(data.get("commitment_id", ""))
        elif action == "add_task":
            self.add_task(
                description=data.get("description", ""),
                priority=data.get("priority", 0.5),
                session_id=data.get("session_id", ""),
                user_id=data.get("user_id", ""),
            )
        elif action == "complete_task":
            self.complete_task(data.get("task_id", ""))

    # ── Internal helpers ──────────────────────────────────────

    def _age_threads(self) -> None:
        now = time.time()
        stale_threshold = self.cfg.thread_stale_hours * 3600
        for t in self._threads.values():
            if t.status == ThreadStatus.OPEN:
                if (now - t.last_active) > stale_threshold:
                    t.status = ThreadStatus.STALE

    def _age_commitments(self) -> None:
        now = time.time()
        expire_threshold = self.cfg.commitment_expire_hours * 3600
        for c in self._commitments.values():
            if c.status == CommitmentStatus.PENDING:
                if (now - c.created_at) > expire_threshold:
                    c.status = CommitmentStatus.EXPIRED

    def _enforce_thread_cap(self) -> None:
        open_threads = [t for t in self._threads.values() if t.status == ThreadStatus.OPEN]
        if len(open_threads) > self.cfg.max_open_threads:
            open_threads.sort(key=lambda t: t.last_active)
            for t in open_threads[: len(open_threads) - self.cfg.max_open_threads]:
                t.status = ThreadStatus.STALE

    def _enforce_commitment_cap(self) -> None:
        pending = [c for c in self._commitments.values() if c.status == CommitmentStatus.PENDING]
        if len(pending) > self.cfg.max_commitments:
            pending.sort(key=lambda c: c.created_at)
            for c in pending[: len(pending) - self.cfg.max_commitments]:
                c.status = CommitmentStatus.EXPIRED

    def _enforce_task_cap(self) -> None:
        pending = [t for t in self._tasks.values() if not t.completed]
        if len(pending) > self.cfg.max_pending_tasks:
            pending.sort(key=lambda t: t.priority)
            for t in pending[: len(pending) - self.cfg.max_pending_tasks]:
                t.completed = True
                t.completed_at = time.time()

    def _publish(self) -> None:
        if self.bus:
            self.bus.publish("continuity_updated", self.stats())
