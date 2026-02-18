"""
System resource monitor — polls CPU, RAM, GPU, and VRAM usage.

Publishes ``runtime_stats`` events on a timer so the top-bar gauges
and the System-tab chart receive live data.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Optional

import psutil
from PyQt6.QtCore import QObject, QTimer

from .events import EventBus
from .plugin_manager import PluginManager


def _gpu_stats() -> tuple[Optional[str], Optional[str]]:
    """Return (gpu_usage%, vram_usage%) strings, or (None, None)."""
    try:
        # Try pynvml first (fast, no subprocess)
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_pct = f"{util.gpu}%"
        vram_used_mb = mem.used / (1024 ** 2)
        vram_total_mb = mem.total / (1024 ** 2)
        vram_pct = f"{vram_used_mb / vram_total_mb * 100:.0f}%"
        pynvml.nvmlShutdown()
        return gpu_pct, vram_pct
    except Exception:
        pass

    # Fallback: nvidia-smi CLI
    if shutil.which("nvidia-smi") is None:
        return None, None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=4,
            text=True,
        )
        parts = out.strip().split(",")
        if len(parts) >= 3:
            gpu_pct = f"{parts[0].strip()}%"
            used = float(parts[1].strip())
            total = float(parts[2].strip())
            vram_pct = f"{used / total * 100:.0f}%" if total else "-"
            return gpu_pct, vram_pct
    except Exception:
        pass

    return None, None


class SystemMonitor(QObject):
    """
    Polls system resources every *interval_ms* and publishes
    ``runtime_stats`` events on the shared EventBus.

    The payload matches what ``CenterPanel`` and ``SystemTab`` expect::

        {"CPU": "12%", "RAM": "48%", "GPU": "35%", "VRAM": "62%",
         "Health": "active"}
    """

    def __init__(
        self,
        event_bus: EventBus,
        plugin_manager: PluginManager,
        interval_ms: int = 3000,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._bus = event_bus
        self._pm = plugin_manager
        self._start_time = time.monotonic()

        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)

    # ── public ──────────────────────────────────────────────────

    def start(self) -> None:
        self._tick()          # immediate first sample
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    # ── private ─────────────────────────────────────────────────

    def _tick(self) -> None:
        cpu = f"{psutil.cpu_percent(interval=0):.0f}%"
        ram = f"{psutil.virtual_memory().percent:.0f}%"

        gpu, vram = _gpu_stats()

        # Determine health from plugin state
        plugin = self._pm.active_plugin
        if plugin is not None and plugin.is_connected():
            health = "active"
            try:
                model = plugin.active_model()
                model_name = model.name if model else "connected"
            except Exception:
                model_name = "connected"
        elif plugin is not None:
            health = "warning"
            model_name = "disconnected"
        else:
            health = "standby"
            model_name = "no provider"

        # Uptime
        elapsed = time.monotonic() - self._start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            uptime = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            uptime = f"{minutes}m {seconds}s"
        else:
            uptime = f"{seconds}s"

        data: dict[str, str] = {
            "CPU": cpu,
            "RAM": ram,
            "Health": health,
            "Model": model_name,
            "Uptime": uptime,
        }
        if gpu is not None:
            data["GPU"] = gpu
        if vram is not None:
            data["VRAM"] = vram

        self._bus.publish("runtime_stats", data)
