"""
LLM registry — standalone JSON file that stores metadata for every
saved model (local *and* online).

The registry lives at ``llm_registry.json`` next to ``config.json``.
Each entry is keyed by a unique **id** (slug derived from the name)
so look-ups are O(1).

Schema per entry::

    {
        "id":              "llama3-8b",
        "name":            "Llama 3 8B",
        "mode":            "local",          # "local" | "online"
        "path":            "/models/llama3.gguf",
        "address":         "http://localhost:8080/v1",
        "provider":        "",               # "OpenAI", "Anthropic", …
        "api_key":         "",
        "endpoint":        "",
        "model_id":        "",
        "size_bytes":      4_500_000_000,
        "size_label":      "4.5 GB",
        "compute":         "GPU",            # "GPU" | "CPU" | "Unknown"
        "parameters":      "",               # e.g. "8B", "70B"
        "quant":           "",               # e.g. "Q4_K_M", "FP16"
        "notes":           "",
        "stop_tokens":     [],               # extra stop/artifact tokens to strip
        "chat_template":   "auto",           # "auto"|"chatml"|"llama3"|"mistral"|…
        "needs_formatting": True,            # strip artifact tokens before display
    }

The module never imports Qt — it is pure-Python so core logic or
a future headless mode can use it too.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_REGISTRY_PATH = Path("llm_registry.json")

# GGUF filenames often encode quant + param info, e.g.
#   "llama-3-8b-instruct.Q4_K_M.gguf"
_QUANT_RE = re.compile(
    r"[._-](Q[0-9]_[A-Z0-9_]+|F16|FP16|FP32|BF16|GPTQ|AWQ|GGML)", re.IGNORECASE
)
_PARAM_RE = re.compile(r"(\d+\.?\d*)\s*[Bb]", re.IGNORECASE)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} PB"


def _guess_compute(name: str, path: str) -> str:
    """Best-effort guess from filename conventions."""
    low = (name + path).lower()
    if "cpu" in low:
        return "CPU"
    if "gpu" in low or "cuda" in low:
        return "GPU"
    # GGUF / GGML models are typically CPU-friendly but can offload
    ext = Path(path).suffix.lower() if path else ""
    if ext in (".gguf", ".ggml", ".bin"):
        return "CPU"
    if ext in (".safetensors", ".pt", ".pth"):
        return "GPU"
    return "Unknown"


def _guess_quant(filename: str) -> str:
    m = _QUANT_RE.search(filename)
    return m.group(1).upper() if m else ""


def _guess_params(filename: str) -> str:
    m = _PARAM_RE.search(filename)
    return f"{m.group(1)}B" if m else ""


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

class LLMRegistry:
    """Read / write ``llm_registry.json``."""

    def __init__(self, path: Path | str = _REGISTRY_PATH) -> None:
        self._path = Path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    # -- persistence ---------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._entries = raw
            except (json.JSONDecodeError, OSError):
                self._entries = {}

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._entries, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    # -- queries -------------------------------------------------------

    def all(self) -> list[dict[str, Any]]:
        return list(self._entries.values())

    def get(self, entry_id: str) -> dict[str, Any] | None:
        return self._entries.get(entry_id)

    def find_by_name(self, name: str) -> dict[str, Any] | None:
        for e in self._entries.values():
            if e.get("name") == name:
                return e
        return None

    # -- mutations -----------------------------------------------------

    def upsert_local(
        self,
        name: str,
        path: str,
        address: str = "",
        *,
        compute: str = "",
        parameters: str = "",
        quant: str = "",
        notes: str = "",
        stop_tokens: list[str] | None = None,
        chat_template: str | None = None,
        needs_formatting: bool | None = None,
    ) -> dict[str, Any]:
        """Add or update a **local** model entry.

        Auto-detects size, quant, param count, and compute type from
        the file when not provided explicitly.  Formatting fields
        (stop_tokens, chat_template, needs_formatting) are preserved
        from the existing entry when not supplied.
        """
        entry_id = _slugify(name)
        filename = Path(path).name if path else ""
        existing = self._entries.get(entry_id, {})

        size_bytes = 0
        if path and os.path.isfile(path):
            size_bytes = os.path.getsize(path)

        entry: dict[str, Any] = {
            "id": entry_id,
            "name": name,
            "mode": "local",
            "path": path,
            "address": address,
            "provider": "",
            "api_key": "",
            "endpoint": "",
            "model_id": "",
            "size_bytes": size_bytes,
            "size_label": _human_size(size_bytes) if size_bytes else "-",
            "compute": compute or _guess_compute(name, path),
            "parameters": parameters or _guess_params(filename),
            "quant": quant or _guess_quant(filename),
            "notes": notes,
            # Formatting fields — use supplied value, else preserve existing,
            # else use sensible default
            "stop_tokens": (
                stop_tokens if stop_tokens is not None
                else existing.get("stop_tokens", [])
            ),
            "chat_template": (
                chat_template if chat_template is not None
                else existing.get("chat_template", "auto")
            ),
            "needs_formatting": (
                needs_formatting if needs_formatting is not None
                else existing.get("needs_formatting", True)
            ),
        }
        self._entries[entry_id] = entry
        self._save()
        return entry

    def upsert_online(
        self,
        name: str,
        provider: str,
        api_key: str = "",
        endpoint: str = "",
        model_id: str = "",
        *,
        notes: str = "",
        stop_tokens: list[str] | None = None,
        chat_template: str | None = None,
        needs_formatting: bool | None = None,
    ) -> dict[str, Any]:
        """Add or update an **online** API entry."""
        entry_id = _slugify(name)
        existing = self._entries.get(entry_id, {})
        entry: dict[str, Any] = {
            "id": entry_id,
            "name": name,
            "mode": "online",
            "path": "",
            "address": endpoint or provider,
            "provider": provider,
            "api_key": api_key,
            "endpoint": endpoint,
            "model_id": model_id,
            "size_bytes": 0,
            "size_label": "-",
            "compute": "Cloud",
            "parameters": "",
            "quant": "",
            "notes": notes,
            "stop_tokens": (
                stop_tokens if stop_tokens is not None
                else existing.get("stop_tokens", [])
            ),
            "chat_template": (
                chat_template if chat_template is not None
                else existing.get("chat_template", "auto")
            ),
            # Online APIs are clean by default
            "needs_formatting": (
                needs_formatting if needs_formatting is not None
                else existing.get("needs_formatting", False)
            ),
        }
        self._entries[entry_id] = entry
        self._save()
        return entry

    def patch(self, entry_id: str, **fields: Any) -> dict[str, Any] | None:
        """Update specific fields of an existing entry without touching others."""
        entry = self._entries.get(entry_id)
        if entry is None:
            return None
        entry.update(fields)
        self._save()
        return entry

    def remove(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)
        self._save()

    def rename(self, old_id: str, new_name: str) -> dict[str, Any] | None:
        entry = self._entries.pop(old_id, None)
        if entry is None:
            return None
        new_id = _slugify(new_name)
        entry["id"] = new_id
        entry["name"] = new_name
        self._entries[new_id] = entry
        self._save()
        return entry
