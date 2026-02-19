"""
llama.cpp server plugin — connects to (or starts) a local llama-server.

Default endpoint: ``http://localhost:8080``

If a model path is provided and no server is already running on the
target port, the plugin automatically launches ``llama-server`` as a
managed subprocess.  The process is terminated on disconnect.
"""

from __future__ import annotations

import atexit
import shutil
import socket
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Any

from .openai_compat import OpenAICompatPlugin


class Plugin(OpenAICompatPlugin):
    _default_base_url = "http://localhost:8080"
    _plugin_name = "llama.cpp (Local)"
    _plugin_version = "1.1.0"

    def __init__(self) -> None:
        super().__init__()
        self._server_proc: subprocess.Popen | None = None
        atexit.register(self._stop_server)

    # ── Lifecycle ─────────────────────────────────────────────

    def connect(self, config: dict[str, Any]) -> None:
        model_path = config.get("model_path", "")
        server_binary = config.get("llama_server_path", "").strip()

        # Resolve target host/port from the base_url config
        raw_url = config.get("base_url", "").strip().rstrip("/")
        if raw_url.endswith("/v1"):
            raw_url = raw_url[:-3]
        base = raw_url or self._default_base_url
        parsed = urllib.parse.urlparse(base)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8080

        # n_gpu_layers: -1 = all on GPU, 0 = CPU only
        n_gpu_layers: int = int(config.get("n_gpu_layers", 0))

        # Auto-start the server if we have a local model and port is free
        if model_path and Path(model_path).is_file():
            if not self._is_port_in_use(host, port):
                self._launch_server(
                    model_path, host, port, server_binary,
                    n_gpu_layers=n_gpu_layers,
                )

        # Standard OpenAI-compat connect (verify + fetch models)
        super().connect(config)

        # Extra llama.cpp-specific health check
        health = self._get_json(f"{self._base_url}/health")
        status = health.get("status", "")
        if status and status != "ok":
            self._connected = False
            raise ConnectionError(
                f"llama.cpp server reports status: {status}"
            )

    def disconnect(self) -> None:
        super().disconnect()
        self._stop_server()

    # ── Server process management ─────────────────────────────

    def _launch_server(
        self, model_path: str, host: str, port: int,
        server_binary: str = "",
        n_gpu_layers: int = 0,
    ) -> None:
        binary = self._find_binary(server_binary)
        if not binary:
            raise ConnectionError(
                "Cannot find 'llama-server' on this system.\n\n"
                "Install llama.cpp and make sure 'llama-server' is in your "
                "PATH, set a custom binary path in the LLM settings, or "
                "start the server manually before connecting."
            )

        cmd = [
            binary,
            "--model", model_path,
            "--host", host,
            "--port", str(port),
            "--n-gpu-layers", str(n_gpu_layers),
        ]

        self._server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Block until the server finishes loading the model
        self._wait_for_ready(f"http://{host}:{port}")

    def _stop_server(self) -> None:
        proc = self._server_proc
        if proc is None:
            return
        self._server_proc = None
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def _wait_for_ready(self, base_url: str, timeout: int = 180) -> None:
        """Poll ``/health`` until the server reports ``ok``."""
        deadline = time.monotonic() + timeout
        health_url = f"{base_url}/health"

        while time.monotonic() < deadline:
            # If the process died, surface its stderr
            if self._server_proc and self._server_proc.poll() is not None:
                stderr = ""
                if self._server_proc.stderr:
                    stderr = self._server_proc.stderr.read().decode(
                        "utf-8", errors="replace",
                    )[:500]
                code = self._server_proc.returncode
                self._server_proc = None
                raise ConnectionError(
                    f"llama-server exited with code {code}.\n{stderr}"
                )

            try:
                data = self._get_json(health_url)
                if data.get("status") == "ok":
                    return
            except Exception:
                pass

            time.sleep(1)

        # Timed out — clean up
        self._stop_server()
        raise ConnectionError(
            f"llama-server did not finish loading within {timeout}s. "
            "The model may be too large for available memory."
        )

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _find_binary(custom_path: str = "") -> str | None:
        """Locate the ``llama-server`` executable.

        If *custom_path* is provided and points to an executable file it is
        returned immediately, bypassing the automatic search.
        """
        if custom_path:
            p = Path(custom_path)
            if p.is_file():
                return str(p)

        found = shutil.which("llama-server")
        if found:
            return found

        # Check common install locations
        for p in (
            Path.home() / ".local" / "bin" / "llama-server",
            Path("/usr/local/bin/llama-server"),
            Path("/usr/bin/llama-server"),
        ):
            if p.is_file():
                return str(p)

        return None

    @staticmethod
    def _is_port_in_use(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex((host, port)) == 0

    def get_config_schema(self) -> dict[str, Any]:
        schema = super().get_config_schema()
        schema["base_url"]["default"] = "http://localhost:8080"
        # llama.cpp doesn't need an API key
        schema.pop("api_key", None)
        schema["llama_server_path"] = {
            "type": "path",
            "label": "llama-server Binary",
            "default": "",
            "description": (
                "Full path to the llama-server executable. "
                "Leave blank to search PATH and common install locations."
            ),
        }
        return schema
