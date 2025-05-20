# === ollama_manager.py ===

# === Imports ===
from __future__ import annotations

import os
import subprocess
import time
from typing import Final

import requests

# === Constants ===
OLLAMA_HOST: Final[str] = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHECK_URL: Final[str] = f"{OLLAMA_HOST}/api/tags"


def _is_server_up(timeout: float = 1.5) -> bool:
    """Check if the Ollama server is responding."""
    try:
        requests.get(CHECK_URL, timeout=timeout)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False


def _start_server(detach: bool = True) -> subprocess.Popen | None:
    """
    Starts `ollama serve` if it is not already running.
    Returns the subprocess if started, otherwise None.
    """
    if _is_server_up():
        return None

    # Suppress stdout/stderr output when running detached
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    cmd = ["ollama", "serve"]
    proc = subprocess.Popen(cmd, **kwargs) if detach else subprocess.Popen(cmd)

    # Poll the server to wait until it's responsive
    for _ in range(15):
        if _is_server_up():
            return proc
        time.sleep(0.6)

    raise RuntimeError("Failed to start Ollama server.")


def _warm_up(model: str) -> None:
    """
    Sends a simple prompt to ensure the model is loaded into memory.
    If it fails, prints a warning but does not interrupt execution.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": "ping", "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        _ = resp.json()["response"]
    except Exception as exc:
        print(f"Warm-up skipped ({exc})")


def ensure_model(model: str) -> None:
    """
    Checks if the model is already downloaded locally.
    If not, downloads it using `ollama pull <model>`.
    """
    try:
        resp = requests.get(CHECK_URL, timeout=3).json()
        available = {m["name"] for m in resp.get("models", [])}
        if model not in available:
            print(f"Downloading model '{model}' …")
            subprocess.run(["ollama", "pull", model], check=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to check or pull model '{model}': {exc}")


def prepare_ollama(model: str) -> None:
    """
    Ensures the Ollama backend is ready:
    1. Starts the server (if needed).
    2. Downloads the model (if missing).
    3. Loads the model into memory.
    """
    _start_server()
    ensure_model(model)
    _warm_up(model)
    print("Ollama is ready!")
