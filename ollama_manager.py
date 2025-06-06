"""Utility helpers to start and prepare an Ollama backend.

The functions here are imported by *main.py* to
- start the local Ollama server (if not already running)
- make sure a given model is downloaded
- warm‑up the model so the first real request is fast
"""

from __future__ import annotations

import logging

# === Imports ===
import os
import subprocess
import time
from subprocess import Popen
from typing import Final, Optional

import requests

# === Logging ===
logger = logging.getLogger(__name__)

# === Constants ===
OLLAMA_HOST: Final[str] = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHECK_URL: Final[str] = f"{OLLAMA_HOST}/api/tags"

# === Helpers ===


def _is_server_up(timeout: float = 1.5) -> bool:
    """Return True if the Ollama HTTP endpoint responds within *timeout* seconds."""
    try:
        requests.get(CHECK_URL, timeout=timeout)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False


def _start_server(detach: bool = True) -> Optional[Popen[bytes]]:
    """Run ``ollama serve`` unless it is already running.

    Returns the subprocess handle if a new server was started, otherwise None.
    """
    if _is_server_up():
        logger.info("Ollama server already running at %s", OLLAMA_HOST)
        return None

    cmd = ["ollama", "serve"]
    logger.info("Starting Ollama server …")

    proc: Popen[bytes]
    if detach:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        proc = subprocess.Popen(cmd)

    # Poll until responsive (≈9 s total)
    for _ in range(15):
        if _is_server_up():
            logger.info("Ollama server is up and responsive")
            return proc
        time.sleep(0.6)

    logger.error("Failed to start Ollama server – timeout reached")
    raise RuntimeError("Failed to start Ollama server.")


def _warm_up(model: str) -> None:
    """Send a dummy prompt so the model is loaded into memory."""
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": "ping", "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        snippet = resp.json().get("response", "")[:40]
        logger.debug("Model '%s' warm‑up response: %s", model, snippet)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Warm‑up skipped (%s)", exc, exc_info=False)


def ensure_model(model: str) -> None:
    """Download *model* via ``ollama pull`` if it is not yet available locally."""
    try:
        resp = requests.get(CHECK_URL, timeout=3)
        resp.raise_for_status()
        available = {m["name"] for m in resp.json().get("models", [])}
        if model not in available:
            logger.info("Downloading model '%s' … this may take a while", model)
            subprocess.run(["ollama", "pull", model], check=True)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to check or pull model '%s'", model)
        raise RuntimeError(f"Failed to check or pull model '{model}': {exc}") from exc


def prepare_ollama(model: str) -> None:
    """Ensure server is running, model present, and warmed up."""
    logger.info("Preparing Ollama backend for model '%s'", model)
    _start_server()
    ensure_model(model)
    _warm_up(model)
    logger.info("Ollama backend ready! ✔")
