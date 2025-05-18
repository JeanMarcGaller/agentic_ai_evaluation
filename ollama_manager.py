# ollama_manager.py
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Final

import requests

OLLAMA_HOST: Final[str] = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHECK_URL: Final[str] = f"{OLLAMA_HOST}/api/tags"


def _is_server_up(timeout: float = 1.5) -> bool:
    try:
        requests.get(CHECK_URL, timeout=timeout)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False


def _start_server(detach: bool = True) -> subprocess.Popen:
    """
    Startet `ollama serve` (falls nicht aktiv) und gibt den Prozess zurück.
    Läuft er schon, wird `None` zurückgegeben.
    """
    if _is_server_up():
        return None

    # `ollama serve` hängt sich nicht auf STDOUT ⇒ gleich in den Hintergrund schicken
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    cmd = ["ollama", "serve"]
    proc = subprocess.Popen(cmd, **kwargs) if detach else subprocess.Popen(cmd)

    # kleines Polling, bis Server Ports offen sind
    for _ in range(15):
        if _is_server_up():
            return proc
        time.sleep(0.6)

    raise RuntimeError("Ollama-Server konnte nicht gestartet werden.")


def _warm_up(model: str) -> None:
    """
    Ein schneller Prompt über die REST-API lädt das Modell in den RAM.
    Schlägt der Call fehl, wird nur gewarnt – das Haupt­programm läuft weiter.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": "ping", "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        _ = resp.json()["response"]      # Antwort kommt nur bei success
    except Exception as exc:             # breche nicht ab, nur Hinweis
        print(f"⚠️  Warm-up übersprungen ({exc})")


def ensure_model(model: str = "llama3.1") -> None:
    """
    Prüft, ob das Modell bereits auf der Platte liegt;
    lädt es andernfalls per `ollama pull <model>`.
    """
    resp = requests.get(CHECK_URL, timeout=3).json()
    available = {m["name"] for m in resp.get("models", [])}
    if model not in available:
        print(f"📦  Lade Modell '{model}' …")
        subprocess.run(["ollama", "pull", model], check=True)


def prepare_ollama(model: str = "llama3.1") -> None:
    _start_server()
    ensure_model(model)
    _warm_up(model)
    print("✅  Ollama bereit.")

