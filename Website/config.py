"""Environment-driven configuration for the website.

Every setting has a working default, so ``python -m Website.app`` runs locally with no
environment at all. A hosting platform (Render, Docker, systemd) overrides behaviour by
setting the variable by name — the code reads ``os.environ`` and nothing is hardcoded in
the request path.

Design notes:

* Relative paths are resolved against the repository root, so an override such as
  ``MODELS_DIR=Models`` behaves the same from any working directory.
* Malformed numbers fall back to the default instead of crashing the process at import
  time: a typo in a dashboard field should never take the site down.
* This project needs **no API keys or secrets**. ``SECRET_KEY`` is read only so that
  Flask sessions work if they are ever added; the app functions without it.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBSITE_DIR = Path(__file__).resolve().parent

# Optional convenience for local runs: read `.env` if python-dotenv is installed.
# Real environment variables always win, and on hosted platforms there is no .env.
try:  # pragma: no cover - trivial import guard
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover
    pass


def _path(name: str, default: Path) -> Path:
    """Read a path setting; relative values are resolved against the project root."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def _int(name: str, default: int, minimum: int = 1) -> int:
    """Read an integer setting, falling back to ``default`` when missing or invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "debug"}


# --------------------------------------------------------------------- server
HOST = os.environ.get("HOST", "127.0.0.1").strip() or "127.0.0.1"
PORT = _int("PORT", 5000)
DEBUG = _bool("FLASK_DEBUG", False)
LOG_LEVEL = (os.environ.get("LOG_LEVEL") or "INFO").strip().upper()
# Optional: only needed if Flask sessions are introduced later.
SECRET_KEY = os.environ.get("SECRET_KEY") or None

# --------------------------------------------------------------------- models
MODELS_DIR = _path("MODELS_DIR", PROJECT_ROOT / "Models")
METRICS_PATH = _path("METRICS_PATH", PROJECT_ROOT / "Evaluation" / "metrics" / "metrics.json")

# --------------------------------------------------------------- upload policy
UPLOAD_DIR = _path("UPLOAD_DIR", WEBSITE_DIR / "static" / "uploads")
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 8)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024  # Flask rejects anything larger with 413
MAX_PIXELS = _int("MAX_IMAGE_PIXELS", 40_000_000)  # decompression-bomb guard
MIN_SIDE = _int("MIN_IMAGE_SIDE", 64)
UPLOAD_TTL_SECONDS = _int("UPLOAD_TTL_SECONDS", 60 * 60)
