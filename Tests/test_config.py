"""Environment-variable configuration contract.

The site is configured entirely through the environment, with working defaults. These
tests pin the behaviour that matters operationally: a missing or malformed value must
never take the app down, and relative paths must resolve against the project root.
"""
from __future__ import annotations

from pathlib import Path

from Website import config


def test_defaults_match_the_documented_values() -> None:
    assert config.MAX_UPLOAD_BYTES == config.MAX_UPLOAD_MB * 1024 * 1024
    assert config.MAX_UPLOAD_MB > 0
    assert config.MAX_PIXELS > 0
    assert config.MIN_SIDE > 0
    assert config.UPLOAD_TTL_SECONDS > 0
    assert config.UPLOAD_DIR.is_absolute()
    assert config.MODELS_DIR.name == "Models"


def test_integer_override_is_used(monkeypatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_MB", "3")
    assert config._int("MAX_UPLOAD_MB", 8) == 3


def test_malformed_integer_falls_back_instead_of_crashing(monkeypatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_MB", "eight")
    assert config._int("MAX_UPLOAD_MB", 8) == 8
    monkeypatch.setenv("MAX_UPLOAD_MB", "-5")
    assert config._int("MAX_UPLOAD_MB", 8) == 8


def test_relative_path_resolves_against_project_root(monkeypatch) -> None:
    monkeypatch.setenv("MODELS_DIR", "Models")
    assert config._path("MODELS_DIR", Path("/tmp")) == config.PROJECT_ROOT / "Models"


def test_absolute_path_is_left_alone(monkeypatch) -> None:
    absolute = Path(config.PROJECT_ROOT) / "elsewhere"
    monkeypatch.setenv("MODELS_DIR", str(absolute))
    assert config._path("MODELS_DIR", Path("/tmp")) == absolute


def test_boolean_flags_accept_common_spellings(monkeypatch) -> None:
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("FLASK_DEBUG", value)
        assert config._bool("FLASK_DEBUG") is True
    monkeypatch.setenv("FLASK_DEBUG", "0")
    assert config._bool("FLASK_DEBUG") is False
