"""YAML configuration loading and project-root resolution."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Repository root (directory containing `configs/`)."""
    env = os.environ.get("RELIABILITY_LAB_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir():
            return parent
    return Path.cwd()


def load_env() -> None:
    """Load a .env file if python-dotenv is installed. Never raises."""
    try:
        from dotenv import load_dotenv

        load_dotenv(project_root() / ".env")
    except Exception:
        pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_models_config(path: str | Path = "configs/models.yaml") -> dict[str, Any]:
    return load_yaml(path)


def load_experiments_config(path: str | Path = "configs/experiments.yaml") -> dict[str, Any]:
    return load_yaml(path)


def load_evaluation_config(path: str | Path = "configs/evaluation.yaml") -> dict[str, Any]:
    return load_yaml(path)
