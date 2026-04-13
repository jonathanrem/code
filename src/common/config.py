"""
Common configuration utilities for the ML pipeline.

This module provides functions to load and access configuration from config.yaml.
"""
from pathlib import Path

def load_config(path: Path) -> dict:
    """Load configuration from a YAML file."""
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"warning: failed to load config: {exc}")
        return {}

def cfg_get(cfg: dict, key: str, default):
    """Get a value from the configuration dictionary using dot notation."""
    node = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

def cfg_path(cfg: dict, key: str, default: str) -> Path:
    """Get a path from the configuration, resolving relative to BASE_DIR."""
    value = cfg_get(cfg, key, default)
    path_value = Path(value)
    return path_value if path_value.is_absolute() else BASE_DIR / path_value

# Global configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"
CONFIG = load_config(CONFIG_PATH)