"""
Common model utilities for the ML pipeline.

This module provides functions to load models and metadata.
"""
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple
from .config import CONFIG, cfg_path

def load_model(model_path: Path):
    """Load a pickled model."""
    with open(model_path, "rb") as f:
        return pickle.load(f)

def load_metadata(metadata_path: Path) -> Dict:
    """Load metadata from JSON file."""
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}

def get_model_and_metadata(run_id: str = None) -> Tuple:
    """Get model and metadata paths based on run_id or latest run."""
    if run_id:
        runs_dir = cfg_path(CONFIG, "paths.runs_dir", "runs")
        run_dir = runs_dir / run_id
        model_path = run_dir / "model.pkl"
        metadata_path = run_dir / "metadata.json"
    else:
        # Fallback to legacy paths
        model_path = cfg_path(CONFIG, "paths.model", "model.pkl")
        metadata_path = cfg_path(CONFIG, "paths.metadata", "metadata.json")

    model = load_model(model_path)
    metadata = load_metadata(metadata_path)
    return model, metadata, model_path, metadata_path