import os
import json
from typing import Optional, Dict

# Shared directory for cross-service artifacts (mounted via docker-compose)
SHARED_DIR = os.environ.get("SHARED_DIR", "./shared")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _example_path(model_type: str) -> str:
    _ensure_dir(SHARED_DIR)
    return os.path.join(SHARED_DIR, f"{model_type}_example.json")


def write_example(model_type: str, example: Dict) -> None:
    """Persist a sample request/example for a given model type to the shared dir."""
    path = _example_path(model_type)
    with open(path, "w") as f:
        json.dump(example, f, indent=2)


def read_example(model_type: str) -> Optional[Dict]:
    """Read a previously stored example for a given model type, if any."""
    path = _example_path(model_type)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

