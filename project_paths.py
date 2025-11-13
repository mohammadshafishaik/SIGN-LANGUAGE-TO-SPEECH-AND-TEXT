"""
Project paths helper to avoid hardcoded absolute directories.

Assumes this file lives at the repository root. All other modules
should import from here instead of embedding user-specific paths.
"""
from pathlib import Path

# Repository root (folder containing this file)
ROOT_DIR: Path = Path(__file__).resolve().parent

# Common subdirectories
DATASET_DIR: Path = ROOT_DIR / "dataset"
DATASETS_LOCAL_DIR: Path = ROOT_DIR / "datasets_local.nosync"
CHECKPOINTS_DIR: Path = ROOT_DIR / "checkpoints"
LOGS_DIR: Path = ROOT_DIR / "logs"
RESULTS_DIR: Path = ROOT_DIR / "results"
MODELS_DIR: Path = ROOT_DIR / "models"


def ensure_default_dirs() -> None:
    """Create frequently used directories if they don't exist."""
    for d in [DATASET_DIR, CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ROOT_DIR",
    "DATASET_DIR",
    "DATASETS_LOCAL_DIR",
    "CHECKPOINTS_DIR",
    "LOGS_DIR",
    "RESULTS_DIR",
    "MODELS_DIR",
    "ensure_default_dirs",
]
