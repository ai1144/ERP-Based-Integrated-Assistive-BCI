"""Local configuration for the anonymized research-code copy."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("BCI_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()
EEG_DATA_DIR = Path(os.environ.get("BCI_EEG_DATA_DIR", str(DATA_DIR / "eeg"))).expanduser()
BACKEND_URL = os.environ.get("BCI_BACKEND_URL", "http://127.0.0.1:8000")
