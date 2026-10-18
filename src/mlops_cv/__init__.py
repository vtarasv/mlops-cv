import os
from pathlib import Path

__version__ = "0.1.0"

# Repo-local ultralytics settings.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / ".ultralytics"))
