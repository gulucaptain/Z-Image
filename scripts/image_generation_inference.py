"""Compatibility entry point for the unified generation service."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zimage_app.cli import main

if __name__ == "__main__":
    main(default_save_steps=False)
