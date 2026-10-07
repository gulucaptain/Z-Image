"""Generate images through the shared application service."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zimage_app.cli import main

if __name__ == "__main__":
    main()
