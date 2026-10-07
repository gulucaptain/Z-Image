"""Unified Gradio entry point: python app.py --port 7860."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from zimage_app.settings import Settings, browse_directories, prepare_temp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    prepare_temp(settings)
    from zimage_app.ui import build_app

    demo = build_app(settings)
    demo.queue(max_size=32, default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share,
        allowed_paths=[str(path) for path in browse_directories(settings).values()], show_error=True,
    )


if __name__ == "__main__":
    main()
