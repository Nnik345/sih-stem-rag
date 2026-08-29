#!/usr/bin/env python3
"""Launch the local RAG visualizer on 127.0.0.1.

    python scripts/run_visualizer.py
    python scripts/run_visualizer.py --port 8000 --no-browser
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Reload on Python source changes.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window.")
    args = parser.parse_args()

    dist = PROJECT_ROOT / "frontend" / "dist" / "index.html"
    if not dist.is_file():
        print("Frontend production build not found at frontend/dist/.", file=sys.stderr)
        print("Build it locally, then re-run this launcher:", file=sys.stderr)
        print(file=sys.stderr)
        print("  cd frontend", file=sys.stderr)
        print("  npm install", file=sys.stderr)
        print("  npm run build", file=sys.stderr)
        print("  cd ..", file=sys.stderr)
        print("  python scripts/run_visualizer.py", file=sys.stderr)
        return 2

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        webbrowser.open(url)

    uvicorn.run(
        "rag_visualizer.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
