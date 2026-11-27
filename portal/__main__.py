"""Run the portal: python -m portal [--host 127.0.0.1] [--port 8000]"""
from __future__ import annotations

import argparse
import logging
import os


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m portal", description="aitesteval web portal")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 inside a container")
    # Hosts such as Render inject $PORT; an explicit --port still wins.
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT") or 8000),
                    help="default: $PORT if set, else 8000")
    ap.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    uvicorn.run("portal.app:get_app", factory=True, host=args.host, port=args.port, reload=args.reload,
                proxy_headers=True)


if __name__ == "__main__":
    main()
