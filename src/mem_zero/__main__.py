from __future__ import annotations

import sys

import uvicorn

from .config import Config


def main() -> None:
    config = Config.from_env()
    uvicorn.run(
        "mem_zero.server:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
